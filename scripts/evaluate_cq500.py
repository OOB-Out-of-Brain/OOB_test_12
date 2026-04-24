"""CQ500 3-class 평가.

주의: CQ500 GT는 ICH(출혈) 라벨만 존재. ischemic 라벨이 없다.
따라서 3-class 모델을 쓰되 스캔 단위 GT는 hemorrhagic vs non-hemorrhagic(=normal+ischemic)으로 환산.

출력 (results/cq500_3class/):
  - summary.csv           (scan, gt_hem, pred_cls, pred_hem, max_conf, max_hem_pct, max_isch_pct)
  - metrics.txt           (hemorrhagic 이진 + 3-class 분포)
  - correct/{normal,hemorrhagic}/
  - wrong/{normal_to_ischemic, normal_to_hemorrhagic,
           hemorrhagic_to_normal, hemorrhagic_to_ischemic}/
"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom

from inference.pipeline import StrokePipeline
from inference.visualization import _build_figure
from scripts._eval_common import classify_bucket, save_3panel, ensure_bucket_dirs


CQ500_DIR = Path("./data/raw/cq500")
OUT_DIR = Path("./results/cq500_3class")


def save_scan_panel(orig_np, result, out_path: Path, gt_tag: str, dpi: int = 100):
    """대표 슬라이스에 3-panel figure (원본 + 확률바 + overlay) 저장."""
    if orig_np is None or result is None:
        return
    fig = _build_figure(orig_np, result, alpha=0.45)
    mark = "✓" if (gt_tag == "hem" and result.class_name == "hemorrhagic") or \
                  (gt_tag == "nonhem" and result.class_name != "hemorrhagic") else "✗"
    fig.suptitle(
        f"GT={gt_tag}  →  Pred={result.class_name.upper()} ({result.confidence:.1%}) {mark}",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def apply_brain_window(hu_arr: np.ndarray, center: int = 40, width: int = 80) -> np.ndarray:
    lo, hi = center - width / 2, center + width / 2
    x = np.clip(hu_arr, lo, hi)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def dicom_to_png(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = arr * slope + intercept
    return apply_brain_window(hu)


def parse_gt(reads_csv: Path) -> dict:
    gt = {}
    if not reads_csv.exists():
        return gt
    with open(reads_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name")
            if not name:
                continue
            cols = ["R1:ICH", "R2:ICH", "R3:ICH"]
            votes = [int(row.get(c, 0) or 0) for c in cols if c in row]
            if not votes:
                continue
            gt[name] = 1 if sum(votes) >= 2 else 0
    return gt


def evaluate_scan(pipe: StrokePipeline, scan_dir: Path,
                  max_slices: int = 12) -> dict:
    """한 스캔 평가. 중앙 슬라이스 ~12개 → 스캔 단위 요약.

    스캔 라벨 결정: 슬라이스 중 하나라도 hemorrhagic → scan=hemorrhagic.
    그 외: ischemic이 우세하면 ischemic, 아니면 normal.
    """
    dcm_files = sorted(scan_dir.rglob("*.dcm"))
    if not dcm_files:
        return {"pred_cls": "unknown", "n_slices": 0,
                "max_conf": 0, "max_hem_pct": 0, "max_isch_pct": 0}

    n = len(dcm_files)
    if n > max_slices:
        start = (n - max_slices) // 2
        sel = dcm_files[start:start + max_slices]
    else:
        sel = dcm_files

    per_slice_pred = []
    max_conf_hem = 0.0
    max_hem_pct = 0.0
    max_isch_pct = 0.0
    # 클래스별 가장 확신 높은 슬라이스의 (이미지, result) 보존
    best_by_cls = {"normal": None, "ischemic": None, "hemorrhagic": None}
    best_conf_by_cls = {"normal": 0.0, "ischemic": 0.0, "hemorrhagic": 0.0}

    for dcm in sel:
        try:
            u8 = dicom_to_png(dcm)
        except Exception:
            continue
        rgb = np.stack([u8] * 3, axis=-1)
        r = pipe.run(rgb)
        per_slice_pred.append(r.class_name)
        if r.hemorrhagic_area_pct > max_hem_pct:
            max_hem_pct = r.hemorrhagic_area_pct
        if r.ischemic_area_pct > max_isch_pct:
            max_isch_pct = r.ischemic_area_pct
        if r.class_name == "hemorrhagic" and r.confidence > max_conf_hem:
            max_conf_hem = r.confidence
        if r.confidence > best_conf_by_cls[r.class_name]:
            best_conf_by_cls[r.class_name] = r.confidence
            best_by_cls[r.class_name] = (rgb, r)

    # 스캔 단위 결정 규칙
    if "hemorrhagic" in per_slice_pred:
        pred_cls = "hemorrhagic"
    elif per_slice_pred.count("ischemic") >= max(2, len(per_slice_pred) // 4):
        pred_cls = "ischemic"
    else:
        pred_cls = "normal"

    # scan 대표 슬라이스: pred_cls 대표 우선, 없으면 hem > isch > normal 순
    best_pair = (best_by_cls[pred_cls]
                 or best_by_cls["hemorrhagic"]
                 or best_by_cls["ischemic"]
                 or best_by_cls["normal"])

    return {
        "pred_cls": pred_cls,
        "n_slices": len(sel),
        "max_conf": max_conf_hem,
        "max_hem_pct": max_hem_pct,
        "max_isch_pct": max_isch_pct,
        "best_pair": best_pair,  # (rgb_image, PipelineResult) or None
    }


def save_error_sample(scan_name: str, overlay, out_dir: Path):
    if overlay is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_dir / f"{scan_name}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CQ500_DIR.exists() or not any(CQ500_DIR.iterdir()):
        print(f"CQ500 데이터 없음: {CQ500_DIR}")
        print(f"   scripts/download_cq500.py 실행 필요")
        sys.exit(1)

    gt_map = parse_gt(CQ500_DIR / "reads.csv")
    if not gt_map:
        print("reads.csv 없거나 파싱 실패")
        sys.exit(1)
    print(f"GT 라벨: {len(gt_map)}개 스캔")

    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt="./checkpoints/segmentor/best_segmentor.pth"
            if Path("./checkpoints/segmentor/best_segmentor.pth").exists()
            else None,
    )

    # 스캔 폴더 찾기: 최대 2단계(qct01/CQ500CT101 CQ500CT101) 까지 내려가서 CQ500CT* 디렉토리 수집.
    # reads.csv 키가 "CQ500-CT-101" 이라 폴더명 "CQ500CT101 CQ500CT101" → "CQ500-CT-101" 로 정규화.
    import re as _re
    def normalize_scan_name(folder_name: str) -> str:
        # "CQ500CT101 CQ500CT101" → "CQ500CT101"
        first = folder_name.split()[0] if folder_name else folder_name
        # "CQ500CT101" → "CQ500-CT-101"
        m = _re.match(r"CQ500CT(\d+)", first)
        if m:
            return f"CQ500-CT-{m.group(1)}"
        return first

    scan_dirs = []
    for p in CQ500_DIR.rglob("CQ500CT*"):
        if p.is_dir():
            scan_dirs.append(p)
    scan_dirs.sort()
    print(f"스캔 폴더: {len(scan_dirs)}개 (재귀 수집)")

    # 버킷 폴더 미리 생성 (CQ500 은 ischemic GT 없음 → correct/ischemic 만들지 않음)
    ensure_bucket_dirs(OUT_DIR, include_ischemic_correct=False)

    summary_csv = OUT_DIR / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan", "gt_hem", "pred_cls", "pred_hem",
                    "max_conf", "max_hem_pct", "max_isch_pct", "n_slices"])

        y_true, y_pred = [], []
        per_class_pred = {"normal": 0, "ischemic": 0, "hemorrhagic": 0}
        unmatched = 0

        for i, scan_dir in enumerate(scan_dirs, 1):
            name = normalize_scan_name(scan_dir.name)
            if name not in gt_map:
                unmatched += 1
                continue
            gt_hem = gt_map[name]
            r = evaluate_scan(pipe, scan_dir)
            pred_cls = r["pred_cls"]
            pred_hem = 1 if pred_cls == "hemorrhagic" else 0
            per_class_pred[pred_cls] = per_class_pred.get(pred_cls, 0) + 1

            y_true.append(gt_hem)
            y_pred.append(pred_hem)

            w.writerow([name, gt_hem, pred_cls, pred_hem,
                        f"{r['max_conf']:.3f}",
                        f"{r['max_hem_pct']:.2f}", f"{r['max_isch_pct']:.2f}",
                        r["n_slices"]])

            # GT 변환: CQ500 은 hemorrhagic / non-hemorrhagic 이진 → normal(non-hem) / hemorrhagic
            gt_name = "hemorrhagic" if gt_hem == 1 else "normal"
            bucket = classify_bucket(gt_name, pred_cls, has_ischemic_gt=False)

            # 대표 슬라이스 3-panel figure 저장
            pair = r.get("best_pair")
            if pair is not None:
                rgb, res = pair
                save_3panel(
                    rgb, res,
                    OUT_DIR / bucket / f"{name}.png",
                    gt_name, dpi=100,
                )
            else:
                # 슬라이스 전부 처리 실패 — 빈 placeholder 라도 남겨서 누락 표시
                placeholder = OUT_DIR / bucket / f"{name}_NO_DICOM.txt"
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.write_text(f"scan={name}\npred_cls={pred_cls}\ngt_hem={gt_hem}\n")

            if i % 10 == 0:
                print(f"  {i}/{len(scan_dirs)} 처리됨...")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = len(y_true)

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    acc  = (tp + tn) / max(total, 1)

    report = f"""\
CQ500 3-class 평가 리포트 (총 {total} 스캔)
=============================================
※ CQ500 GT는 ICH(출혈) 라벨만 존재. 아래 sens/spec은 hemorrhagic vs non-hemorrhagic 이진 지표.

[3-class 예측 분포 (스캔 단위)]
  normal       : {per_class_pred.get("normal", 0):>5}
  ischemic     : {per_class_pred.get("ischemic", 0):>5}
  hemorrhagic  : {per_class_pred.get("hemorrhagic", 0):>5}

[Hemorrhagic 이진 평가]
Confusion matrix:
                예측 non-hem     예측 hemorrhagic
실제 non-hem     {tn:>6}         {fp:>6} ← 오탐(FP)
실제 hemorrhagic {fn:>6}         {tp:>6} ← 누락(FN)

Sensitivity : {sens:.4f}  (출혈을 출혈로)
Specificity : {spec:.4f}
PPV         : {ppv:.4f}
NPV         : {npv:.4f}
Accuracy    : {acc:.4f}

Normal 오탐율(FP rate)     : {fp / max(tn + fp, 1):.4f}
Hemorrhagic 누락율(FN rate): {fn / max(tp + fn, 1):.4f}
"""
    print(report)
    (OUT_DIR / "metrics.txt").write_text(report)
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
