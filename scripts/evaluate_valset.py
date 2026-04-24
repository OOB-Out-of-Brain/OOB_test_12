"""Val set 3-class 상세 평가 (FP/FN/오분류 리포트).

학습 중 backprop에 사용되지 않은 검증 세트 전체(≈2361장)를 환자 단위 split 기준으로 평가.
3-class confusion matrix + per-class precision/recall + 오분류 샘플 저장.

출력 (results/valset_3class/):
  - summary.csv
  - metrics.txt
  - errors/gt_X_pred_Y/ 폴더들 (잘못 예측된 샘플 시각화)
"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.combined_dataset import build_combined_dataloaders, CLASS_NAMES
from inference.pipeline import StrokePipeline
from inference.visualization import _build_figure
from scripts._eval_common import classify_bucket, save_3panel, ensure_bucket_dirs


OUT_DIR = Path("./results/valset_3class")


def save_dual_panel(orig_np, result, out_path: Path, gt_name: str, dpi: int = 100):
    """brain_test 스타일 3-panel figure (원본 + 3-class 확률바 + 병변 overlay) + GT/Pred 표시."""
    fig = _build_figure(orig_np, result, alpha=0.45)
    mark = "✓" if gt_name == result.class_name else "✗"
    fig.suptitle(
        f"GT={gt_name}  →  Pred={result.class_name.upper()} ({result.confidence:.1%}) {mark}",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 버킷 폴더 미리 생성 (val set 은 ischemic GT 있으니 correct/ischemic 포함)
    ensure_bucket_dirs(OUT_DIR, include_ischemic_correct=True)

    print("Val set 로딩 (3-class)...")
    _, val_loader, _ = build_combined_dataloaders(
        ct_root="./data/raw/ct_hemorrhage/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
        tekno21_cache="./data/raw/tekno21",
        image_size=224, batch_size=1, num_workers=0,
    )
    val_ds = val_loader.dataset
    samples = val_ds.samples  # [(source, ref, label), ...]
    print(f"총 {len(samples)} 샘플 평가\n")

    seg_ckpt = Path("./checkpoints/segmentor/best_segmentor.pth")
    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt=str(seg_ckpt) if seg_ckpt.exists() else None,
    )

    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    summary_rows = []

    hf = val_ds.hf
    for i, (source, ref, gt) in enumerate(samples):
        if source in ("ct", "bhsd"):
            img = np.array(Image.open(ref).convert("RGB"))
            name = Path(ref).name
        else:
            item = hf[ref]
            im = item["image"]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            img = np.array(im.convert("RGB"))
            name = f"tk_{ref}.png"

        r = pipe.run(img)
        pred = r.class_idx
        cm[gt, pred] += 1

        summary_rows.append([
            source, name,
            CLASS_NAMES[gt], CLASS_NAMES[pred],
            f"{r.confidence:.3f}",
            f"{r.class_probs.get('normal', 0):.3f}",
            f"{r.class_probs.get('ischemic', 0):.3f}",
            f"{r.class_probs.get('hemorrhagic', 0):.3f}",
            f"{r.ischemic_area_pct:.2f}",
            f"{r.hemorrhagic_area_pct:.2f}",
        ])

        # 버킷(정상/출혈/허혈/오탐/오류/혼동)별 폴더 배치. tekno21 val 은 ischemic GT 있음.
        bucket = classify_bucket(CLASS_NAMES[gt], CLASS_NAMES[pred], has_ischemic_gt=True)
        save_3panel(
            img, r,
            OUT_DIR / bucket / f"{source}_{i:05d}_{Path(name).stem}.png",
            CLASS_NAMES[gt], dpi=100,
        )

        if (i + 1) % 200 == 0:
            acc_so_far = np.trace(cm) / max(cm.sum(), 1)
            print(f"  {i+1}/{len(samples)}  acc={acc_so_far:.4f}")

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "file", "gt", "pred", "confidence",
                    "p_normal", "p_ischemic", "p_hemorrhagic",
                    "ischemic_pct", "hemorrhagic_pct"])
        w.writerows(summary_rows)

    # 메트릭
    total = int(cm.sum())
    acc = np.trace(cm) / max(total, 1)

    per_class = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        support = int(cm[c, :].sum())
        per_class.append((CLASS_NAMES[c], precision, recall, support))

    # 보고서
    header = "GT\\Pred     " + "  ".join(f"{n:>12}" for n in CLASS_NAMES)
    cm_lines = [header]
    for c in range(n_classes):
        row = f"{CLASS_NAMES[c]:<10}  " + "  ".join(f"{cm[c, p]:>12d}" for p in range(n_classes))
        cm_lines.append(row)

    pc_lines = [f"{'class':<12} {'precision':>10} {'recall':>10} {'support':>10}"]
    for name, pr, rc, sup in per_class:
        pc_lines.append(f"{name:<12} {pr:>10.4f} {rc:>10.4f} {sup:>10d}")

    report = f"""\
Val set 3-class 평가 (총 {total} 샘플)
=============================================

Confusion matrix (rows=GT, cols=Pred)
{chr(10).join(cm_lines)}

Per-class metrics
{chr(10).join(pc_lines)}

Overall Accuracy : {acc:.4f}

결과 버킷 저장 경로 (방향성 9-버킷):
  {OUT_DIR}/correct/{{normal,ischemic,hemorrhagic}}/         (정답)
  {OUT_DIR}/wrong/normal_to_ischemic/                         (정상→허혈 오탐)
  {OUT_DIR}/wrong/normal_to_hemorrhagic/                      (정상→출혈 오탐)
  {OUT_DIR}/wrong/ischemic_to_normal/                         (허혈→정상 놓침)
  {OUT_DIR}/wrong/hemorrhagic_to_normal/                      (출혈→정상 놓침)
  {OUT_DIR}/wrong/ischemic_to_hemorrhagic/                    (허혈→출혈 혼동)
  {OUT_DIR}/wrong/hemorrhagic_to_ischemic/                    (출혈→허혈 혼동)
"""
    print("\n" + report)
    (OUT_DIR / "metrics.txt").write_text(report)
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
