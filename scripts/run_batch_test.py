"""폴더 내 이미지를 3-class 배치 추론 + 결과 저장.

사용법:
    python scripts/run_batch_test_3class.py --input-dir /Users/bari/Downloads/brain_test \
                                             --output-dir results/brain_test_3class/

파일 이름에서 GT를 추측해 비교 출력 (선택):
    - "nomal"/"normal"  → normal
    - "iskemi"/"ischem" → ischemic
    - 그 외 (EDH/ICH/SAH/SDH 등) → hemorrhagic
  --no-gt-from-name 옵션을 주면 비교하지 않고 예측만 출력.
"""

import argparse, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from inference.pipeline import StrokePipeline


def infer_gt_from_name(name: str):
    low = name.lower()
    if re.search(r"nomal|normal", low):
        return "normal"
    if re.search(r"iskemi|ischem|isch", low):
        return "ischemic"
    if re.search(r"edh|ich|sah|sdh|ihv|hemorr|bleed|출혈", low):
        return "hemorrhagic"
    return None


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("모델 로딩...")
    pipe = StrokePipeline(
        classifier_ckpt=args.cls_ckpt,
        segmentor_ckpt=args.seg_ckpt if Path(args.seg_ckpt).exists() else None,
    )
    if pipe.segmentor is None:
        print(f"  (세그멘터 ckpt 없음 → 분류만 진행)")

    images = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        and not p.name.startswith("result_")
    ])
    if not images:
        print(f"이미지 없음: {in_dir}")
        return
    print(f"이미지 {len(images)}개 추론 → {out_dir}\n")

    header = f"{'파일':<24} {'예측':<12} {'신뢰도':>7}  {'확률(N/I/H)':<24} {'lesion px':<20}"
    if not args.no_gt_from_name:
        header = f"{'GT':<12} " + header
    print(header)
    print("-" * len(header))

    summary = {"normal": 0, "ischemic": 0, "hemorrhagic": 0}
    cm = {}  # (gt, pred) → count
    for img_path in images:
        r = pipe.run(img_path)
        summary[r.class_name] = summary.get(r.class_name, 0) + 1

        probs_str = " / ".join(
            f"{r.class_probs.get(c, 0):.2f}"
            for c in ["normal", "ischemic", "hemorrhagic"]
        )
        lesion_str = ""
        if r.ischemic_area_px:
            lesion_str += f"isch={r.ischemic_area_px}px "
        if r.hemorrhagic_area_px:
            lesion_str += f"hem={r.hemorrhagic_area_px}px"

        gt = None
        prefix = ""
        if not args.no_gt_from_name:
            gt = infer_gt_from_name(img_path.stem)
            prefix = f"{(gt or '-'):<12} "
            if gt is not None:
                cm[(gt, r.class_name)] = cm.get((gt, r.class_name), 0) + 1

        print(f"{prefix}{img_path.name:<24} {r.class_name:<12} "
              f"{r.confidence:>6.1%}  {probs_str:<24} {lesion_str}")

        # 시각화 저장
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        orig = np.array(Image.open(img_path).convert("RGB"))
        axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(r.overlay_image if r.overlay_image is not None else orig)
        color = {"normal": "green", "ischemic": "blue", "hemorrhagic": "red"}.get(r.class_name, "black")
        title = f"{r.class_name.upper()} ({r.confidence:.1%})"
        if gt is not None:
            tag = "✓" if gt == r.class_name else "✗"
            title = f"GT={gt} | {title} {tag}"
        axes[1].set_title(title, color=color)
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{img_path.stem}_result.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\n" + "=" * 70)
    print(f"  총 {len(images)}개 | normal {summary.get('normal', 0)} | "
          f"ischemic {summary.get('ischemic', 0)} | hemorrhagic {summary.get('hemorrhagic', 0)}")

    if cm:
        print("\n  라벨(파일명 기반) vs 예측:")
        classes = ["normal", "ischemic", "hemorrhagic"]
        header = "  GT\\Pred    " + "  ".join(f"{c:>12}" for c in classes)
        print(header)
        total_correct = 0
        total_seen = 0
        for g in classes:
            row = f"  {g:<10} " + "  ".join(
                f"{cm.get((g, p), 0):>12d}" for p in classes
            )
            print(row)
            for p in classes:
                total_seen += cm.get((g, p), 0)
            total_correct += cm.get((g, g), 0)
        if total_seen:
            print(f"\n  Accuracy (라벨 인식된 {total_seen}개): {total_correct / total_seen:.4f}")

    print(f"  저장: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cls-ckpt", default="./checkpoints/classifier/best_classifier.pth")
    p.add_argument("--seg-ckpt", default="./checkpoints/segmentor/best_segmentor.pth")
    p.add_argument("--no-gt-from-name", action="store_true",
                   help="파일명에서 GT 추측 비활성화")
    args = p.parse_args()
    main(args)
