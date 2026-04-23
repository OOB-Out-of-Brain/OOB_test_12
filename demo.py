"""단일 CT 이미지 3-class 추론 데모.

사용법:
    python demo.py --image path/to/ct.png
    python demo.py --image path/to/ct.png --output results/result.png
    python demo.py --image path/to/ct.png --cls_ckpt checkpoints/classifier/best_classifier.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from inference.pipeline import StrokePipeline
from inference.visualization import save_visualization


def main(args):
    cls_ckpt = args.cls_ckpt or "checkpoints/classifier/best_classifier.pth"
    seg_ckpt = args.seg_ckpt or "checkpoints/segmentor/best_segmentor.pth"

    if not Path(cls_ckpt).exists():
        print(f"❌ classifier 체크포인트 없음: {cls_ckpt}")
        print("   먼저 학습 실행: python training/train_classifier.py --epochs 50")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"❌ 이미지 없음: {args.image}")
        sys.exit(1)

    print(f"\n이미지: {args.image}")
    print("모델 로딩 중...")
    pipeline = StrokePipeline(
        classifier_ckpt=cls_ckpt,
        segmentor_ckpt=seg_ckpt if Path(seg_ckpt).exists() else None,
    )
    if pipeline.segmentor is None:
        print(f"  (segmentor ckpt 없음 → 분류만 수행, overlay 생략)")

    print("추론 실행 중...\n")
    result = pipeline.run(args.image)

    print("=" * 50)
    print(result)
    print("=" * 50)

    output_path = args.output or f"results/{Path(args.image).stem}_result.png"
    orig_np = np.array(Image.open(args.image).convert("RGB"))
    save_visualization(orig_np, result, output_path)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뇌 CT 3-class 분석 데모")
    parser.add_argument("--image", required=True, help="CT 이미지 경로")
    parser.add_argument("--output", default=None, help="결과 PNG 저장 경로")
    parser.add_argument("--cls_ckpt", default=None, help="분류 모델 체크포인트")
    parser.add_argument("--seg_ckpt", default=None, help="분할 모델 체크포인트")
    args = parser.parse_args()
    main(args)
