"""
데이터셋 준비 스크립트.

  tekno21 : HuggingFace에서 자동 캐시 (학습 시 자동 다운로드)
  AISD    : 수동 다운로드 안내 + 폴더 구조 검증

실행:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import zipfile
import shutil


AISD_RAW = Path("./data/raw/aisd")
TEKNO21_CACHE = Path("./data/raw/tekno21")


def check_aisd():
    images_dir = AISD_RAW / "images"
    masks_dir = AISD_RAW / "masks"

    print("\n── AISD 데이터셋 확인 ────────────────────────────────────────")
    if images_dir.exists() and masks_dir.exists():
        n_img = len(list(images_dir.glob("*.png")))
        n_mask = len(list(masks_dir.glob("*.png")))
        print(f"  이미지: {n_img}개  마스크: {n_mask}개")
        if n_img > 0 and n_mask > 0:
            print("  ✅ AISD 데이터셋 준비 완료")
            return True
    print("""
  ❌ AISD 데이터셋이 없습니다. 아래 절차로 수동 다운로드하세요:

  1. GitHub에서 요청:
     https://github.com/GriffinLiang/AISD
     → "Dataset Download" 링크 또는 이메일 요청

  2. 다운로드 후 압축 해제:
     AISD.zip → data/raw/aisd/

  3. 폴더 구조 확인:
     data/raw/aisd/
       images/   ← CT PNG 파일들 (xxx.png)
       masks/    ← 분할 마스크 PNG 파일들 (xxx.png)

  4. 이 스크립트 다시 실행하여 확인
""")
    return False


def check_tekno21():
    print("\n── tekno21 데이터셋 확인 ────────────────────────────────────")
    print("  HuggingFace에서 학습 시 자동 다운로드됩니다.")
    print("  (BTX24/tekno21-brain-stroke-dataset-multi)")
    try:
        from datasets import load_dataset
        print("  캐시 확인 중...")
        ds = load_dataset(
            "BTX24/tekno21-brain-stroke-dataset-multi",
            split="train",
            cache_dir=str(TEKNO21_CACHE),
        )
        print(f"  ✅ tekno21 전체: {len(ds)}개 (train/val은 학습 시 자동 분리)")
        print(f"  피처: {list(ds.features.keys())}")
        label_key = next(k for k in ["label", "labels", "category", "class"]
                         if k in ds.features)
        import numpy as np
        labels = [int(ds[i][label_key]) for i in range(len(ds))]
        counts = np.bincount(labels)
        for i, c in enumerate(counts):
            from data.classifier_dataset import CLASS_NAMES
            print(f"    {CLASS_NAMES[i]}: {c}개")
        return True
    except Exception as e:
        print(f"  ⚠️  다운로드 중 오류: {e}")
        return False


def main():
    print("=" * 60)
    print("  뇌졸중 AI 프로젝트 데이터셋 준비")
    print("=" * 60)

    ok_tekno = check_tekno21()

    ok_aisd = check_aisd()

    print("\n── 요약 ──────────────────────────────────────────────────────")
    print(f"  tekno21 (분류): {'✅' if ok_tekno else '❌'}")
    print(f"  AISD (분할)   : {'✅' if ok_aisd else '❌ 수동 다운로드 필요'}")

    if ok_tekno:
        print("\n분류 학습 가능:  python training/train_classifier.py")
    if ok_aisd:
        print("분할 학습 가능:  python training/train_segmentor.py")


if __name__ == "__main__":
    main()
