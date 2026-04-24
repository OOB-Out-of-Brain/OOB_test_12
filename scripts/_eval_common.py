"""테스트 스크립트 공통 유틸: 결과 분류(버킷팅) + 3-panel figure 저장.

방향성 9-버킷 체계 (GT → Pred):
  correct/normal
  correct/ischemic
  correct/hemorrhagic
  wrong/normal_to_ischemic       (정상인데 허혈로 예측 → 오탐)
  wrong/normal_to_hemorrhagic    (정상인데 출혈로 예측 → 오탐)
  wrong/ischemic_to_normal       (허혈인데 정상으로 예측 → 놓침)
  wrong/hemorrhagic_to_normal    (출혈인데 정상으로 예측 → 놓침)
  wrong/ischemic_to_hemorrhagic  (허혈인데 출혈로 예측 → 병변 혼동)
  wrong/hemorrhagic_to_ischemic  (출혈인데 허혈로 예측 → 병변 혼동)

ischemic GT 가 없는 데이터셋(brain_test 일부·CQ500)에선 has_ischemic_gt=False 로 호출:
  GT=ischemic 인 경우가 원천적으로 존재하지 않으므로 ischemic_to_* 버킷은 사용되지 않음.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.visualization import _build_figure


_ALL_BUCKETS = [
    "correct/normal",
    "correct/ischemic",
    "correct/hemorrhagic",
    "wrong/normal_to_ischemic",
    "wrong/normal_to_hemorrhagic",
    "wrong/ischemic_to_normal",
    "wrong/hemorrhagic_to_normal",
    "wrong/ischemic_to_hemorrhagic",
    "wrong/hemorrhagic_to_ischemic",
]


def classify_bucket(gt_name: str, pred_name: str, has_ischemic_gt: bool = True) -> str:
    """GT/Pred 조합을 버킷 경로(상대)로 매핑."""
    if gt_name == pred_name:
        return f"correct/{gt_name}"
    # 방향성 보존: {gt}_to_{pred}
    return f"wrong/{gt_name}_to_{pred_name}"


def save_3panel(orig_np, result, out_path: Path, gt_name: str,
                dpi: int = 100, suptitle_prefix: str = ""):
    """원본 + (분류 확률 + 뇌 영역 구성) + 병변 overlay 3-panel figure 저장."""
    if orig_np is None or result is None:
        return
    fig = _build_figure(orig_np, result, alpha=0.45)
    mark = "O" if gt_name == result.class_name else "X"
    prefix = f"{suptitle_prefix}  " if suptitle_prefix else ""
    fig.suptitle(
        f"{prefix}GT={gt_name}  ->  Pred={result.class_name.upper()} "
        f"({result.confidence:.1%}) [{mark}]",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def ensure_bucket_dirs(root: Path, include_ischemic_correct: bool = True):
    """모든 버킷 폴더를 미리 만들어둔다 (빈 폴더도 생성).
    include_ischemic_correct=False 면 GT=ischemic 관련 버킷은 만들지 않음.
    """
    for sub in _ALL_BUCKETS:
        if not include_ischemic_correct and (sub == "correct/ischemic" or sub.startswith("wrong/ischemic_to_")):
            continue
        (root / sub).mkdir(parents=True, exist_ok=True)
