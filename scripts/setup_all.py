"""원샷 데이터 셋업: 학습용 4종 + (선택) CQ500 외부 테스트셋.

신규 사용자가 레포를 clone 한 후 학습 준비를 끝내기 위한 진입점.
체크포인트는 학습 완료 후 생성되므로 이 스크립트 대상 아님 —
`training/train_classifier.py` 등을 직접 실행해서 만들면 된다.

기본 (학습용 데이터 4종):
    python scripts/setup_all.py

CQ500 외부 테스트셋까지 (+28GB, aria2c 필요):
    python scripts/setup_all.py --with-cq500
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(cmd: list) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-cq500", action="store_true",
                        help="CQ500 외부 테스트셋도 다운로드 (~28GB, aria2c 필요)")
    args = parser.parse_args()

    print("=" * 60)
    print("  OOB_test_11 데이터 셋업")
    print("=" * 60)

    rc = run([sys.executable, str(HERE / "download_data.py")])
    if args.with_cq500:
        rc |= run([sys.executable, str(HERE / "download_cq500.py")])

    print("\n" + "=" * 60)
    if rc == 0:
        print("  ✅ 데이터 준비 완료")
        print("=" * 60)
        print("\n학습 시작:")
        print("  python training/train_classifier.py --epochs 50")
        print("  python training/train_segmentor.py  --epochs 80")
        print("\n추론/평가:")
        print("  python scripts/run_batch_test.py --input-dir imgs/ --output-dir results/")
        print("  python scripts/evaluate_valset.py")
        print("  python scripts/evaluate_cq500.py   # CQ500 받은 경우")
    else:
        print("  ⚠️ 일부 단계 실패 — 위 로그 확인")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
