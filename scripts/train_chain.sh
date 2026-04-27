#!/usr/bin/env bash
# 분류기 학습 끝나면 자동으로 세그멘터 학습 이어 시작.
# MPS 단일 자원이라 동시 학습은 둘 다 느려짐 → 순차 실행이 정답.
#
# 사용:
#   bash scripts/train_chain.sh                    # 50 + 80 epoch (기본)
#   CLS_EPOCHS=30 SEG_EPOCHS=50 bash scripts/train_chain.sh
#
# 백그라운드로 돌리려면:
#   nohup bash scripts/train_chain.sh > logs/chain.log 2>&1 &
#
# 이미 분류기 PID 파일이 있으면 분류기 끝날 때까지 기다린 다음 세그멘터만 시작.

set -e

CLS_EPOCHS="${CLS_EPOCHS:-50}"
SEG_EPOCHS="${SEG_EPOCHS:-80}"
PY="${PY:-venv/bin/python}"
TS=$(date "+%Y%m%d_%H%M%S")

mkdir -p logs

# 1. 분류기 — 이미 돌고 있으면 그 PID 를 기다리고, 아니면 새로 시작
if [[ -f logs/train_classifier.pid ]] && ps -p "$(cat logs/train_classifier.pid)" > /dev/null 2>&1; then
    CLS_PID=$(cat logs/train_classifier.pid)
    echo "[chain] 이미 실행 중인 분류기 학습 발견 (PID=$CLS_PID) — 끝날 때까지 대기"
else
    CLS_LOG="logs/train_classifier_${TS}.log"
    echo "[chain] 분류기 학습 시작 ($CLS_EPOCHS epoch) → $CLS_LOG"
    nohup "$PY" -u training/train_classifier.py --epochs "$CLS_EPOCHS" > "$CLS_LOG" 2>&1 &
    CLS_PID=$!
    echo "$CLS_PID" > logs/train_classifier.pid
    echo "$CLS_LOG" > logs/train_classifier.logpath
fi

# 2. 분류기 종료 대기 (성공·실패 둘 다 진행 — 세그는 분류기 ckpt 안 써도 됨)
echo "[chain] PID $CLS_PID 종료 대기..."
while ps -p "$CLS_PID" > /dev/null 2>&1; do
    sleep 30
done
CLS_RC=0  # 종료 코드 모름 (백그라운드 + nohup), 그냥 진행
echo "[chain] 분류기 학습 종료. (PID $CLS_PID)"

# 3. 세그멘터 — 분류기 끝난 직후 시작
TS2=$(date "+%Y%m%d_%H%M%S")
SEG_LOG="logs/train_segmentor_${TS2}.log"
echo "[chain] 세그멘터 학습 시작 ($SEG_EPOCHS epoch) → $SEG_LOG"
nohup "$PY" -u training/train_segmentor.py --epochs "$SEG_EPOCHS" > "$SEG_LOG" 2>&1 &
SEG_PID=$!
echo "$SEG_PID" > logs/train_segmentor.pid
echo "$SEG_LOG" > logs/train_segmentor.logpath
echo "[chain] 세그멘터 PID $SEG_PID — 모니터: tail -f $SEG_LOG"

# 4. 세그멘터 종료 대기 (이 스크립트가 nohup 으로 돌고 있을 때만 의미 있음)
while ps -p "$SEG_PID" > /dev/null 2>&1; do
    sleep 30
done
echo "[chain] 세그멘터 학습 종료. 전체 chain 완료."
echo "[chain] 다음: python scripts/evaluate_external_test.py"
