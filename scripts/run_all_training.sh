#!/usr/bin/env bash
# 세 변형을 순차 학습. 통제 실험이므로 steps/batch/lr/모델크기를 전부 동일하게 고정한다.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
S=50000

echo "===== [1/3] abs : SILK 재현 (절대값 예측) ====="
python scripts/04_train.py --variant abs       --steps $S

echo "===== [2/3] delta : SLERP 위 잔차 예측 ====="
python scripts/04_train.py --variant delta     --steps $S

echo "===== [3/3] delta_vel : 잔차 + 속도 보조손실 ====="
python scripts/04_train.py --variant delta_vel --steps $S

echo "ALL TRAINING DONE"
