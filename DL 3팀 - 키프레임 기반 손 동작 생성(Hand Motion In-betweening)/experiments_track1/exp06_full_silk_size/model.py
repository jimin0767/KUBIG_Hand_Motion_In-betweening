"""exp06: SILK 원 논문 그대로의 크기(d_model=1024, n_layers=6, n_heads=8, d_ff=4096).

exp01(d_model=512, ~19M)에서 "모델 키우면 유일하게 확실히 좋아진다"는 결론이 나왔으니,
그 흐름 그대로 원 논문 크기까지 밀어붙여서 개선이 어디까지 이어지는지 확인한다
(CLAUDE.md "아직 안 한 것" 항목). 이신영님 SILK 재구현 노트북의 하이퍼파라미터
(SS_D_MODEL=1024, SS_N_HEADS=8, SS_N_LAYERS=6, SS_D_FF=4096)와 동일.

구조는 sl_model_a.SILKHand와 완전히 동일(코드 중복 방지), 하이퍼파라미터만 다르게 생성.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # data_signlang/
from sl_model_a import SILKHand, l1_loss  # noqa: E402

__all__ = ["build_model", "l1_loss"]


def build_model():
    return SILKHand(d_model=1024, n_layers=6, n_heads=8, d_ff=4096, max_len=64, dropout=0.1)
