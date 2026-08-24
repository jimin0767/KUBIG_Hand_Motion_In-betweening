"""
exp01: 트랙 A 베이스라인(`../../sl_model_a.py`) 대비 모델 용량만 키운 변형.

베이스라인: d_model=256, n_layers=4, n_heads=4, d_ff=1024  (~3.2M 파라미터)
이 실험:   d_model=512, n_layers=6, n_heads=8, d_ff=2048  (~4배 이상 파라미터)

가설: SILK 논문은 "모델 복잡도보다 데이터 양이 중요하다"고 주장했다. 우리 데이터(647만
프레임)는 원 논문 LaFAN1(4.5시간)보다 큰 자릿수라, 베이스라인 3.2M 모델이 이미 용량
병목일 수도 있다. 이 실험은 그 가설을 우리 데이터에서 직접 검증한다 — 학습 스텝 수는
베이스라인과 동일(18000)하게 고정해 "모델 크기만 바꿨을 때"의 순수 효과를 본다.

구조는 sl_model_a.SILKHand와 완전히 동일(코드 중복 방지), 하이퍼파라미터만 다르게 생성.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # data_signlang/ 를 path에 추가
from sl_model_a import SILKHand, l1_loss  # noqa: E402

__all__ = ["build_model", "l1_loss"]


def build_model():
    return SILKHand(d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=64, dropout=0.1)
