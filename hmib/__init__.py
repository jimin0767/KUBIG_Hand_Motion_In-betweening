"""Hand Motion In-betweening — 공용 파이프라인 (KUBIG DL 3팀).

0816 회의 확정 규약을 코드로 고정한 패키지.
  - 데이터: How2Sign 단독 (CSL-Daily 제외)
  - 표현:   MANO 15관절 x 6D 회전 = 90차원, 위치/손목 채널 없음
  - 윈도우: context 10 + transition T + target 1, stride 5
  - 평가:   T in {5,10,20,30}, 지표 L2Q / L2P(MANO FK) / NPSS, gap 구간만 채점
"""
__all__ = ["rotation", "mano", "metrics", "data", "baselines", "models"]

N_JOINTS = 15
FEAT_DIM = N_JOINTS * 6          # 90
CTX = 10                          # context 프레임 수
EVAL_T_VALUES = (5, 10, 20, 30)
TRAIN_T_RANGE = (5, 30)
