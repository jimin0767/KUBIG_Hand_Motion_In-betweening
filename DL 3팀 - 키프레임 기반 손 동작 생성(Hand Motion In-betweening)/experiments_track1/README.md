# KUBIG_Hand_Motion_In-betweening (Context Transformer track)

SILK(2025)의 single-Transformer-encoder in-betweening 레시피를 손(hand) 회전 데이터(rot6D,
MANO 기반 15관절, How2Sign)에 그대로 적용한 결정적(deterministic) 회귀 트랙. 모델 크기와
손실 함수만 바꿔가며 비교한 6개 버전을 기록한다.

## v0. baseline (`sl_model_a.py`, d_model=256)

**SILK 구조 그대로**

- 4층·4헤드 Transformer 인코더(`d_model=256`, `d_ff=1024`), 단일 L1 손실
- 위치 인코딩은 절대 위치가 아니라 **목표 키프레임까지 남은 거리 기준**으로 구현
  (윈도우 길이가 T마다 바뀌는 구조에서, 같은 절대 인덱스가 "목표 프레임"과 "gap 중간"을
  오가며 학습 신호가 충돌하는 문제를 피하기 위함)

**성능**

T=5: L2Q=0.0143 L2P=0.0009 NPSS=0.0012 (n=147748)

T=10: L2Q=0.0352 L2P=0.0022 NPSS=0.0068 (n=143382)

T=20: L2Q=0.0656 L2P=0.0042 NPSS=0.0203 (n=134664)

T=30: L2Q=0.0845 L2P=0.0055 NPSS=0.0325 (n=126206)

![gt vs baseline](results/gt_vs_baseline_135_T20.gif)

## v1. 모델 확장 (`exp01_bigger_model/`, d_model=512, 19M)

**구현 내용**

- 구조는 baseline과 완전히 동일, `d_model=512, n_layers=6, n_heads=8, d_ff=2048`로 용량만 확장
- 동기: in-betweening 관련 선행 연구 다수가 "모델 용량보다 데이터 양이 성능을 더 좌우한다"고
  주장하는데, 이게 손 도메인(관절 수·데이터 규모가 몸 전체보다 훨씬 작음)에서도 성립하는지
  직접 확인하려는 목적. 학습 스텝·데이터·손실함수는 baseline과 전부 고정.

**성능**

T=5: L2Q=0.0121 L2P=0.0007 NPSS=0.0010 (n=147748)

T=10: L2Q=0.0322 L2P=0.0020 NPSS=0.0063 (n=143382)

T=20: L2Q=0.0627 L2P=0.0040 NPSS=0.0196 (n=134664)

T=30: L2Q=0.0822 L2P=0.0054 NPSS=0.0320 (n=126206)

![gt vs exp01](results/gt_vs_exp01_135_T20.gif)

## v2. 잔차 예측 (`exp02_residual_prediction/`)

**구현 내용**

- 절대값 대신 SLERP 보간 대비 잔차(residual)를 예측하도록 타깃만 교체 (Δ-Interpolator식
  파라미터화). 모델 구조·크기는 baseline과 동일
- 추론 시 최종 예측 = 모델 출력(잔차) + SLERP reference로 복원

**성능**

T=5: L2Q=0.0156 L2P=0.0009 NPSS=0.0012 (n=147748)

T=10: L2Q=0.0353 L2P=0.0022 NPSS=0.0067 (n=143382)

T=20: L2Q=0.0652 L2P=0.0042 NPSS=0.0200 (n=134664)

T=30: L2Q=0.0845 L2P=0.0055 NPSS=0.0324 (n=126206)

![gt vs exp02](results/gt_vs_exp02_135_T20.gif)

## v3. 다양성 유도 손실 (`exp04_diversity_loss/`)

**구현 내용**

- 기존 L1 손실에 "예측이 GT보다 덜 움직이면 벌점" one-sided hinge를 추가 — 회귀 모델이
  다봉분포를 평균내면서 동작이 밋밋해지는 문제(확률적/생성 모델 계열이 지적하는 약점)를,
  아키텍처를 바꾸지 않고 손실 항 하나로 완화할 수 있는지 실험
- gap 구간의 프레임간 상대회전 크기(속도)를 예측/정답 각각 구해 비교, 예측 쪽이 부족한
  만큼만 벌점(정답보다 더 움직이는 건 벌점 없음 — 그건 L1이 이미 담당)

**성능**

T=5: L2Q=0.0143 L2P=0.0009 NPSS=0.0012 (n=147748)

T=10: L2Q=0.0352 L2P=0.0022 NPSS=0.0069 (n=143382)

T=20: L2Q=0.0657 L2P=0.0042 NPSS=0.0223 (n=134664)

T=30: L2Q=0.0849 L2P=0.0055 NPSS=0.0382 (n=126206)

![gt vs exp04](results/gt_vs_exp04_135_T20.gif)

## v4. 속도 매칭 손실 (`exp05_velocity_loss/`)

**구현 내용**

- v3의 더 엄격한 버전. 속도의 "크기"만 맞추는 게 아니라, gap 구간 프레임간 상대회전을
  방향까지 포함한 벡터 그대로 L1으로 맞춤 (양은 맞아도 패턴이 틀리면 v3는 못 잡지만
  이건 잡아냄)

**성능**

T=5: L2Q=0.0157 L2P=0.0010 NPSS=0.0013 (n=147748)

T=10: L2Q=0.0358 L2P=0.0022 NPSS=0.0069 (n=143382)

T=20: L2Q=0.0663 L2P=0.0043 NPSS=0.0204 (n=134664)

T=30: L2Q=0.0852 L2P=0.0056 NPSS=0.0326 (n=126206)

![gt vs exp05](results/gt_vs_exp05_135_T20.gif)

## v5. SILK 원 논문 크기 (`exp06_full_silk_size/`, d_model=1024, 76M)

**구현 내용**

- v1에서 개선이 계속되길래, SILK 원 논문이 실제로 쓴 크기(`d_model=1024, n_layers=6,
  n_heads=8, d_ff=4096`)까지 그대로 밀어붙여서 용량 확장 효과가 어디서 한계에 부딪히는지 확인
- 다른 변수는 v1과 동일하게 고정

**성능**

T=5: L2Q=0.0116 L2P=0.0007 NPSS=0.0010 (n=147748)

T=10: L2Q=0.0316 L2P=0.0020 NPSS=0.0062 (n=143382)

T=20: L2Q=0.0621 L2P=0.0040 NPSS=0.0195 (n=134664)

T=30: L2Q=0.0817 L2P=0.0053 NPSS=0.0318 (n=126206)

![gt vs exp06](results/gt_vs_exp06_135_T20.gif)

## 데이터 / 표현 공통사항

- How2Sign(SignSparK 전처리, MANO rot6D) 단독, 오른손 + 왼손(오른손 규약으로 미러링) 둘 다
  사용 (`preprocess_offset5/windowed_dataset.py`)
- 시작점 offset=5 간격 슬라이딩 학습 윈도우(SILK 논문 실측값), 학습 시 가림 길이 T는 5~30
  사이 매 샘플마다 균일 랜덤, 평가는 T=5/10/20/30 고정

## 참고 문헌

**SILK (arXiv:2506.09075)**

모델 백본 — 6층 Transformer 인코더, 단일 L1 손실, 학습 시 가림 길이 5~30 균일 샘플링.
데이터 슬라이스 offset=5도 이 논문 원문 실측값을 그대로 따름.

## 재현

```bash
# 필요한 것: SignSparK How2Sign LMDB (경로는 windowed_dataset.py 상단 DATA_ROOT에서 설정)
cd experiments/exp01_bigger_model   # 또는 exp02/exp04/exp05/exp06_full_silk_size
python train_offset5.py --steps 127560 --batch_size 64
```
