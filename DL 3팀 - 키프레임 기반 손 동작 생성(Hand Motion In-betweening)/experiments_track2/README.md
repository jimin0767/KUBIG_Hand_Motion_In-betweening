# KUBIG_Hand_Motion_In-betweening
# Hand Motion In-betweening (How2Sign · MANO 기반)

How2Sign 수어 영상의 손 동작(MANO 기반 회전값)에서, 컨텍스트(앞 프레임들)와
타겟(마지막 프레임) 사이에 비어 있는 구간(gap)을 자연스럽게 채우는 "손 동작 in-betweening" 문제를
다룹니다.

노트북: `Handmotioninbetweening_0823.ipynb`

## 개요

- **베이스라인**: SILK 구조 Transformer 인코더 (`d_model=1024, nhead=8, num_layers=6, dim_feedforward=4096`)
- **Two-stage Transformer** (Qin et al., SIGGRAPH Asia 2022 기반): Context Transformer(=baseline, freeze)가
  거친 예측을 만들고 Detail Transformer가 이를 다듬는 2단계 구조
- **다중 키프레임 구조** (자체 확장, 과제 3): gap 중간 지점을 추가 조건으로 노출시켜 하나의 gap을 두 개로
  분할하는 구조
- **MANO 관절 시각화**: GT/Baseline/Two-stage/다중 키프레임 4개를 3D 스켈레톤 GIF로 비교
- **개선 시도**: 다중 키프레임의 트레이드오프를 완화하기 위한 loss 가중치 조정(soft-loss) 실험

## 노트북 구성

| 섹션 | 내용 |
|---|---|
| 1 | 환경 설정 (레포 클론, 패키지 설치) |
| 2 | 데이터 준비 (Google Drive에서 복원) |
| 3 | 좌표계 변환(왼손→오른손 flip) + LMDB 디코딩 유틸 |
| 4 | Index 파일 로드 (train/dev/test) |
| 5 | Dataset / DataLoader |
| 6 | 입력 특징 변환 (`build_silk_features`) |
| 7 | 모델 정의 (`SilkHandEncoderV2`, `RelativePositionalEncoding`) |
| 8 | 손실 함수 |
| 9 | MANO 레이어 로드 + 평가지표 (L2Q · L2P-MANO · NPSS) |
| 10 | T-버킷 평가 루프 |
| 11 | Baseline(Context Transformer) 학습 |
| 12 | Two-stage Transformer (Context + Detail) |
| 13 | 다중 키프레임 구조 결합 |
| 14 | MANO 시각화 (GIF 비교) |
| 15 | 다중 키프레임 개선 시도 — soft-loss |

## 실행 환경 / 데이터 준비

Google Colab 실행을 전제로 합니다 (GPU 필요, A100 권장 — T4에서는 `train_loader`의 `batch_size`를
64→32로 낮추는 것을 권장). Google Drive에 아래 구조로 데이터가 준비되어 있어야 합니다.

```
MyDrive/KUBIG/contest/
├── train_index.npz
├── dev_index.npz
├── test_index.npz
├── MANO_RIGHT.pkl
├── MANO_LEFT.pkl
├── checkpoints/                              # 학습 시 자동 생성, 모델 체크포인트 저장 위치
└── HandMotionInbetweening_data_backup/
    └── lmdb/
        ├── train/How2Sign_reopt_train.lmdb
        ├── dev/How2Sign_reopt_dev.lmdb
        └── test/How2Sign_reopt_test.lmdb
```

셀은 위에서 아래로 순서대로 실행하면 됩니다. 각 학습 루프는 Google Drive에 체크포인트를 저장하고
재실행 시 자동으로 이어서 학습하므로, Colab 런타임이 끊겨도 처음부터 다시 돌릴 필요는 없습니다.
Section 14에서 생성되는 비교 GIF는 `MyDrive/KUBIG/contest/viz_T{5,10,20,30}.gif`에 저장됩니다.

## 평가 설정

test셋 기준, gap 길이 T=5/10/20/30 버킷별로 평가 (표본 수 n=15,360, 전 모델 동일 조건). 지표는 모두
낮을수록 좋습니다.

- **L2Q**: 회전값(quaternion) 오차
- **L2P**: MANO forward kinematics 기반 실제 관절 위치 오차
- **NPSS**: 움직임의 시간적 자연스러움 (주파수 스펙트럼 기반)

## 결과 요약

| T | 지표 | Baseline | Two-stage | 다중 키프레임 |
|---|---|---|---|---|
| 5 | L2Q / L2P / NPSS | 0.0235 / 0.0013 / 0.0043 | 0.0196 / 0.0011 / 0.0039 | 0.0207 / 0.0011 / 0.0043 |
| 10 | L2Q / L2P / NPSS | 0.0407 / 0.0023 / 0.0167 | 0.0382 / 0.0021 / 0.0162 | 0.0389 / 0.0021 / 0.0167 |
| 20 | L2Q / L2P / NPSS | 0.0688 / 0.0039 / 0.0448 | 0.0671 / 0.0039 / 0.0445 | 0.0674 / 0.0039 / 0.0451 |
| 30 | L2Q / L2P / NPSS | 0.0887 / 0.0052 / 0.0683 | 0.0865 / 0.0050 / 0.0677 | 0.0868 / 0.0051 / 0.0682 |

**주요 발견**

- Two-stage는 baseline 대비 모든 T·모든 지표에서 일관되게 개선됩니다. gap이 짧을수록(T=5) 개선폭이
  크고, 길어질수록 줄어드는 경향이 있습니다 (Context 단계의 coarse 예측 품질이 병목으로 추정).
- 다중 키프레임 구조는 위치 정확도(L2Q/L2P)를 Two-stage보다 더 개선하지만, 자연스러움(NPSS)은 모든
  T에서 소폭 악화됩니다 — 중간 키프레임 경계에서 "이음매"가 생기는 것이 원인으로, MANO 관절 GIF의
  프레임 간 움직임량 분석으로도 같은 지점에서 확인했습니다.
- 이 트레이드오프를 완화하기 위해 중간 키프레임 위치에 낮은 loss 가중치(0.3)를 주는 soft-loss를
  시도했으나, Two-stage보다 모든 T·모든 지표에서 더 나쁘게 나와 이 방향은 채택하지 않았습니다
  (Section 15).

## 시각화

Section 14에서 생성한 GT / Baseline / Two-stage / 다중 키프레임 비교 GIF입니다 

**T=20**
<img width="1000" height="1000" alt="viz_matched_fixed" src="https://github.com/user-attachments/assets/abd9d458-b913-4271-85f0-ed489609971c" />


## 참고

- Two-stage Transformer 구조: Qin et al., *"Motion In-betweening via Two-Stage Transformers"*,
  SIGGRAPH Asia 2022
- 입력 특징 변환 / 모델 크기 설정은 팀 SILK 파이프라인과 동일하게 맞췄습니다.
