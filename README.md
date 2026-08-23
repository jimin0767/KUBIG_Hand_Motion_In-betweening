# KUBIG_Hand_Motion_In-betweening


## v1. Diffusion SILK 1 (diffusion_silk_notebook.ipynb)

**SILK 구조 유지**

- **6층·8헤드 Transformer 인코더**(`d_model=1024`, `d_ff=4096`, Pre-LN)를 그대로 가져다 씀
- 새 아키텍처를 처음부터 만든 게 아니라, **"결정적 회귀 모델"을 "diffusion denoiser"로 용도 전환**
- 달라진 부분
    1. **입력에 timestep 임베딩이 더해짐** — "지금이 노이즈 제거 과정의 몇 번째 단계인지"를 모델에 알려줌 (위치 인코딩과 같은 sinusoidal 방식이지만, 대상이 "시간축 위치"가 아니라 "노이즈 단계")
    2. **입력에 "관측여부" 플래그 채널이 추가됨** — 그 프레임이 컨텍스트/목표/보너스 키프레임인지, 아니면 예측해야 할 gap인지 표시
 

**구현 내용**

1. x0 예측(노이즈가 아니라): MDM/CondMDI 계열의 특징을 그대로 따라, 노이즈 ε이 아니라 원본 회전값(x0)을 직접 예측하도록 설계했어.

2. Cosine noise schedule: 학습 때 정답에 노이즈를 섞는 방식으로 표준 cosine beta schedule을 씀 (1000스텝 기준).

3. Inpainting 방식 조건화(CondMDI 핵심 메커니즘): 매 학습 스텝마다

4. 학습 손실: x0 예측값과 정답 사이의 단일 L1, 전체 시퀀스에 대해 (SILK 트랙과 동일한 철학 — 마스킹된 gap 구간만이 아니라 전체 시퀀스 기준).

5. 추론(샘플링): 순수 1000스텝 순차 diffusion은 비현실적으로 느려서(배치당 forward 1000회), DDIM 스타일로 50스텝만 균일 간격으로 골라 밟는 서브샘플링을 적용 — 매 스텝 관측 구간을 그 시점에 맞게 다시 노이즈 낀 정답으로 재치환하면서 진행.

**성능**

T=5: L2Q=0.0258 L2P=0.0014 NPSS=0.0040 (n=3840)

T=10: L2Q=0.0566 L2P=0.0031 NPSS=0.0232 (n=3840)

T=20: L2Q=0.0919 L2P=0.0051 NPSS=0.0613 (n=3840)

T=30: L2Q=0.1178 L2P=0.0067 NPSS=0.0909 (n=3840)


https://github.com/user-attachments/assets/02f18bfb-378b-4ae0-a5b3-e28c55033729



## v2. Diffusion SILK 2 (diffusion_silk_notebook_keyframe.ipynb)

**구현 내용**

- 이전 실험에 keyframe 구조 강화
- 세그먼트 시작/중간/끝
- 학습 때만

**성능**

T=5: L2Q=0.0255 L2P=0.0014 NPSS=0.0040 (n=3840)

T=10: L2Q=0.0516 L2P=0.0029 NPSS=0.0197 (n=3840)

T=20: L2Q=0.0868 L2P=0.0051 NPSS=0.0538 (n=3840)

T=30: L2Q=0.1119 L2P=0.0066 NPSS=0.0825 (n=3840)

https://github.com/user-attachments/assets/aa00aeb9-dd2c-4e8d-b7a4-1e8120fbee6c


## v3. Diffusion SILK 3 (diffusion_silk_notebook_vel_loss.ipynb)

**구현 내용**

- 속도 손실 추가

**성능**

T=5: L2Q=0.0276 L2P=0.0015 NPSS=0.0050 (n=3840)

T=10: L2Q=0.0587 L2P=0.0032 NPSS=0.0290 (n=3840)

T=20: L2Q=0.1061 L2P=0.0058 NPSS=0.0897 (n=3840)

T=30: L2Q=0.1372 L2P=0.0074 NPSS=0.1308 (n=3840)


https://github.com/user-attachments/assets/7b1a2c81-830e-4caa-ac5d-22c0cadbbb3f




## v4. Flow matching SILK (flow_silk_notebook.ipynb)

**구현 내용**

- v2 백본 사용
- DDPM을 flow matching으로 변경
- 샘플링 시 n_step=20

**성능**

T=5: L2Q=0.0199 L2P=0.0011 NPSS=0.0037 (n=3840)

T=10: L2Q=0.0415 L2P=0.0023 NPSS=0.0179 (n=3840)

T=20: L2Q=0.0727 L2P=0.0041 NPSS=0.0489 (n=3840)

T=30: L2Q=0.1018 L2P=0.0057 NPSS=0.0808 (n=3840)


T=5: L2Q=0.0306 L2P=0.0017 NPSS=0.0072 (n=147748)
T=10: L2Q=0.0569 L2P=0.0032 NPSS=0.0276 (n=143382)
T=20: L2Q=0.0959 L2P=0.0056 NPSS=0.0719 (n=134664)
T=30: L2Q=0.1206 L2P=0.0070 NPSS=0.1055 (n=126206)

https://github.com/user-attachments/assets/db986186-7ff1-4e12-851c-397d11190e52



## 참고 문헌

**SILK (arXiv:2506.09075)**

모델 백본 — 6층 Transformer 인코더를 diffusion/flow denoiser로 재활용. 목표-상대 위치 인코딩도 이 계보에서 가져옴

**RMIB (Harvey et al. 2020, LaFAN1 논문)**

time-to-arrival(ztta) 설계 — 우리 목표-상대 위치 인코딩이 이 방식과 문헌적으로 일치함을 확인하는 근거로 사용

**CondMDI (Cohan et al. 2024, arXiv:2405.11126, SIGGRAPH)**

핵심 방법론 — 학습 때 관측 구간을 강제로 정답으로 치환하는 inpainting 조건화 메커니즘. "단순 추론시점 임퓨테이션보다 학습 자체에 마스킹 패턴을 가르치는 게 낫다"는 이 논문의 발견을 그대로 채택

**MotionGPT3 (2026)**
모션 생성 도메인에서 flow matching이 diffusion보다 훨씬 적은 스텝(4~8)으로 수렴한다는 것을 실증 비교 — 4번 버전의 n_steps=20 채택 근거

