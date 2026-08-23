"""SILK 스타일 Transformer 인코더 — 절대값 예측 / 잔차(Δ) 예측 두 변형.

설계 메모
---------
1. 전역/국소 분리 projection은 쓰지 않는다.
   SILK의 최대 ablation 이득(17%)은 "전신 root 궤적과 관절 회전을 분리"해서 나온 것인데,
   우리 90차원에는 root(손목)가 아예 없다. 앞 6차원은 손목이 아니라 index_00 관절이므로
   (팀 슬랙에서 확인됨), 여기서 분리를 하면 "검지 하나만 특별 취급"이 될 뿐이다.
   따라서 181차원을 단일 projection으로 받는다.

2. 위치 인코딩은 0816 회의 결정을 따른다 — 목표 프레임이 항상 rel=0이 되는
   상대 위치 임베딩 테이블. gap 길이가 매번 달라도 "목표까지 몇 프레임 남았는가"가
   항상 같은 값으로 표현된다.

3. Δ 변형은 SLERP 보간값 위에 잔차를 얹는다. 김석우님이 T=45(학습 범위 밖)에서
   절대값 예측 모델만 무너지고 잔차 예측은 SLERP 수준을 유지한다고 보고한 구조다.
   SLERP가 바닥을 깔아주기 때문에 잔차가 부정확해도 발산하지 않는다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import FEAT_DIM

IN_DIM = FEAT_DIM * 2 + 1        # 상태 90 + 속도 90 + 관측플래그 1 = 181


class TargetRelativePE(nn.Module):
    """목표 프레임 기준 상대 위치 임베딩 (0816 회의 확정안)."""

    def __init__(self, d_model: int, max_rel: int = 64):
        super().__init__()
        self.max_rel = max_rel
        self.emb = nn.Embedding(2 * max_rel + 1, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        idx = (rel + self.max_rel).clamp(0, 2 * self.max_rel)
        return h + self.emb(idx)


class SilkHandEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_ff: int = 2048, dropout: float = 0.1, max_rel: int = 64,
                 residual: bool = False):
        super().__init__()
        self.residual = residual
        self.in_proj = nn.Linear(IN_DIM, d_model)
        self.pe = TargetRelativePE(d_model, max_rel)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, FEAT_DIM)
        nn.init.zeros_(self.head.bias)
        if residual:
            # 학습 초기 출력이 정확히 SLERP가 되도록 잔차 헤드를 0에서 시작
            nn.init.zeros_(self.head.weight)

    def forward(self, feats: torch.Tensor, rel: torch.Tensor, valid: torch.Tensor,
                base: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pe(self.in_proj(feats), rel)
        h = self.encoder(h, src_key_padding_mask=~valid)
        out = self.head(h)
        if self.residual:
            assert base is not None, "residual 모델에는 SLERP base가 필요합니다"
            out = base + out
        return out


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """gap이면서 패딩이 아닌 프레임에서만 L1."""
    m = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - gt).abs() * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]


def velocity_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """프레임간 변화량(속도)에 대한 보조 손실.

    "위치는 대충 맞는데 움직임이 밋밋한" 회귀 특유의 문제를 직접 겨냥한다
    (0816 회의 해법 5번). gap 내부의 인접 프레임 쌍만 본다.
    """
    dp = pred[:, 1:] - pred[:, :-1]
    dg = gt[:, 1:] - gt[:, :-1]
    m = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).to(pred.dtype)
    return ((dp - dg).abs() * m).sum() / m.sum().clamp_min(1.0) / pred.shape[-1]
