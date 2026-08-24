"""
트랙 A 모델 — SILK(2025)의 레시피를 손 스켈레톤 규모로 축소 적용한 단일 트랜스포머 인코더.

SILK 원 논문 대비 이 스파이크에서 의도적으로 단순화/변경한 부분 (정직하게 명시):
  - 위치 인코딩: **목표 키프레임까지 남은 거리** 기준 학습된 임베딩(2026-08-16 변경, 팀원(이신영)
    코드에서 배움). 이전엔 절대 위치(프레임을 앞에서부터 센 번호)를 썼는데, 윈도우 길이
    L=ctx+T+1이 T(5~30)에 따라 매번 바뀌는 우리 구조에서는 같은 절대 인덱스가 T값에 따라
    "목표 프레임"이었다가 "gap 중간"이었다가 역할이 계속 바뀌는 문제가 있었다(팀원이 실측으로
    확인한 버그). 목표 프레임을 항상 거리 0으로 고정하면 이 충돌이 원천적으로 사라진다.
    주의: `collate()`가 배치 내 짧은 시퀀스를 뒤쪽에 패딩하므로, 거리는 배치 최대길이 L이 아니라
    **샘플별 실제 길이**(`pad_mask.sum()`, 항상 목표=마지막 유효 프레임) 기준으로 계산해야 한다.
  - 관측 여부(observed) 1채널을 입력에 추가함 (원 논문은 0-채움 패턴 자체로 암묵적으로 알림).
  - 우리는 위치 정보가 없어(MANO shape 파라미터 미보유) 회전만 다룸. 원 논문의 "9J+4" 출력에서
    위치(3J) 성분이 빠진 축소판.
  - d_model=1024→256, layer=6→4로 축소 (손 스켈레톤 규모+스파이크 목적).

주의: 이 변경으로 위치 임베딩의 의미가 바뀌었으므로, 이전(절대 PE) 체크포인트는 shape은
호환되지만 값의 의미가 달라 그대로 불러오면 안 된다 — 전부 재학습이 필요하다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sl_dataset import FEAT_FULL, FEAT_STATE


class SILKHand(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_len=64, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(FEAT_FULL, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, FEAT_STATE)
        self.norm_in = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        """x: (B,L,FEAT_FULL), pad_mask: (B,L) True=실제 프레임(패딩 아님).
        목표 키프레임은 항상 각 샘플의 마지막 '실제' 프레임(=pad_mask가 True인 마지막 위치)이라,
        거기까지 남은 거리를 위치 정보로 쓴다. 배치 최대길이가 아니라 샘플별 실제 길이 기준."""
        B, L, _ = x.shape
        lengths = pad_mask.sum(dim=1)                       # (B,) 샘플별 실제 프레임 수
        t = torch.arange(L, device=x.device)[None, :]        # (1,L)
        dist_to_target = (lengths[:, None] - 1) - t          # (B,L) 목표=0, 과거로 갈수록 증가
        dist_to_target = dist_to_target.clamp(min=0, max=self.pos_emb.num_embeddings - 1)
        h = self.in_proj(x) + self.pos_emb(dist_to_target)   # (B,L,d_model)
        h = self.norm_in(h)
        h = self.encoder(h, src_key_padding_mask=~pad_mask)
        return self.out_proj(h)  # (B,L,FEAT_STATE)


def l1_loss(pred, target, pad_mask):
    """패딩 프레임 제외한 L1. SILK처럼 전 채널 균등 가중치, 프레임 전체(관측+결측)에 대해 계산."""
    err = (pred - target).abs().sum(-1)          # (B,L)
    err = err * pad_mask.float()
    return err.sum() / pad_mask.float().sum().clamp(min=1)
