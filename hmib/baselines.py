"""학습 없는 기준선 — SLERP와 Hold.

0816 회의에서 "SLERP 대비 상대 개선율로 평가한다"고 정해놓고 정작 이 숫자가 없었다.
여기가 팀 전체 결과표의 원점이 된다.

Hold(마지막 context 프레임 유지)를 같이 두는 이유: SLERP를 못 이기는 모델이라도
Hold보다 나은지는 알아야 "동작을 배우긴 했다"를 주장할 수 있다. 김석우님의
InterHand2.6M 제로샷 실험에서도 Hold가 하한선 역할을 했다.
"""
from __future__ import annotations

import torch

from . import CTX, N_JOINTS
from .rotation import (matrix_to_quaternion, matrix_to_rot6d, quaternion_slerp,
                       quaternion_to_matrix, rot6d_to_matrix)


def slerp_fill(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """마지막 context 프레임과 target 프레임 사이를 구면 선형보간으로 채운다.

    x: (B, L, 90) — 관측 프레임만 실제로 참조한다.
    T: (B,) gap 길이  ->  (B, L, 90)
    """
    B, L, D = x.shape
    q0 = matrix_to_quaternion(rot6d_to_matrix(x[:, CTX - 1].reshape(B, N_JOINTS, 6)))
    idx_last = (CTX + T).clamp(max=L - 1)                        # target 프레임 위치
    q1 = matrix_to_quaternion(rot6d_to_matrix(
        x[torch.arange(B, device=x.device), idx_last].reshape(B, N_JOINTS, 6)))
    steps = torch.arange(L, device=x.device).view(1, L, 1, 1).to(x.dtype)
    denom = (T + 1).view(B, 1, 1, 1).to(x.dtype)
    t = ((steps - (CTX - 1)) / denom).clamp(0.0, 1.0)            # (B,L,1,1)
    q = quaternion_slerp(q0[:, None].expand(-1, L, -1, -1),
                         q1[:, None].expand(-1, L, -1, -1), t)
    return matrix_to_rot6d(quaternion_to_matrix(q)).reshape(B, L, D)


def hold_fill(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """마지막 context 프레임을 그대로 유지하는, 아무것도 안 하는 기준선."""
    B, L, D = x.shape
    return x[:, CTX - 1:CTX].expand(B, L, D).clone()
