"""6D 회전 표현 유틸 (torch, GPU 가능).

컨벤션 — pytorch3d 표준(행 기준, stack dim=-2)을 따른다.
6D = 회전행렬의 앞 두 '행'. SignSparK 공식 소스(pose_datasets_lmdb.py)와 동일하며,
팀 슬랙에서 합의한 규약이다.

geodesic 오차와 SLERP은 행/열 규약에 불변이지만, MANO forward kinematics는
불변이 아니다(R vs R^T). 그래서 FK를 쓰는 L2P에서는 이 규약을 반드시 지켜야 한다.
"""
from __future__ import annotations

import torch

# 왼손(SMPLX-left) -> 오른손(WiLoR) 좌표계 변환 마스크.
# R_flip @ R @ R_flip, R_flip = diag(1,-1,-1)  <=>  원소별 s_i*s_j 곱, s=(1,-1,-1).
# 이 3x3 마스크는 대칭이라 행/열 규약과 무관하게 앞 두 행(=6D)에 그대로 적용된다.
LEFT_FLIP_6D = torch.tensor([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). Gram-Schmidt, 행 기준(pytorch3d)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6). 앞 두 행을 이어붙임."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def flip_left_to_right(feat: torch.Tensor) -> torch.Tensor:
    """(..., J*6) 왼손 feature를 오른손 규약으로 변환."""
    shp = feat.shape
    return (feat.reshape(*shp[:-1], -1, 6) * LEFT_FLIP_6D.to(feat)).reshape(shp)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) [w,x,y,z]. 수치 안정을 위해 최대 성분 분기 사용."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # 네 후보의 제곱을 모두 만들고 가장 큰 것을 고른다 (pytorch3d와 같은 전략)
    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1).clamp_min(0.0).sqrt()
    quats = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    best = q_abs.argmax(dim=-1)
    q = torch.gather(quats, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    denom = torch.gather(q_abs, -1, best[..., None]).clamp_min(1e-8) * 2.0
    return q / denom


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) [w,x,y,z] -> (..., 3, 3)."""
    q = _normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle. MANO FK 입력용."""
    q = matrix_to_quaternion(R)
    q = torch.where(q[..., :1] < 0, -q, q)   # w>=0으로 정규화 -> 각도가 [0, pi]에 들어옴
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    # sin_half -> 0 에서의 특이점 처리 (테일러 전개)
    scale = torch.where(sin_half > 1e-6, angle / sin_half.clamp_min(1e-8),
                        torch.full_like(angle, 2.0))
    return xyz * scale


def geodesic_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """두 회전 사이 측지 각(라디안). (..., 3, 3) x2 -> (...)."""
    rel = torch.matmul(R1.transpose(-1, -2), R2)
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))


def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """구면 선형보간. q0,q1: (..., 4), t: (..., 1) -> (..., 4)."""
    q0, q1 = _normalize(q0), _normalize(q1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)          # double-cover: 짧은 호를 택함
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near = sin_theta < 1e-6                      # 거의 같은 회전이면 lerp로 대체
    w0 = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-8))
    w1 = torch.where(near, t, torch.sin(t * theta) / sin_theta.clamp_min(1e-8))
    return _normalize(w0 * q0 + w1 * q1)


def slerp_6d(start: torch.Tensor, end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """6D 회전 두 개 사이를 SLERP. start/end: (B, J, 6), t: (B, T, 1, 1) -> (B, T, J, 6)."""
    q0 = matrix_to_quaternion(rot6d_to_matrix(start))[:, None]   # (B,1,J,4)
    q1 = matrix_to_quaternion(rot6d_to_matrix(end))[:, None]
    q = quaternion_slerp(q0.expand(-1, t.shape[1], -1, -1),
                         q1.expand(-1, t.shape[1], -1, -1), t)
    return matrix_to_rot6d(quaternion_to_matrix(q))
