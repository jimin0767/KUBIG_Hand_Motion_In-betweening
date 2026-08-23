"""평가지표 — L2Q · L2P(MANO FK) · NPSS · geodesic. 0816 회의 정의를 그대로 따름.

회의록 코드는 윈도우 하나씩 numpy로 도는 형태였다. 여기서는 같은 수식을 GPU 배치로
벡터화했다(같은 T의 윈도우끼리 묶으면 gap 길이가 같아 배치가 그대로 성립).
`_reference_*` 함수로 회의록 원본 구현과 수치가 일치하는지 검증한다.

모든 지표는 gap(생성 구간) 프레임만 대상으로 한다.
"""
from __future__ import annotations

import numpy as np
import torch

from . import rotation as rot
from .mano import ManoFK

N_JOINTS = 15


def l2q(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """쿼터니언 L2 거리. (B, T, 90) -> (B,). double-cover 보정 포함."""
    B, T, _ = pred.shape
    qp = rot.matrix_to_quaternion(rot.rot6d_to_matrix(pred.reshape(B, T, N_JOINTS, 6)))
    qg = rot.matrix_to_quaternion(rot.rot6d_to_matrix(gt.reshape(B, T, N_JOINTS, 6)))
    d = torch.minimum((qp - qg).norm(dim=-1), (qp + qg).norm(dim=-1))
    return d.reshape(B, -1).mean(dim=-1)


def l2p(pred: torch.Tensor, gt: torch.Tensor, fk: ManoFK) -> torch.Tensor:
    """MANO forward kinematics 기반 관절 위치 오차(미터). (B, T, 90) -> (B,)."""
    B, T, _ = pred.shape
    jp = fk(rot.rot6d_to_matrix(pred.reshape(B, T, N_JOINTS, 6)))   # (B,T,16,3)
    jg = fk(rot.rot6d_to_matrix(gt.reshape(B, T, N_JOINTS, 6)))
    return (jp - jg).norm(dim=-1).reshape(B, -1).mean(dim=-1)


def geodesic_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """측지 회전 오차(도). 석우님 실험의 주 지표와 같은 정의. (B, T, 90) -> (B,)."""
    B, T, _ = pred.shape
    Rp = rot.rot6d_to_matrix(pred.reshape(B, T, N_JOINTS, 6))
    Rg = rot.rot6d_to_matrix(gt.reshape(B, T, N_JOINTS, 6))
    return torch.rad2deg(rot.geodesic_angle(Rp, Rg)).reshape(B, -1).mean(dim=-1)


def motion_magnitude(seq: torch.Tensor) -> torch.Tensor:
    """gap 구간의 프레임간 평균 회전 속도(도/frame). (B, T, 90) -> (B,).

    0816 회의 7번에서 지적한 "가려진 구간에서 모델이 멈춰 있으려 한다"를 직접 재는 값이다.
    예측의 이 값을 정답의 값으로 나눈 비율이 1보다 한참 작으면, 모델이 평균으로
    회귀해 움직임을 죽이고 있다는 뜻이다(regression to the mean).
    """
    B, T, _ = seq.shape
    if T < 2:
        return torch.zeros(B, device=seq.device)
    R = rot.rot6d_to_matrix(seq.reshape(B, T, N_JOINTS, 6))
    return torch.rad2deg(rot.geodesic_angle(R[:, :-1], R[:, 1:])).reshape(B, -1).mean(-1)


def npss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Normalized Power Spectrum Similarity. (B, T, D) -> (B,).

    회의록 구현과 동일: 시간축 FFT 파워 -> 특징별 정규화 -> |차이| 합 -> gt 에너지 가중.
    """
    pf = torch.fft.fft(pred, dim=1).abs() ** 2                          # (B,T,D)
    gf = torch.fft.fft(gt, dim=1).abs() ** 2
    pn = pf / (pf.sum(dim=1, keepdim=True) + 1e-8)
    gn = gf / (gf.sum(dim=1, keepdim=True) + 1e-8)
    diff = (gn - pn).abs().sum(dim=1)                                  # (B,D)
    w = gf.sum(dim=1)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return (diff * w).sum(dim=1)


# ---------------------------------------------------------------------------
# 회의록 원본(numpy, 윈도우 1개씩) 구현 — 벡터화 버전 검증용
# ---------------------------------------------------------------------------
def _reference_npss(pred_seq: np.ndarray, gt_seq: np.ndarray) -> float:
    pred_fft = np.abs(np.fft.fft(pred_seq, axis=0)) ** 2
    gt_fft = np.abs(np.fft.fft(gt_seq, axis=0)) ** 2
    gt_norm = gt_fft / (gt_fft.sum(axis=0, keepdims=True) + 1e-8)
    pred_norm = pred_fft / (pred_fft.sum(axis=0, keepdims=True) + 1e-8)
    diff = np.abs(gt_norm - pred_norm).sum(axis=0)
    weight = gt_fft.sum(axis=0)
    weight = weight / (weight.sum() + 1e-8)
    return float((diff * weight).sum())


def _reference_l2q(pred_90: np.ndarray, gt_90: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    def to_mat(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).double()
        return rot.rot6d_to_matrix(t.reshape(-1, 6)).numpy()

    def to_q(m):
        q = Rotation.from_matrix(m).as_quat()      # scipy: [x,y,z,w]
        return q[:, [3, 0, 1, 2]]

    pq, gq = to_q(to_mat(pred_90)), to_q(to_mat(gt_90))
    d = np.minimum(np.linalg.norm(pq - gq, axis=-1), np.linalg.norm(pq + gq, axis=-1))
    return float(d.mean())
