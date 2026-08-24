"""
수어(sign language) 손 동작 in-betweening 공용 코어 모듈.

SHREC/DHG 때 쓴 `../data_shrec_dhg/ib_core.py`와 짝을 이루지만 표현이 다르다 — 여기 데이터
(CSL-Daily, How2Sign)는 위치가 아니라 **관절 회전(rot6D)** 으로 저장돼 있다(SignSparK 전처리본).
그래서 뼈 길이 문제 자체가 없고, 회전 공간에서 바로 지표를 정의할 수 있다.

이 폴더(`data_signlang/`)는 `data_shrec_dhg/`와 완전히 독립적으로 동작하도록 만들어져 있다
(두 라운드를 명확히 구분하기 위해, `ib_core.py`를 import하지 않고 필요한 회전 유틸을
아래에 직접 구현함 — rot_to_6d/rot_from_6d/_slerp_rot은 ib_core.py의 것과 동일한 열(column)
기반 구현. 참고: geodesic 오차·SLERP 보간 모두 행/열 규약에 수학적으로 불변이라
(대각합 항등식 tr(AᵀB)=tr(ABᵀ), SO(3)에서의 켤레 대칭성으로 실측 검증 완료) 이 규약을
써도 결과 수치는 정확하다. 다만 3D로 실제 손 모양을 그릴 때는 SignSparK의 진짜 규약(행 기반)을
써야 하고, 그건 `visualize_sign_sample.py`가 별도로 구현한다.

## 관절 수 불일치 (실측으로 확인)
CSL-Daily는 손 하나당 16관절(96차원), How2Sign은 15관절(90차원)이다.
관절별 프레임간 회전각을 실측 비교한 결과, 인덱스 0~14는 두 데이터셋에서 크기 패턴이
거의 동일했고, CSL-Daily의 인덱스15만 유독 회전량이 컸다(평균 8.8도/frame, 다음으로 큰
관절의 1.5배) — 손 전체의 전역 방향(global wrist orientation)으로 추정된다.
→ **두 데이터셋을 합쳐 쓸 땐 공통 15관절(0~14)만 사용한다.** CSL-Daily만 쓸 때는
   COMMON_JOINTS=16으로 인덱스15(전역 방향)까지 포함할 수 있다.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np

LMDB_ROOT = Path(r"C:\tmp\signspark_data")

N_JOINTS_RAW = {"CSL-Daily": 16, "How2Sign": 15}
COMMON_JOINTS = 15          # 두 데이터셋을 합칠 때 쓰는 공통 관절 수 (앞 15개)
FEAT_DIM = COMMON_JOINTS * 6  # = 90, 회전만 (위치 정보 없음 — MANO shape 파라미터가 없어 FK 불가)


# ---------------------------------------------------------------------------
# 회전 유틸 (ib_core.py와 동일한 열 기반 구현 — 이 폴더를 독립적으로 유지하기 위해 복제)
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """(...,3,3) 회전행렬 → 6D 연속 표현 (Zhou et al. 2019). 앞 두 열을 이어붙인 것."""
    return R[..., :, :2].transpose(*range(R.ndim - 2), -1, -2).reshape(*R.shape[:-2], 6)


def rot_from_6d(d6: np.ndarray) -> np.ndarray:
    """6D → 회전행렬. Gram-Schmidt로 직교화."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _slerp_rot(R0: np.ndarray, R1: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """두 회전행렬 사이를 축-각 보간. (len(ws),3,3) 반환."""
    dR = R0.T @ R1
    tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    ang = np.arccos(tr)
    if ang < 1e-6:
        return np.repeat(R0[None], len(ws), axis=0)
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
    out = []
    for w in ws:
        a = ang * w
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        out.append(R0 @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K))
    return np.stack(out)


# ---------------------------------------------------------------------------
def open_lmdb(dataset: str, split: str):
    import lmdb
    path = LMDB_ROOT / split / f"{dataset}_reopt_{split}.lmdb"
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True, max_readers=8)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__meta__"))
    return env, list(meta["clip_ids"])


def load_clip(env, cid) -> dict:
    with env.begin() as txn:
        raw = txn.get(cid.encode() if isinstance(cid, str) else cid)
    return dict(np.load(io.BytesIO(raw), allow_pickle=True))


def hand_state(feat: np.ndarray, hand: str) -> np.ndarray:
    """(T, J_raw*6) → (T, COMMON_JOINTS*6). 공통 관절만 잘라낸다.
    hand는 'left'|'right' — 표기용, 실제 처리는 동일 (SignSparK 컨벤션상 left도 이미
    right-hand 프레임으로 통일하는 옵션이 있지만, 이 스파이크에서는 원본 그대로 씀)."""
    T = len(feat)
    J = feat.shape[1] // 6
    return feat.reshape(T, J, 6)[:, :COMMON_JOINTS].reshape(T, COMMON_JOINTS * 6)


def rel_rotation_6d(state6d: np.ndarray) -> np.ndarray:
    """(T, J*6) 상태에서 프레임간 상대회전(각속도 역할)을 6D로. 첫 프레임은 항등회전(=0 아님, 6D의
    항등에 해당하는 [1,0,0,0,1,0])으로 채운다. SILK의 '속도는 입력에만' 트릭을 회전 공간에 맞게 구현."""
    T = len(state6d)
    J = state6d.shape[1] // 6
    R = rot_from_6d(state6d.reshape(T, J, 6))
    R_prev = np.concatenate([R[:1], R[:-1]], axis=0)
    Rrel = np.einsum("tjik,tjil->tjkl", R_prev, R)  # R_prev^T @ R_cur
    return rot_to_6d(Rrel).reshape(T, J * 6)


def geodesic_error_deg(pred6d: np.ndarray, gt6d: np.ndarray) -> float:
    """두 rot6D 시퀀스 사이의 평균 측지 회전 오차 (도). SHREC 때의 L2P에 대응하는 핵심 지표."""
    T = len(pred6d)
    J = pred6d.shape[1] // 6
    Rp = rot_from_6d(pred6d.reshape(T, J, 6))
    Rg = rot_from_6d(gt6d.reshape(T, J, 6))
    Rrel = np.einsum("tjik,tjil->tjkl", Rp, Rg)
    tr = np.clip((np.einsum("tjii->tj", Rrel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)).mean())


# ---------------------------------------------------------------------------
# 학습 없는 기준선 (SLERP / hold) — SHREC 때 run_baselines.py와 같은 역할
# ---------------------------------------------------------------------------
def baseline_hold(win6d: np.ndarray, ctx: int) -> np.ndarray:
    n_trans = len(win6d) - ctx - 1
    return np.repeat(win6d[ctx - 1][None], n_trans, axis=0)


def baseline_slerp(win6d: np.ndarray, ctx: int) -> np.ndarray:
    """관절별 SLERP. 관절축/프레임축 모두 벡터화(파이썬 루프 없음) — 2026-08-16, exp02가 이
    함수를 매 샘플(`__getitem__`)마다 호출하면서 CPU 병목(260ms/step, 베이스라인의 9배)이던
    걸 계기로 재작성. 옛 루프 버전과 실제 데이터 기준 최대오차 4.44e-16(기계정밀도)로 일치
    검증(`test_slerp_vectorized.py`), 호출 1회당 15.7배 빨라짐(3.92ms -> 0.25ms)."""
    n_trans = len(win6d) - ctx - 1
    J = win6d.shape[1] // 6
    R0 = rot_from_6d(win6d[ctx - 1].reshape(J, 6))   # (J,3,3)
    R1 = rot_from_6d(win6d[-1].reshape(J, 6))         # (J,3,3)
    ws = np.arange(1, n_trans + 1) / (n_trans + 1)    # (n_trans,)

    dR = np.einsum("jik,jil->jkl", R0, R1)            # (J,3,3), R0^T @ R1
    tr = np.clip((np.einsum("jii->j", dR) - 1) / 2, -1, 1)
    ang = np.arccos(tr)                               # (J,)
    small = ang < 1e-6                                # 회전량 거의 0인 관절 (축이 정의 안 됨)
    safe_sin = np.where(small, 1.0, np.sin(ang))       # 0으로 나누기 방지, small인 곳은 아래서 덮어씀

    axis = np.stack([dR[:, 2, 1] - dR[:, 1, 2],
                      dR[:, 0, 2] - dR[:, 2, 0],
                      dR[:, 1, 0] - dR[:, 0, 1]], axis=-1) / (2 * safe_sin[:, None])  # (J,3)

    z = np.zeros(J)
    K = np.stack([
        np.stack([z, -axis[:, 2], axis[:, 1]], axis=-1),
        np.stack([axis[:, 2], z, -axis[:, 0]], axis=-1),
        np.stack([-axis[:, 1], axis[:, 0], z], axis=-1),
    ], axis=-2)                                        # (J,3,3) 각 관절의 skew-symmetric 행렬
    K2 = np.einsum("jik,jkl->jil", K, K)                # (J,3,3), K@K

    a = ang[:, None] * ws[None, :]                      # (J,n_trans)
    sin_a = np.sin(a)[:, :, None, None]
    cos_a = np.cos(a)[:, :, None, None]
    R_delta = np.eye(3) + sin_a * K[:, None] + (1 - cos_a) * K2[:, None]  # (J,n_trans,3,3), Rodrigues
    R_out = np.einsum("jab,jtbc->jtac", R0, R_delta)    # (J,n_trans,3,3), R0 @ R_delta
    R_out = np.where(small[:, None, None, None], R0[:, None], R_out)  # 정지 관절은 R0 그대로

    out = R_out.transpose(1, 0, 2, 3)                   # (n_trans,J,3,3) -- 기존 반환 shape 유지
    return rot_to_6d(out).reshape(n_trans, J * 6)


BASELINES = {"hold": baseline_hold, "slerp": baseline_slerp}
