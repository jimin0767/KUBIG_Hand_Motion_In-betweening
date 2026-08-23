"""MANO forward kinematics — smplx / chumpy 없이 직접 구현.

왜 직접 구현했나
----------------
공식 MANO pkl은 chumpy 객체를 담고 있고, chumpy는 Python 3.11+에서 import 자체가
안 된다(제거된 inspect.getargspec 사용). 그래서 smplx 경로가 이 환경에서는 막힌다.

대신 우리에게 필요한 건 '관절 위치'뿐이고, betas=0이면 필요한 값이 전부 순수
ndarray로 pkl 안에 들어있다:
  - J             (16, 3)  rest pose 관절 위치  (= J_regressor @ v_template, betas=0)
  - kintree_table (2, 16)  부모 인덱스
mesh vertex(LBS, shapedirs, posedirs)는 L2P 계산에 필요 없으므로 건드리지 않는다.
따라서 이 FK 결과는 smplx가 betas=0, global_orient=0, transl=0으로 낼 값과 동일하다.

검증: 첫 뼈 길이가 0.09064로, 팀(이신영)이 smplx 경로로 측정한 값과 일치한다.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


class _ChumpyStub:
    """pkl 안의 chumpy 객체를 흡수하는 자리표시자 (우리는 안 쓰는 shapedirs 뿐)."""

    def __setstate__(self, state):  # noqa: D105
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def load_mano(side: str = "RIGHT") -> dict:
    path = ASSET_DIR / f"MANO_{side.upper()}.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            return _Unpickler(f, encoding="latin1").load()


class ManoFK:
    """15관절 회전 -> 16관절 3D 위치. global_orient=0, betas=0, transl=0 고정.

    입력은 회전행렬 (..., 15, 3, 3), 출력은 (..., 16, 3) [미터].
    관절 순서: 0=wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb.
    """

    def __init__(self, side: str = "RIGHT", device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32):
        m = load_mano(side)
        J = np.asarray(m["J"], dtype=np.float64)                 # (16, 3)
        parents = np.asarray(m["kintree_table"], dtype=np.int64)[0].copy()
        parents[0] = -1                                          # 루트 표시
        self.parents = parents
        self.device, self.dtype = torch.device(device), dtype
        self.J = torch.tensor(J, dtype=dtype, device=self.device)
        # 부모로부터의 상대 위치 (rest bone vector)
        rel = J.copy()
        rel[1:] = J[1:] - J[parents[1:]]
        self.rel = torch.tensor(rel, dtype=dtype, device=self.device)   # (16, 3)

    def __call__(self, rotmats: torch.Tensor) -> torch.Tensor:
        """(..., 15, 3, 3) -> (..., 16, 3)."""
        lead = rotmats.shape[:-3]
        R = rotmats.reshape(-1, 15, 3, 3).to(self.dtype)
        N = R.shape[0]
        eye = torch.eye(3, dtype=self.dtype, device=R.device).expand(N, 1, 3, 3)
        R = torch.cat([eye, R], dim=1)                            # 손목(=항등) 붙여 16개
        rel = self.rel.to(R.device).expand(N, 16, 3)

        g_rot = [R[:, 0]]
        g_pos = [rel[:, 0]]
        for j in range(1, 16):
            p = self.parents[j]
            g_rot.append(torch.matmul(g_rot[p], R[:, j]))
            g_pos.append(g_pos[p] + torch.einsum("nij,nj->ni", g_rot[p], rel[:, j]))
        out = torch.stack(g_pos, dim=1)                           # (N, 16, 3)
        return out.reshape(*lead, 16, 3)

    def to(self, device):
        self.device = torch.device(device)
        self.J = self.J.to(device)
        self.rel = self.rel.to(device)
        return self

    def global_transforms(self, rotmats: torch.Tensor):
        """(..., 15, 3, 3) -> (전역 회전 (N,16,3,3), 관절 위치 (N,16,3))."""
        R = rotmats.reshape(-1, 15, 3, 3).to(self.dtype)
        N = R.shape[0]
        eye = torch.eye(3, dtype=self.dtype, device=R.device).expand(N, 1, 3, 3)
        R = torch.cat([eye, R], dim=1)
        rel = self.rel.to(R.device).expand(N, 16, 3)
        g_rot, g_pos = [R[:, 0]], [rel[:, 0]]
        for j in range(1, 16):
            p = self.parents[j]
            g_rot.append(torch.matmul(g_rot[p], R[:, j]))
            g_pos.append(g_pos[p] + torch.einsum("nij,nj->ni", g_rot[p], rel[:, j]))
        return torch.stack(g_rot, 1), torch.stack(g_pos, 1)


class ManoLBS(ManoFK):
    """관절 위치에 더해 메쉬 정점까지 복원 (시각화용). betas=0.

    shapedirs(유일한 chumpy 객체)는 betas=0이면 쓰이지 않으므로, posedirs·weights·
    v_template·faces만으로 공식 MANO와 동일한 메쉬가 나온다.
    """

    def __init__(self, side: str = "RIGHT", device="cpu", dtype=torch.float32):
        super().__init__(side, device, dtype)
        m = load_mano(side)
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=self.device)  # noqa: E731
        self.v_template = t(m["v_template"])              # (778, 3)
        self.posedirs = t(m["posedirs"])                  # (778, 3, 135)
        self.weights = t(m["weights"])                    # (778, 16)
        self.faces = np.asarray(m["f"], dtype=np.int64)   # (1538, 3)

    def vertices(self, rotmats: torch.Tensor) -> torch.Tensor:
        """(..., 15, 3, 3) -> (N, 778, 3)."""
        R = rotmats.reshape(-1, 15, 3, 3).to(self.dtype)
        N = R.shape[0]
        eye = torch.eye(3, dtype=self.dtype, device=R.device)
        # pose blend shape: (R - I)를 펼친 135차원
        pose_feat = (R - eye).reshape(N, 135)
        v = self.v_template[None] + torch.einsum("vdp,np->nvd", self.posedirs, pose_feat)
        g_rot, g_pos = self.global_transforms(R)                       # (N,16,3,3),(N,16,3)
        Jr = self.J[None].expand(N, 16, 3)
        t_off = g_pos - torch.einsum("nkij,nkj->nki", g_rot, Jr)       # (N,16,3)
        Rw = torch.einsum("vk,nkij->nvij", self.weights, g_rot)
        tw = torch.einsum("vk,nkj->nvj", self.weights, t_off)
        return torch.einsum("nvij,nvj->nvi", Rw, v) + tw
