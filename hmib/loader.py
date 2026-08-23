"""학습용 고속 배치 샘플러.

DataLoader + per-sample 파이썬 루프 대신, 배치 하나를 numpy fancy indexing 한 번으로
만든다. 윈도우 최대 길이가 41(=10+30+1)로 고정이라 (B,41,90) 버퍼에 바로 채울 수 있다.

train 캐시(약 3.8GB)는 통째로 RAM에 올린다. memmap 랜덤 액세스의 페이지 폴트를
없애서 GPU가 놀지 않게 하기 위함이다. RAM 31GB 환경에서 안전하다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import CTX, FEAT_DIM, N_JOINTS, TRAIN_T_RANGE
from .data import CACHE_DIR
from .rotation import LEFT_FLIP_6D

LMAX = CTX + TRAIN_T_RANGE[1] + 1        # 41
_FLIP = LEFT_FLIP_6D.numpy().astype(np.float32)


class TrainBatcher:
    def __init__(self, index_path: Path, split: str = "train", seed: int = 0,
                 apply_flip: bool = True, in_ram: bool = True):
        meta = dict(np.load(CACHE_DIR / f"{split}_meta.npz", allow_pickle=True))
        total = int(meta["total_frames"])
        path = CACHE_DIR / f"{split}_feat.bin"
        if in_ram:
            self.feat = np.fromfile(path, dtype=np.float32).reshape(total, FEAT_DIM)
        else:
            self.feat = np.memmap(path, dtype=np.float32, mode="r", shape=(total, FEAT_DIM))
        idx = np.load(index_path, allow_pickle=True)
        offs, lens = meta["offsets"], meta["lengths"]
        ci, hd, st = idx["clip_idx"], idx["hand"], idx["start"]
        base = (offs[ci, hd] + st).astype(np.int64)

        # 평가용 인덱스(dev/test)는 start가 L=CTX+T+1 기준이라 T=5짜리 윈도우는
        # 클립 끝에서 16프레임 거리에 있을 수 있다. 학습 배처는 T를 최대 30까지
        # 뽑으므로(L=41) 그런 윈도우는 클립 경계를 넘어 '다음 클립'을 읽게 된다.
        # -> 41프레임이 확보되지 않는 윈도우는 버린다.
        room = (offs[ci, hd] + lens[ci]).astype(np.int64) - base
        keep = room >= LMAX
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"  [{split}] 41프레임 미확보 윈도우 {n_drop:,}개 제외 "
                  f"({n_drop/len(keep)*100:.1f}%) -- 클립 경계 침범 방지")
        self.base = base[keep]
        self.hand = hd[keep].astype(np.int8)
        self.n = len(self.base)
        self.rng = np.random.default_rng(seed)
        self.apply_flip = apply_flip
        self.grid = np.arange(LMAX, dtype=np.int64)[None, :]

    def batch(self, B: int, device: str = "cuda") -> dict:
        i = self.rng.integers(0, self.n, size=B)
        T = self.rng.integers(TRAIN_T_RANGE[0], TRAIN_T_RANGE[1] + 1, size=B)
        L = CTX + T + 1
        valid = self.grid < L[:, None]                                   # (B,41)
        pos = self.base[i][:, None] + np.where(valid, self.grid, 0)
        x = self.feat[pos.ravel()].reshape(B, LMAX, FEAT_DIM).copy()
        x *= valid[:, :, None]
        if self.apply_flip:
            left = self.hand[i] == 1
            if left.any():
                x[left] = (x[left].reshape(-1, LMAX, N_JOINTS, 6) * _FLIP
                           ).reshape(-1, LMAX, FEAT_DIM)
        obs = (self.grid < CTX) | (self.grid == (L - 1)[:, None])
        obs &= valid
        rel = np.where(valid, self.grid - (L - 1)[:, None], 0)
        to = lambda a, d: torch.from_numpy(np.ascontiguousarray(a)).to(device, dtype=d)  # noqa: E731
        return {"x": to(x, torch.float32),
                "obs": to(obs, torch.bool),
                "valid": to(valid, torch.bool),
                "rel": to(rel, torch.long),
                "T": to(T, torch.long)}
