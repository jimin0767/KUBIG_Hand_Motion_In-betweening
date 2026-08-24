"""index/*.npz(정수 인덱스)를 실제 학습 가능한 PyTorch Dataset으로 복원.

build_index.py가 만든 인덱스는 (클립, 손, 시작위치[, T]) 정수 조합만 담고 있어서, 원본
LMDB에 접근할 수 있어야 실제 프레임 데이터를 복원할 수 있다 (LMDB 자체는 각자 SignSparK
공식 경로에서 받아야 함 -- README.md 참고). x/y/mask 구성은 `../sl_dataset.py`와 동일한
규약(FEAT_FULL=181, FEAT_STATE=90)이라 `sl_model_a.SILKHand`에 코드 변경 없이 바로 쓸 수 있다.

공유해서 쓸 때: 아래 DATA_ROOT만 각자 컴퓨터의 SignSparK LMDB 경로로 바꾸면 됨 (build_index.py
상단의 DATA_ROOT와 같은 값이어야 함 -- 인덱스를 만들 때 쓴 경로와 실제 로드할 경로가 다르면 안 됨).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================================
# 여기만 바꾸면 됩니다 -- build_index.py의 DATA_ROOT와 동일하게 맞춰야 함.
DATA_ROOT = Path(r"C:\tmp\signspark_data")
# ============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data_signlang/
import sl_core as SL  # noqa: E402
SL.LMDB_ROOT = DATA_ROOT
from sl_dataset import FEAT_FULL, FEAT_STATE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "exp03_both_hands"))
from flip_utils import flip_left_to_right  # noqa: E402

HERE = Path(__file__).parent
DATASET = "How2Sign"
CTX = 10

__all__ = ["TrainWindowDataset", "EvalWindowDataset", "collate"]


def _load_clips(split: str, clip_ids: np.ndarray) -> dict:
    """(clip_id, hand) -> state(T,90) 배열. 원본 SignHandDataset과 동일하게 전부 미리 로드
    (__getitem__마다 LMDB 여는 것보다 훨씬 빠름)."""
    env, _ = SL.open_lmdb(DATASET, split)
    cache = {}
    for cid in clip_ids:
        clip = SL.load_clip(env, cid)
        right = SL.hand_state(clip["right_features"], "right")
        left_raw = SL.hand_state(clip["left_features"], "left")
        left = flip_left_to_right(left_raw.astype(np.float64)).astype(np.float32)
        cache[(cid, 0)] = right
        cache[(cid, 1)] = left
    env.close()
    return cache


def _make_sample(state: np.ndarray, st: int, T: int) -> dict:
    L = CTX + T + 1
    win = state[st:st + L].astype(np.float32)

    vel = SL.rel_rotation_6d(win.astype(np.float64)).astype(np.float32)
    mask = np.zeros(L, dtype=bool); mask[:CTX] = True; mask[-1] = True
    state_in = win.copy(); state_in[~mask] = 0.0
    vel_in = vel.copy(); vel_in[~mask] = 0.0; vel_in[-1] = 0.0  # 2026-08-21: target 위치 속도도 0으로 -- 안 그러면 target absolute+velocity 조합으로 마지막 gap 프레임 역산 가능(실제 검증된 누수)
    obs = mask.astype(np.float32)[:, None]
    x = np.concatenate([state_in, vel_in, obs], axis=-1)
    return {"x": torch.from_numpy(x), "y": torch.from_numpy(win),
            "mask": torch.from_numpy(mask), "ctx": CTX, "trans": T}


class TrainWindowDataset(Dataset):
    """train_index.npz 기반. __getitem__마다 T~Uniform[5,30]을 새로 뽑는다(고정 분할 아님 --
    시작점 grid(offset=5)는 고정, 그 위에서 길이만 매번 랜덤. SILK 논문 방식)."""

    def __init__(self, trans=(5, 30), seed=0):
        d = np.load(HERE / "index" / "train_index.npz", allow_pickle=True)
        self.clip_ids, self.clip_idx = d["clip_ids"], d["clip_idx"]
        self.hand, self.start = d["hand"], d["start"]
        self.trans_range = trans
        self.rng = np.random.default_rng(seed)
        self._cache = _load_clips("train", self.clip_ids)

    def __len__(self):
        return len(self.clip_idx)

    def __getitem__(self, i):
        cid = self.clip_ids[self.clip_idx[i]]
        state = self._cache[(cid, int(self.hand[i]))]
        T = int(self.rng.integers(self.trans_range[0], self.trans_range[1] + 1))
        return _make_sample(state, int(self.start[i]), T)


class EvalWindowDataset(Dataset):
    """dev_index.npz / test_index.npz 기반. T가 인덱스에 이미 고정돼 저장돼 있음."""

    def __init__(self, split: str):
        assert split in ("dev", "test")
        d = np.load(HERE / "index" / f"{split}_index.npz", allow_pickle=True)
        self.clip_ids, self.clip_idx = d["clip_ids"], d["clip_idx"]
        self.hand, self.start, self.T = d["hand"], d["start"], d["T"]
        self._cache = _load_clips(split, self.clip_ids)

    def __len__(self):
        return len(self.clip_idx)

    def __getitem__(self, i):
        cid = self.clip_ids[self.clip_idx[i]]
        state = self._cache[(cid, int(self.hand[i]))]
        return _make_sample(state, int(self.start[i]), int(self.T[i]))


def collate(batch):
    L = max(b["x"].shape[0] for b in batch)
    n = len(batch)
    x = torch.zeros(n, L, FEAT_FULL)
    y = torch.zeros(n, L, FEAT_STATE)
    obs = torch.zeros(n, L, dtype=torch.bool)
    pad = torch.zeros(n, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        li = b["x"].shape[0]
        x[i, :li] = b["x"]; y[i, :li] = b["y"]
        obs[i, :li] = b["mask"]; pad[i, :li] = True
    return {"x": x, "y": y, "obs_mask": obs, "pad_mask": pad,
            "trans": torch.tensor([b["trans"] for b in batch])}
