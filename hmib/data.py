"""데이터 캐시 빌드 + Dataset/collate.

캐시 설계
---------
LMDB를 학습 중에 직접 읽으면 매 샘플마다 npz 압축 해제가 반복된다(에폭당 163만 윈도우).
대신 한 번만 훑어서 모든 손 시퀀스를 하나의 연속 float32 배열로 펼쳐두고, 학습·평가는
memmap 슬라이싱만 한다. train 캐시가 약 3.8GB라 RAM 31GB 환경에서는 통째로 페이지
캐시에 올라가 디스크 I/O가 사실상 사라진다.

float16으로 줄이지 않은 이유: 6D 값이 [-1,1] 범위인데 fp16 해상도가 약 1e-3이라
우리가 재려는 L2Q(0.02~0.09)의 소수점 셋째 자리를 오염시킨다.

왼손은 캐시에 원본(SMPLX-left) 그대로 저장하고, flip은 Dataset에서 적용한다.
이렇게 해야 flip 유무 비교 실험이 가능하고, 캐시가 원본에 충실하게 남는다.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import CTX, FEAT_DIM, N_JOINTS, TRAIN_T_RANGE
from .rotation import LEFT_FLIP_6D

DATA_ROOT = Path(r"C:\tmp\signspark_data")
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DATASET = "How2Sign"

_FLIP_NP = LEFT_FLIP_6D.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 캐시 빌드
# ---------------------------------------------------------------------------
def open_lmdb(split: str):
    import lmdb
    path = DATA_ROOT / split / f"{DATASET}_reopt_{split}.lmdb"
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True, max_readers=16)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__meta__"))
    return env, list(meta["clip_ids"])


def build_cache(split: str, index_clip_ids=None, verbose: bool = True) -> dict:
    """LMDB -> cache/{split}_feat.bin + {split}_meta.npz. 이미 있으면 건너뛴다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = CACHE_DIR / f"{split}_feat.bin"
    meta_path = CACHE_DIR / f"{split}_meta.npz"
    if bin_path.exists() and meta_path.exists():
        if verbose:
            print(f"[{split}] 캐시 이미 존재 -> 건너뜀")
        return dict(np.load(meta_path, allow_pickle=True))

    env, lmdb_ids = open_lmdb(split)
    if index_clip_ids is not None:
        same = list(index_clip_ids) == lmdb_ids
        if not same:
            raise AssertionError(
                f"[{split}] 팀 인덱스의 clip_ids 순서가 LMDB와 다릅니다 -- clip_idx가 어긋납니다.")
        if verbose:
            print(f"[{split}] 팀 인덱스 clip_ids {len(lmdb_ids):,}개 순서 일치 확인")

    offsets = np.zeros((len(lmdb_ids), 2), dtype=np.int64)
    lengths = np.zeros(len(lmdb_ids), dtype=np.int32)
    cursor = 0
    with env.begin() as txn, open(bin_path, "wb") as out:
        for i, cid in enumerate(lmdb_ids):
            clip = dict(np.load(io.BytesIO(txn.get(cid.encode())), allow_pickle=True))
            n = 0
            for h, key in enumerate(("right_features", "left_features")):
                f = np.asarray(clip[key], dtype=np.float32)
                J = f.shape[1] // 6
                f = f.reshape(len(f), J, 6)[:, :N_JOINTS].reshape(len(f), FEAT_DIM)
                offsets[i, h] = cursor
                out.write(np.ascontiguousarray(f).tobytes())
                cursor += len(f)
                n = len(f)
            lengths[i] = n
            if verbose and (i + 1) % 5000 == 0:
                print(f"  {i+1:,}/{len(lmdb_ids):,}  ({cursor*FEAT_DIM*4/1e9:.2f} GB)")
    env.close()
    meta = {"clip_ids": np.array(lmdb_ids, dtype=object),
            "offsets": offsets, "lengths": lengths,
            "total_frames": np.int64(cursor)}
    np.savez(meta_path, **meta)
    if verbose:
        print(f"[{split}] 캐시 완료: {cursor:,} 프레임, {cursor*FEAT_DIM*4/1e9:.2f} GB")
    return meta


def load_cache(split: str):
    meta = dict(np.load(CACHE_DIR / f"{split}_meta.npz", allow_pickle=True))
    feat = np.memmap(CACHE_DIR / f"{split}_feat.bin", dtype=np.float32, mode="r",
                     shape=(int(meta["total_frames"]), FEAT_DIM))
    return feat, meta


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """팀 공용 인덱스(train/dev/test_index.npz)를 그대로 소비하는 윈도우 로더.

    train: (clip, hand, start)만 있고 T는 매 샘플 [5,30] 균일 랜덤.
    eval : (clip, hand, start, T) 고정.
    """

    def __init__(self, split, index_path, apply_flip=True, subset=None):
        self.split = split
        self.feat, self.meta = load_cache(split)
        idx = np.load(index_path, allow_pickle=True)
        self.clip_idx = idx["clip_idx"]
        self.hand = idx["hand"]
        self.start = idx["start"]
        self.T = idx["T"] if "T" in idx.files else None
        if subset is not None:
            self.clip_idx = self.clip_idx[subset]
            self.hand = self.hand[subset]
            self.start = self.start[subset]
            if self.T is not None:
                self.T = self.T[subset]
        self.offsets = self.meta["offsets"]
        self.apply_flip = apply_flip
        self.is_train = self.T is None

    def __len__(self):
        return len(self.clip_idx)

    def __getitem__(self, i):
        c = int(self.clip_idx[i])
        h = int(self.hand[i])
        st = int(self.start[i])
        if self.is_train:
            T = int(np.random.randint(TRAIN_T_RANGE[0], TRAIN_T_RANGE[1] + 1))
        else:
            T = int(self.T[i])
        L = CTX + T + 1
        off = int(self.offsets[c, h])
        w = np.array(self.feat[off + st: off + st + L], dtype=np.float32)
        if h == 1 and self.apply_flip:
            w = (w.reshape(L, N_JOINTS, 6) * _FLIP_NP).reshape(L, FEAT_DIM)
        return torch.from_numpy(w), T


def collate(batch):
    """가변 길이 윈도우를 패딩해 배치로. 목표 프레임 기준 상대 위치(rel)도 만든다."""
    Ts = torch.tensor([T for _, T in batch], dtype=torch.long)
    Lmax = int((CTX + Ts + 1).max())
    B = len(batch)
    x = torch.zeros(B, Lmax, FEAT_DIM)
    obs = torch.zeros(B, Lmax, dtype=torch.bool)      # 관측(context/target) 여부
    valid = torch.zeros(B, Lmax, dtype=torch.bool)    # 패딩이 아닌 실제 프레임
    rel = torch.zeros(B, Lmax, dtype=torch.long)      # 0816 회의: 목표 프레임이 항상 0
    for b, (w, T) in enumerate(batch):
        L = CTX + T + 1
        x[b, :L] = w
        valid[b, :L] = True
        obs[b, :CTX] = True
        obs[b, L - 1] = True
        rel[b, :L] = torch.arange(L) - (L - 1)
    return {"x": x, "obs": obs, "valid": valid, "rel": rel, "T": Ts}


def make_model_input(x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """(B,L,90) + 관측마스크 -> (B,L,181) = 0채움 상태 90 + 속도 90 + 관측플래그 1.

    속도는 반드시 0-채움 이후의 값으로 계산한다. 순서를 바꾸면 gap 구간의 정답이
    속도 채널을 통해 모델로 새어 들어간다(오은하 문서에서 지적된 leakage).
    """
    s = x * obs.unsqueeze(-1)
    v = torch.zeros_like(s)
    v[:, 1:] = s[:, 1:] - s[:, :-1]
    return torch.cat([s, v, obs.unsqueeze(-1).to(x.dtype)], dim=-1)
