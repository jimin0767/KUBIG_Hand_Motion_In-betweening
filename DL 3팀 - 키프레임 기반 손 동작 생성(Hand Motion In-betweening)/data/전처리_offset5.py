"""How2Sign 손동작 in-betweening — offset=5 전처리 인덱스 빌더 (단일 파일, 공유용)

SILK 논문(arXiv:2506.09075) 실측값 기준 프로토콜:
  - 데이터: How2Sign 단독 (CSL-Daily는 저자 공지 미해결 버그 + 관절수 불일치로 제외)
  - 양손: 오른손 원본 + 왼손(오른손 규약으로 뒤집음) 둘 다 인덱스에 포함
  - context: 10프레임 고정
  - 학습(train): 시작점을 offset=5 간격으로 두고, 실제 가림 길이 T는 5~30 사이에서
    매 샘플마다 새로 균일 랜덤 추출 (연속적, 고정 분할 아님) -> 인덱스엔 (clip, hand, start)만
    저장, T는 저장 안 함(나중에 학습 시점에 매번 다시 뽑음)
  - 평가(dev/test): T={5,10,20,30} 고정, offset=5로 겹침 허용 슬라이딩 -> (clip, hand, start, T)
  - 움직임 필터(수어 사이 정지 구간 제외)는 학습 인덱스에만 적용, 평가엔 미적용(전수 채점)

이 스크립트는 실제 프레임 데이터를 복사하지 않는다 -- 원본 LMDB의 어느 클립, 어느 지점에서
몇 프레임을 뜯어 쓸지를 가리키는 (정수) 인덱스만 만든다. 그래서 결과물이 작다(원본 LMDB
12GB와 무관, index/*.npz는 수십MB 이내).

공유해서 쓸 때: 아래 DATA_ROOT만 각자 컴퓨터의 SignSparK LMDB 경로로 바꾸면 됨
(`{DATA_ROOT}/{split}/How2Sign_reopt_{split}.lmdb` 구조를 기대함).

사용:
    python 전처리_offset5.py                    # train+dev+test 전부
    python 전처리_offset5.py --splits test        # 특정 split만
    python 전처리_offset5.py --out D:\내경로\index  # 결과 저장 위치도 바꾸고 싶으면
"""
from __future__ import annotations

import argparse
import io
import pickle
import time
from pathlib import Path

import numpy as np

# ============================================================================
# 여기만 바꾸면 됩니다 -- 각자 SignSparK LMDB가 실제로 있는 경로.
DATA_ROOT = Path(r"C:\tmp\signspark_data")
# 결과(인덱스) 저장 위치. 기본값은 이 스크립트가 있는 폴더 밑의 index/ 폴더.
OUT_DIR = Path(__file__).parent / "index"
# ============================================================================

DATASET = "How2Sign"
CTX = 10
OFFSET = 5
TRAIN_TRANS_RANGE = (5, 30)   # 학습: 이 범위에서 매번 랜덤 T
EVAL_T_VALUES = [5, 10, 20, 30]
MIN_SPEED_DEG = 0.3           # 학습 인덱스에만 적용
COMMON_JOINTS = 15            # How2Sign은 애초에 15관절 (CSL-Daily의 16번째=전역방향 채널 없음)


# ---------------------------------------------------------------------------
# sl_core.py에서 이 스크립트가 실제로 쓰는 부분만 그대로 가져옴 (열(column) 기반 6D 규약)
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def rot_from_6d(d6: np.ndarray) -> np.ndarray:
    """6D -> 회전행렬. Gram-Schmidt로 직교화."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def open_lmdb(dataset: str, split: str):
    import lmdb
    path = DATA_ROOT / split / f"{dataset}_reopt_{split}.lmdb"
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True, max_readers=8)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__meta__"))
    return env, list(meta["clip_ids"])


def load_clip(env, cid) -> dict:
    with env.begin() as txn:
        raw = txn.get(cid.encode() if isinstance(cid, str) else cid)
    return dict(np.load(io.BytesIO(raw), allow_pickle=True))


def hand_state(feat: np.ndarray) -> np.ndarray:
    """(T, J_raw*6) -> (T, COMMON_JOINTS*6). How2Sign은 원래 15관절이라 사실상 그대로 통과."""
    T = len(feat)
    J = feat.shape[1] // 6
    return feat.reshape(T, J, 6)[:, :COMMON_JOINTS].reshape(T, COMMON_JOINTS * 6)


# ---------------------------------------------------------------------------
# 실제 인덱스 빌드 로직
# ---------------------------------------------------------------------------
def _speed_cumsum(state: np.ndarray) -> np.ndarray:
    """관절 평균 프레임간 회전각의 누적합 -- 구간 평균 속도를 O(1)에 구하기 위한 전처리."""
    R = rot_from_6d(state.reshape(len(state), COMMON_JOINTS, 6))
    Rrel = np.einsum("tjik,tjil->tjkl", R[:-1], R[1:])
    tr = np.clip((np.einsum("tjii->tj", Rrel) - 1) / 2, -1, 1)
    speed = np.degrees(np.arccos(tr)).mean(-1)
    return np.concatenate([[0.0], np.cumsum(speed)])


def build_train_index(split: str):
    env, ids = open_lmdb(DATASET, split)
    L = CTX + TRAIN_TRANS_RANGE[1] + 1  # 어떤 T(<=30)를 뽑아도 안전하게 들어가도록 최대길이로 필터
    clip_idx, hand_arr, start_arr = [], [], []
    n_drop = 0
    t0 = time.time()
    for i, cid in enumerate(ids):
        clip = load_clip(env, cid)
        for hand_flag, key in ((0, "right_features"), (1, "left_features")):
            state = hand_state(clip[key])
            if len(state) < L:
                continue
            csum = _speed_cumsum(state)
            for st in range(0, len(state) - L + 1, OFFSET):
                if (csum[st + L - 1] - csum[st]) / (L - 1) < MIN_SPEED_DEG:
                    n_drop += 1
                    continue
                clip_idx.append(i); hand_arr.append(hand_flag); start_arr.append(st)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(ids)} ... {time.time()-t0:.0f}s")
    env.close()
    print(f"[{split}] 인덱스 {len(clip_idx):,}개 (움직임 필터로 제외 {n_drop:,}개), {time.time()-t0:.1f}s")
    return {
        "clip_ids": np.array(ids, dtype=object),          # 표: 정수 인덱스 -> 실제 clip_id 문자열
        "clip_idx": np.array(clip_idx, dtype=np.int32),
        "hand": np.array(hand_arr, dtype=np.int8),         # 0=right, 1=left(뒤집어 씀)
        "start": np.array(start_arr, dtype=np.int32),
    }


def build_eval_index(split: str):
    env, ids = open_lmdb(DATASET, split)
    clip_idx, hand_arr, start_arr, t_arr = [], [], [], []
    t0 = time.time()
    for i, cid in enumerate(ids):
        clip = load_clip(env, cid)
        for hand_flag, key in ((0, "right_features"), (1, "left_features")):
            T_len = len(clip[key])
            for T in EVAL_T_VALUES:
                L = CTX + T + 1
                for st in range(0, T_len - L + 1, OFFSET):
                    clip_idx.append(i); hand_arr.append(hand_flag); start_arr.append(st); t_arr.append(T)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(ids)} ... {time.time()-t0:.0f}s")
    env.close()
    print(f"[{split}] 인덱스 {len(clip_idx):,}개, {time.time()-t0:.1f}s")
    return {
        "clip_ids": np.array(ids, dtype=object),
        "clip_idx": np.array(clip_idx, dtype=np.int32),
        "hand": np.array(hand_arr, dtype=np.int8),
        "start": np.array(start_arr, dtype=np.int32),
        "T": np.array(t_arr, dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        print(f"\n=== {split} ===")
        data = build_train_index(split) if split == "train" else build_eval_index(split)
        out = out_dir / f"{split}_index.npz"
        np.savez_compressed(out, **data)
        print(f"저장: {out}  ({out.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
