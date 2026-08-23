"""LMDB -> 연속 배열 캐시 변환 + 팀 공용 인덱스 정합성 검증.

팀이 이미 만들어 배포한 {train,dev,test}_index.npz를 그대로 쓰려면, 그 안의
clip_idx가 가리키는 클립 순서가 우리 LMDB 순서와 정확히 같아야 한다.
여기서 clip_ids를 대조해 어긋나면 즉시 실패시킨다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib.data import DATA_ROOT, build_cache  # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")


def main():
    splits = sys.argv[1:] or ["dev", "test", "train"]
    for split in splits:
        lmdb_path = DATA_ROOT / split / f"How2Sign_reopt_{split}.lmdb"
        if not lmdb_path.exists():
            print(f"[{split}] LMDB 없음 -> 건너뜀 ({lmdb_path})")
            continue
        idx_path = TEAM_INDEX_DIR / f"{split}_index.npz"
        clip_ids = None
        if idx_path.exists():
            clip_ids = np.load(idx_path, allow_pickle=True)["clip_ids"]
        print(f"\n=== {split} ===")
        build_cache(split, index_clip_ids=clip_ids)


if __name__ == "__main__":
    main()
