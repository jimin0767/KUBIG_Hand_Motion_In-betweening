"""SLERP / Hold 기준선 산출 — 팀 결과표의 원점.

부가로 flip 유무 비교를 같이 낸다. 팀 슬랙에 "공용 전처리에 왼손 flip이 빠진 것
아니냐"는 미해결 질문이 있었는데, 기준선 수치로 그 영향을 정량화할 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib.baselines import hold_fill, slerp_fill      # noqa: E402
from hmib.evaluate import (SUBSET_PER_T, SUBSET_SEED, evaluate, format_table,
                           make_subset, save)          # noqa: E402
from hmib.mano import ManoFK                           # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    index_path = TEAM_INDEX_DIR / f"{split}_index.npz"
    subset = make_subset(index_path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / f"eval_subset_{split}.npz",
             **{f"T{t}": v for t, v in subset.items()},
             seed=SUBSET_SEED, per_t=SUBSET_PER_T)
    print(f"평가 부분집합 저장: results/eval_subset_{split}.npz "
          f"(seed={SUBSET_SEED}, T당 {SUBSET_PER_T}개)")
    print(f"  T별 개수: {{ {', '.join(f'{t}: {len(v):,}' for t, v in subset.items())} }}\n")

    fk = ManoFK("RIGHT", device=DEVICE)
    meta = {"split": split, "seed": SUBSET_SEED, "per_t": SUBSET_PER_T,
            "protocol": "context=10, target=1, gap only, How2Sign, 6D row convention"}

    for flip in (True, False):
        tag = "flip" if flip else "noflip"
        for name, fn in (("SLERP", slerp_fill), ("Hold", hold_fill)):
            res = evaluate(lambda b, f=fn: f(b["x"], b["T"]), split, index_path,
                           subset, device=DEVICE, apply_flip=flip, fk=fk)
            print(format_table(res, f"--- {name}  [{tag}]  split={split}"))
            print()
            save(res, RESULTS / f"baseline_{name.lower()}_{tag}_{split}.json",
                 {**meta, "method": name, "apply_flip": flip})


if __name__ == "__main__":
    main()
