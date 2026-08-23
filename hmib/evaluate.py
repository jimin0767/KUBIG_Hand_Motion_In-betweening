"""평가 하네스 — 팀 공용 프로토콜.

0816 회의 규약: T in {5,10,20,30}, gap 구간만 채점, 지표 L2Q / L2P / NPSS.
여기에 geodesic(도)을 추가로 낸다 — 김석우님 실험의 주 지표라 대조가 가능해진다.

핵심 설계: 평가 표본을 고정 시드로 뽑고 그 인덱스를 파일로 남긴다.
지금 팀은 신영 n=3,840 / 은하 n=320, n=15,360으로 서로 다른 부분집합을 써서
승패가 모델 차이인지 표본 차이인지 구분되지 않는다. 같은 인덱스 파일을 쓰면
그 문제가 사라진다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CTX, EVAL_T_VALUES
from .data import WindowDataset, collate
from .mano import ManoFK
from .metrics import geodesic_deg, l2p, l2q, motion_magnitude, npss

SUBSET_SEED = 20260820          # 팀 공용 시드 (날짜)
SUBSET_PER_T = 5000             # T값 하나당 평가 윈도우 수


def make_subset(index_path: Path, per_t: int = SUBSET_PER_T,
                seed: int = SUBSET_SEED) -> dict:
    """T값별로 고정 시드 부분집합을 뽑는다. {T: 원본 인덱스 배열}."""
    idx = np.load(index_path, allow_pickle=True)
    T = idx["T"]
    rng = np.random.default_rng(seed)
    out = {}
    for t in EVAL_T_VALUES:
        pool = np.where(T == t)[0]
        k = min(per_t, len(pool))
        out[int(t)] = np.sort(rng.choice(pool, size=k, replace=False))
    return out


@torch.no_grad()
def evaluate(fill_fn, split: str, index_path: Path, subset: dict,
             device: str = "cuda", batch_size: int = 256,
             apply_flip: bool = True, fk: ManoFK | None = None) -> dict:
    """fill_fn(batch) -> pred (B,L,90). gap 프레임만 채점해 T별 지표를 낸다."""
    fk = fk or ManoFK("RIGHT", device=device)
    results = {}
    for t, sel in subset.items():
        ds = WindowDataset(split, index_path, apply_flip=apply_flip, subset=sel)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=0)
        acc = {"L2Q": [], "L2P": [], "NPSS": [], "GEO": [], "MOT": [], "MOT_GT": []}
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = fill_fn(batch)
            gt, T = batch["x"], batch["T"]
            L = CTX + int(T[0]) + 1
            gap = slice(CTX, L - 1)                       # 생성 대상 구간만
            p, g = pred[:, gap], gt[:, gap]
            acc["L2Q"].append(l2q(p, g).cpu())
            acc["L2P"].append(l2p(p, g, fk).cpu())
            acc["NPSS"].append(npss(p, g).cpu())
            acc["GEO"].append(geodesic_deg(p, g).cpu())
            acc["MOT"].append(motion_magnitude(p).cpu())
            acc["MOT_GT"].append(motion_magnitude(g).cpu())
        results[t] = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
        results[t]["n"] = int(len(ds))
        results[t]["MOT_RATIO"] = results[t]["MOT"] / max(results[t]["MOT_GT"], 1e-8)
    return results


def format_table(results: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'T':>4} {'n':>7} {'L2Q':>9} {'L2P(m)':>9} {'NPSS':>9} {'geo(deg)':>10}"
                 f" {'움직임비':>9}")
    for t in sorted(results):
        r = results[t]
        lines.append(f"{t:>4} {r['n']:>7,} {r['L2Q']:>9.4f} {r['L2P']:>9.5f} "
                     f"{r['NPSS']:>9.4f} {r['GEO']:>10.3f} {r.get('MOT_RATIO', float('nan')):>9.3f}")
    return "\n".join(lines)


def save(results: dict, path: Path, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": {str(k): v for k, v in results.items()}, "meta": meta or {}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
