"""T=45 외삽 평가 — 학습 범위(T<=30) 밖에서 무슨 일이 일어나는가.

김석우님이 회의 전 자료에서 보고한 핵심 발견을 공용 프로토콜에서 재현한다:
  "절대값 예측 모델은 T=45에서 SLERP보다 42~52% 나빠지고, 잔차(delta) 예측만
   SLERP 수준을 유지한다."

test 인덱스에는 T={5,10,20,30}만 있으므로 같은 규칙(stride 5)으로 T=45 인덱스를
새로 만든다. 캐시의 clip 길이만 있으면 되므로 LMDB를 다시 열 필요는 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX                                              # noqa: E402
from hmib.baselines import hold_fill, slerp_fill                  # noqa: E402
from hmib.data import CACHE_DIR, make_model_input                 # noqa: E402
from hmib.evaluate import evaluate, format_table                  # noqa: E402
from hmib.mano import ManoFK                                      # noqa: E402
from hmib.models import SilkHandEncoder                           # noqa: E402

RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_EXTRA = 45
OFFSET = 5
PER_T = 5000
SEED = 20260820


def build_t45_index(split="test") -> Path:
    """팀 전처리와 같은 규칙(stride 5, 양손)으로 T=45 평가 인덱스를 만든다."""
    out = RESULTS / f"{split}_index_T45.npz"
    if out.exists():
        return out
    meta = dict(np.load(CACHE_DIR / f"{split}_meta.npz", allow_pickle=True))
    lengths = meta["lengths"]
    L = CTX + T_EXTRA + 1
    ci, hd, st, tt = [], [], [], []
    for i, ln in enumerate(lengths):
        for h in (0, 1):
            for s in range(0, int(ln) - L + 1, OFFSET):
                ci.append(i); hd.append(h); st.append(s); tt.append(T_EXTRA)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, clip_ids=meta["clip_ids"],
                        clip_idx=np.array(ci, np.int32), hand=np.array(hd, np.int8),
                        start=np.array(st, np.int32), T=np.array(tt, np.int32))
    print(f"T=45 인덱스 생성: {len(ci):,}개 윈도우 -> {out.name}")
    return out


def main():
    split = "test"
    idx_path = build_t45_index(split)
    T = np.load(idx_path, allow_pickle=True)["T"]
    rng = np.random.default_rng(SEED)
    pool = np.where(T == T_EXTRA)[0]
    subset = {T_EXTRA: np.sort(rng.choice(pool, size=min(PER_T, len(pool)), replace=False))}
    fk = ManoFK("RIGHT", device=DEVICE)
    all_res = {}

    for name, fn in (("SLERP", slerp_fill), ("Hold", hold_fill)):
        all_res[name] = evaluate(lambda b, f=fn: f(b["x"], b["T"]), split, idx_path,
                                 subset, device=DEVICE, fk=fk)
        print(format_table(all_res[name], f"--- {name}  T=45"))

    for ck_path in sorted(RESULTS.glob("ckpt_*.pt")):
        tag = ck_path.stem.replace("ckpt_", "")
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        a = ck["args"]
        m = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                            dim_ff=4 * a["d_model"],
                            residual=a["variant"].startswith("delta")).to(DEVICE)
        m.load_state_dict(ck["model"]); m.eval()

        def fill(b, m=m, variant=a["variant"]):
            base = slerp_fill(b["x"], b["T"]) if variant.startswith("delta") else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=DEVICE == "cuda"):
                return m(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base).float()

        all_res[tag] = evaluate(fill, split, idx_path, subset, device=DEVICE, fk=fk)
        print(format_table(all_res[tag], f"--- {tag}  T=45 (학습 범위 밖)"))

    base = all_res["SLERP"][T_EXTRA]["L2Q"]
    lines = ["", "=" * 62, "T=45 (학습 범위 밖) — SLERP 대비", "=" * 62,
             f"{'method':<14}{'L2Q':>9}{'개선%':>10}{'NPSS':>10}{'geo(deg)':>11}"]
    for k, r in all_res.items():
        v = r[T_EXTRA]
        imp = (base - v["L2Q"]) / base * 100
        lines.append(f"{k:<14}{v['L2Q']:>9.4f}{imp:>+9.1f}%{v['NPSS']:>10.4f}{v['GEO']:>11.3f}")
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "extrapolation_T45.txt").write_text(txt, encoding="utf-8")
    (RESULTS / "extrapolation_T45.json").write_text(
        json.dumps({k: {str(t): v for t, v in r.items()} for k, r in all_res.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
