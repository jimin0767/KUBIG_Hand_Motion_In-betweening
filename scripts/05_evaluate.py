"""학습된 체크포인트를 공용 프로토콜로 평가하고 기준선과 비교한 표를 만든다.

    python scripts/05_evaluate.py                 # results/의 모든 ckpt_*.pt
    python scripts/05_evaluate.py abs delta       # 지정한 태그만
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib.baselines import hold_fill, slerp_fill                  # noqa: E402
from hmib.data import make_model_input                            # noqa: E402
from hmib.evaluate import evaluate, format_table, make_subset, save  # noqa: E402
from hmib.mano import ManoFK                                      # noqa: E402
from hmib.models import SilkHandEncoder                           # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = "test"


def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = ck["args"]
    m = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                        dim_ff=4 * a["d_model"],
                        residual=a["variant"].startswith("delta")).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, a


def model_fill(model, variant):
    def fn(b):
        base = slerp_fill(b["x"], b["T"]) if variant.startswith("delta") else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base)
        return out.float()
    return fn


def main():
    tags = sys.argv[1:] or [p.stem.replace("ckpt_", "")
                            for p in sorted(RESULTS.glob("ckpt_*.pt"))]
    index_path = TEAM_INDEX_DIR / f"{SPLIT}_index.npz"
    subset = make_subset(index_path)
    fk = ManoFK("RIGHT", device=DEVICE)
    all_res = {}

    for name, fn in (("SLERP", slerp_fill), ("Hold", hold_fill)):
        p = RESULTS / f"baseline_{name.lower()}_flip_{SPLIT}.json"
        if p.exists():
            all_res[name] = {int(k): v for k, v in
                             json.loads(p.read_text(encoding="utf-8"))["results"].items()}
        else:
            all_res[name] = evaluate(lambda b, f=fn: f(b["x"], b["T"]), SPLIT,
                                     index_path, subset, device=DEVICE, fk=fk)

    for tag in tags:
        ck = RESULTS / f"ckpt_{tag}.pt"
        if not ck.exists():
            print(f"체크포인트 없음: {ck}")
            continue
        model, a = load_model(ck)
        res = evaluate(model_fill(model, a["variant"]), SPLIT, index_path, subset,
                       device=DEVICE, fk=fk)
        all_res[tag] = res
        save(res, RESULTS / f"eval_{tag}_{SPLIT}.json",
             {"variant": a["variant"], "steps": a["steps"], "split": SPLIT})
        print(format_table(res, f"--- {tag} ({a['variant']}, {a['steps']:,} steps)"))
        print()

    # 비교표 (SLERP 대비 개선율)
    lines = ["", "=" * 96,
             f"SLERP 대비 개선율  (음수 = SLERP보다 나쁨)   split={SPLIT}", "=" * 96,
             f"{'method':<14}" + "".join(f"{'T='+str(t):>19}" for t in (5, 10, 20, 30)),
             f"{'':<14}" + "".join(f"{'L2Q':>9}{'개선%':>10}" for _ in range(4))]
    base = all_res["SLERP"]
    for name, res in all_res.items():
        row = f"{name:<14}"
        for t in (5, 10, 20, 30):
            q = res[t]["L2Q"]
            imp = (base[t]["L2Q"] - q) / base[t]["L2Q"] * 100
            row += f"{q:>9.4f}{imp:>+9.1f}%" if name != "SLERP" else f"{q:>9.4f}{'—':>10}"
        lines.append(row)
    out = "\n".join(lines)
    print(out)
    (RESULTS / f"comparison_{SPLIT}.txt").write_text(out, encoding="utf-8")
    (RESULTS / f"comparison_{SPLIT}.json").write_text(
        json.dumps({k: {str(t): v for t, v in r.items()} for k, r in all_res.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
