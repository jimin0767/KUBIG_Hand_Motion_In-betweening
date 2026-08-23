"""전체 test셋 평가 — 김석우님이 8/23에 전수 평가로 넘어가서 눈높이를 맞춘다.

그분 보고 n(T=5:147,748 / 10:143,382 / 20:134,664 / 30:126,206)이 우리 공용
test_index.npz의 T별 전체 개수와 정확히 일치한다. 즉 같은 인덱스 파일의 전수다.
부분집합(5,000)과 전수는 우리 안정성 검정상 +-0.4% 안에서 같아야 하지만,
직접 비교표를 만들려면 같은 모수로 재는 게 깔끔하다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib.baselines import hold_fill, slerp_fill                     # noqa: E402
from hmib.data import make_model_input                               # noqa: E402
from hmib.evaluate import evaluate, format_table, make_subset, save  # noqa: E402
from hmib.mano import ManoFK                                         # noqa: E402
from hmib.models import SilkHandEncoder                              # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = "test"
ALL = 10**9          # per_t를 크게 주면 make_subset이 해당 T의 전체를 반환


def main():
    index_path = TEAM_INDEX_DIR / f"{SPLIT}_index.npz"
    subset = make_subset(index_path, per_t=ALL)
    print("전수 평가 대상:", {t: f"{len(v):,}" for t, v in subset.items()}, flush=True)
    fk = ManoFK("RIGHT", device=DEVICE)
    out = {}

    for name, fn in (("SLERP", slerp_fill), ("Hold", hold_fill)):
        out[name] = evaluate(lambda b, f=fn: f(b["x"], b["T"]), SPLIT, index_path,
                             subset, device=DEVICE, fk=fk, batch_size=512)
        print(format_table(out[name], f"--- {name} (전수)"), flush=True)

    for ck_path in sorted(RESULTS.glob("ckpt_*.pt")):
        tag = ck_path.stem.replace("ckpt_", "")
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        a = ck["args"]
        m = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                            dim_ff=4 * a["d_model"],
                            residual=a["variant"].startswith("delta")).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.eval()

        def fill(b, m=m, variant=a["variant"]):
            base = slerp_fill(b["x"], b["T"]) if variant.startswith("delta") else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=DEVICE == "cuda"):
                return m(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base).float()

        out[tag] = evaluate(fill, SPLIT, index_path, subset, device=DEVICE, fk=fk,
                            batch_size=512)
        print(format_table(out[tag], f"--- {tag} (전수)"), flush=True)

    (RESULTS / "fulltest_test.json").write_text(
        json.dumps({k: {str(t): v for t, v in r.items()} for k, r in out.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("저장: results/fulltest_test.json")


if __name__ == "__main__":
    main()
