"""SILK 스타일 인코더 학습 — 절대값(abs) / 잔차(delta) 변형.

    python scripts/04_train.py --variant abs   --steps 40000
    python scripts/04_train.py --variant delta --steps 40000
    python scripts/04_train.py --variant delta_vel --steps 40000   # + 속도 보조손실
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX                                          # noqa: E402
from hmib.baselines import slerp_fill                         # noqa: E402
from hmib.data import make_model_input                        # noqa: E402
from hmib.loader import TrainBatcher                          # noqa: E402
from hmib.models import SilkHandEncoder, masked_l1, velocity_l1  # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
RESULTS = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_lr(step, total, base_lr, warmup=1000, floor=0.05):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))


@torch.no_grad()
def quick_dev_loss(model, batcher, variant, n_batches=20, B=128):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        b = batcher.batch(B, DEVICE)
        gap = b["valid"] & ~b["obs"]
        base = slerp_fill(b["x"], b["T"]) if variant.startswith("delta") else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            pred = model(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base)
        tot += masked_l1(pred.float(), b["x"], gap).item()
    model.train()
    return tot / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="abs", choices=["abs", "delta", "delta_vel"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vel_weight", type=float, default=0.5)
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.variant
    RESULTS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_b = TrainBatcher(TEAM_INDEX_DIR / "train_index.npz", "train",
                           seed=0, apply_flip=not args.no_flip)
    dev_b = TrainBatcher(TEAM_INDEX_DIR / "dev_index.npz", "dev",
                         seed=1, apply_flip=not args.no_flip)
    print(f"train 윈도우 {train_b.n:,}개 / dev {dev_b.n:,}개  (flip={not args.no_flip})")

    model = SilkHandEncoder(d_model=args.d_model, num_layers=args.layers,
                            nhead=args.heads, dim_ff=4 * args.d_model,
                            residual=args.variant.startswith("delta")).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"모델: {args.variant}  파라미터 {n_par/1e6:.1f}M  d={args.d_model} "
          f"L={args.layers} H={args.heads}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    log = []
    t0 = time.time()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        b = train_b.batch(args.batch, DEVICE)
        gap = b["valid"] & ~b["obs"]
        base = slerp_fill(b["x"], b["T"]) if args.variant.startswith("delta") else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            pred = model(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base)
        pred = pred.float()
        loss = masked_l1(pred, b["x"], gap)
        if args.variant == "delta_vel":
            loss = loss + args.vel_weight * velocity_l1(pred, b["x"], gap)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()

        if step % 1000 == 0:
            dl = quick_dev_loss(model, dev_b, args.variant)
            el = time.time() - t0
            print(f"  step {step:6,}/{args.steps:,}  train {run_loss/1000:.5f}  "
                  f"dev {dl:.5f}  {el/60:.1f}min  ({step/el:.1f} it/s)", flush=True)
            log.append({"step": step, "train": run_loss / 1000, "dev": dl, "sec": el})
            run_loss = 0.0
        if step % 10000 == 0:                      # 중간 저장 (크래시 대비)
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "n_params": n_par, "step": step}, RESULTS / f"ckpt_{tag}.pt")

    ckpt = RESULTS / f"ckpt_{tag}.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "n_params": n_par}, ckpt)
    (RESULTS / f"trainlog_{tag}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"저장: {ckpt}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
