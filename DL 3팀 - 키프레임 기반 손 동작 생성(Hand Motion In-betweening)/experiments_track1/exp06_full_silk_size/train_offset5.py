"""exp06(SILK 원 논문 크기, d_model=1024)를 새 전처리(offset=5, 양손, How2Sign)로 학습.
`../../preprocess_offset5/train.py`와 동일 루프, 모델만 `model.build_model()`(원 논문 크기)로 교체.

사용:
    python train_offset5.py --steps 127560 --batch_size 64
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # data_signlang/
import sl_core as SL  # noqa: E402
from sl_model_a import l1_loss  # noqa: E402
from model import build_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preprocess_offset5"))
from windowed_dataset import TrainWindowDataset, EvalWindowDataset, collate  # noqa: E402

HERE = Path(__file__).parent


def eval_dev(model, dev_ds, device, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dev_ds), min(n, len(dev_ds)), replace=False)
    sub = torch.utils.data.Subset(dev_ds, idx)
    dl = DataLoader(sub, batch_size=128, shuffle=False, collate_fn=collate)
    model.eval()
    tot_err, n_win = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device); pad = batch["pad_mask"].to(device)
            obs = batch["obs_mask"].numpy(); y = batch["y"].numpy()
            pred = model(x, pad).cpu().numpy()
            for b in range(len(pred)):
                gap = np.where(~obs[b] & batch["pad_mask"][b].numpy())[0]
                if len(gap) == 0:
                    continue
                err = SL.geodesic_error_deg(pred[b, gap], y[b, gap])
                tot_err += err; n_win += 1
    model.train()
    return tot_err / max(n_win, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=127560)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--dev_eval_every", type=int, default=10000)
    ap.add_argument("--save", default=str(HERE / "cache" / "ckpt_offset5.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    train_ds = TrainWindowDataset(trans=(5, 30))
    print(f"train 로드: {len(train_ds):,}개, {time.time()-t0:.1f}s")
    t0 = time.time()
    dev_ds = EvalWindowDataset("dev")
    print(f"dev 로드: {len(dev_ds):,}개, {time.time()-t0:.1f}s")

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                     num_workers=0, drop_last=True)
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)

    model.train()
    it = iter(dl)
    losses = []
    t0 = time.time()
    nan_step = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        x = batch["x"].to(device); y = batch["y"].to(device)
        pad = batch["pad_mask"].to(device)

        pred = model(x, pad)
        loss = l1_loss(pred, y, pad)

        if not torch.isfinite(loss):
            nan_step = step
            print(f"!! step {step}: loss가 NaN/Inf ({loss.item()}) — 학습 중단")
            break

        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            recent = np.mean(losses[-args.log_every:])
            print(f"step {step:6d}/{args.steps}  loss(L1,rot6D)={recent:.4f}  grad_norm={gnorm:.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if step % args.dev_eval_every == 0:
            dev_err = eval_dev(model, dev_ds, device)
            print(f"   -> [dev, step {step}] geodesic 오차: {dev_err:.2f}도")

    dt = time.time() - t0
    print(f"\n학습 시간: {dt:.1f}s ({dt/max(step,1)*1000:.1f}ms/step)")

    if nan_step is not None:
        print("결과: 불안정 — NaN 발생, 원인 조사 필요")
        return

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
    print(f"가중치 저장: {args.save}")

    dev_err_final = eval_dev(model, dev_ds, device, n=5000)
    print(f"\n[최종 dev 채점, 5000개 표본] geodesic 오차: {dev_err_final:.2f}도")


if __name__ == "__main__":
    main()
