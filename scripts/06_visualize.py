"""MANO 메쉬 비교 애니메이션 — GT vs SLERP vs 학습 모델들.

움직임이 큰 test 윈도우를 골라 렌더링한다. 0816 회의 7번에서 지적한
"gap 안에서 쥐었다 폈다 하는데 모델이 멈춰 있으려 한다" 문제가 실제로
보이는지 확인하는 것이 목적이라, 정적인 윈도우를 고르면 의미가 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from matplotlib.animation import PillowWriter                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX, N_JOINTS, rotation as rot                   # noqa: E402
from hmib.baselines import slerp_fill                             # noqa: E402
from hmib.data import WindowDataset, collate, make_model_input    # noqa: E402
from hmib.mano import ManoLBS                                     # noqa: E402

# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pick_dynamic_windows(split, index_path, T=20, k=3, pool=400, seed=3):
    """gap 구간의 회전 속도가 큰 윈도우를 고른다."""
    idx = np.load(index_path, allow_pickle=True)
    cand = np.where(idx["T"] == T)[0]
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(cand, size=pool, replace=False))
    ds = WindowDataset(split, index_path, subset=sel)
    batch = collate([ds[i] for i in range(len(ds))])
    x = batch["x"]
    R = rot.rot6d_to_matrix(x.reshape(len(x), -1, N_JOINTS, 6))
    speed = torch.rad2deg(rot.geodesic_angle(R[:, :-1], R[:, 1:])).mean(-1)
    gap_speed = speed[:, CTX:CTX + T].mean(-1)
    top = torch.topk(gap_speed, k).indices.numpy()
    return [ds[i] for i in top], gap_speed[top].numpy()


def render(seqs, labels, T, out_path, title):
    """seqs: [(L,90) 텐서] 리스트를 나란히 애니메이션."""
    lbs = ManoLBS("RIGHT")
    L = seqs[0].shape[0]
    verts = [lbs.vertices(rot.rot6d_to_matrix(s.reshape(L, N_JOINTS, 6))).numpy()
             for s in seqs]
    allv = np.concatenate(verts)
    lo, hi = allv.reshape(-1, 3).min(0), allv.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1

    n = len(seqs)
    fig = plt.figure(figsize=(3.1 * n, 3.6))
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    writer = PillowWriter(fps=10)
    FIGS.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=90):
        for f in range(L):
            in_gap = CTX <= f < L - 1
            for ax, v, lab in zip(axes, verts, labels):
                ax.clear()
                ax.plot_trisurf(v[f][:, 0], v[f][:, 1], v[f][:, 2],
                                triangles=lbs.faces, color="#d98cA0" if in_gap else "#8fb8de",
                                edgecolor="none", shade=True, alpha=0.95)
                ax.set_xlim(c[0] - r, c[0] + r)
                ax.set_ylim(c[1] - r, c[1] + r)
                ax.set_zlim(c[2] - r, c[2] + r)
                ax.set_axis_off()
                ax.set_title(lab, fontsize=10)
                ax.view_init(elev=18, azim=-70)
            fig.suptitle(f"{title}   frame {f+1}/{L}"
                         f"{'   [생성 구간]' if in_gap else '   (관측)'}", fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    print("저장:", out_path)


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    index_path = TEAM_INDEX_DIR / "test_index.npz"
    wins, speeds = pick_dynamic_windows("test", index_path, T=T, k=2)

    tags = [p.stem.replace("ckpt_", "") for p in sorted(RESULTS.glob("ckpt_*.pt"))]
    from hmib.models import SilkHandEncoder
    models = {}
    for tag in tags:
        ck = torch.load(RESULTS / f"ckpt_{tag}.pt", map_location=DEVICE, weights_only=False)
        a = ck["args"]
        m = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                            dim_ff=4 * a["d_model"],
                            residual=a["variant"].startswith("delta")).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.eval()
        models[tag] = (m, a["variant"])

    for wi, (w, _) in enumerate(wins):
        b = collate([(w, T)])
        b = {k: v.to(DEVICE) for k, v in b.items()}
        seqs, labels = [b["x"][0].cpu()], ["GT"]
        sl = slerp_fill(b["x"], b["T"])
        merged = b["x"].clone()
        merged[:, CTX:CTX + T] = sl[:, CTX:CTX + T]
        seqs.append(merged[0].cpu()); labels.append("SLERP")
        with torch.no_grad():
            for tag, (m, variant) in models.items():
                base = sl if variant.startswith("delta") else None
                out = m(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base).float()
                mg = b["x"].clone()
                mg[:, CTX:CTX + T] = out[:, CTX:CTX + T]
                seqs.append(mg[0].cpu()); labels.append(tag)
        render(seqs, labels, T,
               FIGS / f"compare_T{T}_win{wi}.gif",
               f"T={T}  gap 평균속도 {speeds[wi]:.1f}deg/frame")


if __name__ == "__main__":
    main()
