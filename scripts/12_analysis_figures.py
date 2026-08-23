"""분석 그림 3종 — 지금까지 gap 전체 평균으로만 보던 것을 쪼개서 본다.

(3) gap 내 프레임별 오차 프로파일 — gap의 어느 위치가 어려운가
(7) 관절별 오차 히트맵 — 어느 손가락이 어려운가
(8) 윈도우별 개선율 분포 — 평균 뒤에 가려진 승패 분포
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX, N_JOINTS, rotation as rot                   # noqa: E402
from hmib.baselines import hold_fill, slerp_fill                  # noqa: E402
from hmib.data import WindowDataset, collate, make_model_input    # noqa: E402
from hmib.evaluate import make_subset                             # noqa: E402
from hmib.models import SilkHandEncoder                           # noqa: E402

FIGS, RESULTS = ROOT / "figures", ROOT / "results"
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS = (5, 10, 20, 30)
PER_T = 8000          # 프레임/관절 단위로 쪼개므로 전수까지 갈 필요 없음
COL = {"SLERP": "#7A7A7A", "Hold": "#BFBFBF", "abs": "#1F6FB2",
       "delta": "#D9700F", "delta_vel": "#9B4DAF"}
FINGERS = ["검지", "중지", "소지", "약지", "엄지"]
JOINT_LABELS = [f"{f}{i+1}" for f in FINGERS for i in range(3)]


def load_models():
    out = {}
    for ck_path in sorted(RESULTS.glob("ckpt_*.pt")):
        tag = ck_path.stem.replace("ckpt_", "")
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        a = ck["args"]
        m = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                            dim_ff=4 * a["d_model"],
                            residual=a["variant"].startswith("delta")).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.eval()
        out[tag] = (m, a["variant"])
    return out


def l2q_elementwise(pred, gt):
    """(B,T,90) -> (B,T,15) 프레임x관절 단위 쿼터니언 거리."""
    B, T, _ = pred.shape
    qp = rot.matrix_to_quaternion(rot.rot6d_to_matrix(pred.reshape(B, T, N_JOINTS, 6)))
    qg = rot.matrix_to_quaternion(rot.rot6d_to_matrix(gt.reshape(B, T, N_JOINTS, 6)))
    return torch.minimum((qp - qg).norm(dim=-1), (qp + qg).norm(dim=-1))


@torch.no_grad()
def collect():
    """T별로 {method: (프레임프로파일, 관절프로파일, 윈도우별 L2Q)} 수집."""
    models = load_models()
    index_path = TEAM_INDEX_DIR / "test_index.npz"
    subset = make_subset(index_path, per_t=PER_T)
    data = {t: {} for t in TS}
    for t in TS:
        ds = WindowDataset("test", index_path, subset=subset[t])
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
        acc = {}
        for b in dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            L = CTX + t + 1
            gap = slice(CTX, L - 1)
            gt = b["x"][:, gap]
            preds = {"SLERP": slerp_fill(b["x"], b["T"])[:, gap],
                     "Hold": hold_fill(b["x"], b["T"])[:, gap]}
            for tag, (m, variant) in models.items():
                base = slerp_fill(b["x"], b["T"]) if variant.startswith("delta") else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    o = m(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base)
                preds[tag] = o.float()[:, gap]
            for name, p in preds.items():
                e = l2q_elementwise(p, gt)                    # (B,T,15)
                a = acc.setdefault(name, {"frame": [], "joint": [], "win": []})
                a["frame"].append(e.mean(-1).cpu())           # (B,T)
                a["joint"].append(e.mean(1).cpu())            # (B,15)
                a["win"].append(e.reshape(len(e), -1).mean(-1).cpu())   # (B,)
        data[t] = {n: {k: torch.cat(v) for k, v in a.items()} for n, a in acc.items()}
        print(f"  T={t} 수집 완료 (n={len(ds):,})", flush=True)
    return data


# ---------------------------------------------------------------------------
def fig_frame_profile(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    for col, t in enumerate(TS):
        x = np.arange(1, t + 1)
        top, bot = axes[0, col], axes[1, col]
        slerp = data[t]["SLERP"]["frame"].mean(0).numpy()
        for name in ("SLERP", "abs", "delta", "delta_vel"):
            if name not in data[t]:
                continue
            prof = data[t][name]["frame"].mean(0).numpy()
            top.plot(x, prof, color=COL[name], lw=2, label=name,
                     ls="--" if name == "SLERP" else "-")
            if name != "SLERP":
                bot.plot(x, (slerp - prof) / slerp * 100, color=COL[name], lw=2, label=name)
        top.set_title(f"T={t}", fontsize=11)
        for ax in (top, bot):
            ax.grid(alpha=0.3)
            ax.set_xlim(1, t)
        bot.axhline(0, color="#C0392B", lw=1.2)
        bot.set_ylim(-5, 70)
        bot.set_xlabel("gap 내 프레임 위치  (1 = context 직후, 마지막 = target 직전)")
    axes[0, 0].set_ylabel("L2Q (낮을수록 좋음)")
    axes[1, 0].set_ylabel("SLERP 대비 개선율 (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "gap 내 위치별 오차 — 위: 절대 오차는 중앙이 최악(뒤집힌 U자)\n"
        "아래: 모델의 이득은 context 직후 +63%에서 target 직전 0%까지 단조 감소한다. "
        "T가 달라도 이 감쇠 곡선은 거의 겹친다", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIGS / "analysis_frame_profile.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("저장:", out)


def fig_joint_heatmap(data, method="abs"):
    mat = np.stack([data[t][method]["joint"].mean(0).numpy() for t in TS])   # (4,15)
    rel = mat / np.stack([data[t]["SLERP"]["joint"].mean(0).numpy() for t in TS])

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.0))
    for ax, m, title, cmap, fmt in (
            (axes[0], mat, f"{method} 의 관절별 L2Q (절대값)", "viridis_r", "{:.3f}"),
            (axes[1], rel, f"{method} ÷ SLERP — 1보다 작으면 기준선보다 좋음", "RdBu_r", "{:.2f}")):
        vmax = None if cmap == "viridis_r" else 1.0 + np.abs(1.0 - m).max()
        vmin = None if cmap == "viridis_r" else 1.0 - np.abs(1.0 - m).max()
        im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(15), JOINT_LABELS, fontsize=9)
        ax.set_yticks(range(4), [f"T={t}" for t in TS], fontsize=10)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        fontsize=7.5, color="white" if cmap == "viridis_r" else "black")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        for x in (2.5, 5.5, 8.5, 11.5):
            ax.axvline(x, color="w", lw=1.5)
    fig.suptitle("관절별 난이도 — 손가락 이름은 MANO 순서(검지·중지·소지·약지·엄지), "
                 "숫자는 손바닥에서 손끝 순", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGS / "analysis_joint_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("저장:", out)


def fig_improvement_dist(data, method="abs"):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, t in zip(axes, TS):
        s = data[t]["SLERP"]["win"].numpy()
        m = data[t][method]["win"].numpy()
        imp = (s - m) / np.maximum(s, 1e-9) * 100
        lose = (imp < 0).mean() * 100
        med = float(np.median(imp))
        agg = (s.mean() - m.mean()) / s.mean() * 100      # 표에 실린 집계값과 같은 정의
        ax.hist(imp, bins=60, range=(-60, 100), color=COL[method], alpha=0.85)
        ax.axvline(0, color="#C0392B", lw=1.6)
        ax.axvline(med, color="k", ls="--", lw=1.4)
        ax.set_title(f"T={t}   집계 {agg:+.1f}%   중앙값 {med:+.1f}%\n"
                     f"기준선에 진 윈도우 {lose:.1f}%", fontsize=10.5)
        ax.set_xlabel("SLERP 대비 L2Q 개선율 (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("윈도우 수")
    fig.suptitle(f"{method} — 윈도우별 개선율 분포. 집계값이 좋아도 4분의 1 정도의 윈도우는 기준선에 진다\n"
                 f"(x축 -60~100%로 자름. 정지에 가까운 윈도우는 SLERP 오차가 0에 가까워 비율이 발산하므로 "
                 f"대표값은 중앙값을 쓴다)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    out = FIGS / "analysis_improvement_dist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("저장:", out)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    d = collect()
    fig_frame_profile(d)
    fig_joint_heatmap(d)
    fig_improvement_dist(d)
