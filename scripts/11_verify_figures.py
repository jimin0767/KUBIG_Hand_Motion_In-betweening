"""검증 내용을 눈으로 보여주는 그림 2종 (메쉬 렌더링).

(2) 6D 규약 오류 — 같은 프레임을 행/열 기준으로 디코딩해 나란히 렌더링.
    열 기준은 검지가 손등 쪽으로 꺾인다(해부학적으로 불가능한 과신전).
(6) 클립 경계 침범 버그 — 평가 인덱스를 학습 루프에 쓸 때 윈도우가 다음 블록을
    침범하는 순간을 프레임간 속도 스파이크와 메쉬로 보여준다.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX, N_JOINTS, rotation as rot          # noqa: E402
from hmib.data import CACHE_DIR, load_cache              # noqa: E402
from hmib.mano import ManoLBS                            # noqa: E402

FIGS = ROOT / "figures"
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
LMAX = 41
TIPS = [3, 6, 9, 12, 15]


def _mesh(ax, lbs, v, color, title=None):
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=lbs.faces,
                    color=color, edgecolor="none", shade=True)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-70)
    if title:
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# (2) 6D 규약 오류
# ---------------------------------------------------------------------------
def fig_convention(n_frames=4, seed=0):
    feat, meta = load_cache("test")
    offs, lens = meta["offsets"], meta["lengths"]
    rng = np.random.default_rng(seed)

    # 오른손만, 검지 첫 관절 회전이 큰 프레임을 고른다 (차이가 잘 드러남)
    cands = []
    for c in rng.choice(len(lens), size=60, replace=False):
        ln = int(lens[c])
        x = torch.from_numpy(np.array(feat[int(offs[c, 0]):int(offs[c, 0]) + ln]))
        R = rot.rot6d_to_matrix(x.reshape(ln, N_JOINTS, 6))
        ang = torch.rad2deg(rot.matrix_to_axis_angle(R).norm(dim=-1))   # (T,15)
        score = ang[:, 0]                                              # index_00
        for t in torch.topk(score, min(3, ln)).indices.tolist():
            cands.append((float(score[t]), x[t]))
    cands.sort(key=lambda z: -z[0])
    picks = [cands[i] for i in np.linspace(0, min(len(cands), 40) - 1, n_frames).astype(int)]

    lbs = ManoLBS("RIGHT")
    rest = lbs.vertices(torch.eye(3).expand(1, N_JOINTS, 3, 3))[0].numpy()
    rest_tip = lbs(torch.eye(3).expand(1, N_JOINTS, 3, 3))[0][TIPS].norm(dim=-1)

    fig = plt.figure(figsize=(3.0 * n_frames + 3.0, 6.9))
    nc = n_frames + 1
    ax = fig.add_subplot(2, nc, 1, projection="3d")
    _mesh(ax, lbs, rest, "#c9c9c9", "rest (참고)")
    ax = fig.add_subplot(2, nc, nc + 1, projection="3d")
    _mesh(ax, lbs, rest, "#c9c9c9")

    for i, (ang, x6) in enumerate(picks):
        R_row = rot.rot6d_to_matrix(x6.reshape(1, N_JOINTS, 6))
        R_col = R_row.transpose(-1, -2)
        for r, (R, color) in enumerate(((R_row, "#7fae7f"), (R_col, "#c2717a"))):
            v = lbs.vertices(R)[0].numpy()
            tip = lbs(R)[0][TIPS].norm(dim=-1)
            curled = int((tip < rest_tip).sum())
            ax = fig.add_subplot(2, nc, r * nc + i + 2, projection="3d")
            head = f"검지 회전 {ang:.0f}도\n" if r == 0 else ""
            _mesh(ax, lbs, v, color, f"{head}굽은 손가락 {curled}/5")

    fig.text(0.012, 0.72, "행 기준\n(pytorch3d)", fontsize=12, weight="bold",
             color="#3d6b3d", va="center", ha="left")
    fig.text(0.012, 0.26, "열 기준\n(전치)", fontsize=12, weight="bold",
             color="#9c4a53", va="center", ha="left")
    fig.suptitle("6D 회전 규약 판정 — 같은 프레임을 두 방식으로 디코딩\n"
                 "열 기준은 손가락이 손등 쪽으로 꺾인다 (해부학적으로 불가능한 과신전)",
                 fontsize=13)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.02,
                        wspace=0.02, hspace=0.16)
    out = FIGS / "verify_6d_convention.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("저장:", out)


# ---------------------------------------------------------------------------
# (6) 클립 경계 침범 버그
# ---------------------------------------------------------------------------
def fig_boundary_bug():
    feat, meta = load_cache("dev")
    offs, lens = meta["offsets"], meta["lengths"]
    idx = np.load(TEAM_INDEX_DIR / "dev_index.npz", allow_pickle=True)
    ci, hd, st = idx["clip_idx"], idx["hand"], idx["start"]
    base = (offs[ci, hd] + st).astype(np.int64)
    room = (offs[ci, hd] + lens[ci]).astype(np.int64) - base

    # 41프레임을 읽으면 경계를 넘되, 넘기 전 구간이 충분히 남는 윈도우
    ok = np.where((room >= 18) & (room <= 26))[0]
    j = int(ok[len(ok) // 2])
    b, r = int(base[j]), int(room[j])
    print(f"  선택한 윈도우: 남은 프레임 {r}개인데 41프레임을 읽음 -> {41 - r}프레임이 경계 밖")

    w = torch.from_numpy(np.array(feat[b:b + LMAX], dtype=np.float32))
    R = rot.rot6d_to_matrix(w.reshape(LMAX, N_JOINTS, 6))
    speed = torch.rad2deg(rot.geodesic_angle(R[:-1], R[1:])).mean(-1).numpy()

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.30, wspace=0.05)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.arange(1, LMAX), speed, color="#1F6FB2", lw=1.8, marker="o", ms=3.5)
    ax.axvline(r, color="#C0392B", ls="--", lw=1.6)
    ax.annotate(f"클립 경계 (프레임 {r})\n이후는 다른 블록의 데이터",
                xy=(r, speed.max() * 0.92), xytext=(r + 2.0, speed.max() * 0.92),
                fontsize=10, color="#C0392B", va="center")
    ax.axvspan(r, LMAX - 1, color="#C0392B", alpha=0.07)
    ax.set_xlabel("윈도우 내 프레임")
    ax.set_ylabel("프레임간 회전 (도)")
    ax.set_title("평가 인덱스를 학습 루프에 쓸 때 — 윈도우가 클립 경계를 넘는 순간", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, LMAX - 1)

    lbs = ManoLBS("RIGHT")
    show = [max(0, r - 2), r - 1, r, min(LMAX - 1, r + 1)]
    labels = ["경계 2프레임 전", "경계 직전", "경계 직후", "경계 1프레임 후"]
    verts = lbs.vertices(R[show]).numpy()
    allv = verts.reshape(-1, 3)
    c, rad = (allv.min(0) + allv.max(0)) / 2, (allv.max(0) - allv.min(0)).max() / 2 * 0.72
    for k, (f, lab) in enumerate(zip(show, labels)):
        ax = fig.add_subplot(gs[1, k], projection="3d")
        _mesh(ax, lbs, verts[k], "#8fb8de" if f < r else "#d98ca0", lab)
        ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad)
        ax.set_zlim(c[2] - rad, c[2] + rad)

    fig.suptitle("클립 경계 침범 버그 — 에러 없이 조용히 다른 클립을 정답으로 읽는다 (dev의 8.4%)",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGS / "verify_boundary_bug.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("저장:", out)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["convention", "boundary"]
    if "convention" in which:
        fig_convention()
    if "boundary" in which:
        fig_boundary_bug()
