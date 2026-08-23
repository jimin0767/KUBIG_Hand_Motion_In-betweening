"""README용 비교 영상 — 이신영님 shinyoung 브랜치와 같은 형식으로 제작.

형식(diffusion_silk_visualization.ipynb의 animate_hand_comparison과 동일):
  - 1x2 패널, 왼쪽 GT / 오른쪽 예측
  - MANO 16관절 골격 (scatter + bone line), figsize (10,5), interval 120ms
  - GT는 항상 파랑. 예측은 gap 구간에서 빨강, 관측 구간에서 파랑
  - 제목에 프레임 번호와 [GAP] 표기
  - 축 범위는 GT/예측 전체 점으로 고정

세 변형(abs / delta / delta_vel)이 같은 윈도우를 쓰도록 고정해서 서로도 비교 가능하게 했다.

    python scripts/14_readme_videos.py            # T=20, mp4
    python scripts/14_readme_videos.py 30 gif     # T=30, gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from matplotlib.animation import FuncAnimation                    # noqa: E402
from torch.utils.data import DataLoader                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import CTX, N_JOINTS, rotation as rot                   # noqa: E402
from hmib.baselines import slerp_fill                             # noqa: E402
from hmib.data import WindowDataset, collate, make_model_input    # noqa: E402
from hmib.mano import ManoFK                                      # noqa: E402
from hmib.models import SilkHandEncoder                           # noqa: E402

FIGS, RESULTS = ROOT / "figures", ROOT / "results"
VIDEO_DIR = FIGS / "videos"
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MANO 부모 인덱스 (0=손목). 손목을 제외한 15개 뼈를 그린다.
MANO_PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
BONE_JOINTS = list(range(1, 16))

LABEL = {"abs": "SILK abs (절대값 예측)",
         "delta": "SILK delta (SLERP 잔차)",
         "delta_vel": "SILK delta_vel (잔차 + 속도손실)"}


def find_dynamic_window(T=20, pool=6000, seed=42):
    """gap 평균 회전속도가 가장 큰 윈도우를 고른다.

    이신영님 코드는 8.0도/frame 임계값을 쓰지만, 우리 속도 정의(관절 평균 측지각)로는
    T=20에서 6,000개를 훑어도 8.0을 넘는 윈도우가 하나도 없다(최대 6.8). 속도 정의가
    다른 것으로 보여 임계값 대신 '표본 내 최댓값'을 쓰고, 분포상 위치를 같이 출력한다.
    """
    index_path = TEAM_INDEX_DIR / "test_index.npz"
    idx = np.load(index_path, allow_pickle=True)
    cand = np.where(idx["T"] == T)[0]
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(cand, size=min(pool, len(cand)), replace=False))
    ds = WindowDataset("test", index_path, subset=sel)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate)
    speeds = []
    for b in dl:
        x = b["x"]
        R = rot.rot6d_to_matrix(x.reshape(len(x), -1, N_JOINTS, 6))
        s = torch.rad2deg(rot.geodesic_angle(R[:, :-1], R[:, 1:])).mean(-1)
        speeds.append(s[:, CTX:CTX + T].mean(-1))
    gap_speed = torch.cat(speeds)
    k = int(torch.argmax(gap_speed))
    pct = float((gap_speed <= gap_speed[k]).float().mean() * 100)
    return ds[k], float(gap_speed[k]), int(sel[k]), pct


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


def animate(gt_joints, pred_joints, gap_mask, title_pred, out_path, interval=120):
    fig = plt.figure(figsize=(10, 5))
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pr = fig.add_subplot(122, projection="3d")

    all_pts = np.concatenate([gt_joints, pred_joints], axis=0)
    lims = [(all_pts[..., i].min(), all_pts[..., i].max()) for i in range(3)]

    def draw(ax, pts, color, title):
        ax.clear()
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_title(title, fontsize=11)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=25)
        for j in BONE_JOINTS:
            p = MANO_PARENT[j]
            ax.plot([pts[j, 0], pts[p, 0]], [pts[j, 1], pts[p, 1]],
                    [pts[j, 2], pts[p, 2]], c=color)

    def update(t):
        is_gap = bool(gap_mask[t])
        tag = " [GAP]" if is_gap else ""
        draw(ax_gt, gt_joints[t], "tab:blue", f"GT (frame {t}{tag})")
        draw(ax_pr, pred_joints[t], "tab:red" if is_gap else "tab:blue",
             f"{title_pred} (frame {t}{tag})")

    anim = FuncAnimation(fig, update, frames=gt_joints.shape[0], interval=interval)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".mp4":
        anim.save(str(out_path), writer="ffmpeg", fps=round(1000 / interval), dpi=110)
    else:
        anim.save(str(out_path), writer="pillow", fps=round(1000 / interval), dpi=90)
    plt.close(fig)
    print("저장:", out_path)


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ext = sys.argv[2] if len(sys.argv) > 2 else "mp4"

    (w, T_w), speed, orig_idx, pct = find_dynamic_window(T=T)
    print(f"윈도우 선택: test 인덱스 {orig_idx}, gap 평균속도 {speed:.1f}도/frame "
          f"(표본 6,000개 중 상위 {100-pct:.2f}%)")

    b = {k: v.to(DEVICE) for k, v in collate([(w, T)]).items()}
    L = CTX + T + 1
    gap_mask = np.zeros(L, dtype=bool)
    gap_mask[CTX:L - 1] = True

    fk = ManoFK("RIGHT")
    gt_joints = fk(rot.rot6d_to_matrix(
        b["x"][0].cpu().reshape(L, N_JOINTS, 6))).numpy()

    sl = slerp_fill(b["x"], b["T"])
    for tag, (m, variant) in load_models().items():
        with torch.no_grad():
            base = sl if variant.startswith("delta") else None
            out = m(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base).float()
        merged = b["x"].clone()
        merged[:, CTX:L - 1] = out[:, CTX:L - 1]      # 관측 구간은 정답 유지
        pred_joints = fk(rot.rot6d_to_matrix(
            merged[0].cpu().reshape(L, N_JOINTS, 6))).numpy()
        animate(gt_joints, pred_joints, gap_mask, LABEL.get(tag, tag),
                VIDEO_DIR / f"{tag}_T{T}.{ext}")


if __name__ == "__main__":
    main()
