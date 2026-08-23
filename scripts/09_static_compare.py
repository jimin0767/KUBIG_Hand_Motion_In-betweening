"""gap 중앙 프레임에서의 정지 비교 이미지 (문서/PPT용).

애니메이션은 발표 때 좋지만 문서에는 정지 이미지가 필요하다.
움직임이 큰 윈도우의 gap 한가운데를 골라 GT / SLERP / 모델들을 나란히 렌더링한다.
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

from hmib import CTX, N_JOINTS, rotation as rot                   # noqa: E402
from hmib.baselines import slerp_fill                             # noqa: E402
from hmib.data import collate, make_model_input                   # noqa: E402
from hmib.mano import ManoLBS                                     # noqa: E402
from hmib.models import SilkHandEncoder                           # noqa: E402

RESULTS, FIGS = ROOT / "results", ROOT / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 저장소 안의 index/ 를 우선 사용하고, 없으면 팀 공유 폴더로 폴백
TEAM_INDEX_DIR = (ROOT / "index" if (ROOT / "index" / "test_index.npz").exists()
                  else ROOT.parent / "topic" / "discussion")


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sys.argv = sys.argv[:1]
    import importlib.util
    spec = importlib.util.spec_from_file_location("viz", ROOT / "scripts" / "06_visualize.py")
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)

    wins, speeds = viz.pick_dynamic_windows("test", TEAM_INDEX_DIR / "test_index.npz",
                                            T=T, k=1, pool=400)
    w, _ = wins[0]
    b = {k: v.to(DEVICE) for k, v in collate([(w, T)]).items()}
    sl = slerp_fill(b["x"], b["T"])

    seqs, labels = [b["x"][0].cpu()], ["GT (정답)"]
    m2 = b["x"].clone(); m2[:, CTX:CTX + T] = sl[:, CTX:CTX + T]
    seqs.append(m2[0].cpu()); labels.append("SLERP")
    for ck_path in sorted(RESULTS.glob("ckpt_*.pt")):
        tag = ck_path.stem.replace("ckpt_", "")
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        a = ck["args"]
        mdl = SilkHandEncoder(d_model=a["d_model"], num_layers=a["layers"], nhead=a["heads"],
                              dim_ff=4 * a["d_model"],
                              residual=a["variant"].startswith("delta")).to(DEVICE)
        mdl.load_state_dict(ck["model"]); mdl.eval()
        with torch.no_grad():
            base = sl if a["variant"].startswith("delta") else None
            out = mdl(make_model_input(b["x"], b["obs"]), b["rel"], b["valid"], base).float()
        mg = b["x"].clone(); mg[:, CTX:CTX + T] = out[:, CTX:CTX + T]
        seqs.append(mg[0].cpu()); labels.append(tag)

    lbs = ManoLBS("RIGHT")
    L = CTX + T + 1
    frames = [CTX, CTX + T // 3, CTX + 2 * T // 3, CTX + T - 1]     # gap 안 4개 시점
    verts = [lbs.vertices(rot.rot6d_to_matrix(s.reshape(L, N_JOINTS, 6))).numpy()
             for s in seqs]
    allv = np.concatenate(verts).reshape(-1, 3)
    c, r = (allv.min(0) + allv.max(0)) / 2, (allv.max(0) - allv.min(0)).max() / 2 * 1.05

    nr, nc = len(seqs), len(frames)
    fig = plt.figure(figsize=(2.5 * nc, 2.5 * nr))
    for i, (v, lab) in enumerate(zip(verts, labels)):
        for j, f in enumerate(frames):
            ax = fig.add_subplot(nr, nc, i * nc + j + 1, projection="3d")
            ax.plot_trisurf(v[f][:, 0], v[f][:, 1], v[f][:, 2], triangles=lbs.faces,
                            color="#d98ca0", edgecolor="none", shade=True)
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_axis_off(); ax.view_init(elev=18, azim=-70)
            if i == 0:
                ax.set_title(f"gap {j*T//3 + 1}/{T} 프레임", fontsize=10)
            if j == 0:
                ax.text2D(-0.10, 0.5, lab, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=11, weight="bold")
    fig.suptitle(f"gap 구간 생성 결과 비교  (T={T}, gap 평균속도 {speeds[0]:.1f}deg/frame)",
                 fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = FIGS / f"static_compare_T{T}.png"
    fig.savefig(out, dpi=115)
    print("저장:", out)


if __name__ == "__main__":
    main()
