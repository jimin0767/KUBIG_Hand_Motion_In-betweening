"""6D 규약 재판정 — 메쉬 대신 골격으로, 그리고 손바닥 방향으로 수치 판정.

메쉬는 손가락이 손바닥을 파고들면(자기교차) 뭉개져 보여서 굽힘/폄 판단이 어렵다.
여기서는 두 가지로 다시 본다.

  (a) 골격 렌더링 — 뼈와 관절만 그리므로 손가락이 어느 쪽으로 꺾이는지 명확
  (b) 손바닥 방향 투영 — MANO 공식 hands_mean(실제 스캔에서 얻은 '약간 굽힌 평균 손')이
      움직이는 방향을 굽힘(+)으로 정의하고, 두 규약이 어느 쪽으로 가는지 부호로 판정

(b)는 사람의 눈이나 렌더링 각도에 의존하지 않는다.
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

from hmib import N_JOINTS, rotation as rot          # noqa: E402
from hmib.data import load_cache                     # noqa: E402
from hmib.mano import ManoFK, load_mano              # noqa: E402

FIGS = ROOT / "figures"
TIPS = [3, 6, 9, 12, 15]
FINGER_CHAINS = [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 10, 11, 12], [0, 13, 14, 15]]
FINGER_NAMES = ["검지", "중지", "소지", "약지", "엄지"]


def axis_angle_to_matrix(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=aa.dtype).expand_as(K)
    return I + torch.sin(th)[..., None] * K + (1 - torch.cos(th))[..., None] * (K @ K)


def right_hand_frames(n=40000, seed=0):
    feat, meta = load_cache("test")
    offs, lens = meta["offsets"], meta["lengths"]
    rng = np.random.default_rng(seed)
    out = []
    for c in rng.choice(len(lens), size=1200, replace=False):
        ln = int(lens[c])
        out.append(np.array(feat[int(offs[c, 0]):int(offs[c, 0]) + ln]))
        if sum(len(a) for a in out) >= n:
            break
    return torch.from_numpy(np.concatenate(out)[:n].astype(np.float32))


def main():
    fk = ManoFK("RIGHT")
    m = load_mano("RIGHT")
    hands_mean = torch.tensor(np.asarray(m["hands_mean"], dtype=np.float32)).reshape(N_JOINTS, 3)

    J_rest = fk(torch.eye(3).expand(1, N_JOINTS, 3, 3))[0]                    # (16,3)
    J_mean = fk(axis_angle_to_matrix(hands_mean)[None])[0]                    # 굽힌 평균 손

    # --- 굽힘 방향 정의: hands_mean 자세에서 손끝이 이동한 방향 ---
    flex_dir = (J_mean[TIPS] - J_rest[TIPS])                                  # (5,3)
    flex_dir = flex_dir / flex_dir.norm(dim=-1, keepdim=True)

    print("=" * 78)
    print("굽힘 방향 정의 — MANO 공식 hands_mean(실제 스캔 평균 손)이 rest에서 움직인 방향")
    print("=" * 78)
    d_mean = (J_mean[TIPS] - J_rest[TIPS]).norm(dim=-1)
    print(f"  hands_mean에서 손끝 이동량(m): {d_mean.numpy().round(4)}")
    print("  -> 이 방향으로 가면 굽힘(+), 반대로 가면 폄/과신전(-)")

    x = right_hand_frames()
    print(f"\n오른손 실데이터 {len(x):,} 프레임으로 두 규약을 판정")
    print("=" * 78)
    print(f"{'규약':<22}{'굽힘 투영 평균(m)':>20}{'굽힘(+) 프레임 비율':>22}")
    print("-" * 78)
    results = {}
    for name, dec in (("행 기준 (pytorch3d)", rot.rot6d_to_matrix),
                      ("열 기준 (전치)", lambda d: rot.rot6d_to_matrix(d).transpose(-1, -2))):
        R = dec(x.reshape(-1, N_JOINTS, 6))
        J = fk(R)                                                             # (N,16,3)
        disp = J[:, TIPS] - J_rest[TIPS]                                      # (N,5,3)
        proj = (disp * flex_dir).sum(-1)                                      # (N,5) 굽힘 방향 성분
        results[name] = proj
        print(f"{name:<22}{proj.mean():>20.4f}{(proj > 0).float().mean()*100:>21.1f}%")
    print("-" * 78)
    print("  굽힘 투영이 양수 = 손끝이 hands_mean과 같은 쪽(손바닥 쪽)으로 이동")
    print("  사람 손은 굽힘(약 90도+)이 폄(약 10도)보다 훨씬 크므로 양수여야 정상")

    print("\n손가락별 굽힘(+) 프레임 비율")
    print(f"{'규약':<22}" + "".join(f"{n:>10}" for n in FINGER_NAMES))
    for name, proj in results.items():
        print(f"{name:<22}" + "".join(f"{v*100:>9.1f}%"
                                      for v in (proj > 0).float().mean(0).tolist()))

    # ------------------------------------------------------------------
    # 골격 렌더링 (자기교차 없는 명확한 비교)
    # ------------------------------------------------------------------
    ang = torch.rad2deg(rot.matrix_to_axis_angle(
        rot.rot6d_to_matrix(x.reshape(-1, N_JOINTS, 6))).norm(dim=-1))[:, 0]
    picks = torch.topk(ang, 3).indices.tolist()

    views = [(18, -70), (0, -90), (75, -90)]
    view_names = ["비스듬히", "옆에서 (굽힘이 가장 잘 보임)", "위에서"]
    fig = plt.figure(figsize=(4.6 * len(views), 10.5))
    nr, nc = 3, len(views)

    def draw(ax, J, color, title=None):
        J = J.numpy()
        for ch in FINGER_CHAINS:
            ax.plot(J[ch, 0], J[ch, 1], J[ch, 2], "-o", color=color, lw=2.4, ms=4.5)
        ax.scatter(*J[0], color="k", s=45)
        rng_ = 0.11
        c = J[0]
        ax.set_xlim(c[0] - rng_, c[0] + rng_); ax.set_ylim(c[1] - rng_, c[1] + rng_)
        ax.set_zlim(c[2] - rng_, c[2] + rng_)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=11)

    R_row = rot.rot6d_to_matrix(x[picks[0]].reshape(1, N_JOINTS, 6))
    rows = [("rest (기준자세)", J_rest, "#9a9a9a"),
            ("행 기준 (pytorch3d)", fk(R_row)[0], "#2e7d32"),
            ("열 기준 (전치)", fk(R_row.transpose(-1, -2))[0], "#b3323c")]
    for r, (lab, J, color) in enumerate(rows):
        for cix, ((el, az), vn) in enumerate(zip(views, view_names)):
            ax = fig.add_subplot(nr, nc, r * nc + cix + 1, projection="3d")
            draw(ax, J, color, vn if r == 0 else None)
            ax.view_init(elev=el, azim=az)
            if cix == 0:
                ax.text2D(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=12, weight="bold", color=color)
    fig.suptitle("같은 프레임의 손 골격 — 메쉬 대신 뼈만 그려 굽힘 방향을 명확히 본다\n"
                 "가운데 줄(행 기준)은 손가락이 손바닥 쪽으로 말리고, "
                 "아래 줄(열 기준)은 반대쪽으로 젖혀진다", fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.93])
    out = FIGS / "verify_6d_convention_skeleton.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print("\n저장:", out)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    main()
