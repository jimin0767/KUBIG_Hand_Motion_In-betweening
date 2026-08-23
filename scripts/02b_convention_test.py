"""6D 행/열 규약 결정 테스트 — MANO 공식 hands_mean을 정답 기준으로 사용.

전치(행 vs 열)는 회전의 역이므로 axis-angle의 부호가 뒤집힌다. 따라서 "어느 쪽이
굽힘 방향인가"만 정하면 규약이 확정된다.

MANO pkl의 hands_mean(45차원 axis-angle)은 MANO 저자들이 실제 손 스캔에서 얻은
'편안하게 살짝 굽힌 평균 손' 자세다. 이건 규약 논쟁과 무관한 외부 정답이므로,
데이터의 평균 포즈가 hands_mean과 같은 방향인지 반대 방향인지 보면 판정된다.

오른손만 사용한다(왼손은 좌표계가 달라 섞이면 판정이 오염됨).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hmib import N_JOINTS, rotation as rot          # noqa: E402
from hmib.data import load_cache                     # noqa: E402
from hmib.mano import ManoFK, load_mano              # noqa: E402

TIPS = [3, 6, 9, 12, 15]
FINGER = ["index", "middle", "pinky", "ring", "thumb"]


def right_hand_frames(split="test", n=60000, seed=0):
    """오른손 프레임만 무작위 추출."""
    feat, meta = load_cache(split)
    offs, lens = meta["offsets"], meta["lengths"]
    rng = np.random.default_rng(seed)
    out = []
    for c in rng.choice(len(lens), size=1500, replace=False):
        ln = int(lens[c])
        o = int(offs[c, 0])                    # hand 0 = right
        out.append(np.array(feat[o:o + ln]))
        if sum(len(a) for a in out) >= n:
            break
    return torch.from_numpy(np.concatenate(out)[:n].astype(np.float32))


def main():
    x = right_hand_frames()
    print(f"오른손 프레임 {len(x):,}개 표본\n")

    m = load_mano("RIGHT")
    hands_mean = torch.tensor(np.asarray(m["hands_mean"], dtype=np.float32)).reshape(N_JOINTS, 3)

    print("=" * 74)
    print("[결정 테스트] 데이터 평균 포즈 vs MANO 공식 hands_mean 방향 일치")
    print("=" * 74)
    for name, dec in (("행 기준 (pytorch3d / SignSparK 공식)", rot.rot6d_to_matrix),
                      ("열 기준 (전치)", lambda d: rot.rot6d_to_matrix(d).transpose(-1, -2))):
        R = dec(x.reshape(-1, N_JOINTS, 6))
        aa = rot.matrix_to_axis_angle(R)                      # (N,15,3)
        mean_aa = aa.mean(0)                                  # (15,3)
        cos = torch.nn.functional.cosine_similarity(mean_aa, hands_mean, dim=-1)
        agree = (cos > 0).float().mean()
        print(f"\n  {name}")
        print(f"    관절별 코사인 유사도 평균 {cos.mean():+.4f}   양(+)인 관절 {int(agree*15)}/15")
        print(f"    관절별: {np.round(cos.numpy(), 2)}")

    print("\n" + "=" * 74)
    print("[보조 테스트] 손끝-손목 거리 — 굽힘(작아짐) vs 폄(커짐), 오른손만")
    print("=" * 74)
    fk = ManoFK("RIGHT")
    rest = fk(torch.eye(3).expand(1, N_JOINTS, 3, 3))[0]
    rest_tip = rest[TIPS].norm(dim=-1)
    # hands_mean 자세의 손끝 거리 = "정상적으로 살짝 굽힌 손"의 기준점
    th = hands_mean.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = hands_mean / th
    K = torch.zeros(N_JOINTS, 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    R_mean = (torch.eye(3) + torch.sin(th)[..., None] * K
              + (1 - torch.cos(th))[..., None] * (K @ K))
    mean_tip = fk(R_mean[None])[0][TIPS].norm(dim=-1)
    print(f"  rest(항등) 손끝거리       {rest_tip.numpy().round(4)}")
    print(f"  MANO hands_mean 손끝거리  {mean_tip.numpy().round(4)}   <- 정상적으로 굽힌 손")

    for name, dec in (("행 기준", rot.rot6d_to_matrix),
                      ("열 기준", lambda d: rot.rot6d_to_matrix(d).transpose(-1, -2))):
        j = fk(dec(x.reshape(-1, N_JOINTS, 6)))
        tip = j[:, TIPS].norm(dim=-1)
        below = (tip < rest_tip).float().mean(0)
        print(f"  {name}: 평균 {tip.mean(0).numpy().round(4)}  "
              f"rest보다 가까운 비율 {np.round(below.numpy()*100,1)}")
    print("\n  손가락(엄지 제외)은 해부학적으로 폄이 거의 불가능하다.")
    print("  -> rest보다 '가까운' 비율이 높은 쪽이 올바른 규약.")


if __name__ == "__main__":
    main()
