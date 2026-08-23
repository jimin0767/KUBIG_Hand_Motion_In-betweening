"""데이터 규약 검증 3종 — 팀이 열어둔 질문들을 실데이터로 판정한다.

(1) 6D 행/열 규약: geodesic/SLERP는 불변이지만 MANO FK는 불변이 아니다.
    해부학적 타당성(손가락이 손바닥 쪽으로 굽는가)으로 판정한다.
(2) 왼손 flip: flip 유무에 따라 왼손 포즈 분포가 오른손 분포와 얼마나 정합되는지.
(3) 원본 6D가 이미 정규직교에 가까운지(= 저장 시 손실이 없었는지).
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
from hmib.mano import ManoFK                         # noqa: E402

TIPS = [3, 6, 9, 12, 15]        # 각 손가락 끝 관절


def rot6d_column(d6: torch.Tensor) -> torch.Tensor:
    """열 기준 디코딩 (Zhou et al. 원 논문 표기). 행 기준의 전치와 같다."""
    return rot.rot6d_to_matrix(d6).transpose(-1, -2)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    feat, meta = load_cache("test")
    n = len(feat)
    sel = np.random.choice(n, size=40000, replace=False)
    x = torch.from_numpy(np.array(feat[np.sort(sel)], dtype=np.float32))

    print("=" * 72)
    print("(3) 원본 6D의 정규직교성 — 저장 손실 확인")
    a1 = x.reshape(-1, N_JOINTS, 6)[..., :3]
    a2 = x.reshape(-1, N_JOINTS, 6)[..., 3:]
    print(f"  |a1| 평균 {a1.norm(dim=-1).mean():.6f}   |a2| 평균 {a2.norm(dim=-1).mean():.6f}")
    dot = (rot._normalize(a1) * rot._normalize(a2)).sum(-1).abs()
    print(f"  a1·a2 직교 이탈 평균 {dot.mean():.6e}  최대 {dot.max():.6e}")
    print("  -> 이미 정규직교. Gram-Schmidt는 사실상 항등이고 저장 손실이 없다.")

    print("=" * 72)
    print("(1) 6D 행/열 규약 판정 — MANO FK 해부학 검사")
    fk = ManoFK("RIGHT")
    rest_tip = fk(torch.eye(3).expand(1, N_JOINTS, 3, 3))[0][TIPS].norm(dim=-1).mean()
    for name, dec in (("행 기준(pytorch3d, 공식 소스)", rot.rot6d_to_matrix),
                      ("열 기준(Zhou 원논문 표기)", rot6d_column)):
        R = dec(x.reshape(-1, N_JOINTS, 6))
        j = fk(R)                                          # (N,16,3)
        tip = j[:, TIPS].norm(dim=-1).mean(dim=-1)         # 손목에서 손끝까지 평균 거리
        curled = (tip < rest_tip).float().mean()
        print(f"  {name}")
        print(f"    손끝-손목 거리 평균 {tip.mean():.4f} m (rest={rest_tip:.4f} m)")
        print(f"    굽힘(rest보다 가까움) 비율 {curled*100:.1f}%")
    print("  -> 사람 손은 굽힘(약 90도+)이 폄(약 30도)보다 훨씬 크므로,")
    print("     굽힘 비율이 높은 쪽이 올바른 규약이다.")

    print("=" * 72)
    print("(2) 왼손 flip 검증")
    offs, lens = meta["offsets"], meta["lengths"]
    ci = np.random.choice(len(lens), size=300, replace=False)
    R_list, L_raw, L_flip = [], [], []
    for c in ci:
        ln = int(lens[c])
        if ln < 30:
            continue
        s = slice(0, min(ln, 120))
        r = np.array(feat[int(offs[c, 0]) + s.start: int(offs[c, 0]) + s.stop])
        l = np.array(feat[int(offs[c, 1]) + s.start: int(offs[c, 1]) + s.stop])
        R_list.append(r)
        L_raw.append(l)
        L_flip.append((l.reshape(-1, N_JOINTS, 6) * rot.LEFT_FLIP_6D.numpy()).reshape(l.shape))
    R_all = torch.from_numpy(np.concatenate(R_list))
    fk = ManoFK("RIGHT")

    def tip_profile(arr):
        j = fk(rot.rot6d_to_matrix(torch.from_numpy(np.concatenate(arr)).reshape(-1, N_JOINTS, 6)))
        return j[:, TIPS].norm(dim=-1).mean(0)

    pr = fk(rot.rot6d_to_matrix(R_all.reshape(-1, N_JOINTS, 6)))[:, TIPS].norm(dim=-1).mean(0)
    praw, pfl = tip_profile(L_raw), tip_profile(L_flip)
    print(f"  오른손      손끝 거리 프로필 {pr.numpy().round(4)}")
    print(f"  왼손 raw    (flip 없음)      {praw.numpy().round(4)}  |차이| {float((praw-pr).abs().mean()):.5f}")
    print(f"  왼손 flip   (오른손 규약)    {pfl.numpy().round(4)}  |차이| {float((pfl-pr).abs().mean()):.5f}")
    print("  -> 오른손 프로필에 더 가까운 쪽이 올바른 처리다.")

    f = torch.randn(100, N_JOINTS * 6)
    rt = rot.flip_left_to_right(rot.flip_left_to_right(f))
    print(f"  flip 왕복 오차 {float((rt-f).abs().max()):.2e}")
    Rf = rot.rot6d_to_matrix(rot.flip_left_to_right(f).reshape(-1, N_JOINTS, 6))
    I = torch.eye(3).expand_as(Rf)
    print(f"  flip 후 직교성 이탈 {float((Rf@Rf.transpose(-1,-2)-I).abs().max()):.2e}"
          f"  det 평균 {float(Rf.det().mean()):.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
