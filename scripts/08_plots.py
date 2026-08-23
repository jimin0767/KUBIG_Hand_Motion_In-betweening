"""결과 그림 — 지표별 T 곡선, SLERP 대비 개선율, 학습 곡선."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
TS = [5, 10, 20, 30]
# 기준선은 회색(의도적으로 뒤로 물러남), 모델 3종만 categorical 색상.
# 색각 검증 통과 조합 (dataviz validate_palette.js: 전 항목 PASS).
STYLE = {"SLERP": ("#7A7A7A", "--", "o"), "Hold": ("#BFBFBF", ":", "s"),
         "abs": ("#1F6FB2", "-", "^"), "delta": ("#D9700F", "-", "v"),
         "delta_vel": ("#9B4DAF", "-", "D")}


def main():
    comp = json.loads((RESULTS / "comparison_test.json").read_text(encoding="utf-8"))
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1) 지표별 T 곡선
    metrics = [("L2Q", "L2Q (회전, 낮을수록 좋음)"), ("L2P", "L2P (MANO FK 위치, m)"),
               ("NPSS", "NPSS (움직임 리듬)"), ("GEO", "측지 오차 (도)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (key, title) in zip(axes, metrics):
        for name, res in comp.items():
            c, ls, mk = STYLE.get(name, ("#333333", "-", "x"))
            ax.plot(TS, [res[str(t)][key] for t in TS], ls, color=c, marker=mk,
                    label=name, lw=2, ms=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("gap 길이 T (프레임)")
        ax.set_xticks(TS)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("How2Sign test · 고정 시드 부분집합 T당 5,000 윈도우 · gap 구간만 채점",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "metric_curves.png", dpi=130)
    plt.close(fig)

    # 2) SLERP 대비 개선율
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [n for n in comp if n not in ("SLERP",)]
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [(comp["SLERP"][str(t)]["L2Q"] - comp[name][str(t)]["L2Q"])
                / comp["SLERP"][str(t)]["L2Q"] * 100 for t in TS]
        ax.bar(np.arange(len(TS)) + i * w, vals, w, label=name,
               color=STYLE.get(name, ("#333333",))[0])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(np.arange(len(TS)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([f"T={t}" for t in TS])
    ax.set_ylabel("SLERP 대비 L2Q 개선율 (%)")
    ax.set_title("기준선 대비 개선율 — 양수면 SLERP를 이김")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "improvement_vs_slerp.png", dpi=130)
    plt.close(fig)

    # 3) 학습 곡선
    logs = sorted(RESULTS.glob("trainlog_*.json"))
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for p in logs:
            tag = p.stem.replace("trainlog_", "")
            d = json.loads(p.read_text(encoding="utf-8"))
            c = STYLE.get(tag, ("#333333",))[0]
            ax.plot([r["step"] for r in d], [r["train"] for r in d], color=c, lw=1.5,
                    label=f"{tag} train")
            ax.plot([r["step"] for r in d], [r["dev"] for r in d], color=c, lw=1.5,
                    ls="--", alpha=0.7, label=f"{tag} dev")
        ax.set_xlabel("step")
        ax.set_ylabel("masked L1 (gap 구간)")
        ax.set_title("학습 곡선")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "training_curves.png", dpi=130)
        plt.close(fig)

    # 4) T=45 외삽
    p45 = RESULTS / "extrapolation_T45.json"
    if p45.exists():
        d = json.loads(p45.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(7.5, 4))
        allT = TS + [45]
        for name in d:
            if name not in comp:
                continue
            c, ls, mk = STYLE.get(name, ("#333333", "-", "x"))
            ys = [comp[name][str(t)]["L2Q"] for t in TS] + [d[name]["45"]["L2Q"]]
            ax.plot(allT, ys, ls, color=c, marker=mk, label=name, lw=2, ms=5)
        ax.axvline(30, color="k", ls=":", lw=1)
        ax.text(30.5, ax.get_ylim()[1] * 0.97, "학습 범위 한계 (T=30)", fontsize=9, va="top")
        ax.set_xlabel("gap 길이 T")
        ax.set_ylabel("L2Q")
        ax.set_title("학습 범위 밖(T=45) 외삽 거동")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "extrapolation_T45.png", dpi=130)
        plt.close(fig)

    print("그림 저장 ->", FIGS)


if __name__ == "__main__":
    main()
