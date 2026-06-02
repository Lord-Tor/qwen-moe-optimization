"""Фигуры для слайдов из summary.json (aggregate_all.py).
  fig1: кривые Δ(MULT) по доменам — показывает оптимум вмешательства (перевёрнутая U).
  fig2: bar-chart Δ optimal-bias vs random-control, со звёздочками значимости McNemar.
Запуск: python scripts/plot_figures.py --summary results/summary/Qwen1.5-MoE-A2.7B_summary.json
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sig_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out_dir", default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.summary) as f:
        S = json.load(f)
    model = S["model"]
    domains = S["domains"]

    # ---- fig1: Δ(MULT) кривые ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for subj, e in domains.items():
        scan = e.get("scan_delta_by_mult")
        if not scan:
            continue
        mults = sorted(float(m) for m in scan)
        deltas = [scan[str(m)] if str(m) in scan else scan[m] for m in mults]
        ax.plot(mults, deltas, marker="o", label=subj)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("bias multiplier (log)")
    ax.set_ylabel("Δ accuracy vs baseline (п.п.)")
    ax.set_title(f"{model}: оптимум вмешательства по множителю")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    f1 = f"{args.out_dir}/{model}_fig1_multiplier_curve.png"
    fig.tight_layout(); fig.savefig(f1, dpi=150); plt.close(fig)
    print(f"[saved] {f1}")

    # ---- fig2: bias vs random, с значимостью ----
    subj_list = [s for s, e in domains.items() if "delta_opt" in e]
    x = np.arange(len(subj_list))
    w = 0.38
    bias_d = [domains[s]["delta_opt"] for s in subj_list]
    rand_d = [domains[s].get("delta_random", 0) for s in subj_list]
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, bias_d, w, label="gradient bias (opt)", color="#2a6f97")
    b2 = ax.bar(x + w/2, rand_d, w, label="random control", color="#bbbbbb")
    # звёздочки значимости над bias-барами
    for i, s in enumerate(subj_list):
        p = domains[s].get("mcnemar_p")
        star = sig_stars(p)
        y = bias_d[i]
        ax.text(x[i] - w/2, y + (0.3 if y >= 0 else -0.8), star,
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in subj_list], fontsize=8)
    ax.set_ylabel("Δ accuracy vs baseline (п.п.)")
    ax.set_title(f"{model}: gradient bias vs random control (* = McNemar p<0.05)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    f2 = f"{args.out_dir}/{model}_fig2_bias_vs_random.png"
    fig.tight_layout(); fig.savefig(f2, dpi=150); plt.close(fig)
    print(f"[saved] {f2}")


if __name__ == "__main__":
    main()
