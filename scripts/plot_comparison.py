"""Сравнительный график Qwen vs OLMoE: Δ accuracy по доменам, две модели рядом.
Берёт два summary.json (из aggregate_all.py).

Запуск:
  python scripts/plot_comparison.py \
      --summaries results/summary/Qwen1.5-MoE-A2.7B_summary.json \
                  results/summary/OLMoE-1B-7B-0924_summary.json \
      --out results/summary/comparison_qwen_vs_olmoe.png
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")

DOMAIN_LABELS = {
    "abstract_algebra": "Abstract\nAlgebra",
    "college_mathematics": "College\nMath",
    "high_school_mathematics": "HS Math",
    "formal_logic": "Formal\nLogic",
    "professional_law": "Law",
    "college_biology": "Biology",
    "high_school_world_history": "World\nHistory",
    "moral_scenarios": "Moral\nScenarios",
}


def pretty(name):
    return DOMAIN_LABELS.get(name, name.replace("_", " ").title())


def short(name):
    # короткое имя модели для легенды
    if "OLMoE" in name or "olmoe" in name:
        return "OLMoE-1B-7B"
    if "Qwen" in name or "qwen" in name:
        return "Qwen1.5-MoE"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="2+ summary.json от aggregate_all.py")
    ap.add_argument(
        "--out", default="results/summary/comparison_qwen_vs_olmoe.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    models = []
    for path in args.summaries:
        if not os.path.exists(path):
            print(f"[warn] нет {path} — пропуск")
            continue
        with open(path) as f:
            S = json.load(f)
        models.append((short(S["model"]), S["domains"]))

    if len(models) < 1:
        print("Нет данных.")
        return

    # Только домены, где есть delta_opt хотя бы у одной модели (т.е. был финальный
    # прогон на оптимальном множителе — это math-домены). Non-math домены попадают
    # в summary через baseline/costscan, но у них нет delta_opt -> пустые столбцы.
    all_domains = []
    for _, dom in models:
        for d in dom:
            if d not in all_domains and dom[d].get("delta_opt") is not None:
                all_domains.append(d)
    if not all_domains:
        print("[warn] нет доменов с delta_opt — нечего сравнивать")
        return

    x = np.arange(len(all_domains))
    n_models = len(models)
    width = 0.8 / n_models
    colors = ["#2a6f97", "#e76f51", "#588157", "#9d4edd"]

    fig, ax = plt.subplots(figsize=(max(9, len(all_domains) * 1.6), 5.5))
    for mi, (mname, dom) in enumerate(models):
        deltas = []
        for d in all_domains:
            e = dom.get(d, {})
            deltas.append(e.get("delta_opt"))
        # bar с пропуском None
        xs = x + (mi - (n_models - 1) / 2) * width
        vals = [v if v is not None else 0 for v in deltas]
        bars = ax.bar(xs, vals, width, label=mname,
                      color=colors[mi % len(colors)])
        # звёздочки значимости
        for xi, d in zip(xs, all_domains):
            e = dom.get(d, {})
            p = e.get("mcnemar_p")
            v = e.get("delta_opt")
            if v is None:
                continue
            star = "*" if (p is not None and p < 0.05) else ""
            if star:
                ax.text(xi, v + (0.3 if v >= 0 else -0.9), star,
                        ha="center", fontsize=12, fontweight="bold")

    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in all_domains], fontsize=8)
    ax.set_ylabel("Δ accuracy при оптимальном множителе (п.п.)")
    ax.set_title(
        "Сравнение моделей: прирост на математических доменах\nот градиентного смещения роутера (* = McNemar p<0.05)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
