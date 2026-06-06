"""Анализ 'цены специализации': сводит costscan по non-math доменам в таблицу
Δ(MULT) и строит график trade-off (math выигрыш vs non-math цена) на одной оси.
"""
import matplotlib.pyplot as plt
import argparse
import glob
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


def load_scan(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[float(r["multiplier"])] = r["delta"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen1.5-MoE-A2.7B")
    ap.add_argument("--scan_dir", default="results/scan")
    ap.add_argument("--out_dir", default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    MN = args.model_name

    # non-math: *_costscan.jsonl ; math: *_scan.jsonl (из прошлого этапа)
    cost = {}
    for fp in sorted(glob.glob(f"{args.scan_dir}/{MN}_*_costscan.jsonl")):
        subj = os.path.basename(fp).replace(
            f"{MN}_", "").replace("_costscan.jsonl", "")
        cost[subj] = load_scan(fp)
    math = {}
    for fp in sorted(glob.glob(f"{args.scan_dir}/{MN}_*_scan.jsonl")):
        subj = os.path.basename(fp).replace(
            f"{MN}_", "").replace("_scan.jsonl", "")
        math[subj] = load_scan(fp)

    if not cost:
        print("Нет *_costscan.jsonl — сначала прогони run_cost_of_specialization.sh")
        return

    # таблица
    mults = sorted({m for d in cost.values() for m in d})
    print("\n" + "=" * 90)
    print("ЦЕНА СПЕЦИАЛИЗАЦИИ: Δ accuracy на non-math при общем MATH-bias")
    print("=" * 90)
    hdr = "domain".ljust(28) + " | " + \
        " | ".join(f"x{m}".ljust(7) for m in mults)
    print(hdr)
    print("-" * len(hdr))
    for subj, d in cost.items():
        line = pretty(subj).replace(chr(10), " ").ljust(28) + " | " + " | ".join(
            (f"{d.get(m):+.1f}".ljust(7) if d.get(m) is not None else "—".ljust(7)) for m in mults)
        print(line)
    print("=" * 90)

    # график trade-off
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for subj, d in math.items():
        ms = sorted(d)
        ax.plot(ms, [d[m] for m in ms], marker="o", lw=2,
                color="tab:blue", alpha=0.5,
                label="MATH (выигрыш)" if subj == list(math)[0] else None)
    for subj, d in cost.items():
        ms = sorted(d)
        ax.plot(ms, [d[m] for m in ms], marker="s", ls="--",
                label=f"{pretty(subj).replace(chr(10),' ')} (цена)")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("bias multiplier (log)")
    ax.set_ylabel("Δ accuracy vs baseline (п.п.)")
    ax.set_title(
        f"{MN}: trade-off — math выигрыш (синие) vs non-math цена (пунктир)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = f"{args.out_dir}/{MN}_fig3_cost_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
