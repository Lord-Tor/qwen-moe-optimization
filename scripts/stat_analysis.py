"""Статзначимость улучшения accuracy: парный McNemar-тест + бутстрап-CI для Δ.
Работает на УЖЕ посчитанных jsonl (CPU, без GPU).

McNemar: для парных бинарных исходов (один и тот же вопрос, baseline vs biased).
  b01 = baseline неверно -> biased верно (исправления)
  b10 = baseline верно   -> biased неверно (поломки)
  Тестирует H0: исправлений и поломок поровну (т.е. Δ=0).
  Используем точный биномиальный вариант (надёжен при малых счётчиках).
Бутстрап: 95% CI для Δaccuracy ресэмплингом по вопросам.
"""
import argparse
import glob
import json
import math
import os
import numpy as np


def binom_two_sided_p(k, n, p=0.5):
    """Точный двусторонний биномиальный тест без scipy.
    P = сумма биномиальных вероятностей всех исходов не вероятнее наблюдаемого."""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
             for i in range(n + 1)]
    obs = probs[k]
    # суммируем хвосты: все исходы с вероятностью <= наблюдаемой (с допуском на float)
    eps = 1e-12
    return float(min(1.0, sum(pr for pr in probs if pr <= obs + eps)))


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def correctness_vec(rows):
    # выровнять по index, чтобы пары точно соответствовали одному вопросу
    by_idx = {r["index"]: int(r["correct"]) for r in rows}
    return by_idx


def mcnemar_exact(base_c, bias_c, idxs):
    b01 = b10 = 0  # 01: base0->bias1 (fix); 10: base1->bias0 (break)
    for i in idxs:
        bb, hb = base_c[i], bias_c[i]
        if bb == 0 and hb == 1:
            b01 += 1
        elif bb == 1 and hb == 0:
            b10 += 1
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # двусторонний точный биномиальный тест с p=0.5
    k = min(b01, b10)
    p = binom_two_sided_p(k, n, 0.5)
    return b01, b10, p


def bootstrap_delta(base_c, bias_c, idxs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idxs = np.array(idxs)
    bvec = np.array([base_c[i] for i in idxs])
    hvec = np.array([bias_c[i] for i in idxs])
    n = len(idxs)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.integers(0, n, n)
        deltas[b] = (hvec[samp].mean() - bvec[samp].mean()) * 100
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def analyze(baseline_path, biased_path, label):
    base = load(baseline_path)
    bias = load(biased_path)
    bc, hc = correctness_vec(base), correctness_vec(bias)
    common = sorted(set(bc) & set(hc))
    if not common:
        return None
    acc_b = np.mean([bc[i] for i in common]) * 100
    acc_h = np.mean([hc[i] for i in common]) * 100
    b01, b10, p = mcnemar_exact(bc, hc, common)
    d_mean, d_lo, d_hi = bootstrap_delta(bc, hc, common)
    return {
        "label": label, "n": len(common),
        "acc_base": round(acc_b, 2), "acc_bias": round(acc_h, 2),
        "delta": round(acc_h - acc_b, 2),
        "fixes(0->1)": b01, "breaks(1->0)": b10,
        "mcnemar_p": round(p, 4),
        "boot_delta_mean": round(d_mean, 2),
        "boot_CI95": f"[{d_lo:.1f}, {d_hi:.1f}]",
        "signif_0.05": "ДА" if p < 0.05 else "нет",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_glob", required=True)
    ap.add_argument("--biased_dir", required=True)
    args = ap.parse_args()

    results = []
    for bf in sorted(glob.glob(args.baseline_glob)):
        subj = os.path.basename(bf).replace("_baseline.jsonl", "")
        # ищем оптимальный biased в biased_dir
        opt = os.path.join(args.biased_dir, f"{subj}_opt.jsonl")
        if os.path.exists(opt):
            r = analyze(bf, opt, subj)
            if r:
                results.append(r)
        # и его random-control, если есть
        rnd = os.path.join(args.biased_dir, f"{subj}_opt_randomctrl.jsonl")
        if os.path.exists(rnd):
            r = analyze(bf, rnd, subj + " [random-ctrl]")
            if r:
                results.append(r)

    if not results:
        print("Нет пар для анализа. Проверь пути.")
        return

    print("\n" + "=" * 110)
    print("СТАТЗНАЧИМОСТЬ: McNemar (парный точный тест) + бутстрап 95% CI для Δaccuracy")
    print("=" * 110)
    cols = ["label", "n", "acc_base", "acc_bias", "delta", "fixes(0->1)",
            "breaks(1->0)", "mcnemar_p", "signif_0.05", "boot_CI95"]
    # печать вручную, без pandas-зависимости
    w = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-" * 110)
    for r in results:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))
    print("=" * 110)
    print("Интерпретация:")
    print("  mcnemar_p < 0.05 -> улучшение статзначимо (исправлений достоверно больше поломок).")
    print("  CI95 не включает 0 -> Δ устойчиво положительна.")
    print(
        "  Сравни строку домена с её [random-ctrl]: у настоящего bias p должно быть меньше.\n")


if __name__ == "__main__":
    main()
