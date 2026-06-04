"""Единый агрегатор результатов проекта в один JSON + CSV.
Сводит: baseline acc, оптимальный MULT и Δ, McNemar p, бутстрап CI,
random-control Δ/p, routing-метрики (Jaccard overlap, confidence shift),
полную кривую Δ(MULT) из скана.

Запуск:
  python scripts/aggregate_all.py --model_name Qwen1.5-MoE-A2.7B
Выход: results/summary/<model>_summary.json и .csv
"""
import argparse
import glob
import json
import math
import os
import numpy as np


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f]


def acc_of(rows):
    return np.mean([int(r["correct"]) for r in rows]) * 100 if rows else None


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def routing_metrics(base, other):
    """Средний Jaccard overlap по слоям и средний сдвиг confidence предсказанного base-ответа."""
    if not base or not other:
        return None, None
    by_idx_o = {r["index"]: r for r in other}
    jac_tot = jac_n = 0
    conf_shifts = []
    for br in base:
        idx = br["index"]
        orr = by_idx_o.get(idx)
        if orr is None:
            continue
        pred = br["pred"]
        if pred in br.get("scores", {}) and pred in orr.get("scores", {}):
            conf_shifts.append(orr["scores"][pred] - br["scores"][pred])
        b_rt = br.get("router_last_token_topk", {})
        o_rt = orr.get("router_last_token_topk", {})
        for lay in b_rt:
            if lay in o_rt:
                jac_tot += jaccard(b_rt[lay]["topk_experts"],
                                   o_rt[lay]["topk_experts"])
                jac_n += 1
    jac = (jac_tot / jac_n * 100) if jac_n else None
    cshift = float(np.mean(conf_shifts)) if conf_shifts else None
    return (round(jac, 2) if jac is not None else None,
            round(cshift, 4) if cshift is not None else None)


def binom_two_sided_p(k, n, p=0.5):
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) * p**i * (1-p)**(n-i) for i in range(n+1)]
    obs = probs[k]
    return float(min(1.0, sum(pr for pr in probs if pr <= obs + 1e-12)))


def mcnemar(base, other):
    if not base or not other:
        return None, None, None
    bc = {r["index"]: int(r["correct"]) for r in base}
    oc = {r["index"]: int(r["correct"]) for r in other}
    common = set(bc) & set(oc)
    b01 = sum(1 for i in common if bc[i] == 0 and oc[i] == 1)
    b10 = sum(1 for i in common if bc[i] == 1 and oc[i] == 0)
    p = binom_two_sided_p(min(b01, b10), b01 + b10)
    return b01, b10, round(p, 4)


def bootstrap_ci(base, other, n_boot=10000, seed=0):
    if not base or not other:
        return None
    bc = {r["index"]: int(r["correct"]) for r in base}
    oc = {r["index"]: int(r["correct"]) for r in other}
    common = sorted(set(bc) & set(oc))
    bv = np.array([bc[i] for i in common])
    ov = np.array([oc[i] for i in common])
    rng = np.random.default_rng(seed)
    n = len(common)
    deltas = np.array([(ov[s].mean() - bv[s].mean()) * 100
                       for s in (rng.integers(0, n, n) for _ in range(n_boot))])
    return [round(float(np.percentile(deltas, 2.5)), 1),
            round(float(np.percentile(deltas, 97.5)), 1)]


def load_scan(path):
    rows = load_jsonl(path)
    if not rows:
        return None
    return {r["multiplier"]: r["delta"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen1.5-MoE-A2.7B")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--final_dir", default="results/final")
    ap.add_argument("--scan_dir", default="results/scan")
    ap.add_argument("--out_dir", default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    MN = args.model_name

    base_files = sorted(glob.glob(f"{args.results_dir}/{MN}_*_baseline.jsonl"))
    summary = {"model": MN, "domains": {}}

    for bf in base_files:
        subj = os.path.basename(bf).replace(
            f"{MN}_", "").replace("_baseline.jsonl", "")
        full = f"{MN}_{subj}"
        base = load_jsonl(bf)
        opt = load_jsonl(f"{args.final_dir}/{full}_opt.jsonl")
        rnd = load_jsonl(f"{args.final_dir}/{full}_opt_randomctrl.jsonl")
        scan = load_scan(f"{args.scan_dir}/{full}_scan.jsonl")

        entry = {"n": len(base) if base else None, "acc_base": round(
            acc_of(base), 2) if base else None}
        if opt:
            entry["acc_opt"] = round(acc_of(opt), 2)
            entry["delta_opt"] = round(acc_of(opt) - acc_of(base), 2)
            b01, b10, p = mcnemar(base, opt)
            entry["mcnemar_fixes"], entry["mcnemar_breaks"], entry["mcnemar_p"] = b01, b10, p
            entry["boot_CI95"] = bootstrap_ci(base, opt)
            jac, cshift = routing_metrics(base, opt)
            entry["jaccard_overlap_pct"], entry["confidence_shift"] = jac, cshift
            entry["significant_0.05"] = bool(p is not None and p < 0.05)
        if rnd:
            entry["acc_random"] = round(acc_of(rnd), 2)
            entry["delta_random"] = round(acc_of(rnd) - acc_of(base), 2)
            _, _, pr = mcnemar(base, rnd)
            entry["random_mcnemar_p"] = pr
        if scan:
            entry["scan_delta_by_mult"] = scan
            valid = {m: d for m, d in scan.items() if d is not None}
            if valid:
                best_m = max(valid, key=lambda m: valid[m])
                entry["scan_best_mult"], entry["scan_best_delta"] = best_m, valid[best_m]
            else:
                entry["scan_best_mult"], entry["scan_best_delta"] = None, None
        summary["domains"][subj] = entry

    out_json = f"{args.out_dir}/{MN}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_json}")

    try:
        import pandas as pd
        flat = []
        for subj, e in summary["domains"].items():
            row = {"domain": subj}
            row.update({k: v for k, v in e.items()
                       if k != "scan_delta_by_mult"})
            flat.append(row)
        df = pd.DataFrame(flat)
        out_csv = f"{args.out_dir}/{MN}_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")
        print("\n" + df.to_string(index=False))
    except Exception as ex:
        print(f"[warn] CSV не записан: {ex}")


if __name__ == "__main__":
    main()
