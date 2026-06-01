import glob
import json
import os
import numpy as np
import pandas as pd


def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def jaccard_similarity(l1, l2):
    s1, s2 = set(l1), set(l2)
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0


def acc(data):
    return sum(1 for r in data if r['correct']) / len(data) * 100 if data else None


def routing_overlap(base, other):
    tot, n = 0.0, 0
    for b, o in zip(base, other):
        br, orr = b.get("router_last_token_topk", {}), o.get(
            "router_last_token_topk", {})
        for lay in br:
            if lay in orr:
                tot += jaccard_similarity(br[lay]["topk_experts"],
                                          orr[lay]["topk_experts"])
                n += 1
    return (tot / n * 100) if n else None


def main():
    base_files = sorted(glob.glob("results/*_baseline.jsonl"))
    rows = []
    for bf in base_files:
        subj = os.path.basename(bf).replace("_baseline.jsonl", "")
        base = load_jsonl(bf)
        ab = acc(base)
        if ab is None:
            continue
        variants = {
            "biased": f"results/{subj}_biased.jsonl",
            "biased_excl": f"results/{subj}_biased_excl.jsonl",
            "random_ctrl": f"results/{subj}_biased_randomctrl.jsonl",
        }
        row = {"Subject": subj, "Acc Base (%)": round(ab, 2)}
        for tag, path in variants.items():
            d = load_jsonl(path)
            if d and len(d) == len(base):
                a = acc(d)
                row[f"{tag} Acc"] = round(a, 2)
                row[f"{tag} Δ"] = round(a - ab, 2)
                ov = routing_overlap(base, d)
                row[f"{tag} Jac%"] = round(ov, 1) if ov is not None else None
            else:
                row[f"{tag} Acc"] = None
                row[f"{tag} Δ"] = None
                row[f"{tag} Jac%"] = None
        rows.append(row)

    if not rows:
        print("Нет данных для анализа.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("QWEN-MOE ROUTING ANALYSIS  (Δ = изменение accuracy vs baseline)")
    print("Сравни 'biased Δ' и 'biased_excl Δ' с 'random_ctrl Δ':")
    print("если эффект bias НЕ превышает random_ctrl — он неспецифичен.")
    print("=" * 100)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
