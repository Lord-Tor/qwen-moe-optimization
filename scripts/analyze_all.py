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


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def analyze_pair(base_file, bias_file):
    base_data = load_jsonl(base_file)
    bias_data = load_jsonl(bias_file)

    if not base_data or not bias_data or len(base_data) != len(bias_data):
        return None

    base_correct = sum(1 for row in base_data if row['correct'])
    bias_correct = sum(1 for row in bias_data if row['correct'])
    total = len(base_data)

    acc_base = (base_correct / total) * 100
    acc_bias = (bias_correct / total) * 100

    total_jaccard = 0.0
    layers_counted = 0
    conf_shifts = []

    for base_row, bias_row in zip(base_data, bias_data):
        pred_char = base_row['pred']
        conf_shifts.append(bias_row['scores'].get(
            pred_char, 0) - base_row['scores'].get(pred_char, 0))

        b_router = base_row.get("router_last_token_topk", {})
        h_router = bias_row.get("router_last_token_topk", {})

        for layer_name in b_router.keys():
            if layer_name in h_router:
                total_jaccard += jaccard_similarity(
                    b_router[layer_name]["topk_experts"], h_router[layer_name]["topk_experts"])
                layers_counted += 1

    avg_jaccard = (total_jaccard / layers_counted *
                   100) if layers_counted > 0 else 100.0
    return {
        "Acc Base (%)": round(acc_base, 2),
        "Acc Bias (%)": round(acc_bias, 2),
        "Delta (%)": round(acc_bias - acc_base, 2),
        "Jaccard Overlap (%)": round(avg_jaccard, 2),
        "Mean Conf Shift": round(np.mean(conf_shifts), 4)
    }


def main():
    base_files = glob.glob("results/*_baseline.jsonl")
    results = []

    for base_file in base_files:
        subject = os.path.basename(base_file).replace('_baseline.jsonl', '')
        bias_file = f"results/{subject}_biased.jsonl"

        metrics = analyze_pair(base_file, bias_file)
        if metrics:
            metrics["Subject"] = subject
            results.append(metrics)

    if not results:
        print("Данные для анализа не найдены.")
        return

    df = pd.DataFrame(results)
    df = df[["Subject", "Acc Base (%)", "Acc Bias (%)",
             "Delta (%)", "Jaccard Overlap (%)", "Mean Conf Shift"]]

    print("\n" + "="*85)
    print("🚀 QWEN-MOE ROUTING ANALYSIS RESULTS")
    print("="*85)
    print(df.to_markdown(index=False))
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
