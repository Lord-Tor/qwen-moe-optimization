import glob
import json
import os
import pandas as pd


def get_accuracy(filepath):
    if not os.path.exists(filepath):
        return None
    correct, total = 0, 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            correct += int(data['correct'])
            total += 1
    return (correct / total * 100) if total > 0 else 0.0


def main():
    print(f"\n{'='*65}")
    print(
        f"| {'Subject':<25} | {'Baseline (%)':<12} | {'Biased (%)':<10} | {'Delta':<6} |")
    print(f"{'='*65}")

    baseline_files = glob.glob("results/*_baseline.jsonl")

    for base_file in baseline_files:
        subject = os.path.basename(base_file).replace('_baseline.jsonl', '')
        bias_file = f"results/{subject}_biased.jsonl"

        acc_base = get_accuracy(base_file)
        acc_bias = get_accuracy(bias_file)

        if acc_base is not None and acc_bias is not None:
            delta = acc_bias - acc_base
            print(
                f"| {subject:<25} | {acc_base:<12.2f} | {acc_bias:<10.2f} | {delta:+.2f}% |")

    print(f"{'='*65}\n")
    print("To visualize a specific subject, run:")
    print("python scripts/plot_results.py --baseline results/SUBJECT_baseline.jsonl --heuristic results/SUBJECT_biased.jsonl\n")


if __name__ == "__main__":
    main()
