import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str,
                        default="results/mmlu_onepass_results.jsonl")
    parser.add_argument("--heuristic", type=str,
                        default="results/math_biased_cluster.jsonl")
    parser.add_argument("--gradient", type=str, default=None)
    args = parser.parse_args()
    df_base = load_results(args.baseline)
    df_heur = load_results(args.heuristic)
    acc_base = df_base['correct'].mean() * 100
    acc_heur = df_heur['correct'].mean() * 100

    print(f"=== Qwen1.5-MoE-A2.7B Accuracy ===")
    print(f"Baseline:  {acc_base:.2f}%")
    print(f"Heuristic: {acc_heur:.2f}%")
    print(f"Delta:     {acc_heur - acc_base:+.2f}%")
    plot_data = [
        {"Method": "Baseline (Default Routing)", "Accuracy": acc_base},
        {"Method": "Frequency Heuristic", "Accuracy": acc_heur}
    ]

    if args.gradient:
        df_grad = load_results(args.gradient)
        acc_grad = df_grad['correct'].mean() * 100
        print(f"Gradient:  {acc_grad:.2f}%")
        print(f"Delta G:   {acc_grad - acc_base:+.2f}%")
        plot_data.append({"Method": "Gradient Bias", "Accuracy": acc_grad})

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(x="Method", y="Accuracy", data=df_plot, palette="viridis")

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("MMLU (High School Mathematics) - MoE Routing Optimization",
              fontsize=16, pad=20)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("")
    plt.ylim(0, max(df_plot["Accuracy"]) + 10)

    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png", dpi=300)
    print("\nPlot saved to results/accuracy_comparison.png")


if __name__ == "__main__":
    main()
