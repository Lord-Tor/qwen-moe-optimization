import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def get_top1_experts_from_log(filepath):
    """Извлекает ID самого вероятного эксперта (Top-1) для каждого слоя во всех вопросах."""
    expert_counts = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            router_info = data.get("router_last_token_topk", {})
            
            # Проходим по всем слоям и берем первого эксперта из topk
            for layer, info in router_info.items():
                top1_expert = info["topk_experts"][0]
                expert_counts[top1_expert] += 1
                
    return expert_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Путь к baseline .jsonl")
    parser.add_argument("--heuristic", type=str, required=True, help="Путь к biased .jsonl")
    args = parser.parse_args()

    # Считаем частоту выбора экспертов
    base_counts = get_top1_experts_from_log(args.baseline)
    heur_counts = get_top1_experts_from_log(args.heuristic)

    # Собираем в DataFrame для удобной отрисовки
    df_base = pd.DataFrame(list(base_counts.items()), columns=['Expert_ID', 'Count'])
    df_base['Method'] = 'Baseline'
    
    df_heur = pd.DataFrame(list(heur_counts.items()), columns=['Expert_ID', 'Count'])
    df_heur['Method'] = 'Biased (x100)'

    df_plot = pd.concat([df_base, df_heur], ignore_index=True)

    # Отрисовка
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # Строим график, где рядом стоят столбцы бейзлайна и смещения для каждого эксперта
    sns.barplot(x="Expert_ID", y="Count", hue="Method", data=df_plot, palette="muted", alpha=0.9)

    plt.title("Top-1 Expert Selection Frequency (Baseline vs Biased)", fontsize=16, pad=15)
    plt.xlabel("Expert ID", fontsize=12)
    plt.ylabel("Selection Count", fontsize=12)
    plt.legend(title="Routing Method")
    
    plt.tight_layout()
    output_img = "expert_distribution_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"Готово! График сохранен как {output_img}")

if __name__ == "__main__":
    main()
