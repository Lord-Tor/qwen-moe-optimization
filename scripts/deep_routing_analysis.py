import json
import argparse
import numpy as np
import pandas as pd


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--biased", required=True)
    args = parser.parse_args()

    base_data = load_jsonl(args.baseline)
    bias_data = load_jsonl(args.biased)

    if len(base_data) != len(bias_data):
        print("Ошибка: файлы разной длины!")
        return

    total_jaccard = 0.0
    layers_counted = 0
    confidence_shifts = []

    for base_row, bias_row in zip(base_data, bias_data):
        # 1. Анализ уверенности в ответе
        pred_char = base_row['pred']
        # Вытаскиваем log_prob выбранного ответа
        base_conf = base_row['scores'][pred_char]
        bias_conf = bias_row['scores'][pred_char]
        confidence_shifts.append(bias_conf - base_conf)

        # 2. Анализ пересечения экспертов (Jaccard)
        base_router = base_row.get("router_last_token_topk", {})
        bias_router = bias_row.get("router_last_token_topk", {})

        for layer_name in base_router.keys():
            if layer_name in bias_router:
                base_experts = base_router[layer_name]["topk_experts"]
                bias_experts = bias_router[layer_name]["topk_experts"]

                total_jaccard += jaccard_similarity(base_experts, bias_experts)
                layers_counted += 1

    avg_jaccard = total_jaccard / layers_counted if layers_counted > 0 else 0
    mean_conf_shift = np.mean(confidence_shifts)

    print("\n" + "="*50)
    print("🧠 DEEP ROUTING ANALYSIS")
    print("="*50)
    print(f"Total Questions Analyzed : {len(base_data)}")
    print(f"Expert Overlap (Jaccard) : {avg_jaccard*100:.2f}%")
    print(f"Mean Confidence Shift    : {mean_conf_shift:+.4f} (log_prob)")

    if avg_jaccard < 0.5:
        print("-> ВЫВОД: Роутер кардинально изменил маршруты (пересечение менее 50%).")
    else:
        print("-> ВЫВОД: Маршрутизация осталась преимущественно прежней.")

    print("="*50 + "\n")


if __name__ == "__main__":
    main()
