"""Объединяет сырые градиенты нескольких доменов в один общий .pt + конфиг.
Взвешенное по числу примеров среднее avg_grads, суммирование freq.
Не требует GPU/модели.

Запуск:
  python scripts/merge_grads.py \
      --raws configs/raw/Qwen1.5-MoE-A2.7B_high_school_mathematics_grad.pt \
             configs/raw/Qwen1.5-MoE-A2.7B_abstract_algebra_grad.pt \
             configs/raw/Qwen1.5-MoE-A2.7B_formal_logic_grad.pt \
             configs/raw/Qwen1.5-MoE-A2.7B_college_mathematics_grad.pt \
      --out_raw configs/raw/Qwen1.5-MoE-A2.7B_MATHALL_grad.pt \
      --out_config configs/Qwen1.5-MoE-A2.7B_MATHALL_normal.json \
      --domain_label math_all
"""
import argparse
import json
import os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raws", nargs="+", required=True)
    ap.add_argument("--out_raw", required=True)
    ap.add_argument("--out_config", required=True)
    ap.add_argument("--domain_label", default="math_all")
    ap.add_argument("--topk_experts", type=int, default=4)
    ap.add_argument("--bias_value", type=float, default=0.2)
    args = ap.parse_args()

    merged_grads = {}   # name -> weighted sum of avg_grads
    merged_freq = {}
    total_proc = 0
    n_exp = None
    used = []

    for path in args.raws:
        if not os.path.exists(path):
            print(f"[warn] нет {path} — пропускаю")
            continue
        d = torch.load(path, map_location="cpu")
        w = d.get("processed", 1)          # вес = число примеров домена
        total_proc += w
        n_exp = d.get("num_experts", n_exp)
        used.append((os.path.basename(path), w))
        for name, g in d["avg_grads"].items():
            g = g.float()
            merged_grads[name] = merged_grads.get(name, torch.zeros_like(g)) + g * w
        for name, fr in d.get("freq", {}).items():
            fr = fr.float()
            merged_freq[name] = merged_freq.get(name, torch.zeros_like(fr)) + fr

    if not merged_grads:
        raise RuntimeError("Ни одного .pt не загружено — проверь пути.")

    # взвешенное среднее
    for name in merged_grads:
        merged_grads[name] = merged_grads[name] / total_proc

    print(f"[merge] объединено доменов: {len(used)}, всего примеров: {total_proc}")
    for fn, w in used:
        print(f"        {fn}: вес {w}")

    os.makedirs(os.path.dirname(args.out_raw) or ".", exist_ok=True)
    torch.save({"avg_grads": merged_grads, "freq": merged_freq,
                "processed": total_proc, "num_experts": n_exp,
                "subject": args.domain_label}, args.out_raw)
    print(f"[saved] объединённые градиенты -> {args.out_raw}")

    # строим обычный конфиг (топ-k самых полезных = самые отрицательные градиенты)
    bias = {}
    for name, avg in merged_grads.items():
        idx = torch.topk(avg, k=args.topk_experts, largest=False).indices
        bias[name] = {str(int(i)): args.bias_value for i in idx}
    cfg = {"domain": args.domain_label, "meta": {"merged_from": [u[0] for u in used],
           "total_examples": total_proc}, "bias": bias}
    os.makedirs(os.path.dirname(args.out_config) or ".", exist_ok=True)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[saved] общий math-конфиг -> {args.out_config} ({len(bias)} слоёв)")


if __name__ == "__main__":
    main()
