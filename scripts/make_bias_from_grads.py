"""Строит bias-конфиг(ы) из УЖЕ собранных сырых градиентов (.pt от build_gradient_bias --save_raw_grads).
Дёшево: не грузит модель, не считает backward. Позволяет получить и обычный, и exclude_dominant
вариант из одного сбора градиентов."""
import argparse
import json
import os
import torch


def build(avg_grads, freq, topk, bias_value, exclude_dominant):
    bias = {}
    for name, avg in avg_grads.items():
        avg = avg.float().clone()
        if exclude_dominant:
            dom = freq.get(name)
            if dom is not None:
                avg[dom > 0] = float("inf")  # доминирующих не берём в "полезные"
        idx = torch.topk(avg, k=topk, largest=False).indices
        bias[name] = {str(int(i)): bias_value for i in idx}
    return bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help=".pt с avg_grads/freq")
    p.add_argument("--out_normal", required=True)
    p.add_argument("--out_exclude", default=None)
    p.add_argument("--topk_experts", type=int, default=4)
    p.add_argument("--bias_value", type=float, default=0.2)
    args = p.parse_args()

    data = torch.load(args.raw, map_location="cpu")
    avg_grads, freq = data["avg_grads"], data.get("freq", {})
    subject = data.get("subject", "unknown")

    normal = {"domain": f"{subject}_grad", "meta": {"exclude_dominant": False},
              "bias": build(avg_grads, freq, args.topk_experts, args.bias_value, False)}
    os.makedirs(os.path.dirname(args.out_normal) or ".", exist_ok=True)
    json.dump(normal, open(args.out_normal, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[bias] normal -> {args.out_normal} ({len(normal['bias'])} layers)")

    if args.out_exclude:
        excl = {"domain": f"{subject}_grad_excl", "meta": {"exclude_dominant": True},
                "bias": build(avg_grads, freq, args.topk_experts, args.bias_value, True)}
        json.dump(excl, open(args.out_exclude, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[bias] exclude -> {args.out_exclude} ({len(excl['bias'])} layers)")


if __name__ == "__main__":
    main()
