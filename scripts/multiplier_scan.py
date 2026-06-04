"""Скан множителя bias на сетке, БЕЗ пересбора градиентов и БЕЗ перезагрузки модели.
Грузит модель один раз, переиспользует уже собранный normal-конфиг, прогоняет все
множители (включая 0.0 = baseline-sanity), пишет компактный jsonl со сводкой Δ(MULT).

Запуск (один домен):
  python scripts/multiplier_scan.py --subject high_school_mathematics \
      --bias_file configs/Qwen1.5-MoE-A2.7B_high_school_mathematics_normal.json \
      --baseline results/Qwen1.5-MoE-A2.7B_high_school_mathematics_baseline.jsonl \
      --out results/scan/Qwen1.5-MoE-A2.7B_high_school_mathematics_scan.jsonl \
      --multipliers 0.5 1 2 5 10 20 50
"""
import argparse
import gc
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import build_prompt, prepare_choice_token_ids

# переиспользуем хук-логику из основного скрипта инференса
try:
    from scripts.qwen_mmlu_biased import make_topk_recompute_hook
except ImportError:
    import sys
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
    from qwen_mmlu_biased import make_topk_recompute_hook


def load_baseline_acc(path):
    if not os.path.exists(path):
        return None
    n = c = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            c += int(r["correct"])
            n += 1
    return (c / n * 100) if n else None


def score_prompt(model, tokenizer, device, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    MAX_SEQ_LEN = 512
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, -MAX_SEQ_LEN:]
        attention_mask = attention_mask[:, -MAX_SEQ_LEN:]
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, return_dict=True)
    nt = out.logits[0, -1]
    lp = torch.log_softmax(nt.float(), dim=-1)
    return {ch: float(lp[tid].item()) for ch, tid in choice_token_ids.items()}


def apply_hooks(model, bias_data, dtype, multiplier, top_k, norm_topk_prob):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        ln = f"layer_{i}"
        if ln in bias_data.get("bias", {}) and hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            bt = torch.zeros(model.config.num_experts, dtype=dtype)
            for eid, val in bias_data["bias"][ln].items():
                bt[int(eid)] = float(val) * multiplier
            hooks.append(layer.mlp.gate.register_forward_hook(
                make_topk_recompute_hook(bt, top_k, norm_topk_prob)))
    return hooks


def eval_acc(model, tokenizer, device, ds, choice_token_ids, answer_map, limit):
    total = correct = 0
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        gold = answer_map[int(ex["answer"])]
        prompt = build_prompt(ex["question"], ex["choices"])
        scores = score_prompt(model, tokenizer, device,
                              prompt, choice_token_ids)
        pred = max(scores, key=scores.get)
        total += 1
        correct += int(pred == gold)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return correct / total * 100 if total else None, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--subject", default="high_school_mathematics")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--bias_file", required=True)
    p.add_argument("--baseline", required=True,
                   help="baseline jsonl для опорной accuracy")
    p.add_argument("--out", required=True)
    p.add_argument("--experts_impl", default="eager")
    p.add_argument("--multipliers", nargs="+", type=float,
                   default=[0.5, 1, 2, 5, 10, 20, 50])
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if torch.cuda.is_available():
        dtype = torch.float16
        n_gpu = torch.cuda.device_count()
        max_mem = ({0: "10GiB", 1: "22GiB"} if n_gpu >= 2 else {0: "22GiB"}) \
            if "qwen" in args.model.lower() else None  # не-Qwen: авто-баланс
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True, device_map="auto",
            max_memory=max_mem, attn_implementation="sdpa", experts_implementation=args.experts_impl,
            offload_folder="/mnt/tank/scratch/" + os.environ.get("USER", "tmp") + "/.offload_qwen")
        device = model.model.embed_tokens.weight.device
    else:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True)
        device = "cpu"
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    top_k = getattr(model.config, "num_experts_per_tok", 4)
    norm_topk_prob = getattr(model.config, "norm_topk_prob", False)

    with open(args.bias_file) as f:
        bias_data = json.load(f)
    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    acc_base = load_baseline_acc(args.baseline)
    if acc_base is None:
        raise RuntimeError(
            f"baseline пуст/не найден: {args.baseline}. Скан невозможен без опорной accuracy.")
    print(f"[scan] subject={args.subject} baseline_acc={acc_base}")

    results = []
    for mult in args.multipliers:
        hooks = apply_hooks(model, bias_data, dtype,
                            mult, top_k, norm_topk_prob)
        acc, n = eval_acc(model, tokenizer, device, ds,
                          choice_token_ids, answer_map, args.limit)
        for h in hooks:
            h.remove()
        delta = (
            acc - acc_base) if (acc is not None and acc_base is not None) else None
        row = {"subject": args.subject, "multiplier": mult, "acc": round(acc, 2),
               "acc_base": round(acc_base, 2) if acc_base else None,
               "delta": round(delta, 2) if delta is not None else None, "n": n}
        results.append(row)
        print(f"[scan] mult={mult:<5} acc={acc:.2f} delta={delta:+.2f}" if delta is not None
              else f"[scan] mult={mult} acc={acc}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[scan] saved -> {args.out}")
    best = max((r for r in results if r["delta"] is not None),
               key=lambda r: r["delta"], default=None)
    if best:
        print(
            f"[scan] BEST: mult={best['multiplier']} delta={best['delta']:+.2f}")


if __name__ == "__main__":
    main()
