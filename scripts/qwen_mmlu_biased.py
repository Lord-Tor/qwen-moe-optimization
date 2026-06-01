import argparse
import gc
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import build_prompt, prepare_choice_token_ids, extract_last_token_topk


# ---------------------------------------------------------------------------
# КОРЕНЬ БАГА (transformers 5.x):
# mlp.gate теперь Qwen2MoeTopKRouter и его forward возвращает КОРТЕЖ
# (router_logits, routing_weights, router_indices). Реальная маршрутизация
# идёт по router_indices (3-й элемент) + routing_weights (2-й).
# Старый хук прибавлял bias к router_logits (1-й элемент) — он влияет ТОЛЬКО
# на aux-loss/логирование, поэтому scores в выходе были идентичны.
#
# Чтобы РЕАЛЬНО изменить маршрут, надо:
#   1) прибавить bias к router_logits,
#   2) ПЕРЕСЧИТАТЬ top-k и routing_weights из смещённых логитов,
#   3) вернуть подменённый кортеж (logits, weights, indices).
# ---------------------------------------------------------------------------


def make_topk_recompute_hook(b_tensor, top_k, norm_topk_prob):
    """Хук для transformers 5.x: пересчитывает выбор экспертов из смещённых логитов."""
    def hook(module, args, output):
        if not (isinstance(output, tuple) and len(output) >= 3):
            # Фолбэк на старый API (gate = nn.Linear -> один тензор логитов):
            # там top-k берётся снаружи, поэтому достаточно сместить логиты.
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits + b_tensor.to(logits.device, logits.dtype)
            return (logits,) + output[1:] if isinstance(output, tuple) else logits

        router_logits, routing_weights, router_indices = output[0], output[1], output[2]
        biased = router_logits + \
            b_tensor.to(router_logits.device, router_logits.dtype)

        probs = torch.softmax(biased.float(), dim=-1)
        new_w, new_idx = torch.topk(probs, top_k, dim=-1)
        if norm_topk_prob:
            new_w = new_w / new_w.sum(dim=-1, keepdim=True)
        new_w = new_w.to(routing_weights.dtype)
        new_idx = new_idx.to(router_indices.dtype)
        return (biased, new_w, new_idx) + tuple(output[3:])
    return hook


def apply_bias_hooks(model, bias_file, dtype, multiplier, top_k, norm_topk_prob):
    with open(bias_file, "r", encoding="utf-8") as f:
        bias_data = json.load(f)

    hooks = []
    for i, layer in enumerate(model.model.layers):
        layer_name = f"layer_{i}"
        if layer_name in bias_data.get("bias", {}) and hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            bias_tensor = torch.zeros(model.config.num_experts, dtype=dtype)
            for exp_id, val in bias_data["bias"][layer_name].items():
                bias_tensor[int(exp_id)] = float(val) * multiplier
            h = layer.mlp.gate.register_forward_hook(
                make_topk_recompute_hook(bias_tensor, top_k, norm_topk_prob))
            hooks.append(h)
    print(f"[bias] Applied recompute hooks to {len(hooks)} layers "
          f"(multiplier={multiplier}, top_k={top_k}, norm_topk={norm_topk_prob}).")
    if len(hooks) == 0:
        raise RuntimeError(
            "0 хуков навешано — проверь имена слоёв в bias_file.")
    return hooks


def make_random_bias_hooks(model, dtype, multiplier, top_k, norm_topk_prob,
                           layer_names, bias_value, seed=0):
    """КОНТРОЛЬ: те же слои и та же величина смещения, но эксперты выбраны СЛУЧАЙНО.
    Если градиентный bias даёт Δ, неотличимую от random-bias, значит эффект — это
    шум от перемаршрутизации вообще, а не от 'доменных' экспертов."""
    g = torch.Generator().manual_seed(seed)
    hooks = []
    for i, layer in enumerate(model.model.layers):
        layer_name = f"layer_{i}"
        if layer_name in layer_names and hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            n = model.config.num_experts
            k = len(layer_names[layer_name])
            idx = torch.randperm(n, generator=g)[:k]
            bias_tensor = torch.zeros(n, dtype=dtype)
            bias_tensor[idx] = bias_value * multiplier
            h = layer.mlp.gate.register_forward_hook(
                make_topk_recompute_hook(bias_tensor, top_k, norm_topk_prob))
            hooks.append(h)
    print(f"[random-bias] Applied to {len(hooks)} layers (control run).")
    return hooks


def onepass_choice_scores(model, tokenizer, device, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    MAX_SEQ_LEN = 512
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, -MAX_SEQ_LEN:]
        attention_mask = attention_mask[:, -MAX_SEQ_LEN:]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_router_logits=True, return_dict=True)

    next_token_logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(next_token_logits.to(torch.float32), dim=-1)
    scores = {ch: float(log_probs[tid].item())
              for ch, tid in choice_token_ids.items()}
    router_info = extract_last_token_topk(outputs.router_logits, k=4)
    return scores, router_info


def run_eval(model, tokenizer, device, ds, choice_token_ids, answer_map, limit, out_path, subject):
    total = correct = 0
    all_scores = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            gold = answer_map[int(ex["answer"])]
            prompt = build_prompt(ex["question"], ex["choices"])
            scores, router_info = onepass_choice_scores(
                model, tokenizer, device, prompt, choice_token_ids)
            pred = max(scores, key=scores.get)
            ok = pred == gold
            total += 1
            correct += int(ok)
            all_scores.append(scores)
            fout.write(json.dumps({"index": i, "subject": subject, "gold": gold,
                                   "pred": pred, "correct": ok, "scores": scores,
                                   "router_last_token_topk": router_info},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return correct, total, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str,
                        default="high_school_mathematics")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="results/math_biased_cluster.jsonl")
    parser.add_argument("--experts_impl", type=str, default="eager")
    parser.add_argument("--bias_file", type=str, required=True)
    parser.add_argument("--bias_multiplier", type=float, default=1.0)
    parser.add_argument("--random_control", action="store_true",
                        help="Доп. прогон со случайным bias (контроль научной валидности)")
    args = parser.parse_args()
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if torch.cuda.is_available():
        dtype = torch.float16
        n_gpu = torch.cuda.device_count()
        max_mem = {0: "10GiB", 1: "22GiB"} if n_gpu >= 2 else {0: "22GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto", max_memory=max_mem, attn_implementation="sdpa",
            offload_folder="/mnt/tank/scratch/" + __import__("os").environ.get("USER", "tmp") + "/.offload_qwen")
        model_device = model.model.embed_tokens.weight.device
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True)
        model_device = "mps"
        model.to(model_device)
    else:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True)
        model_device = "cpu"
        model.to(model_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    model.config._experts_implementation = args.experts_impl  # no-op в стоке, оставлено

    top_k = getattr(model.config, "num_experts_per_tok", 4)
    norm_topk_prob = getattr(model.config, "norm_topk_prob", False)

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # ---- SELF-CHECK на 1 примере: меняет ли инжект выход модели? ----
    sample = ds[0]
    sample_prompt = build_prompt(sample["question"], sample["choices"])
    scores_before, _ = onepass_choice_scores(
        model, tokenizer, model_device, sample_prompt, choice_token_ids)

    hooks = apply_bias_hooks(model, args.bias_file, dtype,
                             args.bias_multiplier, top_k, norm_topk_prob)

    scores_after, _ = onepass_choice_scores(
        model, tokenizer, model_device, sample_prompt, choice_token_ids)
    delta = sum(abs(scores_before[c] - scores_after[c]) for c in "ABCD")
    print(f"[self-check] |Δscores| на 1 примере = {delta:.6f}")
    if delta < 1e-9:
        for h in hooks:
            h.remove()
        raise RuntimeError(
            "SELF-CHECK ПРОВАЛЕН: инжект НЕ меняет выход модели (как в старом баге). "
            "Проверь сигнатуру mlp.gate под свою версию transformers.")
    print("[self-check] OK: инжект влияет на выход модели.")

    # ---- Основной biased-прогон ----
    correct, total, _ = run_eval(model, tokenizer, model_device, ds,
                                 choice_token_ids, answer_map, args.limit,
                                 args.output, args.subject)
    print(f"Done (biased). Accuracy: {correct}/{total} = {correct/total:.4f}")
    for h in hooks:
        h.remove()

    # ---- Контроль на random-bias (та же геометрия, случайные эксперты) ----
    if args.random_control:
        with open(args.bias_file) as f:
            bdata = json.load(f)
        layer_names = {k: list(v.keys())
                       for k, v in bdata.get("bias", {}).items()}
        any_val = next(iter(next(iter(bdata["bias"].values())).values()))
        rhooks = make_random_bias_hooks(model, dtype, args.bias_multiplier,
                                        top_k, norm_topk_prob, layer_names,
                                        bias_value=float(any_val), seed=0)
        rout = args.output.replace(".jsonl", "_randomctrl.jsonl")
        rc, rt, _ = run_eval(model, tokenizer, model_device, ds,
                             choice_token_ids, answer_map, args.limit,
                             rout, args.subject)
        for h in rhooks:
            h.remove()
        print(
            f"Done (random-ctrl). Accuracy: {rc}/{rt} = {rc/rt:.4f}  -> {rout}")
        print("[вывод] Сравни biased vs random-ctrl: если Δ примерно равны — "
              "эффект НЕ специфичен для 'доменных' экспертов.")


if __name__ == "__main__":
    main()
