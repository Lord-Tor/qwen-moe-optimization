import argparse
import gc
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(question, choices):
    return (f"Question: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")


def extract_last_token_topk(router_logits_tuple, k=4):
    result = {}
    if router_logits_tuple is None:
        return result
    for layer_idx, layer_logits in enumerate(router_logits_tuple):
        if layer_logits is None:
            continue
        if layer_logits.dim() == 3:
            logits_last = layer_logits[0, -1]
        elif layer_logits.dim() == 2:
            logits_last = layer_logits[-1]
        elif layer_logits.dim() == 1:
            logits_last = layer_logits
        else:
            continue
        probs = torch.softmax(logits_last, dim=-1)
        kk = min(k, probs.shape[-1])
        topk_vals, topk_idx = torch.topk(probs, k=kk)
        result[f"layer_{layer_idx}"] = {
            "topk_experts": topk_idx.detach().cpu().tolist(),
            "topk_probs": [float(x) for x in topk_vals.detach().cpu().tolist()]
        }
    return result


def prepare_choice_token_ids(tokenizer):
    return {ch: tokenizer(" " + ch, add_special_tokens=False)["input_ids"][0] for ch in ["A", "B", "C", "D"]}


def onepass_choice_scores(model, tokenizer, device, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = inputs["input_ids"].to(
        device), inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_router_logits=True, return_dict=True)
    next_token_logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    scores = {ch: float(log_probs[token_id].item())
              for ch, token_id in choice_token_ids.items()}
    router_info = extract_last_token_topk(outputs.router_logits, k=4)
    return scores, router_info


def apply_bias_hooks(model, bias_file, dtype):
    with open(bias_file, "r", encoding="utf-8") as f:
        bias_data = json.load(f)

    hooks = []
    for i, layer in enumerate(model.model.layers):
        layer_name = f"layer_{i}"
        if layer_name in bias_data["bias"]:
            # Создаем тензор на CPU, хук сам перекинет его на нужный GPU в момент прохода
            bias_tensor = torch.zeros(model.config.num_experts, dtype=dtype)
            for exp_id, val in bias_data["bias"][layer_name].items():
                bias_tensor[int(exp_id)] = float(val)

            def get_hook(b_tensor):
                def hook(module, args, output):
                    # Динамическая синхронизация девайса тензора с выходом текущего слоя
                    return output + b_tensor.to(output.device)
                return hook

            h = layer.mlp.gate.register_forward_hook(get_hook(bias_tensor))
            hooks.append(h)

    print(
        f"[bias] Applied bias hooks from {bias_file} to {len(hooks)} layers.")
    return hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str,
                        default="high_school_mathematics")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=270)
    parser.add_argument("--output", type=str,
                        default="results/mmlu_biased.jsonl")
    parser.add_argument("--experts_impl", type=str,
                        default="batched_mm")  # На NVIDIA eager не нужен
    parser.add_argument("--bias_file", type=str,
                        required=True, help="Path to JSON bias file")
    args = parser.parse_args()

    # Мульти-платформенная инициализация
    if torch.cuda.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="sdpa"
        )
        # Получаем девайс первого слоя
        model_device = model.model.embed_tokens.weight.device
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
        model_device = "mps"
        model.to(model_device)
    else:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
        model_device = "cpu"
        model.to(model_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    model.config._experts_implementation = args.experts_impl

    # Внедряем bias без жесткой привязки к девайсу
    apply_bias_hooks(model, args.bias_file, dtype)

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    total = correct = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break
            q_start = time.perf_counter()
            gold = answer_map[int(ex["answer"])]
            prompt = build_prompt(ex["question"], ex["choices"])

            infer_start = time.perf_counter()
            # Передаем model_device для синхронизации токенов
            scores, router_info = onepass_choice_scores(
                model, tokenizer, model_device, prompt, choice_token_ids)
            infer_time = time.perf_counter() - infer_start

            pred = max(scores, key=scores.get)
            is_correct = pred == gold
            total += 1
            correct += int(is_correct)
            elapsed = time.perf_counter() - q_start

            row = {"index": i, "subject": args.subject, "gold": gold, "pred": pred, "correct": is_correct,
                   "scores": scores, "infer_time_sec": infer_time, "router_last_token_topk": router_info}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{i+1}/{args.limit}] gold={gold} pred={pred} correct={is_correct} acc={correct/total:.3f} infer={infer_time:.3f}s")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"Done. Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
