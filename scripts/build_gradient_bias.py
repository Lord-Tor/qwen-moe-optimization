import argparse
import json
import torch
import gc
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(question, choices):
    return (f"Question: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:")


def prepare_choice_token_ids(tokenizer):
    return {ch: tokenizer(" " + ch, add_special_tokens=False)["input_ids"][0] for ch in ["A", "B", "C", "D"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str, default="college_mathematics")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="configs/math_grad_bias.json")
    parser.add_argument("--experts_impl", type=str, default="batched_mm")
    args = parser.parse_args()

    if torch.cuda.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
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

    for param in model.parameters():
        param.requires_grad = False

    layer_logits_dict = {}

    def capture_logits_hook(name):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            logits = output[0] if is_tuple else output
            new_logits = logits.detach().requires_grad_(True)
            new_logits.retain_grad()
            layer_logits_dict[name] = new_logits
            return (new_logits,) + output[1:] if is_tuple else new_logits
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            h = layer.mlp.gate.register_forward_hook(
                capture_logits_hook(f"layer_{i}"))
            hooks.append(h)

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    loss_fn = torch.nn.CrossEntropyLoss()
    accumulated_grads = defaultdict(lambda: torch.zeros(
        model.config.num_experts, device=model_device))
    processed_examples = 0

    print("=== Starting Gradient Collection ===")
    for i, ex in enumerate(ds):
        if i >= args.limit:
            break

        gold_char = answer_map[int(ex["answer"])]
        prompt = build_prompt(ex["question"], ex["choices"])

        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, use_cache=False)
        next_token_logits = outputs.logits[0, -1, :]

        target_token_id = choice_token_ids[gold_char]
        target = torch.tensor(
            [target_token_id], device=next_token_logits.device)

        loss = loss_fn(next_token_logits.to(
            torch.float32).unsqueeze(0), target)

        model.zero_grad()
        loss.backward()

        for layer_name, logits_tensor in layer_logits_dict.items():
            if logits_tensor.grad is not None:
                grad = logits_tensor.grad[0, -1, :].detach().to(model_device)
                accumulated_grads[layer_name] += grad

        processed_examples += 1
        print(
            f"[{processed_examples}/{args.limit}] Processed. Loss: {loss.item():.4f}")

        layer_logits_dict.clear()
        del outputs, loss, next_token_logits, target
        gc.collect()

    for h in hooks:
        h.remove()

    print("=== Generating Bias Config ===")
    bias_config = {"domain": f"{args.subject}_grad", "bias": {}}

    for layer_name, total_grad in accumulated_grads.items():
        avg_grad = total_grad / processed_examples
        topk_grads, topk_indices = torch.topk(avg_grad, k=4, largest=False)
        layer_bias = {str(idx.item()): 0.2 for idx in topk_indices}
        bias_config["bias"][layer_name] = layer_bias

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(bias_config, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
