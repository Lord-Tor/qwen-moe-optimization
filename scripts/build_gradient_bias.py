import argparse
import json
import torch
import gc
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import build_prompt, prepare_choice_token_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str, default="college_mathematics")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="configs/math_grad_bias.json")
    parser.add_argument("--topk_experts", type=int, default=4)
    parser.add_argument("--bias_value", type=float, default=0.2)
    parser.add_argument("--experts_impl", type=str,
                        default="eager")  # no-op в стоке
    parser.add_argument("--no_checkpoint", action="store_true")
    # Исключать ли уже-доминирующих экспертов (баг №3): смещаем в полезных,
    # но НЕ топовых по частоте, иначе подкрепляем статус-кво.
    parser.add_argument("--exclude_dominant", action="store_true")
    parser.add_argument("--dominant_per_layer", type=int, default=4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        dtype = torch.float16
        max_mem = {0: "10GiB", 1: "22GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto", max_memory=max_mem, attn_implementation="sdpa")
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

    for p in model.parameters():
        p.requires_grad = False
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            # под 5.x веса роутера лежат в gate; имя параметра может отличаться,
            # поэтому включаем grad на все параметры gate.
            for p in layer.mlp.gate.parameters():
                p.requires_grad = True

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if not args.no_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # backward-хук на gate: grad_output[0] = dL/d(router_logits).
    # Надёжно при checkpointing (в отличие от forward-hook + retain_grad).
    step_grads = {}

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            step_grads[name] = g.detach().float().reshape(
                -1, g.shape[-1]).mean(dim=0)
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            hooks.append(layer.mlp.gate.register_full_backward_hook(
                make_bwd_hook(f"layer_{i}")))

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    loss_fn = torch.nn.CrossEntropyLoss()

    n_exp = model.config.num_experts
    accumulated = defaultdict(lambda: torch.zeros(
        n_exp, device=model_device, dtype=torch.float32))
    # частоты выбора экспертов (для --exclude_dominant)
    freq = defaultdict(lambda: torch.zeros(n_exp, dtype=torch.float32))
    processed = 0

    print("=== Starting Gradient Collection ===")
    for i, ex in enumerate(ds):
        if i >= args.limit:
            break
        gold = answer_map[int(ex["answer"])]
        prompt = build_prompt(ex["question"], ex["choices"])
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        MAX_SEQ_LEN = 256
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, -MAX_SEQ_LEN:]
            attention_mask = attention_mask[:, -MAX_SEQ_LEN:]

        step_grads.clear()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_router_logits=True, return_dict=True)
        next_token_logits = outputs.logits[0, -1, :]
        target = torch.tensor([choice_token_ids[gold]],
                              device=next_token_logits.device)
        loss = loss_fn(next_token_logits.float().unsqueeze(0), target)

        model.zero_grad(set_to_none=True)
        loss.backward()

        if not step_grads:
            raise RuntimeError(
                "backward-хук не дал градиентов. Запусти с --no_checkpoint и "
                "проверь requires_grad на mlp.gate.")

        for name, g in step_grads.items():
            accumulated[name] += g.to(model_device)

        # частоты из router_logits этого примера (последний токен)
        if args.exclude_dominant and outputs.router_logits is not None:
            for li, rl in enumerate(outputs.router_logits):
                if rl is None:
                    continue
                last = rl[0, -1] if rl.dim() == 3 else (rl[-1]
                                                        if rl.dim() == 2 else rl)
                idx = torch.topk(last.float(), min(
                    args.dominant_per_layer, last.shape[-1])).indices
                freq[f"layer_{li}"][idx.cpu()] += 1

        processed += 1
        print(
            f"[{processed}/{args.limit}] loss={loss.item():.4f} layers={len(step_grads)}")
        del outputs, loss, next_token_logits, target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    total_norm = sum(float(v.norm()) for v in accumulated.values())
    print(f"[diag] суммарная норма градиентов = {total_norm:.6f}")
    assert total_norm > 0, "Градиенты НУЛЕВЫЕ — захват не сработал."

    print("=== Generating Bias Config ===")
    bias_config = {"domain": f"{args.subject}_grad",
                   "meta": {"exclude_dominant": args.exclude_dominant,
                            "topk_experts": args.topk_experts},
                   "bias": {}}
    for name, total in accumulated.items():
        avg = total / processed
        if args.exclude_dominant:
            # маскируем доминирующих: ставим им +inf градиент, чтобы не попали в "полезные"
            dominant = freq[name] > 0
            avg_masked = avg.clone()
            avg_masked[dominant.to(avg.device)] = float("inf")
            topk_idx = torch.topk(
                avg_masked, k=args.topk_experts, largest=False).indices
        else:
            topk_idx = torch.topk(
                avg, k=args.topk_experts, largest=False).indices
        bias_config["bias"][name] = {
            str(int(idx)): args.bias_value for idx in topk_idx}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(bias_config, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved to {args.output} (layers: {len(bias_config['bias'])})")


if __name__ == "__main__":
    main()
