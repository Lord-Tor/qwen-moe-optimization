import argparse
import json
import torch
import gc
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import build_prompt, prepare_choice_token_ids


# ---------------------------------------------------------------------------
# Захват dL/d(router_logits) НАДЁЖНО в обоих режимах (с/без checkpointing).
#
# Способ: forward-хук на mlp.gate подменяет возвращаемый тензор логитов
# роутера на тот же тензор с retain_grad(), оставляя его в графе. В отличие
# от register_full_backward_hook на под-модуле (который при non-reentrant
# checkpointing не всегда получает grad_output), retained leaf-тензор
# гарантированно накапливает .grad при backward — даже если активации
# перевычисляются.
#
# В transformers 5.x gate возвращает кортеж (router_logits, weights, indices).
# Мы трогаем ТОЛЬКО router_logits (output[0]) — на forward это эквивалентная
# подстановка (тот же тензор), маршрутизацию не меняем, нам нужен лишь grad.
# ---------------------------------------------------------------------------


def attach_logit_capture_hooks(model, store):
    """Вешает forward-хуки, кладёт retained router_logits каждого слоя в store."""
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate")):
            continue

        def make_hook(name):
            def hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                logits = output[0] if is_tuple else output
                if logits.requires_grad:
                    logits.retain_grad()
                    store[name] = logits
                return output  # тот же объект — forward не меняется
            return hook

        hooks.append(layer.mlp.gate.register_forward_hook(
            make_hook(f"layer_{i}")))
    return hooks


def collect_grads(model, tokenizer, ds, choice_token_ids, answer_map,
                  limit, model_device, use_checkpoint):
    """Один проход сбора. Возвращает (accumulated, freq, processed).
    Кидает torch.cuda.OutOfMemoryError наружу, чтобы вызывающий мог фолбэкнуться."""
    if use_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    n_exp = model.config.num_experts
    accumulated = defaultdict(lambda: torch.zeros(
        n_exp, device=model_device, dtype=torch.float32))
    freq = defaultdict(lambda: torch.zeros(n_exp, dtype=torch.float32))
    loss_fn = torch.nn.CrossEntropyLoss()
    store = {}
    hooks = attach_logit_capture_hooks(model, store)
    processed = 0

    try:
        for i, ex in enumerate(ds):
            if i >= limit:
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

            store.clear()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, output_router_logits=True, return_dict=True)
            next_token_logits = outputs.logits[0, -1, :]
            target = torch.tensor([choice_token_ids[gold]],
                                  device=next_token_logits.device)
            loss = loss_fn(next_token_logits.float().unsqueeze(0), target)

            model.zero_grad(set_to_none=True)
            loss.backward()

            captured = 0
            for name, logits in store.items():
                if logits.grad is not None:
                    g = logits.grad.detach().float().reshape(-1,
                                                             logits.shape[-1]).mean(dim=0)
                    accumulated[name] += g.to(model_device)
                    captured += 1

            if captured == 0:
                raise RuntimeError("NO_GRAD_CAPTURED")

            # частоты для exclude_dominant
            if outputs.router_logits is not None:
                for li, rl in enumerate(outputs.router_logits):
                    if rl is None:
                        continue
                    last = rl[0, -1] if rl.dim() == 3 else (rl[-1]
                                                            if rl.dim() == 2 else rl)
                    kk = min(4, last.shape[-1])
                    idx = torch.topk(last.float(), kk).indices
                    freq[f"layer_{li}"][idx.cpu()] += 1

            processed += 1
            print(f"[{processed}/{limit}] loss={loss.item():.4f} captured={captured} "
                  f"(checkpoint={use_checkpoint})")
            del outputs, loss, next_token_logits, target
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    return accumulated, freq, processed


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
    # Режим checkpointing: auto (по умолчанию) пробует БЕЗ него, при OOM включает.
    parser.add_argument("--checkpoint_mode",
                        choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--exclude_dominant", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        dtype = torch.float16
        # На 1 GPU max_memory с ключом 1 упадёт — берём только реально видимые карты.
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 2:
            max_mem = {0: "10GiB", 1: "22GiB"}
        else:
            max_mem = {0: "22GiB"}
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
    model.config._experts_implementation = args.experts_impl

    for p in model.parameters():
        p.requires_grad = False
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            for p in layer.mlp.gate.parameters():
                p.requires_grad = True
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # ---- Выбор стратегии checkpointing с авто-фолбэком ----
    print("=== Starting Gradient Collection ===")
    if args.checkpoint_mode == "on":
        attempts = [True]
    elif args.checkpoint_mode == "off":
        attempts = [False]
    else:  # auto: сначала быстро без checkpointing, при OOM — с ним
        attempts = [False, True]

    accumulated = freq = None
    processed = 0
    last_err = None
    for use_ckpt in attempts:
        try:
            print(f"--- attempt: checkpoint={use_ckpt} ---")
            accumulated, freq, processed = collect_grads(
                model, tokenizer, ds, choice_token_ids, answer_map,
                args.limit, model_device, use_ckpt)
            break
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(
                f"[oom] checkpoint={use_ckpt} -> OOM, чищу кэш и пробую следующий режим")
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                last_err = e
                print(
                    f"[oom] checkpoint={use_ckpt} -> OOM (RuntimeError), пробую дальше")
                gc.collect()
                torch.cuda.empty_cache()
            elif "NO_GRAD_CAPTURED" in str(e):
                last_err = e
                print(
                    f"[no-grad] checkpoint={use_ckpt} не дал градиентов, пробую следующий режим")
            else:
                raise

    if processed == 0:
        raise RuntimeError(
            f"Не удалось собрать градиенты ни в одном режиме. Последняя ошибка: {last_err}")

    total_norm = sum(float(v.norm()) for v in accumulated.values())
    print(
        f"[diag] суммарная норма градиентов = {total_norm:.6f}, processed={processed}")
    assert total_norm > 0, "Градиенты НУЛЕВЫЕ — захват не сработал."

    print("=== Generating Bias Config ===")
    bias_config = {"domain": f"{args.subject}_grad",
                   "meta": {"exclude_dominant": args.exclude_dominant,
                            "topk_experts": args.topk_experts},
                   "bias": {}}
    for name, total in accumulated.items():
        avg = total / processed
        if args.exclude_dominant:
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

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(bias_config, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved to {args.output} (layers: {len(bias_config['bias'])})")


if __name__ == "__main__":
    main()
