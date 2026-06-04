import argparse
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import build_prompt, prepare_choice_token_ids, extract_last_token_topk


def onepass_choice_scores(model, tokenizer, device, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = inputs["input_ids"].to(
        device), inputs["attention_mask"].to(device)

    MAX_SEQ_LEN = 512  # как в qwen_mmlu_biased.py — для сопоставимости baseline/biased
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, -MAX_SEQ_LEN:]
        attention_mask = attention_mask[:, -MAX_SEQ_LEN:]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_router_logits=True, return_dict=True)

    next_token_logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(next_token_logits.to(torch.float32), dim=-1)
    scores = {ch: float(log_probs[token_id].item())
              for ch, token_id in choice_token_ids.items()}
    router_info = extract_last_token_topk(outputs.router_logits, k=4)
    return scores, router_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str,
                        default="high_school_mathematics")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=270)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default="results/mmlu_onepass_results.jsonl")
    parser.add_argument("--experts_impl", type=str, default="eager")
    args = parser.parse_args()
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if torch.cuda.is_available():
        dtype = torch.float16
        n_gpu = torch.cuda.device_count()
        max_mem = ({0: "10GiB", 1: "22GiB"} if n_gpu >= 2 else {0: "22GiB"}) \
            if "qwen" in args.model.lower() else None  # не-Qwen: авто-баланс
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True, device_map="auto",
            max_memory=max_mem, attn_implementation="sdpa", experts_implementation=args.experts_impl,
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

    model.eval()
    model.config._experts_implementation = args.experts_impl

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    total = correct = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            if i < args.offset:
                continue
            if i >= args.offset + args.limit:
                break

            gold = answer_map[int(ex["answer"])]
            prompt = build_prompt(ex["question"], ex["choices"])

            infer_start = time.perf_counter()
            scores, router_info = onepass_choice_scores(
                model, tokenizer, model_device, prompt, choice_token_ids)
            infer_time = time.perf_counter() - infer_start

            pred = max(scores, key=scores.get)
            is_correct = pred == gold
            total += 1
            correct += int(is_correct)

            row = {"index": i, "subject": args.subject, "gold": gold, "pred": pred, "correct": is_correct,
                   "scores": scores, "infer_time_sec": infer_time, "router_last_token_topk": router_info}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{i+1}/{args.offset + args.limit}] gold={gold} pred={pred} correct={is_correct} acc={correct/total:.3f} infer={infer_time:.3f}s")

    print(f"Done. Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
