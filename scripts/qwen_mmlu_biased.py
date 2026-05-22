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


def prepare_choice_token_ids(tokenizer):
    return {ch: tokenizer(" " + ch, add_special_tokens=False)["input_ids"][0] for ch in ["A", "B", "C", "D"]}


def onepass_choice_scores(model, tokenizer, device, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = inputs["input_ids"].to(
        device), inputs["attention_mask"].to(device)

    # Честный паритет с biased скриптом
    MAX_SEQ_LEN = 512
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, -MAX_SEQ_LEN:]
        attention_mask = attention_mask[:, -MAX_SEQ_LEN:]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_router_logits=False, return_dict=True)

    next_token_logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(next_token_logits.to(torch.float32), dim=-1)
    scores = {ch: float(log_probs[token_id].item())
              for ch, token_id in choice_token_ids.items()}
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--subject", type=str,
                        default="high_school_mathematics")
    parser.add_argument("--split", type=str, default="test")
    # Снимаем жесткий лимит
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="results/mmlu_onepass_results.jsonl")
    parser.add_argument("--experts_impl", type=str, default="batched_mm")
    args = parser.parse_args()

    if torch.cuda.is_available():
        dtype = torch.float16
        max_mem = {0: "10GiB", 1: "22GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto", max_memory=max_mem, attn_implementation="sdpa")
        model_device = model.model.embed_tokens.weight.device
    else:
        # Fallbacks for Mac/CPU
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
        model_device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(model_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    model.config._experts_implementation = args.experts_impl

    ds = load_dataset("cais/mmlu", args.subject, split=args.split)
    choice_token_ids = prepare_choice_token_ids(tokenizer)
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    total = correct = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break

            gold = answer_map[int(ex["answer"])]
            prompt = build_prompt(ex["question"], ex["choices"])

            infer_start = time.perf_counter()
            scores = onepass_choice_scores(
                model, tokenizer, model_device, prompt, choice_token_ids)
            infer_time = time.perf_counter() - infer_start

            pred = max(scores, key=scores.get)
            is_correct = pred == gold
            total += 1
            correct += int(is_correct)

            row = {"index": i, "subject": args.subject, "gold": gold, "pred": pred,
                   "correct": is_correct, "scores": scores, "infer_time_sec": infer_time}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{i+1}] gold={gold} pred={pred} correct={is_correct} acc={correct/total:.3f} infer={infer_time:.3f}s")
            gc.collect()

    print(f"Done. Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
