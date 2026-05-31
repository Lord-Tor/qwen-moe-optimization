import torch


def build_prompt(question, choices):
    # БЕЗ хвостового пробела: в BPE Qwen естественное продолжение после "Answer:"
    # — это токен с ведущим пробелом (" A"). Скорим именно их (см. ниже).
    return (f"Question: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer:")


def prepare_choice_token_ids(tokenizer):
    ids = {}
    for ch in ["A", "B", "C", "D"]:
        toks = tokenizer(" " + ch, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            print(f"[warn] ' {ch}' -> {toks} (мультитокен); беру первый")
        ids[ch] = toks[0]
    if len(set(ids.values())) != 4:
        print(f"[warn] КОЛЛИЗИЯ id ответов (скоринг сломан): {ids}")
    return ids


def extract_last_token_topk(router_logits_tuple, k=4):
    """Топ-k экспертов последнего токена из output_router_logits.
    В transformers 5.x это кортеж тензоров логитов (по слою),
    форма (tokens, num_experts) или (batch, seq, num_experts)."""
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
        probs = torch.softmax(logits_last.to(torch.float32), dim=-1)
        kk = min(k, probs.shape[-1])
        topk_vals, topk_idx = torch.topk(probs, k=kk)
        result[f"layer_{layer_idx}"] = {
            "topk_experts": topk_idx.detach().cpu().tolist(),
            "topk_probs": [float(x) for x in topk_vals.detach().cpu().tolist()],
        }
    return result
