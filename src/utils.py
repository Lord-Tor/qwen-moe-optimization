import torch


def build_prompt(question, choices):
    return (f"Question: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: ")


def prepare_choice_token_ids(tokenizer):
    return {ch: tokenizer(ch, add_special_tokens=False)["input_ids"][0] for ch in ["A", "B", "C", "D"]}


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
        probs = torch.softmax(logits_last.to(torch.float32), dim=-1)
        kk = min(k, probs.shape[-1])
        topk_vals, topk_idx = torch.topk(probs, k=kk)
        result[f"layer_{layer_idx}"] = {
            "topk_experts": topk_idx.detach().cpu().tolist(),
            "topk_probs": [float(x) for x in topk_vals.detach().cpu().tolist()]
        }
    return result
