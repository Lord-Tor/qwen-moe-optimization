import json
import sys
from collections import defaultdict, Counter

domain = sys.argv[1]
files = sys.argv[2:]

layer_expert_counts = defaultdict(Counter)

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            routing = row.get("router_last_token_topk", {})
            for layer_name, info in routing.items():
                for expert_id in info.get("topk_experts", []):
                    layer_expert_counts[layer_name][expert_id] += 1

bias = {}
for layer_name, counter in layer_expert_counts.items():
    top_experts = [expert for expert, _ in counter.most_common(4)]
    bias[layer_name] = {str(expert): 0.2 for expert in top_experts}

out = {
    "domain": domain,
    "bias": bias,
}
with open(f"{domain}_bias.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"saved: {domain}_bias.json")
