import json
import sys
from collections import defaultdict, Counter

files = sys.argv[1:]
layer_expert_counts = defaultdict(Counter)
total_examples = 0

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_examples += 1
            routing = row.get("router_last_token_topk", {})
            for layer_name, info in routing.items():
                for expert_id in info.get("topk_experts", []):
                    layer_expert_counts[layer_name][expert_id] += 1

print(f"examples: {total_examples}")
for layer_name in sorted(layer_expert_counts.keys(), key=lambda x: int(x.split('_')[1])):
    top = layer_expert_counts[layer_name].most_common(10)
    print(f"\n{layer_name}")
    for expert_id, cnt in top:
        print(f"  expert {expert_id}: {cnt}")
