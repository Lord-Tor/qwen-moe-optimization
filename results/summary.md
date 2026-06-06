
=================================================================
| Subject                   | Baseline (%) | Biased (%) | Delta  |
=================================================================
| Qwen1.5-MoE-A2.7B_high_school_mathematics | 28.89        | 28.89      | +0.00% |
| Qwen1.5-MoE-A2.7B_abstract_algebra | 31.00        | 31.00      | +0.00% |
| Qwen1.5-MoE-A2.7B_elementary_mathematics | 46.83        | 46.83      | +0.00% |
| Qwen1.5-MoE-A2.7B_formal_logic | 35.71        | 35.71      | +0.00% |
| Qwen1.5-MoE-A2.7B_college_mathematics | 31.00        | 31.00      | +0.00% |
| Qwen1.5-MoE-A2.7B-Chat_high_school_mathematics | 28.89        | 28.89      | +0.00% |
| Qwen1.5-MoE-A2.7B-Chat_abstract_algebra | 24.00        | 24.00      | +0.00% |
| Qwen1.5-MoE-A2.7B-Chat_elementary_mathematics | 39.15        | 39.15      | +0.00% |
| Qwen1.5-MoE-A2.7B-Chat_formal_logic | 38.10        | 38.10      | +0.00% |
| Qwen1.5-MoE-A2.7B-Chat_college_mathematics | 28.00        | 28.00      | +0.00% |
=================================================================

To visualize a specific subject, run:
python scripts/plot_results.py --baseline results/SUBJECT_baseline.jsonl --heuristic results/SUBJECT_biased.jsonl

