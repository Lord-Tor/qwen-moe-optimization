# 🚀 Qwen1.5-MoE Routing Optimization [WIP]

Optimization of Mixture-of-Experts (MoE) routing mechanisms for domain-specific tasks without full fine-tuning.

## 📖 Project Overview
This project investigates how modifying the router logits in the `Qwen1.5-MoE-A2.7B` model affects its performance on specific knowledge domains (e.g., High School Mathematics). By analyzing the natural routing distribution and applying a gradient-free frequency-based bias, we aim to surgically improve domain accuracy while maintaining inference efficiency.

## ⚙️ Repository Structure
- `configs/` — JSON files containing generated domain-specific bias vectors.
- `scripts/` — Executable Python and Bash scripts for inference, evaluation, and analysis.
- `src/` — Core utilities (prompts, hooks, logprob scorers).
- `results/` — Evaluation logs and routing statistics (ignored in version control).

## 🛠 Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt