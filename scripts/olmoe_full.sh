#!/bin/bash
#SBATCH --job-name=olmoe_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_olmoe_full_%j.log

# Запускать ТОЛЬКО после успешного olmoe_smoke (через afterok).
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results configs configs/raw results/scan results/final

MODEL="allenai/OLMoE-1B-7B-0924"
MN=$(basename $MODEL)
SUBJECTS=("high_school_mathematics" "abstract_algebra" "formal_logic" "college_mathematics")
GRAD_LIMIT=40
MULTS="0.5 1 2 5 10 20 50"

run_domain () (
    set -e
    SUBJ="$1"
    DONE="results/${MN}_${SUBJ}.done"
    [ -f "$DONE" ] && { echo "[skip] $SUBJ"; exit 0; }
    echo "######## OLMoE DOMAIN: $SUBJ ########"

    # baseline
    python scripts/qwen_mmlu_onepass.py \
        --model $MODEL --subject $SUBJ \
        --output results/${MN}_${SUBJ}_baseline.jsonl --limit 10000

    # градиенты + конфиг
    python scripts/build_gradient_bias.py \
        --model $MODEL --subject $SUBJ \
        --output configs/${MN}_${SUBJ}_grad.json \
        --save_raw_grads configs/raw/${MN}_${SUBJ}_grad.pt \
        --limit $GRAD_LIMIT --checkpoint_mode auto
    python scripts/make_bias_from_grads.py \
        --raw configs/raw/${MN}_${SUBJ}_grad.pt \
        --out_normal configs/${MN}_${SUBJ}_normal.json \
        --out_exclude configs/${MN}_${SUBJ}_exclude.json

    # скан множителя
    python scripts/multiplier_scan.py \
        --model $MODEL --subject $SUBJ \
        --bias_file configs/${MN}_${SUBJ}_normal.json \
        --baseline results/${MN}_${SUBJ}_baseline.jsonl \
        --out results/scan/${MN}_${SUBJ}_scan.jsonl \
        --multipliers $MULTS

    touch "$DONE"
    echo "[done] $SUBJ"
)

echo "=== OLMoE FULL ($(date)) ==="
FAILED=()
for SUBJ in "${SUBJECTS[@]}"; do
    if run_domain "$SUBJ"; then echo "[ok] $SUBJ"; else echo "[FAIL] $SUBJ"; FAILED+=("$SUBJ"); fi
done

echo "=== ANALYSIS ==="
python scripts/aggregate_all.py --model_name $MN || true
python scripts/plot_figures.py --summary results/summary/${MN}_summary.json || true

echo "=== OLMoE FULL DONE ($(date)) ==="
[ ${#FAILED[@]} -gt 0 ] && echo ">> FAILED: ${FAILED[*]}"
