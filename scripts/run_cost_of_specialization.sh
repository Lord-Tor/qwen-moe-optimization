#!/bin/bash
#SBATCH --job-name=moe_cost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_cost_%j.log

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results results/scan configs configs/raw

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MN=$(basename $MODEL)
MATH_CFG="configs/${MN}_MATHALL_normal.json"

# non-math домены + 1 нейтральный (moral_scenarios)
NONMATH=("professional_law" "high_school_world_history" "college_biology" "moral_scenarios")
MULTS="0.5 1 2 5 10 20 50"

# 0. Объединённый math-конфиг (если ещё не создан) — на CPU, мгновенно
if [ ! -f "$MATH_CFG" ]; then
    echo "[merge] строю общий math-конфиг из 4 доменов"
    python scripts/merge_grads.py \
        --raws configs/raw/${MN}_high_school_mathematics_grad.pt \
               configs/raw/${MN}_abstract_algebra_grad.pt \
               configs/raw/${MN}_formal_logic_grad.pt \
               configs/raw/${MN}_college_mathematics_grad.pt \
        --out_raw configs/raw/${MN}_MATHALL_grad.pt \
        --out_config "$MATH_CFG" \
        --domain_label math_all || { echo "merge FAILED"; exit 1; }
fi

for SUBJ in "${NONMATH[@]}"; do
    echo "######## COST: $SUBJ ########"
    BASE="results/${MN}_${SUBJ}_baseline.jsonl"
    # 1. Baseline non-math домена (если нет)
    if [ ! -f "$BASE" ]; then
        python scripts/qwen_mmlu_onepass.py \
            --model $MODEL --subject $SUBJ \
            --output "$BASE" --limit 10000 --experts_impl eager \
            || { echo "[FAIL] baseline $SUBJ"; continue; }
    fi
    # 2. Скан множителя с ОБЩИМ math-bias на этом non-math домене
    python scripts/multiplier_scan.py \
        --model $MODEL --subject $SUBJ \
        --bias_file "$MATH_CFG" --baseline "$BASE" \
        --out results/scan/${MN}_${SUBJ}_costscan.jsonl \
        --multipliers $MULTS \
        || echo "[FAIL] costscan $SUBJ"
done

echo "=== COST ANALYSIS ==="
python scripts/analyze_cost.py || true
echo "=== COST DONE ==="
