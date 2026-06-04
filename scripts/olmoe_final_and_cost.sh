#!/bin/bash
#SBATCH --job-name=olmoe_par
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_olmoe_par_%j.log

# НЕ полагаемся на set -e (ловушка в if). Каждый шаг проверяется явно.
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results results/final results/scan results/summary configs configs/raw

MODEL="allenai/OLMoE-1B-7B-0924"
MN=$(basename $MODEL)

# Оптимальные множители OLMoE (из скана full)
declare -A BEST_MULT=(
    ["abstract_algebra"]=2.0
    ["college_mathematics"]=1.0
    ["formal_logic"]=2.0
    ["high_school_mathematics"]=2.0
)

############## ЧАСТЬ 1: ФИНАЛ НА ОПТИМАЛЬНОМ MULT + RANDOM-CONTROL ##############
echo "=== PART 1: FINAL @ optimal MULT ($(date)) ==="
for SUBJ in "${!BEST_MULT[@]}"; do
    MULT=${BEST_MULT[$SUBJ]}
    OUT="results/final/${MN}_${SUBJ}_opt.jsonl"
    BIAS="configs/${MN}_${SUBJ}_normal.json"
    if [ -f "results/final/${MN}_${SUBJ}.optdone" ]; then echo "[skip] final $SUBJ"; continue; fi
    if [ ! -f "$BIAS" ]; then echo "[warn] нет $BIAS — пропуск $SUBJ"; continue; fi
    echo "--- FINAL $SUBJ @ MULT=$MULT ---"
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file "$BIAS" --output "$OUT" \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT --random_control \
        && touch "results/final/${MN}_${SUBJ}.optdone" \
        || echo "[FAIL] final $SUBJ"
done

echo "=== СТАТИСТИКА OLMoE ==="
python scripts/stat_analysis.py \
    --baseline_glob "results/${MN}_*_baseline.jsonl" \
    --biased_dir results/final || echo "[warn] stat не отработал"

############## ЧАСТЬ 2: ЦЕНА СПЕЦИАЛИЗАЦИИ ##############
echo "=== PART 2: COST OF SPECIALIZATION ($(date)) ==="
MATH_CFG="configs/${MN}_MATHALL_normal.json"
if [ ! -f "$MATH_CFG" ]; then
    python scripts/merge_grads.py \
        --raws configs/raw/${MN}_high_school_mathematics_grad.pt \
               configs/raw/${MN}_abstract_algebra_grad.pt \
               configs/raw/${MN}_formal_logic_grad.pt \
               configs/raw/${MN}_college_mathematics_grad.pt \
        --out_raw configs/raw/${MN}_MATHALL_grad.pt \
        --out_config "$MATH_CFG" --domain_label math_all \
        || { echo "[FAIL] merge"; }
fi

NONMATH=("professional_law" "high_school_world_history" "college_biology" "moral_scenarios")
MULTS="0.5 1 2 5 10 20 50"
for SUBJ in "${NONMATH[@]}"; do
    BASE="results/${MN}_${SUBJ}_baseline.jsonl"
    if [ -f "results/scan/${MN}_${SUBJ}_costscan.jsonl" ]; then echo "[skip] cost $SUBJ"; continue; fi
    if [ ! -f "$BASE" ]; then
        python scripts/qwen_mmlu_onepass.py \
            --model $MODEL --subject $SUBJ \
            --output "$BASE" --limit 10000 --experts_impl eager \
            || { echo "[FAIL] baseline $SUBJ"; continue; }
    fi
    python scripts/multiplier_scan.py \
        --model $MODEL --subject $SUBJ \
        --bias_file "$MATH_CFG" --baseline "$BASE" \
        --out results/scan/${MN}_${SUBJ}_costscan.jsonl \
        --multipliers $MULTS --experts_impl eager \
        || echo "[FAIL] costscan $SUBJ"
done

############## АНАЛИЗ ##############
echo "=== AGGREGATE + FIGURES + COST ($(date)) ==="
python scripts/aggregate_all.py --model_name $MN || true
python scripts/plot_figures.py --summary results/summary/${MN}_summary.json || true
python scripts/analyze_cost.py --model_name $MN || true

echo "=== OLMoE PARITY DONE ($(date)) ==="
