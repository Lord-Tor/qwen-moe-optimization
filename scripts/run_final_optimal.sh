#!/bin/bash
#SBATCH --job-name=moe_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_final_%j.log

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results/final

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MN=$(basename $MODEL)

# Оптимальные множители из multiplier_scan (пик Δ на каждом домене):
declare -A BEST_MULT=(
    ["high_school_mathematics"]=5.0
    ["abstract_algebra"]=1.0
    ["formal_logic"]=5.0
    ["college_mathematics"]=1.0
)

for SUBJ in "${!BEST_MULT[@]}"; do
    MULT=${BEST_MULT[$SUBJ]}
    echo "######## FINAL: $SUBJ @ MULT=$MULT ########"
    BIAS="configs/${MN}_${SUBJ}_normal.json"
    if [ ! -f "$BIAS" ]; then
        echo "[warn] нет $BIAS — пропускаю"; continue
    fi
    # Полный per-question вывод на оптимальном множителе + random-control.
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file "$BIAS" \
        --output results/final/${MN}_${SUBJ}_opt.jsonl \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT --random_control \
        || echo "[FAIL] $SUBJ"
done

echo "=== STATISTICAL ANALYSIS ==="
python scripts/stat_analysis.py \
    --baseline_glob "results/${MN}_*_baseline.jsonl" \
    --biased_dir results/final \
    || true
echo "=== FINAL DONE ==="
