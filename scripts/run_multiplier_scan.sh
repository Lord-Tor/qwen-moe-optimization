#!/bin/bash
#SBATCH --job-name=moe_mscan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_mscan_%j.log

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results/scan

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MN=$(basename $MODEL)
SUBJECTS=("high_school_mathematics" "abstract_algebra" "formal_logic" "college_mathematics")
MULTS="0.5 1 2 5 10 20 50"

for SUBJ in "${SUBJECTS[@]}"; do
    echo "######## SCAN: $SUBJ ########"
    BIAS="configs/${MN}_${SUBJ}_normal.json"
    BASE="results/${MN}_${SUBJ}_baseline.jsonl"
    OUT="results/scan/${MN}_${SUBJ}_scan.jsonl"
    if [ ! -f "$BIAS" ] || [ ! -f "$BASE" ]; then
        echo "[warn] нет $BIAS или $BASE — пропускаю $SUBJ"
        continue
    fi
    python scripts/multiplier_scan.py \
        --model $MODEL --subject $SUBJ \
        --bias_file "$BIAS" --baseline "$BASE" --out "$OUT" \
        --multipliers $MULTS \
        || echo "[FAIL] scan $SUBJ"
done

echo "=== SCAN ANALYSIS ==="
python scripts/analyze_scan.py || true
echo "=== SCAN DONE ==="
