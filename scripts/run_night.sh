#!/bin/bash
#SBATCH --job-name=moe_night
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=13:30:00
#SBATCH --output=slurm_night_%j.log

set -e
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || exit 1

mkdir -p results configs configs/raw

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MN=$(basename $MODEL)
SUBJECTS=("high_school_mathematics" "abstract_algebra" "formal_logic" "college_mathematics")
GRAD_LIMIT=60
MULT=20.0

echo "=== NIGHT BATCH: $MN ==="
for SUBJ in "${SUBJECTS[@]}"; do
    echo "######## DOMAIN: $SUBJ ########"

    # 1. Baseline (новый код, чистая токенизация)
    python scripts/qwen_mmlu_onepass.py \
        --model $MODEL --subject $SUBJ \
        --output results/${MN}_${SUBJ}_baseline.jsonl \
        --limit 10000 --experts_impl eager

    # 2. Сбор градиентов ОДИН раз -> сырые .pt (служит обоим вариантам bias)
    python scripts/build_gradient_bias.py \
        --model $MODEL --subject $SUBJ \
        --output configs/${MN}_${SUBJ}_grad.json \
        --save_raw_grads configs/raw/${MN}_${SUBJ}_grad.pt \
        --limit $GRAD_LIMIT --checkpoint_mode auto

    # 3. Из одних градиентов делаем обычный И exclude_dominant конфиг (дёшево, без модели)
    python scripts/make_bias_from_grads.py \
        --raw configs/raw/${MN}_${SUBJ}_grad.pt \
        --out_normal  configs/${MN}_${SUBJ}_normal.json \
        --out_exclude configs/${MN}_${SUBJ}_exclude.json

    # 4a. Обычный bias + random-control (--random_control пишет *_randomctrl.jsonl)
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file configs/${MN}_${SUBJ}_normal.json \
        --output results/${MN}_${SUBJ}_biased.jsonl \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT --random_control

    # 4b. exclude_dominant bias (random-control общий, повторно не нужен)
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file configs/${MN}_${SUBJ}_exclude.json \
        --output results/${MN}_${SUBJ}_biased_excl.jsonl \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT
done

echo "=== AGGREGATING ==="
python scripts/analyze_all.py || true

echo "=== NIGHT BATCH DONE ==="