#!/bin/bash
#SBATCH --job-name=olmoe_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_olmoe_smoke_%j.log

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
# НЕ оффлайн: OLMoE и его веса ещё НЕ в кэше, нужно скачать (первый запуск)
# export HF_HUB_OFFLINE=1   # включи после первого успешного скачивания

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

mkdir -p results configs configs/raw

MODEL="allenai/OLMoE-1B-7B-0924"
MN=$(basename $MODEL)

echo "=== [1/2] OLMoE: сбор градиентов (limit=20) — проверка захвата ==="
python scripts/build_gradient_bias.py \
    --model $MODEL --subject college_mathematics \
    --limit 20 \
    --output configs/${MN}_smoke.json \
    --save_raw_grads configs/raw/${MN}_smoke.pt \
    --checkpoint_mode auto || { echo "GRAD FAILED"; exit 1; }

echo "=== [2/2] OLMoE: biased inference + SELF-CHECK (limit=10) ==="
python scripts/qwen_mmlu_biased.py \
    --model $MODEL --subject high_school_mathematics \
    --bias_file configs/${MN}_smoke.json \
    --output results/${MN}_smoke_biased.jsonl \
    --limit 10 --bias_multiplier 5.0 || { echo "INFER/SELFCHECK FAILED"; exit 1; }

echo "=== OLMoE SMOKE OK ==="
echo ">> Захват градиентов и self-check прошли. Архитектура совместима."
