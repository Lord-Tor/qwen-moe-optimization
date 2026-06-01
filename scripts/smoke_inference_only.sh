#!/bin/bash
#SBATCH --job-name=moe_smoke_inf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_smoke_inf_%j.log

set -e
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

# Использует УЖЕ собранный configs/_smoke.json (сбор градиентов не повторяем).
echo "=== SMOKE inference + SELF-CHECK (limit=10) ==="
python scripts/qwen_mmlu_biased.py \
    --subject high_school_mathematics \
    --bias_file configs/_smoke.json \
    --bias_multiplier 20.0 \
    --limit 10

echo "=== DONE ==="
echo ">> Ищи строку '[self-check]'. OK = инжект меняет выход -> главный баг побеждён."