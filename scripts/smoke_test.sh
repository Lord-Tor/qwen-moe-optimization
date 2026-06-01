#!/bin/bash
#SBATCH --job-name=moe_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_smoke_%j.log

set -e
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results configs

echo "=== [1/2] SMOKE: gradient bias (limit=20) ==="
python scripts/build_gradient_bias.py \
    --subject college_mathematics \
    --limit 20 \
    --output configs/_smoke.json \
    --checkpoint_mode auto

echo "=== [2/2] SMOKE: biased inference + SELF-CHECK (limit=10) ==="
python scripts/qwen_mmlu_biased.py \
    --subject high_school_mathematics \
    --bias_file configs/_smoke.json \
    --bias_multiplier 20.0 \
    --limit 10

echo "=== SMOKE DONE ==="
echo ">> Ищи в логе строку '[self-check]'. OK = пайплайн чинён, можно ставить ночь."