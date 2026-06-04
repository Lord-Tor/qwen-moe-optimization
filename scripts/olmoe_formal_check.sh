#!/bin/bash
#SBATCH --job-name=olmoe_flcheck
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_olmoe_flcheck_%j.log

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

MODEL="allenai/OLMoE-1B-7B-0924"
MN=$(basename $MODEL)

echo "=== ПРОВЕРКА ФИКСА ПАМЯТИ на formal_logic (тот домен, что падал) ==="
echo "Если max_memory-фикс сработал — пройдёт без CUDA OOM на длинных промптах."
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

python scripts/qwen_mmlu_onepass.py \
    --model $MODEL --subject formal_logic \
    --output results/${MN}_formal_logic_baseline.jsonl \
    --limit 10000

echo "=== Если выше 'Done. Accuracy' и нет OOM — фикс работает, можно полный olmoe_full ==="
