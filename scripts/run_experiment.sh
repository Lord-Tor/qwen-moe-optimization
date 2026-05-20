#!/bin/bash
#SBATCH --job-name=qwen_moe            
#SBATCH --partition=gpu                
#SBATCH --nodelist=nike                
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=10:00:00                
#SBATCH --output=results/slurm_moe_%j.log

# 1. Явные переменные окружения для вычислительного узла
export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Активация окружения Conda
source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

echo "=== GPU Information ==="
nvidia-smi

echo "=== Starting Inference ==="
python scripts/qwen_mmlu_biased.py \
  --subject high_school_mathematics \
  --bias_file configs/math_full_bias.json \
  --output results/math_biased_cluster.jsonl \
  --limit 270

echo "=== Done ==="