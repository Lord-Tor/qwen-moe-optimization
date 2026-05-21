#!/bin/bash
#SBATCH --job-name=qwen_grad            
#SBATCH --partition=gpu                
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas              
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=05:00:00                
#SBATCH --output=slurm_grad_%j.log      

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

mkdir -p configs

echo "=== GPU Information ==="
nvidia-smi

echo "=== Starting Gradient Collection ==="
python scripts/build_gradient_bias.py --subject high_school_mathematics --limit 100 --output configs/math_grad_bias.json

echo "=== Done ==="