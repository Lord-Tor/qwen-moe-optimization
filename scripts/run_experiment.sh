#!/bin/bash
#SBATCH --job-name=qwen_moe            
#SBATCH --partition=gpu                
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas              
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=58G                      
#SBATCH --time=10:00:00                
#SBATCH --output=slurm_moe_%j.log      

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is NOT available!'"

mkdir -p results configs

echo "=== Starting Inference ==="
python scripts/qwen_mmlu_biased.py --subject high_school_mathematics --bias_file configs/math_grad_bias.json --output results/math_biased_cluster.jsonl --limit 270

echo "=== Done ==="