#!/bin/bash
#SBATCH --job-name=qwen_moe            
#SBATCH --partition=gpu                
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas              
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=64G                      
#SBATCH --time=10:00:00                
#SBATCH --output=slurm_moe_%j.log      

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
# Проверка, что PyTorch видит GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is NOT available! PyTorch failed to link with NVIDIA drivers.'"
# Страховка: создаем нужные папки до запуска Python
mkdir -p results configs

echo "=== GPU Information ==="
nvidia-smi

echo "=== Starting Inference ==="
python scripts/qwen_mmlu_biased.py \
  --subject high_school_mathematics \
  --bias_file configs/math_full_bias.json \
  --output results/math_biased_cluster.jsonl \
  --limit 270

echo "=== Done ==="