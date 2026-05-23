#!/bin/bash
#SBATCH --job-name=moe_search            
#SBATCH --partition=gpu                
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas              
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=58G                      
#SBATCH --time=05:00:00                
#SBATCH --output=slurm_search_%j.log      

set -e

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is NOT available!'" || exit 1

mkdir -p results configs
MODEL="Qwen/Qwen1.5-MoE-A2.7B"
SUBJ="high_school_mathematics"
MODEL_NAME=$(basename $MODEL)

echo "=== STARTING MULTIPLIER SEARCH ==="

MULTIPLIERS=(1.0 5.0 10.0 20.0 50.0 100.0)

for MULT in "${MULTIPLIERS[@]}"; do
    echo "--- Testing Multiplier: $MULT ---"
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL \
        --subject $SUBJ \
        --bias_file configs/${MODEL_NAME}_${SUBJ}_grad.json \
        --output results/${MODEL_NAME}_${SUBJ}_biased_mult_${MULT}.jsonl \
        --limit 270 \
        --experts_impl eager \
        --bias_multiplier $MULT
done

echo "=== SEARCH COMPLETED ==="