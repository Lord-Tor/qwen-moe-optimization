#!/bin/bash
#SBATCH --job-name=moe_massive            
#SBATCH --partition=gpu                
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas              
#SBATCH --gres=gpu:2                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=58G                      
#SBATCH --time=14:00:00                
#SBATCH --output=slurm_massive_%j.log      

set -e

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is NOT available!'" || exit 1

mkdir -p results configs

SUBJECTS=("high_school_mathematics" "abstract_algebra" "elementary_mathematics" "formal_logic" "college_mathematics")
MODELS=("Qwen/Qwen1.5-MoE-A2.7B" "Qwen/Qwen1.5-MoE-A2.7B-Chat")

echo "=== STARTING MASSIVE NIGHT BATCH ==="

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename $MODEL)
    echo "=========================================="
    echo "🚀 TESTING MODEL: $MODEL_NAME"
    echo "=========================================="

    for SUBJ in "${SUBJECTS[@]}"; do
        echo "--- Processing: $SUBJ on $MODEL_NAME ---"
        python scripts/qwen_mmlu_onepass.py --model $MODEL --subject $SUBJ --output results/${MODEL_NAME}_${SUBJ}_baseline.jsonl --limit 10000 --experts_impl eager
        python scripts/build_gradient_bias.py --model $MODEL --subject $SUBJ --output configs/${MODEL_NAME}_${SUBJ}_grad.json --limit 100 --experts_impl eager
        python scripts/qwen_mmlu_biased.py --model $MODEL --subject $SUBJ --bias_file configs/${MODEL_NAME}_${SUBJ}_grad.json --output results/${MODEL_NAME}_${SUBJ}_biased.jsonl --limit 10000 --experts_impl eager
    done
done

echo "📊 AGGREGATING RESULTS..."
python scripts/aggregate_results.py

echo "=== NIGHT BATCH COMPLETED ==="