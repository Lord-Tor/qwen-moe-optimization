#!/bin/bash
#SBATCH --job-name=moe_night
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=laplas,turing,mars,midas
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=58G
#SBATCH --time=13:30:00
#SBATCH --output=slurm_night_%j.log

# ВНИМАНИЕ: НЕТ глобального 'set -e'. Падение одного домена НЕ должно ронять
# остальные. Но ВНУТРИ домена — строгий fail-fast (см. run_domain), чтобы
# недосчитанные/битые файлы не выдавались за результат (приоритет: чистота).

export HF_HOME=/mnt/tank/scratch/$USER/.cache/huggingface
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH
# Всё в кэше -> работаем оффлайн, чтобы ночь не зависела от сети/рейтлимитов HF.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /nfs/home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moe_env
python -c "import torch; assert torch.cuda.is_available()" || { echo "CUDA НЕ доступна"; exit 1; }

mkdir -p results configs configs/raw logs

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
MN=$(basename $MODEL)
SUBJECTS=("high_school_mathematics" "abstract_algebra" "formal_logic" "college_mathematics")
GRAD_LIMIT=40
MULT=20.0
# Внутренний дедлайн: останавливаемся на границе домена, если до конца
# SLURM-лимита осталось меньше, чем нужно на один домен (оценка снизу).
# Это защищает от SLURM-kill ПОСРЕДИ прогона (который оставил бы битый файл).
HARD_LIMIT_SEC=$((13*3600 + 15*60))   # 13:15, на 15 мин раньше SLURM-лимита 13:30
RESERVE_PER_DOMAIN_SEC=$((90*60))      # ~1.5ч резерв на домен (консервативно)
START_TS=$(date +%s)

# Запускает ОДИН домен в субшелле с локальным fail-fast.
# Любая ошибка внутри -> домен помечается FAILED, .done-маркер НЕ ставится,
# выполнение переходит к следующему домену.
run_domain () (
    set -e                      # локальный fail-fast только внутри домена
    SUBJ="$1"
    DONE="results/${MN}_${SUBJ}.done"

    if [ -f "$DONE" ]; then
        echo "[skip] $SUBJ уже завершён ранее ($DONE) — пропускаю."
        exit 0
    fi
    echo "######## DOMAIN: $SUBJ ########"

    # 1. Baseline
    python scripts/qwen_mmlu_onepass.py \
        --model $MODEL --subject $SUBJ \
        --output results/${MN}_${SUBJ}_baseline.jsonl \
        --limit 10000 --experts_impl eager

    # 2. Сбор градиентов ОДИН раз -> .pt
    python scripts/build_gradient_bias.py \
        --model $MODEL --subject $SUBJ \
        --output configs/${MN}_${SUBJ}_grad.json \
        --save_raw_grads configs/raw/${MN}_${SUBJ}_grad.pt \
        --limit $GRAD_LIMIT --checkpoint_mode auto

    # 3. Оба конфига из одних градиентов
    python scripts/make_bias_from_grads.py \
        --raw configs/raw/${MN}_${SUBJ}_grad.pt \
        --out_normal  configs/${MN}_${SUBJ}_normal.json \
        --out_exclude configs/${MN}_${SUBJ}_exclude.json

    # 4a. Обычный bias + random-control
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file configs/${MN}_${SUBJ}_normal.json \
        --output results/${MN}_${SUBJ}_biased.jsonl \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT --random_control

    # 4b. exclude_dominant bias + собственный random-control
    #     (свой контроль: геометрия exclude отличается от normal)
    python scripts/qwen_mmlu_biased.py \
        --model $MODEL --subject $SUBJ \
        --bias_file configs/${MN}_${SUBJ}_exclude.json \
        --output results/${MN}_${SUBJ}_biased_excl.jsonl \
        --limit 10000 --experts_impl eager \
        --bias_multiplier $MULT --random_control

    touch "$DONE"               # маркер: домен полностью и чисто завершён
    echo "[done] $SUBJ -> $DONE"
)

echo "=== NIGHT BATCH: $MN ($(date)) ==="
FAILED=()
STOPPED_EARLY=0
for SUBJ in "${SUBJECTS[@]}"; do
    # Проверка внутреннего дедлайна: хватит ли времени на ещё один домен?
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    REMAIN=$((HARD_LIMIT_SEC - ELAPSED))
    if [ -f "results/${MN}_${SUBJ}.done" ]; then
        :  # уже готов, run_domain сам пропустит — дедлайн не тратим
    elif [ $REMAIN -lt $RESERVE_PER_DOMAIN_SEC ]; then
        echo "[stop] Осталось ${REMAIN}с (< резерва ${RESERVE_PER_DOMAIN_SEC}с на домен)."
        echo "[stop] Не начинаю $SUBJ, чтобы не получить битый файл от SLURM-kill."
        STOPPED_EARLY=1
        break
    fi

    if run_domain "$SUBJ"; then
        echo "[ok] $SUBJ"
    else
        echo "[FAIL] $SUBJ — домен пропущен, его частичные файлы НЕ войдут в анализ (len mismatch)."
        FAILED+=("$SUBJ")
    fi
done

echo "=== AGGREGATING ($(date)) ==="
# analyze_all сам отбрасывает домены с неполными файлами (len != baseline) -> чистота.
python scripts/analyze_all.py || echo "[warn] анализ не отработал, но сырые .jsonl на месте"

echo "=== NIGHT BATCH DONE ($(date)) ==="
if [ $STOPPED_EARLY -eq 1 ]; then
    echo ">> ОСТАНОВЛЕНО ПО ВНУТРЕННЕМУ ДЕДЛАЙНУ (не все домены посчитаны)."
    echo ">> Завершённые домены чисты. Перезапусти скрипт — досчитает остальные."
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ">> УПАВШИЕ ДОМЕНЫ: ${FAILED[*]}"
    echo ">> Перезапусти тот же скрипт — завершённые домены (.done) пропустятся,"
    echo ">> досчитаются только упавшие."
elif [ $STOPPED_EARLY -eq 0 ]; then
    echo ">> Все домены завершены чисто."
fi