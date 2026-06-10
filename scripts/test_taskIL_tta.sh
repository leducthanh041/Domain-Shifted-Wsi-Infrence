#!/bin/bash
#
# Task-IL TTA v2 evaluation runner.

#SBATCH --job-name=test_taskIL_tta
#SBATCH --output=logs/test_taskIL_tta_%j.out
#SBATCH --error=logs/test_taskIL_tta_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mmlab_students/storageStudents/nguyenvd/Thanhld/WSI/MergeSlide_TTA_v1}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
LOG_DIR="${LOG_DIR:-logs}"
CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_ood_eval_num_workers0.yaml}"
CONFIG_REVERSE="${CONFIG_REVERSE:-configs/default_reverse_eval_num_workers0.yaml}"
CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
TASKIL_TTA_ENTRYPOINT="${TASKIL_TTA_ENTRYPOINT:-test_taskIL_tta.py}"

# ---------------------------------------------------------------------------
# TTA v1 hyperparameters
# ---------------------------------------------------------------------------
TTA_M="${TTA_M:-8}"
TTA_K_SUB="${TTA_K_SUB:-300}"
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"
TTA_BETA="${TTA_BETA:-1.0}"
TTA_LR="${TTA_LR:-1e-4}"
TTA_N_STEPS="${TTA_N_STEPS:-1}"
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"
TTA_EPISODIC="${TTA_EPISODIC:-0}"

# ---------------------------------------------------------------------------
# Python binary
# ---------------------------------------------------------------------------
if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints; do
    repo_path="$PROJECT_ROOT/$name"
    local_path="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$repo_path" ]; then
        :
    elif [ -e "$repo_path" ]; then
        echo "[WARN] $repo_path is not a symlink; hot writes should use $local_path"
    else
        ln -s "$local_path" "$repo_path"
    fi
done

mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

EPISODIC_FLAG=""
EPISODIC_LABEL="continual"
if [ "${TTA_EPISODIC}" = "1" ]; then
    EPISODIC_FLAG="--episodic"
    EPISODIC_LABEL="episodic"
fi

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] taskil_tta_entrypoint=$TASKIL_TTA_ENTRYPOINT"
echo "[INFO] === TTA v1 params ==="
echo "[INFO] M=$TTA_M | K_sub=$TTA_K_SUB | top_ratio=$TTA_TOP_RATIO"
echo "[INFO] beta=$TTA_BETA | lr=$TTA_LR | n_steps=$TTA_N_STEPS"
echo "[INFO] entropy_threshold=$TTA_ENTROPY_THRESHOLD | reset=$EPISODIC_LABEL"

check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    if [ -z "$resolved_log" ]; then return 0; fi
    local fd target pid state cmdline
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue
        pid="${fd#/proc/}"; pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue
        state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path is already held by PID $pid state=$state cmd=$cmdline" >&2
        echo "[ERROR] Refusing to reuse this log." >&2
        return 1
    done
}

run_to_logs() {
    local result_log="$1"
    local error_log="$2"
    shift 2
    echo "[INFO] running: $*"
    echo "[INFO] result_log=$result_log"
    echo "[INFO] error_log=$error_log"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$result_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$error_log"
    "$@" >> "$result_log" 2>> "$error_log"
}

# ---------------------------------------------------------------------------
# TTA base args
# ---------------------------------------------------------------------------
TTA_ARGS=(
    # v1 params
    --M                 "$TTA_M"
    --K_sub             "$TTA_K_SUB"
    --top_ratio         "$TTA_TOP_RATIO"
    --beta              "$TTA_BETA"
    --lr                "$TTA_LR"
    --n_steps           "$TTA_N_STEPS"
    --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --verbose_loss
)
if [ -n "$EPISODIC_FLAG" ]; then
    TTA_ARGS+=("$EPISODIC_FLAG")
fi

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=7

# 1. Forward
run_to_logs \
    "$LOG_DIR/result_taskil_tta.log" \
    "$LOG_DIR/error_taskil_tta.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint       "$TASKIL_TTA_ENTRYPOINT" \
        --config           "$CONFIG_FORWARD" \
        --save_dir         ./checkpoints/finetuned \
        --merge_model_path ./checkpoints/merged \
        "${TTA_ARGS[@]}"

# 2. Reverse
#run_to_logs \
#    "$LOG_DIR/result_taskil_tta_re.log" \
#    "$LOG_DIR/error_taskil_tta_re.log" \
#    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
#        --entrypoint       "$TASKIL_TTA_ENTRYPOINT" \
#        --config           "$CONFIG_REVERSE" \
#        --save_dir         ./checkpoints/finetuned_reverse \
#        --merge_model_path ./checkpoints/merged_reverse \
#        "${TTA_ARGS[@]}"

echo "[INFO] finished at $(date)"