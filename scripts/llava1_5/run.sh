#!/bin/bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

SCRIPTS=(
    gqa.sh
    mme.sh
    mmbench.sh
    mmbench_cn.sh
    pope.sh
    sqa.sh
    textvqa.sh
    # vizwiz.sh
    vqav2.sh
)

script_dataset() {
    case "$1" in
        gqa.sh) echo "gqa" ;;
        mme.sh) echo "mme" ;;
        mmbench.sh) echo "mmbench" ;;
        mmbench_cn.sh) echo "mmbench_cn" ;;
        pope.sh) echo "pope" ;;
        sqa.sh) echo "scienceqa" ;;
        textvqa.sh) echo "textvqa" ;;
        vizwiz.sh) echo "vizwiz" ;;
        vqav2.sh) echo "vqav2" ;;
        *) echo "${1%.sh}" ;;
    esac
}

print_summary() {
    local log_file="$1"

    if [[ ! -f "$log_file" ]]; then
        echo "Summary: log file not found: $log_file"
        return
    fi

    local summary
    summary=$(grep -E "^(Total:|Correct:|Accuracy:|IMG-Accuracy:|Samples:|Category:|Precision:|Recall:|F1 score:|Yes ratio:|total results:|category[[:space:]]|perception_score[[:space:]]|cognition_score[[:space:]]|total_score[[:space:]]|mme_score[[:space:]]|pope_macro_f1[[:space:]]|pope_macro_accuracy[[:space:]]|pope_score[[:space:]]|[A-Za-z0-9_/-]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\\.[0-9]+|[A-Za-z0-9_/-]+: [0-9]+/[0-9]+|[0-9]+\\.[0-9]+,)" "$log_file" | tail -n 60 || true)
    if [[ -n "$summary" ]]; then
        echo "Summary from $log_file:"
        echo "$summary"
    else
        echo "Summary: no metric lines found in $log_file"
    fi
}

RESULT_ROOT="${APET_RESULT_ROOT:-$REPO_ROOT/result/llava1_5}"
PIPELINE_LOG_DIR="$RESULT_ROOT/_pipeline_logs"
mkdir -p "$PIPELINE_LOG_DIR"

echo "[$(date '+%F %T')] Start LLaVA-1.5 ApET eval pipeline"
echo "Repo root: $REPO_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "LLAVA_EVAL_DATA_ROOT=${LLAVA_EVAL_DATA_ROOT:-/root/autodl-fs/llava_eval_imgs}"
echo "APET_RESULT_ROOT=$RESULT_ROOT"
echo "Full child stdout/stderr mirror: $PIPELINE_LOG_DIR"
echo

for script in "${SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    if [[ ! -f "$script_path" ]]; then
        echo "[$(date '+%F %T')] Missing script: $script_path" >&2
        exit 1
    fi

    dataset=$(script_dataset "$script")
    child_log="$PIPELINE_LOG_DIR/${script%.sh}.log"
    dataset_log="$RESULT_ROOT/$dataset/run.log"

    echo "[$(date '+%F %T')] >>> Running $script"
    echo "Full log: $dataset_log"
    start_ts=$(date +%s)
    bash "$script_path" > "$child_log" 2>&1
    status=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [[ $status -ne 0 ]]; then
        echo "[$(date '+%F %T')] !!! $script failed with exit code $status after ${elapsed}s" >&2
        echo "Captured child output: $child_log" >&2
        tail -n 80 "$child_log" >&2 || true
        if [[ "${RUN_CONTINUE_ON_ERROR:-0}" != "1" ]]; then
            exit "$status"
        fi
    else
        echo "[$(date '+%F %T')] <<< Finished $script in ${elapsed}s"
        print_summary "$dataset_log"
    fi
    echo
done

echo "[$(date '+%F %T')] All LLaVA-1.5 ApET eval scripts finished"
