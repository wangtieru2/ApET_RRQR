#!/bin/bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/root/autodl-fs/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/ApET/bin/python}"
CKPT="${CKPT:-llava_v15_7b}"
RESULT_ROOT="${APET_RESULT_ROOT:-$REPO_ROOT/result/lmms_eval}"
RUN_ROOT="${LMMS_EVAL_RUN_ROOT:-$RESULT_ROOT/$CKPT}"
DRY_RUN="${DRY_RUN:-0}"

SELECTIONS_STR="${APET_SELECTIONS:-apet_error rrqr}"
TASKS_STR="${LMMS_EVAL_TASKS:-gqa mmbench_en_dev mmbench_cn_dev mme pope scienceqa_img vqav2_val textvqa_val vizwiz_vqa_val}"

read -r -a SELECTIONS <<< "$SELECTIONS_STR"
read -r -a TASKS <<< "$TASKS_STR"

mkdir -p "$RUN_ROOT"

echo "Running selection-parallel 9-benchmark suite"
echo "Repo root: $REPO_ROOT"
echo "Selections (parallel): ${SELECTIONS[*]}"
echo "Tasks per selection (serial): ${TASKS[*]}"
echo "Run root: $RUN_ROOT"
echo "Dry run: $DRY_RUN"
echo

summarize_result() {
    local task="$1"
    local output_path="$2"
    local status="$3"
    local rc="$4"
    "$PYTHON_BIN" - "$task" "$output_path" "$status" "$rc" <<'PY_SUMMARY'
import glob
import json
import os
import sys

task, output_path, status, rc = sys.argv[1:5]
if status != "OK":
    print(f"{task}\tFAILED\trc={rc}")
    raise SystemExit(0)

candidates = sorted(glob.glob(os.path.join(output_path, "**", "*_results.json"), recursive=True), key=os.path.getmtime)
if not candidates:
    print(f"{task}\tOK\tNO_RESULTS_JSON")
    raise SystemExit(0)
path = candidates[-1]
with open(path, "r") as f:
    data = json.load(f)
results = data.get("results", {})
flat = []
for task_name, metrics in results.items():
    if not isinstance(metrics, dict):
        continue
    for key, value in metrics.items():
        if key.endswith("_stderr") or key == "alias":
            continue
        if isinstance(value, (int, float)) or value is None:
            flat.append(f"{task_name}.{key}={value}")
if not flat:
    flat.append("NO_NUMERIC_METRICS")
print(f"{task}\tOK\t" + "; ".join(flat))
PY_SUMMARY
}

run_selection() {
    local selection="$1"
    local selection_dir="$RUN_ROOT/$selection"
    mkdir -p "$selection_dir"
    local all_res="$selection_dir/all_res.log"
    local run_log="$selection_dir/run.log"
    : > "$all_res"
    : > "$run_log"
    echo "===== selection=$selection =====" | tee -a "$run_log"

    for task in "${TASKS[@]}"; do
        local task_dir="$selection_dir/$task"
        mkdir -p "$task_dir"
        echo "[$selection][$task] start" | tee -a "$run_log"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "DRY_RUN APET_SELECTION=$selection LMMS_EVAL_TASKS=$task LMMS_EVAL_OUTPUT_PATH=$task_dir" | tee -a "$run_log"
            printf "%s\tDRY_RUN\t%s\n" "$task" "$task_dir" >> "$all_res"
            continue
        fi

        APET_SELECTION="$selection" \
        LMMS_EVAL_TASKS="$task" \
        LMMS_EVAL_OUTPUT_PATH="$task_dir" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash scripts/lmms_eval/run_llava1_5_apet_official.sh >> "$run_log" 2>&1
        local rc=$?

        if [[ $rc -eq 0 ]]; then
            summarize_result "$task" "$task_dir" "OK" "$rc" | tee -a "$all_res" "$run_log"
            echo "[$selection][$task] done" | tee -a "$run_log"
        else
            summarize_result "$task" "$task_dir" "FAILED" "$rc" | tee -a "$all_res" "$run_log"
            echo "[$selection][$task] failed with rc=$rc; continue" | tee -a "$run_log"
        fi
        echo | tee -a "$run_log" >/dev/null
    done
}

pids=()
for selection in "${SELECTIONS[@]}"; do
    run_selection "$selection" &
    pid="$!"
    pids+=("$pid")
    echo "Started selection=$selection pid=$pid"
done

overall_rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_rc=1
    fi
done

echo "Selection-parallel suite finished. Summary logs:"
for selection in "${SELECTIONS[@]}"; do
    echo "  $RUN_ROOT/$selection/all_res.log"
done

exit "$overall_rc"
