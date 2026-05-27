#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ "$OMP_NUM_THREADS" == "0" ]]; then
    export OMP_NUM_THREADS=1
fi
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CKPT="${CKPT:-llava_v15_7b}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llava_v15_7b}"
DATA_ROOT="${LLAVA_EVAL_DATA_ROOT:-/root/autodl-fs/llava_eval_imgs}"
RESULT_ROOT="${APET_RESULT_ROOT:-$REPO_ROOT/result/llava1_5}"
APET_LAYER_LIST="${APET_LAYER_LIST:-[16]}"
APET_IMAGE_TOKEN_LIST="${APET_IMAGE_TOKEN_LIST:-[64]}"
APET_VISUAL_TOKEN_NUM="${APET_VISUAL_TOKEN_NUM:-64}"
APET_BASIS_TOKEN_NUM="${APET_BASIS_TOKEN_NUM:-10}"
APET_LLM_PRUNING="${APET_LLM_PRUNING:-0}"
case "${APET_LLM_PRUNING,,}" in
    1|true|yes|y|on)
        LLM_PRUNING_ARGS=(--llm_pruning)
        ;;
    0|false|no|n|off)
        LLM_PRUNING_ARGS=(--no-llm_pruning)
        ;;
    *)
        echo "Invalid APET_LLM_PRUNING=$APET_LLM_PRUNING (expected 1/0, true/false, yes/no, on/off)" >&2
        exit 1
        ;;
esac

RESULT_DIR="$RESULT_ROOT/mmbench_cn"
mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/run.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

SPLIT="mmbench_dev_cn_20231003"
MMBDIR="$DATA_ROOT/mmbench"
ANSWER_DIR="$RESULT_DIR/answers/$SPLIT"
UPLOAD_DIR="$RESULT_DIR/answers_upload/$SPLIT"
mkdir -p "$ANSWER_DIR" "$UPLOAD_DIR"

python -m llava.eval.model_vqa_mmbench \
    --model-path "$MODEL_PATH" \
    --question-file "$MMBDIR/$SPLIT.tsv" \
    --answers-file "$ANSWER_DIR/$CKPT.jsonl" \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list "$APET_LAYER_LIST" \
    --image_token_list "$APET_IMAGE_TOKEN_LIST" \
    --visual_token_num "$APET_VISUAL_TOKEN_NUM" \
    --basis_token_num "$APET_BASIS_TOKEN_NUM" \
    "${LLM_PRUNING_ARGS[@]}" \
    --conv-mode vicuna_v1

python scripts/convert_mmbench_for_submission.py \
    --annotation-file "$MMBDIR/$SPLIT.tsv" \
    --result-dir "$ANSWER_DIR" \
    --upload-dir "$UPLOAD_DIR" \
    --experiment "$CKPT"

python scripts/eval_mmbench_local.py \
    --annotation-file "$MMBDIR/$SPLIT.tsv" \
    --result-file "$ANSWER_DIR/$CKPT.jsonl" \
    --output-file "$RESULT_DIR/eval_result.json"
