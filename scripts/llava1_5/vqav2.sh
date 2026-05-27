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

RESULT_DIR="$RESULT_ROOT/vqav2"
mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/run.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"
VQADIR="$DATA_ROOT/vqav2"
ANSWER_DIR="$RESULT_DIR/answers/$SPLIT/$CKPT"
mkdir -p "$ANSWER_DIR"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --question-file "$VQADIR/$SPLIT.jsonl" \
        --image-folder "$VQADIR/test2015" \
        --answers-file "$ANSWER_DIR/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --layer_list "$APET_LAYER_LIST" \
        --image_token_list "$APET_IMAGE_TOKEN_LIST" \
        --visual_token_num "$APET_VISUAL_TOKEN_NUM" \
        --basis_token_num "$APET_BASIS_TOKEN_NUM" \
        "${LLM_PRUNING_ARGS[@]}" \
        --conv-mode vicuna_v1 &
done

for job in $(jobs -p); do
    wait "$job"
done

output_file="$ANSWER_DIR/merge.jsonl"
: > "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$ANSWER_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python scripts/convert_vqav2_for_submission_file.py \
    --src "$output_file" \
    --test-file "$VQADIR/llava_vqav2_mscoco_test2015.jsonl" \
    --dst "$RESULT_DIR/answers_upload/$CKPT.json"
