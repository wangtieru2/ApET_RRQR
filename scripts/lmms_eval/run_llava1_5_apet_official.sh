#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MMBENCH_EVAL_METHOD="${MMBENCH_EVAL_METHOD:-static}"
export HF_HOME="${HF_HOME:-/root/autodl-fs/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
CKPT="${CKPT:-llava_v15_7b}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llava_v15_7b}"
RESULT_ROOT="${APET_RESULT_ROOT:-$REPO_ROOT/result/lmms_eval}"

# Task names can vary across lmms-eval versions. Confirm with:
#   bash scripts/lmms_eval/list_tasks.sh
TASKS="${LMMS_EVAL_TASKS:-mmbench_en_dev,mmbench_cn_dev,textvqa_val}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"

APET_LAYER_LIST="${APET_LAYER_LIST:-[16]}"
APET_IMAGE_TOKEN_LIST="${APET_IMAGE_TOKEN_LIST:-[64]}"
APET_VISUAL_TOKEN_NUM="${APET_VISUAL_TOKEN_NUM:-192}"  ###
APET_BASIS_TOKEN_NUM="${APET_BASIS_TOKEN_NUM:-10}"
APET_SELECTION="${APET_SELECTION:-apet_error}"
OUTPUT_PATH="${LMMS_EVAL_OUTPUT_PATH:-$RESULT_ROOT/$CKPT/$APET_SELECTION}"
APET_LLM_PRUNING="${APET_LLM_PRUNING:-1}"

case "${APET_LLM_PRUNING,,}" in
    1|true|yes|y|on) LLM_PRUNING_VALUE=true ;;
    0|false|no|n|off) LLM_PRUNING_VALUE=false ;;
    *)
        echo "Invalid APET_LLM_PRUNING=$APET_LLM_PRUNING (expected 1/0, true/false, yes/no, on/off)" >&2
        exit 2
        ;;
esac

if ! "$PYTHON_BIN" -c "import lmms_eval" >/dev/null 2>&1; then
    echo "lmms_eval is not importable by PYTHON_BIN=$PYTHON_BIN." >&2
    echo "Install lmms-eval first. Then install the adapter with:" >&2
    echo "  LMMS_EVAL_REPO=/path/to/lmms-eval bash scripts/lmms_eval/install_apet_llava_adapter.sh" >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c "from lmms_eval import models; assert 'apet_llava' in models.list_available_models(include_aliases=True); models.get_model('apet_llava')" >/dev/null 2>&1; then
    echo "apet_llava is not registered in lmms-eval's model registry." >&2
    echo "Install or refresh the adapter with:" >&2
    echo "  LMMS_EVAL_REPO=/path/to/lmms-eval bash scripts/lmms_eval/install_apet_llava_adapter.sh" >&2
    echo "Also ensure this repo is on PYTHONPATH:" >&2
    echo "  export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH" >&2
    exit 2
fi

mkdir -p "$OUTPUT_PATH"

MODEL_ARGS="pretrained=$MODEL_PATH,conv_template=vicuna_v1,model_name=$CKPT,layer_list=$APET_LAYER_LIST,image_token_list=$APET_IMAGE_TOKEN_LIST,visual_token_num=$APET_VISUAL_TOKEN_NUM,basis_token_num=$APET_BASIS_TOKEN_NUM,selection_method=$APET_SELECTION,llm_pruning=$LLM_PRUNING_VALUE"

CMD=(
    "$PYTHON_BIN" -m accelerate.commands.launch
    --num_processes "$NUM_PROCESSES"
    -m lmms_eval
    --model apet_llava
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --log_samples
    --log_samples_suffix "$CKPT"
    --output_path "$OUTPUT_PATH"
)

if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
fi

echo "Running lmms-eval official-style evaluation"
echo "Repo root: $REPO_ROOT"
echo "Tasks: $TASKS"
echo "Output path: $OUTPUT_PATH"
echo "Model args: $MODEL_ARGS"
echo
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
