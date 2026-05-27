#!/bin/bash
set -euo pipefail

export PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/ApET/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llava_v15_7b}"
export CKPT="${CKPT:-llava_v15_7b}"
export APET_SELECTIONS="${APET_SELECTIONS:-apet_error rrqr}"
# export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mme gqa mmbench_en_dev mmbench_cn_dev pope scienceqa_img textvqa_val vizwiz_vqa_val vqav2_val}"

export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mmbench_cn_dev pope scienceqa_img textvqa_val vizwiz_vqa_val vqav2_val}"


bash scripts/lmms_eval/run_9bench_serial.sh
