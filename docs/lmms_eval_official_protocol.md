# Official / lmms-eval Evaluation Path

This repo keeps the existing local evaluators for fast diagnostics, but paper-comparable MMBench/MMBench-CN/TextVQA should be evaluated through an official-style benchmark stack such as lmms-eval.

## Added files

- `lmms_eval_adapters/apet_llava.py`: lmms-eval model adapter registered as `apet_llava`. It loads this repo's LLaVA/ApET model and applies the existing ApET pruning configuration.
- `scripts/lmms_eval/install_apet_llava_adapter.sh`: copies the adapter into a local lmms-eval checkout.
- `scripts/lmms_eval/list_tasks.sh`: lists task names supported by the installed lmms-eval version.
- `scripts/lmms_eval/run_llava1_5_apet_official.sh`: launches lmms-eval for MMBench/MMBench-CN/TextVQA by default.
- `scripts/check_textvqa_prompt_protocol.py`: checks whether TextVQA predictions were generated with OCR-token prompts when the question file expects OCR tokens.

## One-time setup

```bash
# Example only; use your actual checkout path.
export LMMS_EVAL_REPO=/root/autodl-tmp/lmms-eval
export PYTHONPATH=/root/autodl-tmp/mllm_token_compression/main_exp/ApET_baseline_no_llm_pruning:$PYTHONPATH

bash scripts/lmms_eval/install_apet_llava_adapter.sh
```

For recent lmms-eval versions, this script also registers a registry_v2 `ModelManifest`; copying `apet_llava.py` alone is not enough. Re-run the installer whenever lmms-eval is reinstalled or updated.

If lmms-eval is not installed yet, install it in a separate environment or checkout following the official lmms-eval documentation. This repo does not vendor lmms-eval.

## Confirm task names

Task names vary across lmms-eval versions. Check your installed version first:

```bash
PYTHON_BIN=/path/to/python bash scripts/lmms_eval/list_tasks.sh
```

Likely task names are one of these families:

- MMBench EN: `mmbench_en_dev` or `mmbench_en`
- MMBench CN: `mmbench_cn_dev` or `mmbench_cn`
- TextVQA: `textvqa_val` or `textvqa`

Override the defaults with `LMMS_EVAL_TASKS`.

## Run a smoke test

This does not run a full benchmark:

```bash
PYTHON_BIN=/path/to/python \
LIMIT=8 \
LMMS_EVAL_TASKS=mmbench_en_dev,textvqa_val \
bash scripts/lmms_eval/run_llava1_5_apet_official.sh
```

## Run the official-style evaluation

```bash
PYTHON_BIN=/path/to/python \
NUM_PROCESSES=1 \
MODEL_PATH=/root/autodl-tmp/models/llava_v15_7b \
CKPT=llava_v15_7b \
APET_LAYER_LIST='[16]' \
APET_IMAGE_TOKEN_LIST='[64]' \
APET_VISUAL_TOKEN_NUM=64 \
APET_BASIS_TOKEN_NUM=10 \
APET_LLM_PRUNING=0 \
LMMS_EVAL_TASKS=mmbench_en_dev,mmbench_cn_dev,textvqa_val \
bash scripts/lmms_eval/run_llava1_5_apet_official.sh
```

Results are written by lmms-eval under:

```text
result/lmms_eval/<CKPT>/
```

## Metric labeling

Use lmms-eval outputs for paper-comparable fields:

- `MMB Acc.`
- `MMBCN Acc.`
- `TextVQA Acc.`

Use existing local scripts only for diagnostic fields:

- `MMB-local Dev Acc.`
- `MMBCN-local Dev Acc.`
- `TextVQA-local Acc.`

## Notes

- The adapter is evaluation-only and does not modify model forward or token-compression logic.
- The runner defaults to offline Hugging Face mode, matching the local scripts. Set `TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0` if your lmms-eval task needs to download datasets.
- The default task list is conservative but version-dependent. Always verify with `list_tasks.sh`.
