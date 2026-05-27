# Local Evaluation Changes

This repo was adapted to run LLaVA-1.5 ApET evaluation from local parquet datasets.

## Data

- Source parquet root: `/root/autodl-fs/imgs`
- Prepared LLaVA/ApET eval root: `/root/autodl-fs/llava_eval_imgs`
- Prepared count record: `/root/autodl-fs/llava_eval_imgs/prepared_counts.json`

The prepared data covers:

- GQA testdev balanced
- MME
- MMBench EN dev
- MMBench CN dev
- POPE
- ScienceQA image test
- TextVQA validation
- VizWiz validation and test
- VQAv2 testdev images plus full test2015 question order file for submission conversion

## Official / lmms-eval Path

- `lmms_eval_adapters/apet_llava.py`
  - Optional lmms-eval adapter registered as `apet_llava` for official-style MMBench/MMBench-CN/TextVQA evaluation.
- `scripts/lmms_eval/install_apet_llava_adapter.sh`
  - Copies the adapter into a local lmms-eval checkout.
- `scripts/lmms_eval/run_llava1_5_apet_official.sh`
  - Launches lmms-eval with ApET/LLaVA model args. Defaults to `mmbench_en_dev,mmbench_cn_dev,textvqa_val`; override with `LMMS_EVAL_TASKS`.
- `scripts/lmms_eval/list_tasks.sh`
  - Lists task names for the installed lmms-eval version.
- `docs/lmms_eval_official_protocol.md`
  - Usage notes for paper-comparable official/lmms-eval protocol.

## Added Scripts

- `scripts/prepare_llava_eval_from_parquet.py`
  - Converts local HuggingFace parquet datasets into the layouts expected by ApET/LLaVA eval scripts.
  - Uses parallel VQAv2 testdev image extraction via `--vqav2-workers`.
- `scripts/eval_mme_local.py`
  - Scores local MME yes/no predictions by category and overall accuracy.
- `scripts/eval_mmbench_local.py`
  - Scores local MMBench dev predictions against answer labels.
- `scripts/eval_vqa_answers.py`
  - Scores VQA-style multi-answer annotations, used for VizWiz val.
- `scripts/check_textvqa_prompt_protocol.py`
  - Fails fast when an OCR TextVQA question file is evaluated against non-OCR prediction prompts.
- `scripts/convert_vqav2_for_submission_file.py`
  - Converts a merged VQAv2 answer jsonl to an EvalAI-style JSON while allowing result files to live under `result/`.

## Updated Scripts

The following `scripts/llava1_5/*.sh` scripts now default to:

- Model path: `/root/autodl-tmp/models/llava_v15_7b`
- Data root: `/root/autodl-fs/llava_eval_imgs`
- Result root: `<repo>/result/llava1_5`
- Offline Transformers/HF mode enabled
- `OMP_NUM_THREADS=1` when unset or invalid
- Per-dataset terminal logs written to `<repo>/result/llava1_5/<dataset>/run.log`

Updated files:

- `scripts/llava1_5/gqa.sh`
- `scripts/llava1_5/mme.sh`
- `scripts/llava1_5/mmbench.sh`
- `scripts/llava1_5/mmbench_cn.sh`
- `scripts/llava1_5/pope.sh`
- `scripts/llava1_5/sqa.sh`
- `scripts/llava1_5/textvqa.sh`
- `scripts/llava1_5/vizwiz.sh`
- `scripts/llava1_5/vqav2.sh`

## Runtime Overrides

These environment variables can override defaults:

- `MODEL_PATH`
- `CKPT`
- `LLAVA_EVAL_DATA_ROOT`
- `APET_RESULT_ROOT`
- `CUDA_VISIBLE_DEVICES`
- `VIZWIZ_SPLIT` (`val` by default; set `test` for VizWiz test submission output)

Example:

```bash
cd /root/autodl-tmp/mllm_token_compression/main_exp/ApET_baseline
CUDA_VISIBLE_DEVICES=0 bash scripts/llava1_5/gqa.sh
```
