# Evaluation Protocol Audit: MMBench / MMBench-CN / TextVQA

CURRENT_REPO=/root/autodl-tmp/mllm_token_compression/main_exp/ApET_baseline_no_llm_pruning
APET_ORIG_REPO=/root/autodl-tmp/mllm_token_compression/main_exp/ApET
MMTOK_REPO=/root/autodl-tmp/mllm_token_compression/main_exp/MMTok-main

## 1. Executive summary

The current MMBench and MMBench-CN numbers, 70.09 / 66.09, are not official MMBench scores. They are local dev-label, single-pass accuracies computed by `scripts/eval_mmbench_local.py` directly against the TSV `answer` column. The script does not implement MMBench CircularEval and does not call OpenCompass/VLMEvalKit/MMBench server evaluation.

The current TextVQA evaluator is not a simple exact-match evaluator. It uses `TextVQAAccuracyEvaluator`, `EvalAIAnswerProcessor`, 10 human answers, and VQA-style soft accuracy. However, the existing prediction file at `result/llava1_5/textvqa/answers/llava_v15_7b.jsonl` contains plain prompts such as `question\nAnswer the question using a single word or phrase.`, not the OCR-token prompt expected by the current `textvqa.sh`. The existing artifact therefore reflects a non-OCR/local prediction setting even though the data file prepared by the current script includes OCR tokens.

Bottom line: do not directly mix the current local MMBench/MMBCN/TextVQA results with ApET/MMTok paper table numbers. If local values are shown, label them as `MMB-local Dev Acc.`, `MMBCN-local Dev Acc.`, and `TextVQA-local Acc.`.

## 2. MMBench / MMBench-CN audit

### Protocol table

| Item | Current project | Original ApET | MMTok | Conclusion |
|---|---|---|---|---|
| Data file | `/root/autodl-fs/llava_eval_imgs/mmbench/mmbench_dev_20230712.tsv`; `/root/autodl-fs/llava_eval_imgs/mmbench/mmbench_dev_cn_20231003.tsv` | `data/eval/mmbench/mmbench_dev_20230712.tsv`; `data/eval/mmbench/mmbench_dev_cn_20231003.tsv` | No bundled task/data; README says to use lmms-eval adapters | Current uses local prepared dev TSVs |
| Inference script | `scripts/llava1_5/mmbench.sh`, `mmbench_cn.sh` run `llava.eval.model_vqa_mmbench` with `--single-pred-prompt` | Same LLaVA-style inference scripts | lmms-eval model wrappers only | Current and ApET inference shape is similar, but current has extra local scoring |
| Local label accuracy | Yes: `scripts/eval_mmbench_local.py` computes `Correct / Total` from TSV labels | Not present | Not present in repo | Current 70.09 / 66.09 comes from this local evaluator |
| CircularEval | No. `--all-rounds` exists in inference code but current scripts do not use it, and local evaluator does not aggregate circular rounds | Not implemented in bundled scripts | Expected via external lmms-eval/OpenCompass-style task, not in repo | Current result is single-pass, original option order |
| Official/OpenCompass submission | Generates `.xlsx` through `scripts/convert_mmbench_for_submission.py` | Generates `.xlsx` through same converter | Not bundled | Submission file generation exists, official scoring does not |
| lmms-eval task | Not used | Not used | README/install.md instruct adding MMTok wrappers to lmms-eval | MMTok paper protocol likely comes from external lmms-eval/eval stack, not this repo's local scorer |
| Comparable to paper table? | No | The generated upload file may be comparable only after official/OpenCompass evaluation | Paper numbers likely comparable to official/lmms-eval protocol | Current local dev-label Acc. should not be mixed with paper MMB/MMBCN |

### Direct answers

1. Current MMBench inference generates a JSONL and then creates an upload XLSX. The current scripts also run an extra local evaluator afterward.
2. Yes. `scripts/eval_mmbench_local.py` reads TSV `answer` labels and computes local `Correct / Total`.
3. Yes. It evaluates only the original option order from one prediction round.
4. No. It does not implement CircularEval.
5. The parser is simple, not official: it accepts the first character if it is A/B/C/D, otherwise the first standalone A/B/C/D. It does not do option-text matching. EN predictions are almost all clean single letters, so the EN score is not obviously inflated by parser looseness. CN has some `选项B` cases that are actually marked invalid by this parser.
6. Original ApET provides upload-file generation, not local official scoring.
7. The local MMTok source provides lmms-eval wrappers and instructions, but does not bundle MMBench task configs. It expects evaluation through external lmms-eval.
8. The current 70.09 / 66.09 should be marked as `MMBench local dev-label single-pass Acc.`, not official MMBench Acc.

Current MMBench results should be reported as **MMBench/MMBCN local dev-label single-pass Acc.**, and should **not** be directly mixed with ApET/MMTok paper-reported MMB scores.

### Lightweight MMBench parsing check

Existing prediction files:

- EN: `result/llava1_5/mmbench/answers/mmbench_dev_20230712/llava_v15_7b.jsonl`
- CN: `result/llava1_5/mmbench_cn/answers/mmbench_dev_cn_20231003/llava_v15_7b.jsonl`

First-50 EN check:

| Statistic | Count | Ratio |
|---|---:|---:|
| One clear A/B/C/D | 50 / 50 | 100.00% |
| Multiple option letters | 0 / 50 | 0.00% |
| No clear option but parsed | 0 / 50 | 0.00% |
| Option text appears in raw prediction | 1 / 50 | 2.00% |
| Invalid prediction | 0 / 50 | 0.00% |

All EN predictions:

| Statistic | Count | Ratio |
|---|---:|---:|
| One clear A/B/C/D | 4377 / 4377 | 100.00% |
| Multiple option letters | 0 / 4377 | 0.00% |
| No clear option but parsed | 0 / 4377 | 0.00% |
| Option text appears in raw prediction | 14 / 4377 | 0.32% |
| Invalid prediction | 0 / 4377 | 0.00% |

All CN predictions:

| Statistic | Count | Ratio |
|---|---:|---:|
| One clear A/B/C/D | 4308 / 4329 | 99.51% |
| Multiple option letters | 5 / 4329 | 0.12% |
| No clear option but parsed | 1 / 4329 | 0.02% |
| Option text appears in raw prediction | 31 / 4329 | 0.72% |
| Invalid prediction | 15 / 4329 | 0.35% |

Suspicious CN examples include raw predictions like `选项B`, which the current parser does not parse as `B`, and long answers such as `选项 B 是正确的...` where multiple option letters appear in the explanation. These edge cases are too rare to explain the high local score. The main protocol difference is lack of CircularEval/official scoring.

## 3. TextVQA audit

### Protocol table

| Item | Current project | Original ApET | MMTok / lmms-eval | Conclusion |
|---|---|---|---|---|
| Question file | Script points to `/root/autodl-fs/llava_eval_imgs/textvqa/llava_textvqa_val_v051_ocr.jsonl` | `data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl` | External lmms-eval task; optional OCR injection depending on config | Current script is OCR-token capable |
| OCR tokens included? | Yes in prepared question file; no in existing prediction prompts | Yes in script path | lmms-eval TextVQA supports OCR-token prompt injection via task kwargs | Existing result artifact was generated without OCR prompt |
| Annotation file | `/root/autodl-fs/llava_eval_imgs/textvqa/TextVQA_0.5.1_val.json` | `data/eval/textvqa/TextVQA_0.5.1_val.json` | External dataset/task | Current and original ApET evaluator format match |
| Split size | 5000 predictions; 5000 question rows | 5000 val rows expected | TextVQA val commonly 5000 | Split size is expected |
| Evaluator | `llava.eval.eval_textvqa` + `TextVQAAccuracyEvaluator` | Same | lmms-eval VQA tasks use normalized multi-answer scoring | Current evaluator is VQA-style, not exact-only |
| Answer normalization | EvalAI-style lowercase, punctuation, articles, digit words, contractions | Same | Same family of normalization in lmms-eval | Normalization is present |
| Soft VQA accuracy | Yes, leave-one-out 10-answer soft score | Yes | Yes for TextVQA-style eval | Formula is not the source of the low score |
| Comparable to paper VQAText? | Not from the current existing artifact, because predictions lack OCR prompt and are local-run artifacts | Original protocol can be comparable if run as intended | Paper numbers likely use their reported lmms-eval/protocol | Do not compare current 34.49/36.87 directly with paper VQAText |

### Direct answers

1. Current TextVQA question-file in `scripts/llava1_5/textvqa.sh`: `/root/autodl-fs/llava_eval_imgs/textvqa/llava_textvqa_val_v051_ocr.jsonl`.
2. Current TextVQA annotation-file: `/root/autodl-fs/llava_eval_imgs/textvqa/TextVQA_0.5.1_val.json`.
3. The question file contains OCR tokens. Example prompt starts with `OCR tokens: ...\nQuestion: ... Short answer:`.
4. Current evaluator computes normalized VQA soft accuracy through `TextVQAAccuracyEvaluator`, not simple exact match.
5. Yes. It asserts and uses 10 human answers.
6. Yes. It normalizes case, punctuation, articles, digit words, and contractions through `EvalAIAnswerProcessor`.
7. Yes, the prediction string is normalized as a whole. If the model outputs a full sentence, it will usually score 0 unless the whole normalized sentence matches an answer.
8. The result JSONL schema is compatible: `question_id`, `prompt`, `text`, `answer_id`, `model_id`, `metadata`. The evaluator maps `(question_id, processed prompt question)` to annotations.
9. The current artifact is more consistent with a non-OCR local prediction setting than with paper-reported TextVQA/VQAText numbers. It is not exact-match, but it lacks OCR tokens in the prompt.
10. Original ApET TextVQA script/evaluator matches the intended current script: OCR question file plus `TextVQAAccuracyEvaluator`.
11. MMTok itself does not bundle TextVQA task code; it instructs using lmms-eval wrappers. lmms-eval TextVQA uses normalized VQA-style multi-answer scoring and may include OCR tokens depending on task config.

Current TextVQA result 34.49%/36.87% is likely caused by **evaluating an existing non-OCR local prediction artifact and possibly local model/token settings, not by an exact-match evaluator bug**. It should **not** be compared directly with ApET/MMTok VQAText paper scores.

### Lightweight TextVQA post-hoc evaluation

Existing prediction file:

`/root/autodl-tmp/mllm_token_compression/main_exp/ApET_baseline_no_llm_pruning/result/llava1_5/textvqa/answers/llava_v15_7b.jsonl`

Post-hoc checks on this file:

| Check | Value |
|---|---:|
| Samples matched to annotations | 5000 |
| Missing annotations | 0 |
| Predictions whose stored prompt starts with `OCR tokens:` | 0 |
| Predictions longer than 5 words | 34 |
| Recomputed soft VQA accuracy | 36.87% |
| Normalized any-ground-truth exact match | 40.46% |

The user-reported 34.49% is close to this local artifact family, but the current `run.log` and direct recomputation over the present file give 36.87%.

Sampled first 30 TextVQA rows:

| # | question_id | question | raw prediction | normalized prediction | score | ground-truth answers |
|---:|---|---|---|---|---:|---|
| 1 | 003a8ae2ef43b901 | what is the brand of this camera? | Kodak | kodak | 0.0 | nous les gosses; dakota; clos culombu; dakota digital; dakota; dakota; dakota digital; dakota digital; dakota; dakota |
| 2 | b9dc400eb20bad64 | what does the small white text spell? | Drupal | drupal | 0.0 | copenhagen; copenhagen; copenhagen; copenhagen; copenhagen; thursday; copenhagen; copenhagen; copenhagen; copenhagen |
| 3 | 2b538a43dd933fc1 | what kind of beer is this? | Stout | stout | 0.0 | ale; sublimely self-righteous ale; stone; ale; self righteous; ale; ale; ale; ale; ale |
| 4 | 831bcec304a17054 | what brand liquor is on the right? | Bowmore | bowmore | 1.0 | bowmore; bowmore; bowmore; bowmore; bowmore; bowmore; bowmore; bowmore islay; dowmore islay; bowmore islay |
| 5 | 831bcec304a17054 | how long has the drink on the right been aged? | 10 years | 10 years | 1.0 | 10 years; 10 year; 10 years; 10 years; 10 years; 10 years; 10 years; 10 years; martial arts; 10 |
| 6 | 94ad4aad01e27a32 | what number is on the player's jersey? | 22 | 22 | 1.0 | 22 x10 |
| 7 | 181f00d3ee2b2076 | what is the time? | 10:10 | 10:10 | 0.0 | 5:41; 5:41; 8:00; 5:41; 5:40; 5:41; 5:42; 5:41; 8:00; 5:41 |
| 8 | 181f00d3ee2b2076 | what brand of watch is that? | Tissot | tissot | 0.0 | ap; unanswerable; unanswerable; ap; af; unanswerable; audemars; unanswerable; unanswerable; ap |
| 9 | 30f9b48158fc56e6 | who is at the center of all of this? | Woman | woman | 0.0 | bryan owens; alexa curtis; bryan owens x6; mahou; agile experience design makeup |
| 10 | 4ffbb21f379cccbe | who was the photographer? | Lightbringer | lightbringer | 0.0 | philippe molitor x4; clardajne; phillipe molida; l; no; phillipe meltow; philippe molitar |
| 11 | 4dc59c7e57fa8968 | are these switches on or off? | Off | off | 1.0 | off x10 |
| 12 | 45dc5b66e3f3f31b | what candy bar is down there on the bottom? | Milky way | milky way | 0.0 | hersheys; hershey's x9 |
| 13 | fa9ffd5aca1e4e51 | what does the light sign read on the farthest right window? | All day | all day | 0.0 | bud light x9; all 2 liters |
| 14 | fa9ffd5aca1e4e51 | how much for a can of skoal? | 2.99 | 2.99 | 0.0 | 3.82; $3.32; 3.82 x5; $3.82 x3 |
| 15 | 5ce862cbefd8458f | is this denny's? | Yes | yes | 1.0 | yes x8; pet center; one man show |
| 16 | 5ce862cbefd8458f | what color are the letters on this sign? | Yellow | yellow | 0.0 | red x7; first; reds; answering does not require reading text in the image |
| 17 | 270ee71a340f50e8 | what brand is the bottle with red label? | Jack daniels | jack daniels | 0.0 | red label; johnnie walker; jonnie walker; black label; red label; johnny walker; non-reading; red label; johnnie walker; jonnie walker |
| 18 | 7de24c1c61b63f52 | how many percent is shown on the poster? | 50 | 50 | 0.0 | 5 and 10; 0; 0%; 0; 5% and 10%; 0; 0; 0; 10; 0 |
| 19 | 40c6b4dd3caa006f | how many items can you get for $5? | 1 | 1 | 0.0 | 3 x9; 3 for $5 |
| 20 | 40c6b4dd3caa006f | how man price tags are on the bottom shelf? | 1 | 1 | 0.0 | non-reading x5; 4 x5 |
| 21 | cc9c3bff02242301 | what is one of the brands being advertised? | Yamaha | yamaha | 1.0 | yamaha x8; yahama; peugeot |
| 22 | fb39f5b120097669 | what year was this taken? | 2014 | 2014 | 0.0 | 2012 x10 |
| 23 | a3f1fe4ebd510d4e | what kind of comupter is this? | Macbook | macbook | 1.0 | macbook x9; macbook' |
| 24 | a3f1fe4ebd510d4e | what does the screen say to do? | Select | select | 0.6 | select; select your; continue; non-reading; continue; select; continue; select something; select your keyboard; select your keybound |
| 25 | 7b0112f7f5f096c5 | what is written at the top of the yellow sticker on the fridge? | Do not drink | do not drink | 0.0 | warning x8; warning! do not unplug!; smoking |
| 26 | 7b0112f7f5f096c5 | what is the year on the calender? | 2016 | 2016 | 0.0 | 2010 x9; unanswerable |
| 27 | 71e2433bfd38a9dc | what is the name of the runner on the left? | Wills | wills | 0.0 | willis x10 |
| 28 | 71e2433bfd38a9dc | what event is this from? | Olympics | olympics | 0.0 | millrose games x8; hillrose games x2 |
| 29 | 655dcd26ef3108ff | who beamed at him? | Dumbledore | dumbledore | 1.0 | dumbledore x9; look& storng dumbledore |
| 30 | 655dcd26ef3108ff | what is the name of this chapter? | King's cross | king 's cross | 1.0 | king's cross x9; leo |

## 4. Why local results differ from ApET/MMTok paper tables

MMBench/MMBCN differ because the current reported values are local dev-label single-pass accuracies. Official MMBench scoring uses a stricter benchmark protocol, including CircularEval, and often answer extraction/normalization through the official evaluation stack. The current local number answers a different question: "Did the model pick the TSV label on this one original option ordering?"

TextVQA differs for a different reason. The evaluator formula is already the expected VQA soft accuracy. The mismatch is that the existing predictions were generated without OCR-token context, while the current prepared question file and original ApET script are OCR-token based. This can substantially depress TextVQA, because many examples require reading scene text that OCR tokens expose directly.

MMTok is not a drop-in local evaluation repo here. Its source provides model wrappers and says to evaluate through lmms-eval. Therefore, paper numbers from MMTok should be treated as paper-reported protocol numbers, not as numbers produced by the current local scripts.

## 5. What should be used in my paper

For baseline rows such as ApET, MMTok, VisionZip, SparseVLM, and PDrop, use the corresponding paper-reported baseline results under their own reported evaluation protocols. This is defensible and avoids mixing local partial reruns with official or lmms-eval-style values.

For your own method, either:

- evaluate with the same official/lmms-eval protocol used by the paper baselines; or
- clearly mark any local numbers as local and keep them out of a direct apples-to-apples main table.

Do not silently replace paper MMB/MMBCN/TextVQA with the current local values.

## 6. Recommended table header

Method | GQA Acc. | MMB Acc. | MMBCN Acc. | MME P+C | POPE F1 | SQA Acc. | VQAv2 Acc. | TextVQA Acc. | VizWiz Acc. | Avg.

If current local values are included, use:

Method | GQA Acc. | MMB-local Dev Acc. | MMBCN-local Dev Acc. | MME P+C | POPE F1 | SQA Acc. | VQAv2 Acc. | TextVQA-local Acc. | VizWiz Acc. | Avg.

## 7. Recommended footnote

For paper-reported baselines:

"Baseline results are taken from the corresponding papers under their reported evaluation protocols. Following prior VLM token-compression works, MME is reported as perception+cognition score, POPE as F1, and Avg. is computed as the mean relative performance with respect to the full-token vanilla baseline."

If any current local values are mixed in:

"Rows marked with local metrics use local post-hoc/dev-label evaluation and are not official benchmark scores. In particular, MMB-local Dev Acc. and MMBCN-local Dev Acc. are single-pass local TSV-label accuracies without CircularEval, and TextVQA-local Acc. is computed from local prediction artifacts. These numbers should not be directly compared with paper-reported official/lmms-eval scores."

## 8. Action items

1. In all notes/tables, rename current MMBench 70.09 and MMBCN 66.09 to `MMB-local Dev Acc.` and `MMBCN-local Dev Acc.`.
2. Do not use current MMB/MMBCN values as official MMBench scores.
3. Treat current TextVQA 34.49/36.87 as a local artifact score, because the stored predictions contain no OCR tokens.
4. If TextVQA must be rerun for a local method, ensure the prediction prompt actually contains `OCR tokens:` when using `llava_textvqa_val_v051_ocr.jsonl`; otherwise label it non-OCR/local.
5. If comparable MMB/MMBCN numbers are needed, use official/OpenCompass/VLMEvalKit/lmms-eval protocol rather than `scripts/eval_mmbench_local.py`.
6. Keep the current local evaluators as diagnostic tools only; do not patch them to chase paper values.

## Sources checked

- Local current repo scripts/evaluators listed above.
- Original ApET scripts/evaluators listed above.
- MMTok README/install guidance for lmms-eval wrappers.
- MMBench paper: https://arxiv.org/abs/2307.06281
- lmms-eval documentation/GitHub entry points: https://github.com/EvolvingLMMs-Lab/lmms-eval and https://docs.lmms-lab.com/docs
