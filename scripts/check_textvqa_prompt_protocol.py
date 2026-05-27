import argparse
import json
from pathlib import Path


def read_first_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSONL rows found in {path}")


def count_rows(path):
    total = 0
    ocr_prompts = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if str(row.get("prompt", "")).startswith("OCR tokens:"):
                ocr_prompts += 1
    return total, ocr_prompts


def main():
    parser = argparse.ArgumentParser(description="Check TextVQA question/prediction prompt protocol consistency.")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--allow-non-ocr-result", action="store_true")
    args = parser.parse_args()

    question_file = Path(args.question_file)
    result_file = Path(args.result_file)
    first_question = read_first_jsonl(question_file)
    expects_ocr = str(first_question.get("text", "")).startswith("OCR tokens:")

    total, ocr_prompts = count_rows(result_file)
    if total == 0:
        raise ValueError(f"No prediction rows found in {result_file}")

    if expects_ocr and ocr_prompts != total and not args.allow_non_ocr_result:
        raise SystemExit(
            "TextVQA protocol mismatch: question file contains OCR-token prompts, "
            f"but only {ocr_prompts}/{total} prediction prompts start with 'OCR tokens:'. "
            "This result should be labeled TextVQA-local/non-OCR, not paper-comparable VQAText. "
            "Pass --allow-non-ocr-result only for diagnostic local scoring."
        )

    mode = "ocr" if expects_ocr else "plain"
    print(
        "TextVQA prompt protocol check passed: "
        f"question_mode={mode}, prediction_ocr_prompts={ocr_prompts}/{total}"
    )


if __name__ == "__main__":
    main()
