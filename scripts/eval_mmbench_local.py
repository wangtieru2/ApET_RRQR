import argparse
import json
import re

import pandas as pd


OPTION_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_option(text):
    text = str(text or "").strip()
    if not text:
        return ""
    first = text[0].upper()
    if first in {"A", "B", "C", "D"}:
        return first
    match = OPTION_RE.search(text)
    return match.group(1).upper() if match else ""


def main():
    parser = argparse.ArgumentParser(description="Score MMBench dev predictions against local parquet labels.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    df = pd.read_table(args.annotation_file)
    labels = {int(row["index"]): str(row["answer"]).strip().upper() for _, row in df.iterrows()}
    preds = {}
    for line in open(args.result_file, encoding="utf-8"):
        row = json.loads(line)
        preds[int(row["question_id"])] = parse_option(row.get("text", ""))

    details = []
    correct = 0
    for qid, label in labels.items():
        pred = preds.get(qid, "")
        ok = pred == label
        correct += int(ok)
        details.append({"question_id": qid, "prediction": pred, "answer": label, "correct": ok})

    total = len(labels)
    result = {
        "metric_name": "MMBench-local Dev Acc. (single-pass, no CircularEval)",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "details": details,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Metric: MMBench-local Dev Acc. (single-pass, no CircularEval; not official MMBench)")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {result['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
