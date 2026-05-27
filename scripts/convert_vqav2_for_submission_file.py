import argparse
import json
import os

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def main():
    parser = argparse.ArgumentParser(description="Convert VQAv2 jsonl predictions to EvalAI submission JSON.")
    parser.add_argument("--src", required=True, help="Merged LLaVA answer jsonl.")
    parser.add_argument("--test-file", required=True, help="Full VQAv2 test2015 jsonl used for submission ordering.")
    parser.add_argument("--dst", required=True, help="Output submission json.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    results = {}
    error_line = 0
    for line in open(args.src, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            error_line += 1
            continue
        results[row["question_id"]] = row.get("text", "")

    test_split = [json.loads(line) for line in open(args.test_file, encoding="utf-8")]
    print(f"total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}")

    answer_processor = EvalAIAnswerProcessor()
    all_answers = []
    for row in test_split:
        answer = answer_processor(results[row["question_id"]]) if row["question_id"] in results else ""
        all_answers.append({"question_id": row["question_id"], "answer": answer})

    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(all_answers, f)


if __name__ == "__main__":
    main()
