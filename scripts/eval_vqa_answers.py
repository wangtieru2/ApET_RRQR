import argparse
import json

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Score VQA-style predictions with multiple ground-truth answers.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    annotations = json.load(open(args.annotation_file, encoding="utf-8"))["data"]
    annotations = {row["question_id"]: row for row in annotations if row.get("answers")}
    results = [json.loads(line) for line in open(args.result_file, encoding="utf-8") if line.strip()]

    pred_list = []
    details = []
    for result in results:
        qid = result["question_id"]
        if qid not in annotations:
            continue
        pred = result.get("text", "")
        gt_answers = annotations[qid]["answers"]
        pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})
        details.append({"question_id": qid, "prediction": pred, "answers": gt_answers})

    evaluator = TextVQAAccuracyEvaluator()
    accuracy = evaluator.eval_pred_list(pred_list) if pred_list else 0.0
    output = {
        "total": len(pred_list),
        "accuracy": accuracy,
        "details": details,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Samples: {len(pred_list)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
