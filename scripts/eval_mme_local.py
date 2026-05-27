import argparse
import json
from collections import defaultdict


COGNITION_CATEGORIES = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_yes_no(text):
    text = str(text or "").strip().lower()
    first_sentence = text.split(".")[0]
    words = {w.strip(",:;!?()[]{}\"'") for w in first_sentence.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if text.startswith("no"):
        return "no"
    if text.startswith("yes"):
        return "yes"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Score MME yes/no predictions with the official acc/acc_plus score.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    annotations = read_jsonl(args.annotation_file)
    answers = read_jsonl(args.result_file)
    answer_by_qid = {row["question_id"]: row for row in answers}

    details = []
    category_stats = defaultdict(lambda: {"questions": 0, "correct": 0, "images": set(), "image_results": defaultdict(list)})
    correct = 0
    for ann in annotations:
        pred_row = answer_by_qid.get(ann["question_id"], {})
        pred = parse_yes_no(pred_row.get("text", ""))
        label = parse_yes_no(ann["answer"])
        ok = pred == label
        correct += int(ok)
        stat = category_stats[ann["category"]]
        stat["questions"] += 1
        stat["correct"] += int(ok)
        stat["images"].add(ann["image"])
        stat["image_results"][ann["image"]].append(ok)
        details.append({
            "question_id": ann["question_id"],
            "category": ann["category"],
            "prediction": pred,
            "answer": label,
            "raw_prediction": pred_row.get("text", ""),
            "correct": ok,
        })

    categories = {}
    perception_score = 0.0
    cognition_score = 0.0
    for category, stat in sorted(category_stats.items()):
        questions = stat["questions"]
        images = len(stat["images"])
        acc = 100.0 * stat["correct"] / questions if questions else 0.0
        acc_plus_count = sum(1 for oks in stat["image_results"].values() if oks and all(oks))
        acc_plus = 100.0 * acc_plus_count / images if images else 0.0
        score = acc + acc_plus
        categories[category] = {
            "questions": questions,
            "images": images,
            "correct": stat["correct"],
            "acc": acc,
            "acc_plus": acc_plus,
            "score": score,
        }
        if category in COGNITION_CATEGORIES:
            cognition_score += score
        else:
            perception_score += score

    total = len(annotations)
    result = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "categories": categories,
        "perception_score": perception_score,
        "cognition_score": cognition_score,
        "total_score": perception_score + cognition_score,
        "details": details,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("category	questions	images	acc	acc_plus	score")
    for category, stat in categories.items():
        print(f"{category}	{stat['questions']}	{stat['images']}	{stat['acc']:.2f}	{stat['acc_plus']:.2f}	{stat['score']:.2f}")
    final_score = perception_score + cognition_score
    print(f"perception_score	{perception_score:.2f}")
    print(f"cognition_score	{cognition_score:.2f}")
    print(f"total_score	{final_score:.2f}")
    print(f"mme_score	{final_score:.2f}")


if __name__ == "__main__":
    main()
