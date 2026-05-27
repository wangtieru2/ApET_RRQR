import argparse
import ast
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


PROMPT_SUFFIX = "Answer the question using a single word or phrase."


def iter_parquet_rows(parquet_dir, pattern):
    files = sorted(Path(parquet_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {Path(parquet_dir) / pattern}")

    for parquet_file in files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


def parse_list(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def write_images(src_root, split, out_dir):
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    image_src = src_root / f"{split}_images"
    for row in iter_parquet_rows(image_src, "*.parquet"):
        image_id = row["id"]
        image = row["image"]
        image_bytes = image["bytes"]
        if image_bytes is None:
            raise ValueError(f"Image {image_id} has no embedded bytes")

        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists() or image_path.stat().st_size != len(image_bytes):
            image_path.write_bytes(image_bytes)
        count += 1

    return count


def normalize_question(row):
    row = dict(row)
    row["id"] = str(row["id"])
    row["imageId"] = str(row["imageId"])
    row["entailed"] = parse_list(row.get("entailed"))
    row["equivalent"] = parse_list(row.get("equivalent"))
    return row


def write_questions(src_root, split, out_dir, llava_split):
    llava_path = out_dir / f"{llava_split}.jsonl"
    eval_path = out_dir / f"{split}_questions.json"

    questions_for_eval = {}
    count = 0
    instruction_src = src_root / f"{split}_instructions"

    with llava_path.open("w", encoding="utf-8") as llava_file:
        for row in iter_parquet_rows(instruction_src, "*.parquet"):
            row = normalize_question(row)
            qid = row["id"]
            image_id = row["imageId"]
            question = row["question"].strip()

            llava_record = {
                "question_id": qid,
                "image": f"{image_id}.jpg",
                "text": f"{question}\n{PROMPT_SUFFIX}",
            }
            llava_file.write(json.dumps(llava_record, ensure_ascii=False) + "\n")

            questions_for_eval[qid] = {
                "question": row["question"],
                "answer": row["answer"],
                "fullAnswer": row.get("fullAnswer"),
                "imageId": image_id,
                "isBalanced": row["isBalanced"],
                "groups": row["groups"],
                "entailed": row["entailed"],
                "equivalent": row["equivalent"],
                "types": row["types"],
                "annotations": row.get("annotations"),
                "semantic": row["semantic"],
                "semanticStr": row.get("semanticStr"),
            }
            count += 1

    with eval_path.open("w", encoding="utf-8") as eval_file:
        json.dump(questions_for_eval, eval_file, ensure_ascii=False)

    return count


def copy_eval_script(repo_root, out_dir):
    src = repo_root / "llava" / "eval" / "eval.py"
    dst = out_dir / "eval.py"
    if not src.exists():
        raise FileNotFoundError(f"Cannot find GQA eval script: {src}")
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace GQA parquet files to the layout used by ApET/LLaVA GQA eval."
    )
    parser.add_argument("--src", default="/root/autodl-fs/imgs/GQA", help="HuggingFace GQA parquet directory.")
    parser.add_argument("--dst", default="data/eval/gqa", help="Output directory under the ApET repo.")
    parser.add_argument("--split", default="testdev_balanced", help="GQA split prefix, e.g. testdev_balanced.")
    parser.add_argument(
        "--llava-split",
        default="llava_gqa_testdev_balanced",
        help="Output question jsonl basename expected by gqa.sh.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = Path(args.src).expanduser().resolve()
    out_dir = (repo_root / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_count = write_images(src_root, args.split, out_dir)
    question_count = write_questions(src_root, args.split, out_dir, args.llava_split)
    copy_eval_script(repo_root, out_dir)

    print(f"Wrote {image_count} images to {out_dir / 'images'}")
    print(f"Wrote {question_count} LLaVA questions to {out_dir / (args.llava_split + '.jsonl')}")
    print(f"Wrote GQA eval questions to {out_dir / (args.split + '_questions.json')}")
    print(f"Copied eval.py to {out_dir / 'eval.py'}")


if __name__ == "__main__":
    main()
