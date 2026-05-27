import argparse
import ast
import base64
import csv
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


PROMPT_SUFFIX = "Answer the question using a single word or phrase."
OPTIONS = ["A", "B", "C", "D", "E"]


def parquet_files(root, pattern="*.parquet"):
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {Path(root) / pattern}")
    return files


def iter_rows(root, pattern="*.parquet", columns=None, batch_size=1024):
    for parquet_file in parquet_files(root, pattern):
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def image_ext(image, default=".jpg"):
    path = ""
    if isinstance(image, dict):
        path = image.get("path") or ""
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix
    data = image.get("bytes") if isinstance(image, dict) else None
    if data and data.startswith(b"\x89PNG"):
        return ".png"
    return default


def write_image(image, dst, default_ext=".jpg"):
    if not isinstance(image, dict) or image.get("bytes") is None:
        return None
    dst = Path(dst)
    if dst.suffix == "":
        dst = dst.with_suffix(image_ext(image, default_ext))
    ensure_dir(dst.parent)
    data = image["bytes"]
    if dst.exists() and dst.stat().st_size == len(data):
        return dst
    dst.write_bytes(data)
    return dst


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


def normalize_answer(value):
    return str(value or "").strip().lower().rstrip(".")


def prepare_gqa(src_root, dst_root, repo_root):
    src = Path(src_root) / "gqa"
    dst = Path(dst_root) / "gqa"
    images = dst / "images"
    ensure_dir(images)

    image_count = 0
    for row in iter_rows(src / "testdev_balanced_images", columns=["id", "image"]):
        write_image(row["image"], images / f"{row['id']}.jpg")
        image_count += 1

    llava_rows = []
    eval_questions = {}
    for row in iter_rows(src / "testdev_balanced_instructions"):
        qid = str(row["id"])
        image_id = str(row["imageId"])
        question = row["question"].strip()
        llava_rows.append(
            {
                "question_id": qid,
                "image": f"{image_id}.jpg",
                "text": f"{question}\n{PROMPT_SUFFIX}",
            }
        )
        eval_questions[qid] = {
            "question": row["question"],
            "answer": row["answer"],
            "fullAnswer": row.get("fullAnswer"),
            "imageId": image_id,
            "isBalanced": row["isBalanced"],
            "groups": row["groups"],
            "entailed": parse_list(row.get("entailed")),
            "equivalent": parse_list(row.get("equivalent")),
            "types": row["types"],
            "annotations": row.get("annotations"),
            "semantic": row["semantic"],
            "semanticStr": row.get("semanticStr"),
        }

    question_count = write_jsonl(dst / "llava_gqa_testdev_balanced.jsonl", llava_rows)
    write_json(dst / "testdev_balanced_questions.json", eval_questions)
    shutil.copy2(repo_root / "data/eval/gqa/eval.py", dst / "eval.py")
    return {"images": image_count, "questions": question_count}


def prepare_mme(src_root, dst_root):
    src = Path(src_root) / "mme/data"
    dst = Path(dst_root) / "MME"
    image_root = dst / "images"
    llava_rows = []
    ann_rows = []
    per_image_seen = defaultdict(int)
    written_images = set()

    for row in iter_rows(src):
        raw_image = row["question_id"]
        per_image_seen[raw_image] += 1
        qid = f"{raw_image}#{per_image_seen[raw_image]}"
        image_rel = raw_image
        if image_rel not in written_images:
            write_image(row["image"], image_root / image_rel, default_ext=Path(image_rel).suffix or ".jpg")
            written_images.add(image_rel)
        llava_rows.append(
            {
                "question_id": qid,
                "image": image_rel,
                "text": row["question"],
            }
        )
        ann_rows.append(
            {
                "question_id": qid,
                "image": image_rel,
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
            }
        )

    return {
        "images": len(written_images),
        "questions": write_jsonl(dst / "llava_mme.jsonl", llava_rows),
        "annotations": write_jsonl(dst / "mme_annotations.jsonl", ann_rows),
    }


def encode_image(image):
    if not isinstance(image, dict) or image.get("bytes") is None:
        return ""
    return base64.b64encode(image["bytes"]).decode("utf-8")


def prepare_mmbench_one(src_dir, dst_file, pattern="dev-*.parquet"):
    columns = [
        "index",
        "question",
        "hint",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "category",
        "image",
        "source",
        "l2-category",
        "comment",
    ]
    ensure_dir(Path(dst_file).parent)
    count = 0
    with open(dst_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in iter_rows(src_dir, pattern):
            row = dict(row)
            row["image"] = encode_image(row.get("image"))
            row["l2-category"] = row.get("l2-category", row.get("L2-category", ""))
            writer.writerow({k: row.get(k, "") for k in columns})
            count += 1
    return count


def prepare_mmbench(src_root, dst_root):
    dst = Path(dst_root) / "mmbench"
    en = prepare_mmbench_one(Path(src_root) / "mmbench_en/data", dst / "mmbench_dev_20230712.tsv")
    cn = prepare_mmbench_one(Path(src_root) / "mmbench_cn/data", dst / "mmbench_dev_cn_20231003.tsv")
    return {"mmbench_dev_20230712": en, "mmbench_dev_cn_20231003": cn}


def prepare_pope(src_root, dst_root):
    src = Path(src_root) / "pope/data"
    dst = Path(dst_root) / "pope"
    image_root = dst / "val2014"
    label_rows = defaultdict(list)
    llava_rows = []
    written_images = set()

    for row in iter_rows(src):
        category = row["category"]
        qid = f"{category}_{row['question_id']}"
        image_name = f"{row['image_source']}.jpg"
        if image_name not in written_images:
            write_image(row["image"], image_root / image_name)
            written_images.add(image_name)
        llava_rows.append(
            {
                "question_id": qid,
                "image": image_name,
                "text": row["question"],
                "category": category,
            }
        )
        label_rows[category].append({"label": normalize_answer(row["answer"])})

    question_count = write_jsonl(dst / "llava_pope_test.jsonl", llava_rows)
    coco_dir = dst / "coco"
    ensure_dir(coco_dir)
    for category, rows in sorted(label_rows.items()):
        write_jsonl(coco_dir / f"coco_pope_{category}.json", rows)
    return {"images": len(written_images), "questions": question_count, "categories": len(label_rows)}


def scienceqa_prompt(question, choices, hint):
    parts = []
    if hint:
        parts.append(str(hint).strip())
    parts.append(str(question).strip())
    for option, choice in zip(OPTIONS, choices):
        parts.append(f"{option}. {choice}")
    return "\n".join(part for part in parts if part)


def prepare_scienceqa(src_root, dst_root):
    src = Path(src_root) / "scienceqa_img/data"
    dst = Path(dst_root) / "scienceqa"
    image_root = dst / "images/test"
    problems = {}
    split = []
    llava_rows = []

    for idx, row in enumerate(iter_rows(src, "test-*.parquet")):
        pid = str(idx)
        choices = list(row["choices"])
        ext = image_ext(row["image"], ".png")
        image_name = f"image{ext}"
        image_value = None
        if isinstance(row.get("image"), dict) and row["image"].get("bytes") is not None:
            write_image(row["image"], image_root / pid / image_name, default_ext=ext)
            image_value = image_name
        answer_idx = int(row["answer"])
        problems[pid] = {
            "question": row["question"],
            "choices": choices,
            "answer": answer_idx,
            "hint": row.get("hint", ""),
            "image": image_value,
            "task": row.get("task"),
            "grade": row.get("grade"),
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "category": row.get("category"),
            "skill": row.get("skill"),
            "lecture": row.get("lecture"),
            "solution": row.get("solution"),
        }
        split.append(pid)
        prompt = scienceqa_prompt(row["question"], choices, row.get("hint", ""))
        conversations = [
            {"from": "human", "value": f"{prompt}\n<image>" if image_value else prompt},
            {"from": "gpt", "value": OPTIONS[answer_idx]},
        ]
        record = {"id": pid, "conversations": conversations}
        if image_value:
            record["image"] = f"{pid}/{image_name}"
        llava_rows.append(record)

    write_json(dst / "problems.json", problems)
    write_json(dst / "pid_splits.json", {"test": split})
    write_json(dst / "llava_test_CQM-A.json", llava_rows)
    return {"questions": len(llava_rows)}


def prepare_textvqa(src_root, dst_root):
    src = Path(src_root) / "textvqa/data"
    dst = Path(dst_root) / "textvqa"
    image_root = dst / "train_images"
    llava_rows = []
    annotations = []

    for row in iter_rows(src, "validation-*.parquet"):
        image_name = f"{row['image_id']}{image_ext(row['image'], '.jpg')}"
        write_image(row["image"], image_root / image_name)
        ocr_tokens = row.get("ocr_tokens") or []
        ocr_text = ", ".join(str(token) for token in ocr_tokens if str(token).strip())
        prompt = f"OCR tokens: {ocr_text}\nQuestion: {row['question']} Short answer:"
        llava_rows.append(
            {
                "question_id": row["image_id"],
                "image": image_name,
                "text": prompt,
            }
        )
        annotations.append(
            {
                "image_id": row["image_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "answers": row["answers"],
            }
        )

    return {
        "questions": write_jsonl(dst / "llava_textvqa_val_v051_ocr.jsonl", llava_rows),
        "annotations": len(annotations),
        "annotation_file": str(write_json(dst / "TextVQA_0.5.1_val.json", {"data": annotations})),
    }


def prepare_vizwiz_split(src, dst, split):
    image_root = dst / split
    llava_rows = []
    annotations = []
    for row in iter_rows(src, f"{split}-*.parquet"):
        image_name = row["image"].get("path") or f"{row['question_id']}{image_ext(row['image'], '.jpg')}"
        write_image(row["image"], image_root / image_name)
        llava_rows.append(
            {
                "question_id": row["question_id"],
                "image": image_name,
                "text": f"{row['question']}\n{PROMPT_SUFFIX}",
            }
        )
        annotations.append(
            {
                "question_id": row["question_id"],
                "image": image_name,
                "question": row["question"],
                "answers": row["answers"],
                "category": row["category"],
            }
        )
    write_jsonl(dst / f"llava_{split}.jsonl", llava_rows)
    write_json(dst / f"VizWiz_{split}.json", {"data": annotations})
    return len(llava_rows)


def prepare_vizwiz(src_root, dst_root):
    src = Path(src_root) / "vizwiz_vqa/data"
    dst = Path(dst_root) / "vizwiz"
    return {
        "val": prepare_vizwiz_split(src, dst, "val"),
        "test": prepare_vizwiz_split(src, dst, "test"),
    }


def vqav2_image_name(image_id):
    return f"COCO_test2015_{int(image_id):012d}.jpg"


def prepare_vqav2_testdev_file(task):
    file_index, parquet_file, image_root = task
    parquet_file = Path(parquet_file)
    image_root = Path(image_root)
    rows = []
    image_count = 0
    for batch in pq.ParquetFile(parquet_file).iter_batches(
        batch_size=512, columns=["image_id", "question_id", "question", "image"]
    ):
        for row in batch.to_pylist():
            image_name = row["image"].get("path") or vqav2_image_name(row["image_id"])
            written = write_image(row["image"], image_root / image_name)
            image_count += int(written is not None)
            rows.append(
                {
                    "question_id": row["question_id"],
                    "image": image_name,
                    "text": f"{row['question']}\n{PROMPT_SUFFIX}",
                }
            )
    return file_index, image_count, rows


def prepare_vqav2(src_root, dst_root, workers=8):
    src = Path(src_root) / "vqav2/data"
    dst = Path(dst_root) / "vqav2"
    image_root = dst / "test2015"
    ensure_dir(image_root)
    testdev_rows = []
    test_rows = []

    testdev_files = parquet_files(src, "testdev-*.parquet")
    tasks = [(idx, parquet_file, image_root) for idx, parquet_file in enumerate(testdev_files)]
    if workers <= 1:
        completed = [prepare_vqav2_testdev_file(task) for task in tasks]
    else:
        completed = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(prepare_vqav2_testdev_file, task) for task in tasks]
            for future in as_completed(futures):
                file_index, image_count, rows = future.result()
                print(f"[vqav2] shard {file_index + 1}/{len(tasks)}: {len(rows)} questions, {image_count} image writes")
                completed.append((file_index, image_count, rows))
    for _, _, rows in sorted(completed, key=lambda item: item[0]):
        testdev_rows.extend(rows)

    for row in iter_rows(src, "test-*.parquet", columns=["image_id", "question_id", "question"]):
        test_rows.append(
            {
                "question_id": row["question_id"],
                "image": vqav2_image_name(row["image_id"]),
                "text": f"{row['question']}\n{PROMPT_SUFFIX}",
            }
        )

    return {
        "testdev": write_jsonl(dst / "llava_vqav2_mscoco_test-dev2015.jsonl", testdev_rows),
        "test2015": write_jsonl(dst / "llava_vqav2_mscoco_test2015.jsonl", test_rows),
    }

PREPARE = {
    "gqa": prepare_gqa,
    "mme": prepare_mme,
    "mmbench": prepare_mmbench,
    "pope": prepare_pope,
    "scienceqa": prepare_scienceqa,
    "textvqa": prepare_textvqa,
    "vizwiz": prepare_vizwiz,
    "vqav2": prepare_vqav2,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare local HF parquet eval datasets for ApET/LLaVA scripts.")
    parser.add_argument("--src-root", default="/root/autodl-fs/imgs")
    parser.add_argument("--dst-root", default="/root/autodl-fs/llava_eval_imgs")
    parser.add_argument("--vqav2-workers", type=int, default=8, help="Parallel workers for extracting VQAv2 testdev images.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PREPARE.keys()),
        choices=list(PREPARE.keys()),
        help="Datasets to prepare.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary = {}
    for dataset in args.datasets:
        print(f"[prepare] {dataset}")
        if dataset == "gqa":
            summary[dataset] = PREPARE[dataset](args.src_root, args.dst_root, repo_root)
        elif dataset == "vqav2":
            summary[dataset] = PREPARE[dataset](args.src_root, args.dst_root, args.vqav2_workers)
        else:
            summary[dataset] = PREPARE[dataset](args.src_root, args.dst_root)
        print(f"[done] {dataset}: {summary[dataset]}")

    write_json(Path(args.dst_root) / "prepare_summary.json", summary)
    print(f"Wrote summary to {Path(args.dst_root) / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
