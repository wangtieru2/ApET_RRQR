import argparse
import json
import os
import random
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.case_study_compression import TOKEN_STATE, compress_visual_tokens
from llava.utils import disable_torch_init


CASE_MODES = ["no_compression", "full_pruning", "full_merging", "selective_routing"]


def require_file(path: str, name: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Missing {name}: {p}. Please check the path and pass the correct CLI argument.")
    return p


def require_dir_parent(path: str, name: str) -> Path:
    p = Path(path).expanduser()
    parent = p if p.suffix == "" else p.parent
    if not parent.parent.exists():
        raise FileNotFoundError(f"Missing parent directory for {name}: {parent.parent}. Cannot create {parent}.")
    parent.mkdir(parents=True, exist_ok=True)
    return p


def load_queries(query_file: Path):
    text = query_file.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return [loaded]
        if isinstance(loaded, list):
            return loaded
        raise ValueError("Top-level JSON must be a list or dict.")
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {query_file}:{line_no}: {exc}") from exc
        return rows


def resolve_image_path(row, query_file: Path, override_image_path: str = None) -> Path:
    if override_image_path:
        return require_file(override_image_path, "image file")
    image_value = row.get("image")
    if not image_value:
        raise FileNotFoundError(f"Missing image field for question_id={row.get('question_id')}.")
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = query_file.parent / image_path
    return require_file(str(image_path), "image file")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_grid(token_count: int):
    side = int(math_sqrt := round(token_count ** 0.5))
    if side * side == token_count:
        return side, side
    return None


def save_visualization(debug, image_path: Path, out_path: Path, title: str):
    token_state = debug.get("token_state")
    if token_state is None:
        print(f"warning: skip visualization for {title}: no token_state")
        return
    token_state = token_state.detach().cpu().numpy()
    grid = infer_grid(token_state.shape[0])
    if grid is None:
        print(f"warning: skip visualization for {title}: cannot infer square grid from {token_state.shape[0]} tokens")
        return

    h, w = grid
    state_grid = token_state.reshape(h, w)
    color_map = {
        TOKEN_STATE["drop"]: np.array([0.55, 0.55, 0.55, 0.72]),
        TOKEN_STATE["keep"]: np.array([1.0, 0.05, 0.05, 0.72]),
        TOKEN_STATE["merge"]: np.array([0.05, 0.25, 1.0, 0.72]),
        TOKEN_STATE["representative"]: np.array([0.0, 0.9, 0.25, 0.82]),
    }
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for state, color in color_map.items():
        overlay[state_grid == state] = color

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.imshow(Image.fromarray((overlay * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def install_case_encode_images(model, args):
    model._case_debug_queue = []
    model._case_current_mode = "no_compression"

    def case_encode_images(self, images, selection_method="fps"):
        raw_features, _ = self.get_model().get_vision_tower()(images)
        raw_features = raw_features[:, 1:]
        compressed_features = []
        index_masks = []
        batch_debug = []
        for image_features in raw_features:
            compressed, debug = compress_visual_tokens(
                image_features,
                mode=self._case_current_mode,
                budget=args.visual_token_budget,
                basis_token_num=args.basis_token_num,
                return_debug=True,
                weighted_merge=args.weighted_merge,
            )
            projected = self.get_model().mm_projector(compressed.to(raw_features.dtype))
            compressed_features.append(projected)
            index_masks.append(torch.ones(projected.shape[0], dtype=torch.bool, device=projected.device))
            batch_debug.append(debug)
        self._case_debug_queue.append(batch_debug)
        return torch.stack(compressed_features, dim=0), torch.stack(index_masks, dim=0)

    model.encode_images = types.MethodType(case_encode_images, model)


def build_prompt(question: str, model_config, conv_mode: str):
    qs = question
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_one(model, tokenizer, image_processor, args, row, image_path: Path, mode: str):
    model._case_current_mode = mode
    model._case_debug_queue.clear()
    question = row.get("question") or row.get("text")
    if not question:
        raise ValueError(f"Missing question/text field for row: {row}")

    prompt = build_prompt(question, model.config, args.conv_mode)
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor, _ = process_images([image_pil], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    else:
        image_tensor = image_tensor[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    input_ids = input_ids.to(device=args.device, non_blocking=True)
    image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device=args.device, non_blocking=True)

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_pil.size],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.time() - start

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if not model._case_debug_queue:
        raise RuntimeError("No token debug info captured from encode_images.")
    debug = model._case_debug_queue[-1][0]
    return answer, latency, debug


def save_debug(output_dir: Path, row, mode: str, debug):
    case_id = row.get("case_id", "case")
    question_id = row.get("question_id", "question")
    path = output_dir / f"token_debug_{case_id}_{question_id}_{mode}.pt"
    torch.save(debug, path)
    return path


def eval_case_study(args):
    repo_model_path = require_file(os.path.join(args.model_path, "config.json"), "model config").parent
    query_file = require_file(args.query_file, "query file")
    output_dir = require_dir_parent(args.output_dir, "output_dir")

    set_seed(args.seed)
    disable_torch_init()

    queries = load_queries(query_file)
    if not queries:
        raise ValueError(f"No queries found in {query_file}.")

    model_name = get_model_name_from_path(str(repo_model_path))
    device_map = "auto" if args.device == "cuda" else args.device
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(repo_model_path),
        args.model_base,
        model_name,
        llm_pruning=False,
        device=args.device,
        device_map=device_map,
        use_flash_attn=False,
        visual_token_num=576,
    )
    model.eval()
    install_case_encode_images(model, args)

    modes = CASE_MODES if args.compression_mode == "all" else [args.compression_mode]
    predictions_path = output_dir / "predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for row_idx, row in enumerate(queries):
            image_path = resolve_image_path(row, query_file, args.image_path)
            for mode in modes:
                print(f"[{row_idx + 1}/{len(queries)}] {row.get('question_id', row_idx)} | {mode}")
                answer, latency, debug = run_one(model, tokenizer, image_processor, args, row, image_path, mode)
                save_debug(output_dir, row, mode, debug)
                if args.save_visualization and mode != "no_compression":
                    vis_path = output_dir / f"token_vis_{row.get('case_id', 'case')}_{row.get('question_id', row_idx)}_{mode}.png"
                    try:
                        save_visualization(debug, image_path, vis_path, f"{mode} token states")
                    except Exception as exc:
                        print(f"warning: visualization failed for {mode}: {exc}")

                record = {
                    "case_id": row.get("case_id"),
                    "question_id": row.get("question_id"),
                    "category": row.get("category"),
                    "question": row.get("question") or row.get("text"),
                    "answer_ref": row.get("answer_ref"),
                    "image": str(image_path),
                    "method": mode,
                    "visual_token_budget": None if mode == "no_compression" else args.visual_token_budget,
                    "num_visual_tokens_before": debug["original_token_count"],
                    "num_visual_tokens_after": debug["compressed_token_count"],
                    "answer_pred": answer,
                    "latency_sec": latency,
                    "debug_warnings": debug.get("warnings", []),
                }
                pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pred_f.flush()

    print(f"Saved predictions to {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--model-path", type=str, required=True)
    parser.add_argument("--model_base", "--model-base", type=str, default=None)
    parser.add_argument("--query_file", "--query-file", type=str, required=True)
    parser.add_argument("--image_path", "--image-path", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True)
    parser.add_argument("--compression_mode", "--compression-mode", type=str, default="all",
                        choices=CASE_MODES + ["all"])
    parser.add_argument("--visual_token_budget", "--visual-token-budget", type=int, default=128)
    parser.add_argument("--basis_token_num", "--basis-token-num", type=int, default=10)
    parser.add_argument("--save_token_scores", "--save-token-scores", action="store_true")
    parser.add_argument("--save_visualization", "--save-visualization", action="store_true")
    parser.add_argument("--weighted_merge", "--weighted-merge", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv_mode", "--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    eval_case_study(parse_args())
