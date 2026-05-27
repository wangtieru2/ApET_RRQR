import copy
import warnings
from datetime import timedelta
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.eval.pruning_utils import configure_apet_pruning, parse_optional_list
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except Exception as exc:  # pragma: no cover - evaluated inside lmms-eval runtime
    eval_logger.debug(f"LLaVA/ApET import failed. Install this repo on PYTHONPATH first. Error: {exc}")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean-like value, got {value!r}")


def _first_text(contexts):
    return contexts[0] if isinstance(contexts, list) else contexts


def _flatten_visuals(visuals):
    if not visuals:
        return []
    if any(item is None for item in visuals):
        return []
    flattened = []
    for item in visuals:
        if isinstance(item, list):
            flattened.extend([x for x in item if x is not None])
        elif item is not None:
            flattened.append(item)
    return flattened


def _process_images(images, image_processor, config, device):
    processed = process_images(images, image_processor, config)
    if isinstance(processed, tuple):
        image_tensor = processed[0]
    else:
        image_tensor = processed

    if isinstance(image_tensor, list):
        return [item.to(dtype=torch.float16, device=device) for item in image_tensor]
    return image_tensor.to(dtype=torch.float16, device=device)


@register_model("apet_llava")
class ApETLlava(lmms):
    """LLaVA-1.5/ApET adapter for lmms-eval.

    This is intentionally an evaluation-only adapter. It keeps the ApET model
    implementation in this repository untouched and only wires the existing
    pruning configuration into the lmms-eval model API.
    """

    def __init__(
        self,
        pretrained: str = "/root/autodl-tmp/models/llava_v15_7b",
        model_base: Optional[str] = None,
        model_name: Optional[str] = None,
        conv_template: str = "vicuna_v1",
        device: Optional[str] = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        truncate_context: bool = False,
        layer_list: Optional[str] = "[16]",
        image_token_list: Optional[str] = "[64]",
        visual_token_num: Union[int, str] = 64,
        basis_token_num: Union[int, str] = 10,
        llm_pruning: Union[bool, str, None] = False,
        selection_method: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected lmms-eval model_args for apet_llava: {kwargs}")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = _as_bool(use_cache)
        self.truncate_context = _as_bool(truncate_context)

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        model_name = model_name or get_model_name_from_path(pretrained)
        parsed_layer_list = parse_optional_list(layer_list, "layer_list")
        parsed_image_token_list = parse_optional_list(image_token_list, "image_token_list")
        visual_token_num = int(visual_token_num)
        basis_token_num = int(basis_token_num)
        llm_pruning = bool(_as_bool(llm_pruning))

        selected_indices = []
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            model_base,
            model_name,
            llm_pruning,
            device_map=self.device_map,
            visual_token_num=visual_token_num,
            selected_indices=selected_indices,
        )
        self._config = self._model.config
        self._model.eval()

        pruning_args = SimpleNamespace(
            basis_token_num=basis_token_num,
            visual_token_num=visual_token_num,
            selection_method=selection_method,
        )
        configure_apet_pruning(self._model, pruning_args, parsed_layer_list, parsed_image_token_list)
        if type(self._model).__name__ == "LlavaLlamaForCausalLM_X":
            self._model.model.visual_id = 0

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]:
                raise ValueError("Only DDP, FSDP, and DeepSpeed are supported for multi-GPU lmms-eval.")
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                deepspeed_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **deepspeed_kwargs)
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._world_size = 1
            eval_logger.info("Using single process with device_map=auto")
        else:
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1
            eval_logger.info(f"Using single device: {self._device}")

        eval_logger.info(
            "Loaded apet_llava with "
            f"pretrained={pretrained}, conv_template={conv_template}, "
            f"layer_list={parsed_layer_list}, image_token_list={parsed_image_token_list}, "
            f"visual_token_num={visual_token_num}, basis_token_num={basis_token_num}, "
            f"selection_method={selection_method or 'apet_error'}, llm_pruning={llm_pruning}"
        )

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        tokens = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return tokens[-left_truncate_len:] if left_truncate_len else tokens

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except TypeError:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("apet_llava currently supports generation-style lmms-eval tasks only.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            raw_visuals = doc_to_visual(doc)
            visuals = _flatten_visuals(raw_visuals if isinstance(raw_visuals, list) else [raw_visuals])
            context = _first_text(contexts)

            if visuals and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
                context = image_tokens + "\n" + context

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)

            image_tensor = None
            image_sizes = None
            if visuals:
                image_tensor = _process_images(visuals, self._image_processor, self._config, self.device)
                image_sizes = [visual.size for visual in visuals]

            kwargs = copy.deepcopy(gen_kwargs)
            until = kwargs.pop("until", [self.tok_decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            if "image_aspect_ratio" in kwargs:
                self._config.image_aspect_ratio = kwargs.pop("image_aspect_ratio")

            max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
            temperature = float(kwargs.pop("temperature", 0))
            top_p = kwargs.pop("top_p", None)
            num_beams = int(kwargs.pop("num_beams", 1))

            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=self.use_cache,
                )

            text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            for stop in until:
                if stop:
                    text = text.split(stop)[0].strip()
            outputs.append(text)

            if hasattr(self, "cache_hook"):
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            pbar.update(1)

        pbar.close()
        return outputs

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("apet_llava does not implement multi-round generation.")
