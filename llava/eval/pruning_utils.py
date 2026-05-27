import ast
import os


def parse_optional_list(value, name):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "none":
            return None
        parsed = ast.literal_eval(value)
    else:
        parsed = value

    if parsed is None:
        return None
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{name} must be a list or None, got {type(parsed).__name__}: {parsed!r}")
    return list(parsed)


def parse_bool_env(value, name):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, on/off; got {value!r}")




def parse_bool_config(args, attr_name, env_name, default):
    value = getattr(args, attr_name, None)
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return parse_bool_env(str(value), env_name)


def get_apet_selection_method(args):
    value = getattr(args, "selection_method", None)
    if value is None:
        value = getattr(args, "apet_selection", None)
    if value is None:
        value = os.environ.get("APET_SELECTION", "apet_error")
    value = str(value).strip().lower()
    aliases = {"error": "apet_error", "apet": "apet_error", "cpqr": "rrqr"}
    value = aliases.get(value, value)
    if value not in {"apet_error", "rrqr"}:
        raise ValueError("selection_method must be 'apet_error' or 'rrqr', got %r" % value)
    return value


def should_use_llm_pruning(args, layer_list):
    explicit = getattr(args, "llm_pruning", None)
    if explicit is not None:
        return explicit

    env_value = os.environ.get("APET_LLM_PRUNING")
    if env_value is not None:
        return parse_bool_env(env_value, "APET_LLM_PRUNING")

    return layer_list is not None


def configure_apet_pruning(model, args, layer_list, image_token_list):
    selection_method = get_apet_selection_method(args)
    rrqr_center = parse_bool_config(args, "rrqr_center", "APET_RRQR_CENTER", True)
    rrqr_normalize = parse_bool_config(args, "rrqr_normalize", "APET_RRQR_NORMALIZE", False)
    rrqr_sort_indices = parse_bool_config(args, "rrqr_sort_indices", "APET_RRQR_SORT_INDICES", True)

    if hasattr(model, "model"):
        model.model.basis_token_num = args.basis_token_num
        model.model.apet_selection_method = selection_method
        model.model.apet_rrqr_center = rrqr_center
        model.model.apet_rrqr_normalize = rrqr_normalize
        model.model.apet_rrqr_sort_indices = rrqr_sort_indices

    if type(model).__name__ != "LlavaLlamaForCausalLM_X":
        return

    if layer_list is None:
        raise ValueError("LLM pruning requires --layer_list.")
    if image_token_list is None:
        raise ValueError("LLM pruning requires --image_token_list.")

    model.model.layer_list = layer_list
    model.model.image_token_list = image_token_list
    model.model.image_token_list.insert(0, args.visual_token_num)
