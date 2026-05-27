#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-}"
if [[ -z "$LMMS_EVAL_REPO" ]]; then
    echo "Set LMMS_EVAL_REPO to your lmms-eval checkout, e.g.:" >&2
    echo "  export LMMS_EVAL_REPO=/root/autodl-tmp/mllm_token_compression/lmms-eval-main" >&2
    exit 2
fi

MODELS_DIR="$LMMS_EVAL_REPO/lmms_eval/models"
INIT_FILE="$MODELS_DIR/__init__.py"
ADAPTER_SRC="$REPO_ROOT/lmms_eval_adapters/apet_llava.py"
ADAPTER_DST="$MODELS_DIR/apet_llava.py"

if [[ ! -d "$MODELS_DIR" ]]; then
    echo "lmms-eval models directory not found: $MODELS_DIR" >&2
    exit 2
fi

cp "$ADAPTER_SRC" "$ADAPTER_DST"

python3 - "$INIT_FILE" <<'PYEDIT'
from pathlib import Path
import sys

init_file = Path(sys.argv[1])
text = init_file.read_text(encoding="utf-8")

old_block = """
# Optional local ApET/LLaVA adapter.
try:
    from .apet_llava import *  # noqa: F401,F403
except Exception:
    pass
"""
text = text.replace(old_block, "")

manifest_block = """
# Optional local ApET/LLaVA adapter.
# This lmms-eval version uses registry_v2, so importing a legacy
# @register_model module is not enough; register an explicit manifest.
try:
    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="apet_llava",
            simple_class_path="lmms_eval.models.apet_llava.ApETLlava",
        ),
        overwrite=True,
    )
    AVAILABLE_MODELS["apet_llava"] = "ApETLlava"
except Exception as exc:  # pragma: no cover
    logger.warning(f"Failed to register optional apet_llava model: {exc}")
"""

if 'model_id="apet_llava"' not in text and "model_id='apet_llava'" not in text:
    text = text.rstrip() + "\n" + manifest_block

init_file.write_text(text, encoding="utf-8")
PYEDIT

echo "Installed apet_llava adapter:"
echo "  $ADAPTER_DST"
echo "Registered registry_v2 manifest in:"
echo "  $INIT_FILE"
echo
echo "Make sure this repo is on PYTHONPATH when running lmms-eval:"
echo "  export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH"
