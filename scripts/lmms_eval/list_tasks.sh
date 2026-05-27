#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c "import lmms_eval" >/dev/null 2>&1; then
    echo "lmms_eval is not importable by PYTHON_BIN=$PYTHON_BIN." >&2
    echo "Install lmms-eval first, or set PYTHON_BIN to the environment that has it." >&2
    exit 2
fi

"$PYTHON_BIN" -m lmms_eval --tasks list
