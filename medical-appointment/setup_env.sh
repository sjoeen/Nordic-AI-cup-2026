#!/usr/bin/env bash
# Create ./.venv and install the serving + ML stack. Idempotent.
# Usage: bash setup_env.sh            (CPU torch, default)
#        TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip wheel >/dev/null
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --extra-index-url "$TORCH_INDEX" -r requirements-ml.txt
.venv/bin/python -m pip freeze > .venv/pip-freeze.txt
echo "environment ready: $(.venv/bin/python --version)"
