#!/usr/bin/env bash
# Upload only what the submission endpoint needs. From PowerShell, inside survival-simulator/:
#   wsl bash deploy/push.sh
# Optional: wsl bash deploy/push.sh user@host:/path/   (default target below)
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-root@172.104.156.87:/root/Nordic-AI-Cup-2026/survival-simulator/}"
rsync -aL --partial-dir=.rsync-partial --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='venv/' --exclude='__pycache__/' \
  --exclude='.env' --exclude='.pytest_cache/' --exclude='.ipynb_checkpoints/' --exclude='.rsync-partial/' \
  --exclude='logs/' --exclude='results/' --exclude='training/' --exclude='tests/' --exclude='agents/' \
  --exclude='external/finalists/' --exclude='*.ipynb' --exclude='*.zip' --exclude='*.md' \
  --exclude='agent_server.py' --exclude='simulation_server.py' --exclude='local_playground.py' \
  --exclude='conftest.py' --exclude='requirements-dev.txt' \
  ./ "$TARGET"
