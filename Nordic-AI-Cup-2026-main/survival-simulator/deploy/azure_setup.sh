#!/usr/bin/env bash
# One-time setup of the submission endpoint on an Azure Ubuntu VM (22.04/24.04).
#
#   git clone https://github.com/sjoeen/Nordic-AI-cup-2026.git ~/Nordic-AI-cup-2026
#   bash ~/Nordic-AI-cup-2026/Nordic-AI-Cup-2026-main/survival-simulator/deploy/azure_setup.sh '<PASTE_API_KEY_HERE>'
#
# Also required in the Azure portal: Networking -> add inbound port rule TCP 9052.
# Re-run safely after `git pull` to restart with new code.
set -euo pipefail

API_KEY="${1:-}"
PORT="${PORT:-9052}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE=/etc/survival-agent.env
SERVICE=/etc/systemd/system/survival-agent.service

sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/deploy/requirements-server.txt"

if [ -n "$API_KEY" ] || [ ! -f "$ENV_FILE" ]; then
  printf 'NAIC_API_KEY=%s\nPORT=%s\nREQUIRE_API_KEY=0\nINJECT_FAILURE=0\n' "$API_KEY" "$PORT" | sudo tee "$ENV_FILE" > /dev/null
  sudo chmod 600 "$ENV_FILE"
fi

sudo tee "$SERVICE" > /dev/null <<UNIT
[Unit]
Description=Nordic AI Cup survival simulator submission endpoint
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/python submission_server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

# Ubuntu images on Azure usually have ufw inactive; open the port if it is active.
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then sudo ufw allow "$PORT/tcp"; fi

sudo systemctl daemon-reload
sudo systemctl enable survival-agent
sudo systemctl restart survival-agent
sleep 3
curl -s "http://127.0.0.1:$PORT/" && echo
curl -s "http://127.0.0.1:$PORT/api" && echo
echo "Logs: journalctl -u survival-agent -f"
