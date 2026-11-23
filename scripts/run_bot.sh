#!/usr/bin/env bash
# Keep the paper bot running (restart 30 s after any exit) and refresh the HUD data every 60 s.
#   scripts/run_bot.sh
# Logs: logs/bot.log (bot), logs/supervisor.out (restarts + stdout), logs/hud.out (HUD updater).
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs

python dashboard/update_hud.py --loop 60 >> logs/hud.out 2>&1 &
HUD_PID=$!
trap 'kill $HUD_PID 2>/dev/null' EXIT

while true; do
  python -m bot.main >> logs/supervisor.out 2>&1
  echo "$(date -u) bot exited with $?, restarting in 30s" >> logs/supervisor.out
  sleep 30
done
