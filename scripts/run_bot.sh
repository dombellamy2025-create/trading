#!/usr/bin/env bash
# Keep the paper bot running (restart 30 s after any exit) and refresh the HUD data every 60 s.
#   scripts/run_bot.sh
# Safe to run several copies at once: a lock file (logs/bot.lock, logs/hud.lock) lets only one
# copy run the bot / HUD updater; the others wait on standby and take over within ~10 s if the
# active copy dies. This is how the cloud session stays up across its 2-hour job limit
# (two overlapping copies, started about an hour apart).
# Logs: logs/bot.log (bot), logs/supervisor.out (restarts + stdout), logs/hud.out (HUD updater).
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs
STANDBY=75   # flock exit code when another copy holds the lock

(
  while true; do
    flock -n -E $STANDBY logs/hud.lock python dashboard/update_hud.py --loop 60 >> logs/hud.out 2>&1
    sleep 10
  done
) &
HUD_PID=$!
trap 'kill $HUD_PID 2>/dev/null' EXIT

announced=0
while true; do
  flock -n -E $STANDBY logs/bot.lock python -m bot.main >> logs/supervisor.out 2>&1
  rc=$?
  if [ $rc -eq $STANDBY ]; then
    [ $announced -eq 0 ] && echo "$(date -u) supervisor $$ on standby (another copy runs the bot)" >> logs/supervisor.out
    announced=1
    sleep 10
    continue
  fi
  announced=0
  echo "$(date -u) supervisor $$: bot exited with $rc, restarting in 30s" >> logs/supervisor.out
  sleep 30
done
