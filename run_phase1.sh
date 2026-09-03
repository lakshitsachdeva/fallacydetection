#!/bin/bash
# FallacyLens Phase 1 - launcher. Starts the overnight run in a detached tmux
# session named "fallacylens" so it survives laptop/SSH disconnects.
# Usage: bash run_phase1.sh
# Monitor: tmux attach -t fallacylens   (detach again with Ctrl-b d)
# Logs:    tail -f ~/fallacydetection/logs/run_*.log
set -eu
cd "$HOME/fallacydetection"
mkdir -p logs
LOG="$HOME/fallacydetection/logs/run_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t fallacylens 2>/dev/null; then
  echo "tmux session 'fallacylens' already exists. Attach with: tmux attach -t fallacylens"
  exit 1
fi

chmod +x run_inner.sh
tmux new-session -d -s fallacylens "bash $HOME/fallacydetection/run_inner.sh '$LOG'"
echo "Started. Log: $LOG"
echo "Watch with: tail -f '$LOG'"
