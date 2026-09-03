#!/bin/bash
# FallacyLens Phase 1 - inner runner (executed inside tmux).
# Runs the Phase 1 task with the kimi CLI agent. If the agent exits before
# the completion marker exists, re-prompt it to continue (state lives on disk
# via PROGRESS.md), up to MAX_ROUNDS rounds.
LOG="$1"
cd "$HOME/fallacydetection" || exit 1
export PATH="$HOME/.local/bin:$HOME/.kimi-code/bin:$PATH"
# Launchpad-managed kmi (proxied, pre-authenticated). Runs with auto
# permission in -p mode by design (non-interactive never asks).
KIMI="$HOME/.local/bin/kmi"
MAX_ROUNDS=8

for i in $(seq 1 "$MAX_ROUNDS"); do
  if [ -f report/PHASE1_COMPLETE.md ]; then
    echo "=== COMPLETION MARKER FOUND, stopping at round $i $(date) ===" >> "$LOG"
    break
  fi
  if [ "$i" -eq 1 ]; then
    PROMPT="$(cat PHASE1_TASK.md)"
  else
    PROMPT="You are continuing an in-progress, multi-hour task in $HOME/fallacydetection. First read PHASE1_TASK.md (the full task specification) and PROGRESS.md (your own running progress notes), then inspect the current state of the repo. Continue from exactly where things left off. Do NOT redo completed work. Keep going until every deliverable in PHASE1_TASK.md is genuinely complete and verified, then write report/PHASE1_COMPLETE.md and finish."
  fi
  {
    echo ""
    echo "==================== ROUND $i START $(date) ===================="
    "$KIMI" -p "$PROMPT"
    echo "==================== ROUND $i EXIT code=$? $(date) ===================="
  } >> "$LOG" 2>&1
done

echo "=== RUNNER FINISHED $(date) ===" >> "$LOG"
