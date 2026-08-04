#!/bin/bash

# Restart only the TOS UI process, leaving the robot controller and brokers alone.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOS_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$TOS_APP_DIR/logs"
PROCESS_PATTERN='python.*applications/tos_ui/main.py'

mkdir -p "$LOG_DIR"

mapfile -t ui_pids < <(pgrep -f "$PROCESS_PATTERN" 2>/dev/null || true)
if ((${#ui_pids[@]} > 0)); then
    kill -TERM "${ui_pids[@]}" 2>/dev/null || true

    # Allow the UI a few seconds to shut down cleanly.
    for _ in {1..30}; do
        remaining=()
        for pid in "${ui_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
        done
        ((${#remaining[@]} == 0)) && break
        sleep 0.1
    done

    ((${#remaining[@]} > 0)) && kill -KILL "${remaining[@]}" 2>/dev/null || true
fi

nohup "$SCRIPT_DIR/tos_ui.sh" >>"$LOG_DIR/tos_ui_restart.log" 2>&1 &

if command -v notify-send >/dev/null 2>&1; then
    notify-send "TOS UI" "TOS UI has been restarted."
fi

