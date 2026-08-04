#!/bin/bash

# Perform a clean restart of the complete TOS application process tree.
#
# Why this includes the Robot Controller:
# The UI starts Robot Controllers on demand. Restarting Flask alone leaves those
# controllers and their multiprocessing robot workers alive, so old and new
# workers can consume the same RabbitMQ/local queues in parallel.
#
# RabbitMQ and Mosquitto are system services and intentionally remain running.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOS_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$TOS_APP_DIR/logs"
UI_PROCESS_PATTERN='python.*applications/tos_ui/main.py'
CONTROLLER_PROCESS_PATTERN='python.*applications/robot_controller/main.py'
RESTART_LOCK_FILE="/tmp/tos_stack_restart_${USER:-tos}.lock"

mkdir -p "$LOG_DIR"

# Prevent two fast touchscreen taps from running overlapping restart sequences.
exec 8>"$RESTART_LOCK_FILE"
if ! flock -n 8; then
    exit 0
fi

notify_operator() {
    command -v notify-send >/dev/null 2>&1 && notify-send "TOS UI" "$1"
}

# Ask the running controller to stop motion through its normal command path
# before terminating any process. Failure is tolerated because the UI may
# already be unavailable; the operator must still use the robot's normal safety
# controls before invoking a restart while motion is active.
if command -v curl >/dev/null 2>&1; then
    curl --silent --show-error --fail --max-time 4 \
        --request POST \
        --data "message=stop" \
        --data "setup_id=1" \
        http://127.0.0.1:5000/send_command \
        >>"$LOG_DIR/tos_ui_restart.log" 2>&1 || true
    sleep 2
fi

mapfile -t ui_pids < <(pgrep -f "$UI_PROCESS_PATTERN" 2>/dev/null || true)
mapfile -t controller_pids < <(pgrep -f "$CONTROLLER_PROCESS_PATTERN" 2>/dev/null || true)
all_pids=("${ui_pids[@]}" "${controller_pids[@]}")

if ((${#all_pids[@]} > 0)); then
    kill -TERM "${all_pids[@]}" 2>/dev/null || true

    # Give Flask, controllers, and multiprocessing children time to close their
    # sockets and shared memory cleanly. Track the exact captured PIDs so the
    # script never kills an unrelated process that starts later.
    remaining=("${all_pids[@]}")
    for _ in {1..50}; do
        still_running=()
        for pid in "${remaining[@]}"; do
            kill -0 "$pid" 2>/dev/null && still_running+=("$pid")
        done
        remaining=("${still_running[@]}")
        ((${#remaining[@]} == 0)) && break
        sleep 0.1
    done

    if ((${#remaining[@]} > 0)); then
        kill -KILL "${remaining[@]}" 2>/dev/null || true
    fi
fi

nohup "$SCRIPT_DIR/tos_ui.sh" >>"$LOG_DIR/tos_ui_restart.log" 2>&1 &

notify_operator "TOS UI and Robot Controller restarted cleanly."
