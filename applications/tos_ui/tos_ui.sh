#!/bin/bash
# TOS UI Launcher Script

set -u

# Get the absolute path of the tos_app directory (parent of applications)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOS_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Only one Flask UI may run at a time. Multiple UI processes each poll RabbitMQ
# and may race while auto-starting the same Robot Controller. ``flock`` keeps
# the lock for the lifetime of this shell (and therefore the Python UI below).
UI_LOCK_FILE="/tmp/tos_ui_${USER:-tos}.lock"
exec 9>"$UI_LOCK_FILE"
if ! flock -n 9; then
    # The Flask server already owns the lock. A desktop launcher click should
    # still restore the UI when its browser window was closed.
    if command -v wmctrl >/dev/null 2>&1 \
        && window_id="$(wmctrl -l | awk 'tolower($0) ~ /tos ui.*firefox/ {print $1; exit}')" \
        && [[ -n "$window_id" ]]; then
        wmctrl -ia "$window_id"
    elif command -v firefox >/dev/null 2>&1; then
        firefox --new-window "http://127.0.0.1:5000" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:5000" >/dev/null 2>&1 &
    fi
    exit 0
fi

# Detect conda installation
detect_conda_base() {
    local conda_path=$(which conda 2>/dev/null)
    if [[ -n "$conda_path" ]]; then
        dirname "$(dirname "$conda_path")"
    else
        # Look for common conda installation locations
        local common_paths=(
            "$HOME/anaconda3"
            "$HOME/miniconda3"
            "$HOME/miniforge3"
            "$HOME/mambaforge3"
        )
        
        for path in "${common_paths[@]}"; do
            if [[ -d "$path" && -f "$path/bin/conda" ]]; then
                echo "$path"
                return
            fi
        done
        
        echo "$HOME/miniconda3"  # fallback
    fi
}

# Get conda base directory
CONDA_BASE=$(detect_conda_base)

# Activate conda environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate TOS

# Set environment variables
# Desktop launchers normally do not define PYTHONPATH. With ``set -u``, reading
# an unset variable terminates the launcher before Python can start. Append an
# existing value only when it is present.
export PYTHONPATH="$TOS_APP_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOS_ROOT="$TOS_APP_DIR"
export TOS_CONFIG_PATH="$TOS_APP_DIR/config"
export TOS_LOG_PATH="$TOS_APP_DIR/logs"
export TOS_DATA_PATH="$TOS_APP_DIR/../tos_app_data"

# Change to TOS root directory and run the UI
cd "$TOS_APP_DIR"
exec python applications/tos_ui/main.py
