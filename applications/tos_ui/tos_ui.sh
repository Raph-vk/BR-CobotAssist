#!/bin/bash
# TOS UI Launcher Script

# Get the absolute path of the tos_app directory (parent of applications)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOS_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
export PYTHONPATH="$TOS_APP_DIR:$PYTHONPATH"
export TOS_ROOT="$TOS_APP_DIR"
export TOS_CONFIG_PATH="$TOS_APP_DIR/config"
export TOS_LOG_PATH="$TOS_APP_DIR/logs"
export TOS_DATA_PATH="$TOS_APP_DIR/../tos_app_data"

# Change to TOS root directory and run the UI
cd "$TOS_APP_DIR"
python applications/tos_ui/main.py

