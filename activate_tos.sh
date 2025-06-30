#!/bin/bash
# TOS Environment Activation Script

# Get the directory where this script is located
TOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TOS_ROOT"

# Activate conda environment (using detected conda installation)
source "/home/tos-pc3/anaconda3/etc/profile.d/conda.sh"
conda activate TOS

# Add current directory to Python path
export PYTHONPATH="$TOS_ROOT:$PYTHONPATH"

# Set environment variables for TOS
export TOS_ROOT="$TOS_ROOT"
export TOS_CONFIG_PATH="$TOS_ROOT/config"
export TOS_LOG_PATH="$TOS_ROOT/logs"
export TOS_DATA_PATH="$TOS_ROOT/../tos_app_data"

# Set RabbitMQ environment variables (for system-wide installation)
export RABBITMQ_DEFAULT_USER=admin
export RABBITMQ_DEFAULT_PASS=admin

    echo "TOS environment activated using anaconda!"
    echo "Conda installation: /home/tos-pc3/anaconda3"
    echo "Conda environment: /home/tos-pc3/anaconda3/envs/TOS" 
    echo "TOS root directory: $TOS_ROOT"
    echo "Python path: $PYTHONPATH"
    echo ""
    echo "Environment isolation info:"
    echo "  All dependencies installed in conda environment"
    echo "  CMAKE tools: $CONDA_PREFIX/bin/cmake"
    echo "  Libraries: $CONDA_PREFIX/lib"
    echo "  Headers: $CONDA_PREFIX/include"
    echo ""
    echo "RabbitMQ commands (system-wide installation):"
    echo "  Start RabbitMQ: sudo systemctl start rabbitmq-server"
    echo "  Stop RabbitMQ: sudo systemctl stop rabbitmq-server"
    echo "  Check status: sudo systemctl status rabbitmq-server"
    echo "  Management UI: http://localhost:15672 (admin/admin)"
    echo ""
    echo "To run TOS applications:"
    echo "  Robot Controller: python applications/robot_controller/main.py"
    echo "  TOS UI: python applications/tos_ui/main.py"
    echo ""
    echo "To deactivate: conda deactivate"
