#!/bin/bash

###############################################################################
# TOS Application Installer Script
# 
# This script installs all dependencies needed to run the TOS robotic application.
# Excludes Interbotix and ROS modules as requested.
#
# Usage: chmod +x install_tos.sh && ./install_tos.sh [OPTIONS]
#
# Options:
#   --env-name <name>         Set the conda environment name (default: TOS)
#   --force-env-recreate      Force recreation of environment if it exists
#   --keep-existing-env       Keep existing environment if it exists
#   --help                    Show this help message
#
# Author: TOS Development Team
# Date: 2025-06-26
###############################################################################

set -e  # Exit on any error

# Default options
FORCE_ENV_RECREATE=false
KEEP_EXISTING_ENV=false

# Set the name of the conda environment (change this variable to customize)
ENV_NAME="TOS"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --force-env-recreate)
            FORCE_ENV_RECREATE=true
            shift
            ;;
        --keep-existing-env)
            KEEP_EXISTING_ENV=true
            shift
            ;;
        --help)
            echo "TOS Application Installer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-name <name>         Set the conda environment name (default: TOS)"
            echo "  --force-env-recreate      Force recreation of environment if it exists"
            echo "  --keep-existing-env       Keep existing environment if it exists" 
            echo "  --help                    Show this help message"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate conflicting options
if [[ "$FORCE_ENV_RECREATE" == "true" && "$KEEP_EXISTING_ENV" == "true" ]]; then
    log_error "Cannot use both --force-env-recreate and --keep-existing-env"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper function to find TOS root directory
find_tos_root() {
    local current_dir="$(pwd)"
    local search_dir="$current_dir"
    
    # Look for TOS root indicators in current directory and parent directories
    while [[ "$search_dir" != "/" ]]; do
        if [[ -f "$search_dir/applications/tos_ui/main.py" ]] && [[ -f "$search_dir/config/config.yaml" ]]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    
    # Not found
    return 1
}

# Read and validate TOS configuration
read_tos_config() {
    log_info "Reading TOS configuration..."
    
    # Get the absolute path to the TOS application root
    local tos_root
    if ! tos_root=$(find_tos_root); then
        log_error "Cannot find TOS application root directory!"
        log_error "Current directory: $(pwd)"
        log_error "Please ensure you're running this script from within the TOS application directory tree."
        exit 1
    fi
    
    export TOS_ROOT="$tos_root"
    export CONFIG_FILE="$tos_root/config/config.yaml"
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Found TOS root: $TOS_ROOT"
    log_success "Using config file: $CONFIG_FILE"
}

# Parse YAML configuration using Python
parse_config() {
    log_info "Parsing configuration file..."
    
    # Create a temporary Python script to parse YAML
    local parse_script="/tmp/parse_tos_config.py"
    cat > "$parse_script" << 'EOF'
import yaml
import sys
import json

def parse_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract relevant information
        result = {
            'active_setups': config.get('hardware', {}).get('active_setups', []),
            'setups': {},
            'current_setup': config.get('setup', 'setup_1'),
            'rabbitmq_host': config.get('rabbitmq', {}).get('host', 'auto')
        }
        
        # Extract setup information
        hardware = config.get('hardware', {})
        for setup_name in result['active_setups']:
            if setup_name in hardware:
                setup_data = hardware[setup_name]
                result['setups'][setup_name] = {
                    'setup_id': setup_data.get('setup_id'),
                    'tos_ui_host': setup_data.get('tos_ui_host', False),
                    'ip_address': setup_data.get('ip_address', 'localhost')
                }
        
        return result
    except Exception as e:
        print(f"Error parsing config: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python parse_config.py <config_file>", file=sys.stderr)
        sys.exit(1)
    
    result = parse_config(sys.argv[1])
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)
EOF
    
    # Parse the configuration using system Python (should have PyYAML)
    local config_json
    if ! config_json=$(python3 "$parse_script" "$CONFIG_FILE" 2>/dev/null); then
        log_warning "Failed to parse configuration file. Installing PyYAML and retrying..."
        
        # Try to install PyYAML with pip3
        if command -v pip3 &> /dev/null; then
            pip3 install --user PyYAML || {
                log_error "Failed to install PyYAML. Please install it manually: pip3 install PyYAML"
                exit 1
            }
            
            # Retry parsing
            if ! config_json=$(python3 "$parse_script" "$CONFIG_FILE"); then
                log_error "Failed to parse configuration file even after installing PyYAML"
                exit 1
            fi
        else
            log_error "Python3 or pip3 not found. Cannot parse YAML configuration."
            exit 1
        fi
    fi
    
    # Store the parsed configuration
    echo "$config_json" > /tmp/tos_config.json
    
    rm "$parse_script"
    log_success "Configuration parsed successfully"
}

# Validate configuration and determine installation type
validate_config() {
    log_info "Validating configuration..."
    
    local config_json="/tmp/tos_config.json"
    
    # Read parsed configuration
    local active_setups=$(jq -r '.active_setups[]' "$config_json" 2>/dev/null)
    local setup_count=$(jq -r '.active_setups | length' "$config_json" 2>/dev/null)
    
    if [[ -z "$active_setups" ]] || [[ "$setup_count" == "0" ]]; then
        log_error "No active setups found in configuration"
        log_error "Please ensure 'hardware.active_setups' is properly configured"
        exit 1
    fi
    
    log_info "Found $setup_count active setup(s): $(echo $active_setups | tr '\n' ' ')"
    
    # Check for UI hosts
    local ui_hosts=()
    for setup in $active_setups; do
        local is_ui_host=$(jq -r ".setups[\"$setup\"].tos_ui_host" "$config_json")
        if [[ "$is_ui_host" == "true" ]]; then
            ui_hosts+=("$setup")
        fi
    done
    
    if [[ ${#ui_hosts[@]} -eq 0 ]]; then
        log_error "No UI host found! At least one setup must have 'tos_ui_host: True'"
        log_error "Please configure at least one setup as the UI host in the config"
        exit 1
    elif [[ ${#ui_hosts[@]} -gt 1 ]]; then
        log_error "Multiple UI hosts found: ${ui_hosts[*]}"
        log_error "Only one setup should have 'tos_ui_host: True'"
        log_error "Please fix the configuration before proceeding"
        exit 1
    fi
    
    export UI_HOST_SETUP="${ui_hosts[0]}"
    export SETUP_COUNT="$setup_count"
    
    log_success "Configuration validation passed"
    log_info "UI Host Setup: $UI_HOST_SETUP"
}

# Let user choose which setup this installation represents
choose_setup() {
    log_info "Determining setup selection..."
    
    local config_json="/tmp/tos_config.json"
    local active_setups=$(jq -r '.active_setups[]' "$config_json")
    
    if [[ "$SETUP_COUNT" == "1" ]]; then
        # Only one setup, use it automatically
        local single_setup=$(echo "$active_setups" | head -n1)
        export SELECTED_SETUP="$single_setup"
        log_info "Single setup configuration detected, using: $SELECTED_SETUP"
    else
        # Multiple setups, let user choose
        echo ""
        log_info "Multiple setups detected. Please choose which setup this PC represents:"
        echo ""
        
        local setup_array=()
        local counter=1
        
        for setup in $active_setups; do
            local setup_id=$(jq -r ".setups[\"$setup\"].setup_id" "$config_json")
            local ip_address=$(jq -r ".setups[\"$setup\"].ip_address" "$config_json")
            local is_ui_host=$(jq -r ".setups[\"$setup\"].tos_ui_host" "$config_json")
            local role="Robot Controller"
            
            if [[ "$is_ui_host" == "true" ]]; then
                role="UI Host + Robot Controller"
            fi
            
            setup_array+=("$setup")
            echo "$counter) $setup (ID: $setup_id, IP: $ip_address, Role: $role)"
            ((counter++))
        done
        
        echo ""
        while true; do
            read -p "Choose setup (1-${#setup_array[@]}): " choice
            
            if [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -le "${#setup_array[@]}" ]]; then
                export SELECTED_SETUP="${setup_array[$((choice-1))]}"
                break
            else
                log_error "Invalid choice. Please enter a number between 1 and ${#setup_array[@]}"
            fi
        done
    fi
    
    # Determine if this setup is the UI host
    local is_ui_host=$(jq -r ".setups[\"$SELECTED_SETUP\"].tos_ui_host" "$config_json")
    export IS_UI_HOST="$is_ui_host"
    
    # Get RabbitMQ host (IP of the UI host setup)
    local rabbitmq_host_ip=$(jq -r ".setups[\"$UI_HOST_SETUP\"].ip_address" "$config_json")
    export RABBITMQ_HOST_IP="$rabbitmq_host_ip"
    
    log_success "Setup selection complete"
    log_info "Selected Setup: $SELECTED_SETUP"
    log_info "Is UI Host: $IS_UI_HOST"
    log_info "RabbitMQ Host IP: $RABBITMQ_HOST_IP"
}

# Update configuration file with selected setup
update_config_file() {
    log_info "Updating configuration file with selected setup..."
    
    # Create a Python script to update the YAML file
    local update_script="/tmp/update_tos_config.py"
    cat > "$update_script" << EOF
import yaml
import sys

def update_config(config_file, selected_setup):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update the main setup field
        config['setup'] = selected_setup
        
        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("Configuration updated successfully")
        return True
    except Exception as e:
        print(f"Error updating config: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python update_config.py <config_file> <selected_setup>", file=sys.stderr)
        sys.exit(1)
    
    success = update_config(sys.argv[1], sys.argv[2])
    if not success:
        sys.exit(1)
EOF
    
    # Update the configuration
    if python3 "$update_script" "$CONFIG_FILE" "$SELECTED_SETUP"; then
        log_success "Configuration file updated with setup: $SELECTED_SETUP"
    else
        log_error "Failed to update configuration file"
        exit 1
    fi
    
    rm "$update_script"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

# Check Linux distribution
check_linux_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
        log_info "Detected OS: $OS $VER"
    else
        log_error "Cannot determine Linux distribution"
        exit 1
    fi
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get upgrade -y
    elif command -v yum &> /dev/null; then
        sudo yum update -y
    elif command -v pacman &> /dev/null; then
        sudo pacman -Syu --noconfirm
    else
        log_error "Unsupported package manager"
        exit 1
    fi
    
    log_success "System packages updated"
}

# Install minimal system dependencies (only what's absolutely required)
install_system_deps() {
    log_info "Installing minimal system dependencies..."
    
    # Only essential system packages that cannot be installed in conda
    local packages=(
        "build-essential"  # Compiler toolchain
        "wget"            # For downloading files
        "curl"            # For downloading files
        "git"             # Version control (could be conda, but system is more reliable)
        "pkg-config"      # Package configuration
        "libusb-1.0-0-dev"  # USB development headers (hardware access)
        "libudev-dev"     # Device management headers (hardware access)
        "v4l-utils"       # Video4Linux utilities (camera access)
        "jq"              # JSON parsing utility
        "python3-yaml"    # Python YAML library for config parsing
    )
    
    if command -v apt-get &> /dev/null; then
        for package in "${packages[@]}"; do
            log_info "Installing essential system package: $package..."
            sudo apt-get install -y "$package" || log_warning "Failed to install $package"
        done
    else
        log_error "Only apt-based systems are currently supported"
        exit 1
    fi
    
    log_success "Minimal system dependencies installed"
}

# Detect and configure conda installation
detect_conda() {
    if command -v conda &> /dev/null; then
        log_info "Conda installation detected, determining type and path..."
        
        # Get conda info to determine installation path and type
        local conda_info=$(conda info --base 2>/dev/null)
        
        if [[ -n "$conda_info" ]]; then
            export CONDA_BASE_PATH="$conda_info"
            log_info "Conda base path: $CONDA_BASE_PATH"
            
            # Determine if it's Anaconda or Miniconda
            if [[ "$conda_info" == *"anaconda"* ]]; then
                export CONDA_TYPE="anaconda"
                log_info "Detected Anaconda installation"
            elif [[ "$conda_info" == *"miniconda"* ]]; then
                export CONDA_TYPE="miniconda"
                log_info "Detected Miniconda installation"
            else
                # Try to determine by checking directories
                if [[ -d "$conda_info" ]]; then
                    if find "$conda_info" -maxdepth 1 -name "*anaconda*" -type d | grep -q .; then
                        export CONDA_TYPE="anaconda"
                        log_info "Detected Anaconda installation (by directory scan)"
                    else
                        export CONDA_TYPE="miniconda"
                        log_info "Detected Miniconda installation (by directory scan)"
                    fi
                else
                    export CONDA_TYPE="miniconda"
                    log_warning "Could not determine conda type, assuming Miniconda"
                fi
            fi
            
            # Set conda paths based on detected installation
            export CONDA_BIN_PATH="$CONDA_BASE_PATH/bin"
            export CONDA_PROFILE_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"
            export PATH="$CONDA_BIN_PATH:$PATH"
            
            log_success "Using existing $CONDA_TYPE installation at: $CONDA_BASE_PATH"
            return 0
        fi
    fi
    
    # No conda found
    return 1
}

# Install Miniconda if not present
install_miniconda() {
    if detect_conda; then
        return 0
    fi
    
    log_info "Installing Miniconda..."
    
    local miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    local installer="/tmp/miniconda.sh"
    
    wget -q "$miniconda_url" -O "$installer"
    chmod +x "$installer"
    
    # Install Miniconda silently
    "$installer" -b -p "$HOME/miniconda3"
    
    # Set conda environment variables for new installation
    export CONDA_BASE_PATH="$HOME/miniconda3"
    export CONDA_TYPE="miniconda"
    export CONDA_BIN_PATH="$CONDA_BASE_PATH/bin"
    export CONDA_PROFILE_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"
    
    # Initialize conda
    "$CONDA_BIN_PATH/conda" init bash
    
    # Add conda to PATH for current session
    export PATH="$CONDA_BIN_PATH:$PATH"
    
    rm "$installer"
    log_success "Miniconda installed at: $CONDA_BASE_PATH"
}

# Create and setup conda environment
setup_conda_env() {
    log_info "Setting up TOS conda environment..."
    
    # Make sure conda is in PATH (use detected path)
    export PATH="$CONDA_BIN_PATH:$PATH"
    
    # Accept conda Terms of Service before creating environment
    log_info "Checking and accepting conda Terms of Service..."
    
    # Accept ToS for the main channels that are commonly needed
    conda config --set channel_priority strict 2>/dev/null || true
    
    # Accept ToS for main conda channels
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda tos accept --override-channels --channel conda-forge 2>/dev/null || true
    
    log_success "Conda Terms of Service accepted"
    
    # Check if TOS environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        log_warning "TOS conda environment already exists!"
        
        if [[ "$FORCE_ENV_RECREATE" == "true" ]]; then
            log_info "Force recreating TOS environment (--force-env-recreate flag set)..."
            conda env remove -n "${ENV_NAME}" -y
            log_info "Creating new TOS environment..."
            conda create -n "${ENV_NAME}" python=3.9 -y
            log_success "New TOS conda environment created using $CONDA_TYPE"
        elif [[ "$KEEP_EXISTING_ENV" == "true" ]]; then
            log_info "Keeping existing TOS environment (--keep-existing-env flag set)..."
            log_success "Using existing TOS conda environment with $CONDA_TYPE"
        else
            # Interactive mode - ask user
            echo "Do you want to:"
            echo "1) Remove existing environment and create a new one"
            echo "2) Keep existing environment and continue with package installation"
            echo "3) Exit installation"
            read -p "Choose option (1/2/3): " choice
            
            case $choice in
                1)
                    log_info "Removing existing TOS environment..."
                    conda env remove -n "${ENV_NAME}" -y
                    log_info "Creating new TOS environment..."
                    conda create -n "${ENV_NAME}" python=3.9 -y
                    log_success "New TOS conda environment created using $CONDA_TYPE"
                    ;;
                2)
                    log_info "Keeping existing TOS environment..."
                    log_success "Using existing TOS conda environment with $CONDA_TYPE"
                    ;;
                3)
                    log_error "Installation cancelled by user"
                    exit 1
                    ;;
                *)
                    log_error "Invalid choice. Installation cancelled."
                    exit 1
                    ;;
            esac
        fi
    else
        # Create environment with Python 3.9
        log_info "Creating new TOS environment..."
        conda create -n "${ENV_NAME}" python=3.9 -y
        log_success "TOS conda environment created using $CONDA_TYPE"
    fi
    
    # Disable automatic conda base environment activation
    conda config --set auto_activate false
}

# Install Python packages and conda dependencies
install_python_packages() {
    log_info "Installing Python packages and conda dependencies in TOS environment..."
    
    # Activate conda environment using detected path
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"
    
    # Install development tools via conda (instead of system packages)
    log_info "Installing development tools via conda..."
    conda install -y cmake make ninja pkg-config
    
    # Install libraries via conda (instead of system packages) 
    log_info "Installing libraries via conda..."
    conda install -y eigen openssl
    
    # Install GUI and graphics libraries via conda
    log_info "Installing GUI and graphics libraries via conda..."
    conda install -y -c conda-forge gtk3 glfw mesa-libgl-devel-cos6-x86_64 mesa-dri-drivers-cos6-x86_64
    
    # Core scientific packages
    log_info "Installing core scientific packages..."
    conda install -y numpy=1.26.4 scipy matplotlib pandas scikit-learn
    
    # Install packages via pip
    log_info "Installing packages via pip..."
    pip install --upgrade pip
    
    # Core application dependencies
    pip install \
        pyyaml \
        pika \
        h5py \
        opencv-python \
        Pillow \
        tqdm \
        psutil \
        flask \
        flask-cors \
        requests \
        websocket-client \
        ipython
    
    # PyTorch (CPU version - can be changed to GPU if needed)
    log_info "Installing PyTorch..."
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    # pip install torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    # Intel RealSense SDK
    log_info "Installing Intel RealSense Python SDK..."
    pip install pyrealsense2
    
    # C++ binding libraries for building modules
    log_info "Installing C++ binding libraries..."
    pip install pybind11[global] nanobind

    # Installations for robot visualization
    pip install meshcat urdfpy roboticstoolbox-python
    pip install "networkx>=2.8" --upgrade
    pip install imageio[ffmpeg]
    pip install imageio[pyav]
    
    log_success "Python packages and conda dependencies installed"
}

# Install Ruckig trajectory generation library in conda environment
install_ruckig() {
    log_info "Installing Ruckig trajectory generation library in conda environment..."
    
    # Remember the original directory
    local original_dir="$(pwd)"
    
    # First activate conda environment
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"

    # Try to install ruckig from conda-forge first
    log_info "Attempting to install Ruckig from conda-forge..."
    if conda install -c conda-forge ruckig -y 2>/dev/null; then
        log_success "Ruckig installed from conda-forge"
        return 0
    fi
    
    # Try to install via pip first (often easier than building from source)
    log_info "Conda-forge installation failed, trying pip installation..."
    if pip install ruckig 2>/dev/null; then
        log_success "Ruckig installed via pip"
        return 0
    fi
    
    log_info "Pip installation failed, building from source in conda environment..."
    
    # Install a compatible GCC version in conda environment for C++20 support
    log_info "Installing compatible GCC in conda environment for C++20 support..."
    # Use GCC 11 which has good C++20 support but better compatibility
    conda install -c conda-forge gxx_linux-64=11.4.0 gcc_linux-64=11.4.0 -y || {
        log_warning "Failed to install GCC 11, trying system compiler..."
        # Fall back to system compiler
        export CC=$(which gcc)
        export CXX=$(which g++)
        log_info "Using system GCC: $CC"
    }
    
    # Set conda environment paths for building
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export CPPFLAGS="-I$CONDA_PREFIX/include $CPPFLAGS"
    export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"
    
    # Use conda's GCC if available, otherwise use system GCC
    if [ -f "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
        export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        log_info "Using conda GCC: $CC"
        
        # Add compatibility flags for older glibc
        export CXXFLAGS="-D_GNU_SOURCE -std=c++17 $CXXFLAGS"
        export CFLAGS="-D_GNU_SOURCE $CFLAGS"
    else
        export CC=$(which gcc)
        export CXX=$(which g++)
        log_info "Using system GCC: $CC"
        
        # Use C++17 for better compatibility
        export CXXFLAGS="-std=c++17 $CXXFLAGS"
    fi
    
    # Check GCC version
    local gcc_version=$($CC --version | head -n1)
    log_info "GCC version: $gcc_version"
    
    local build_dir="/tmp/ruckig_build"
    
    # Remove existing build directory
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Clone Ruckig repository and checkout a specific version that works better
    log_info "Cloning Ruckig repository..."
    git clone https://github.com/pantor/ruckig.git
    cd ruckig
    
    # Try to checkout a version that has better C++17 compatibility
    git checkout v0.9.2 2>/dev/null || log_warning "Could not checkout v0.9.2, using latest"
    
    # Build and install into conda environment (not system-wide)
    mkdir build
    cd build
    
    # Configure with C++17 for better compatibility
    log_info "Configuring CMake with C++17 support for better compatibility..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
          -DBUILD_PYTHON_MODULE=ON \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -DCMAKE_C_COMPILER="$CC" \
          -DCMAKE_CXX_COMPILER="$CXX" \
          -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE -std=c++17" \
          -DCMAKE_C_FLAGS="-D_GNU_SOURCE" \
          .. || {
        log_warning "C++17 configuration failed, trying minimal configuration..."
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
              -DBUILD_PYTHON_MODULE=OFF \
              -DCMAKE_CXX_STANDARD=14 \
              -DCMAKE_C_COMPILER="$CC" \
              -DCMAKE_CXX_COMPILER="$CXX" \
              ..
    }
    
    # Build with limited parallelism to avoid memory issues
    log_info "Building Ruckig..."
    if ! make -j2; then
        log_warning "Parallel build failed, trying single-threaded build..."
        make clean
        if ! make; then
            log_error "Ruckig compilation failed completely"
            log_warning "Ruckig installation failed, but TOS can work without it (trajectory generation will be limited)"
            log_warning "You can try installing Ruckig manually later with: pip install ruckig"
            
            # Clean up and return to original directory
            cd "$original_dir"
            rm -rf "$build_dir"
            
            # Don't fail the entire installation, just continue without Ruckig
            log_warning "Continuing TOS installation without Ruckig..."
            return 0
        fi
    fi
    
    if ! make install; then
        log_warning "Ruckig installation failed, but compilation succeeded"
        log_warning "You may need to install Ruckig manually later"
        
        # Clean up and return to original directory
        cd "$original_dir"
        rm -rf "$build_dir"
        return 0
    fi
    
    # Install Python bindings via pip as backup
    log_info "Installing Python bindings..."
    pip install ruckig || log_warning "Python bindings installation failed, but C++ library should work"
    
    # Clean up and return to original directory
    cd "$original_dir"
    rm -rf "$build_dir"
    
    log_success "Ruckig installed in conda environment"
}

# Install RabbitMQ system-wide (only for UI hosts)
install_rabbitmq() {
    if [[ "$IS_UI_HOST" != "true" ]]; then
        log_info "Robot-only setup detected - checking for local RabbitMQ to avoid interference..."
        
        # Check if RabbitMQ is installed locally
        if command -v rabbitmq-server &> /dev/null; then
            log_warning "Local RabbitMQ installation found on robot-only setup!"
            log_warning "This can interfere with connection to remote RabbitMQ at: $RABBITMQ_HOST_IP"
            
            echo "Options:"
            echo "1) Remove local RabbitMQ (recommended - prevents interference)"
            echo "2) Stop local RabbitMQ service but keep installed"
            echo "3) Keep local RabbitMQ running (may cause conflicts)"
            read -p "Choose option (1/2/3): " -n 1 -r
            echo
            
            case $REPLY in
                1)
                    log_info "Removing local RabbitMQ to prevent interference..."
                    
                    # Stop the service
                    sudo systemctl stop rabbitmq-server 2>/dev/null || true
                    sudo systemctl disable rabbitmq-server 2>/dev/null || true
                    
                    # Remove RabbitMQ package and all configurations
                    if command -v apt-get &> /dev/null; then
                        sudo apt-get remove --purge -y rabbitmq-server
                        sudo apt-get autoremove -y
                    elif command -v yum &> /dev/null; then
                        sudo yum remove -y rabbitmq-server
                    elif command -v dnf &> /dev/null; then
                        sudo dnf remove -y rabbitmq-server
                    fi
                    
                    # Remove data directories
                    sudo rm -rf /var/lib/rabbitmq/
                    sudo rm -rf /etc/rabbitmq/
                    sudo rm -rf /var/log/rabbitmq/
                    
                    log_success "Local RabbitMQ removed - will connect to remote at: $RABBITMQ_HOST_IP"
                    ;;
                2)
                    log_info "Stopping local RabbitMQ service..."
                    sudo systemctl stop rabbitmq-server 2>/dev/null || true
                    sudo systemctl disable rabbitmq-server 2>/dev/null || true
                    log_success "Local RabbitMQ stopped - will connect to remote at: $RABBITMQ_HOST_IP"
                    ;;
                3)
                    log_warning "Keeping local RabbitMQ running - this may cause port conflicts!"
                    log_warning "Make sure your application connects to: $RABBITMQ_HOST_IP"
                    ;;
                *)
                    log_error "Invalid option. Stopping local RabbitMQ as safety measure."
                    sudo systemctl stop rabbitmq-server 2>/dev/null || true
                    sudo systemctl disable rabbitmq-server 2>/dev/null || true
                    ;;
            esac
        else
            log_success "No local RabbitMQ found - will connect to remote at: $RABBITMQ_HOST_IP"
        fi
        
        return 0
    fi
    
    log_info "Installing RabbitMQ system-wide (UI host setup)..."
    
    # Check if RabbitMQ is already installed and potentially clean it up
    if command -v rabbitmq-server &> /dev/null; then
        log_warning "RabbitMQ is already installed system-wide"
        
        echo "Options:"
        echo "1) Clean reinstall RabbitMQ (removes all data and configurations)"
        echo "2) Keep existing installation and reconfigure for TOS"
        echo "3) Skip RabbitMQ installation"
        read -p "Choose option (1/2/3): " -n 1 -r
        echo
        
        case $REPLY in
            1)
                log_info "Performing clean RabbitMQ reinstall..."
                
                # Stop the service
                sudo systemctl stop rabbitmq-server 2>/dev/null || true
                sudo systemctl disable rabbitmq-server 2>/dev/null || true
                
                # Remove RabbitMQ package and all configurations
                if command -v apt-get &> /dev/null; then
                    sudo apt-get remove --purge -y rabbitmq-server
                    sudo apt-get autoremove -y
                elif command -v yum &> /dev/null; then
                    sudo yum remove -y rabbitmq-server
                elif command -v dnf &> /dev/null; then
                    sudo dnf remove -y rabbitmq-server
                fi
                
                # Remove data directories
                sudo rm -rf /var/lib/rabbitmq/
                sudo rm -rf /etc/rabbitmq/
                sudo rm -rf /var/log/rabbitmq/
                
                log_success "RabbitMQ completely removed"
                ;;
            2)
                log_info "Keeping existing RabbitMQ installation, reconfiguring for TOS..."
                
                # Ensure it's running
                if ! sudo systemctl is-active --quiet rabbitmq-server; then
                    log_info "Starting RabbitMQ service..."
                    sudo systemctl start rabbitmq-server
                    sudo systemctl enable rabbitmq-server
                fi
                
                # Enable management plugin
                log_info "Enabling RabbitMQ management plugin..."
                sudo rabbitmq-plugins enable rabbitmq_management
                
                # Create a user for TOS application
                log_info "Setting up RabbitMQ user for TOS..."
                # Remove user if exists (ignore errors)
                sudo rabbitmqctl delete_user admin 2>/dev/null || true
                # Create new user
                sudo rabbitmqctl add_user admin admin
                sudo rabbitmqctl set_user_tags admin administrator
                sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
                
                log_success "RabbitMQ reconfigured for TOS"
                log_info "RabbitMQ Management UI available at: http://localhost:15672"
                log_info "TOS credentials: admin/admin"
                log_info "Service control: sudo systemctl start/stop/restart rabbitmq-server"
                return 0
                ;;
            3)
                log_info "Skipping RabbitMQ installation as requested"
                log_warning "Make sure RabbitMQ is properly configured for TOS manually"
                return 0
                ;;
            *)
                log_error "Invalid option. Skipping RabbitMQ installation."
                return 1
                ;;
        esac
    fi
    
    # Install RabbitMQ system-wide
    log_info "Installing RabbitMQ server system-wide..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian installation
        sudo apt-get update
        sudo apt-get install -y rabbitmq-server
        
        # Start and enable the service
        sudo systemctl start rabbitmq-server
        sudo systemctl enable rabbitmq-server
        
        # Wait a moment for service to start
        sleep 3
        
        # Enable management plugin
        log_info "Enabling RabbitMQ management plugin..."
        sudo rabbitmq-plugins enable rabbitmq_management
        
        # Create a user for TOS application
        log_info "Setting up RabbitMQ user for TOS..."
        sudo rabbitmqctl add_user admin admin
        sudo rabbitmqctl set_user_tags admin administrator
        sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
        
        log_success "RabbitMQ installed and configured system-wide"
        log_info "RabbitMQ Management UI available at: http://localhost:15672"
        log_info "TOS credentials: admin/admin"
        log_info "Service control: sudo systemctl start/stop/restart rabbitmq-server"
        log_info "Check status: sudo systemctl status rabbitmq-server"
        
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS installation
        sudo yum install -y rabbitmq-server
        sudo systemctl start rabbitmq-server
        sudo systemctl enable rabbitmq-server
        sudo rabbitmq-plugins enable rabbitmq_management
        sudo rabbitmqctl add_user admin admin
        sudo rabbitmqctl set_user_tags admin administrator
        sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
        log_success "RabbitMQ installed system-wide"
        
    elif command -v pacman &> /dev/null; then
        # Arch Linux installation
        sudo pacman -S --noconfirm rabbitmq
        sudo systemctl start rabbitmq-server
        sudo systemctl enable rabbitmq-server
        sudo rabbitmq-plugins enable rabbitmq_management
        sudo rabbitmqctl add_user admin admin
        sudo rabbitmqctl set_user_tags admin administrator
        sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
        log_success "RabbitMQ installed system-wide"
        
    else
        log_error "Unsupported package manager for RabbitMQ installation"
        log_info "Please install RabbitMQ manually for your distribution"
        return 1
    fi
}

# Install Intel RealSense SDK in conda environment (skip system-wide build)
install_realsense_sdk() {
    log_info "Installing Intel RealSense SDK in conda environment..."
    
    # Activate conda environment
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"
    
    # Try to install librealsense from conda-forge first
    log_info "Attempting to install Intel RealSense from conda-forge..."
    if conda install -c conda-forge librealsense -y 2>/dev/null; then
        log_success "Intel RealSense SDK installed from conda-forge"
        return 0
    fi
    
    log_warning "Conda-forge installation failed, using Python-only SDK..."
    log_info "The Python SDK (pyrealsense2) is already installed via pip"
    log_info "For full C++ SDK support, you may need to install it manually later"
    
    # Create a simple udev rules file for RealSense devices (minimal system impact)
    log_info "Setting up minimal udev rules for RealSense devices..."
    sudo tee /etc/udev/rules.d/99-realsense-libusb.rules > /dev/null <<EOF
# Intel RealSense device rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad*", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b*", MODE="0666", GROUP="plugdev"
EOF
    
    # Reload udev rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    
    log_success "Intel RealSense Python SDK configured (Python-only)"
}

# Build TOS C++ modules using conda environment tools
build_cpp_modules() {
    log_info "Building TOS C++ modules using conda environment tools..."
    
    # Activate conda environment to ensure all tools are available
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"
    
    # Set conda environment paths for building
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export CPPFLAGS="-I$CONDA_PREFIX/include $CPPFLAGS"
    export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    
    # Get the absolute path to the TOS application root
    local tos_root
    if ! tos_root=$(find_tos_root); then
        log_error "Cannot find TOS application root directory!"
        log_error "Current directory: $(pwd)"
        log_error "Please ensure you're running this script from within the TOS application directory tree."
        exit 1
    fi
    
    # Change to the TOS root directory
    cd "$tos_root"
    log_info "TOS root directory: $tos_root"
    log_info "Using conda environment tools from: $CONDA_PREFIX"
    
    # Check if we're on a network file system (which can cause compilation issues)
    local filesystem_type=$(df -T "$tos_root" | tail -n 1 | awk '{print $2}')
    local use_local_build=false
    
    if [[ "$filesystem_type" == "cifs" ]] || [[ "$filesystem_type" == "nfs" ]] || [[ "$tos_root" == *"gvfs"* ]]; then
        log_warning "Detected network file system ($filesystem_type). Building in local temporary directory to avoid compilation issues."
        use_local_build=true
    fi
    
    # Build FANUC robot control loop module
    if [[ -d "$tos_root/modules/robot/fanuc/cpp" ]]; then
        log_info "Building FANUC control loop module with conda tools..."
        
        if [[ "$use_local_build" == "true" ]]; then
            # Copy source to local temp directory for building
            local temp_build_dir="/tmp/tos_cpp_build_$$"
            local fanuc_source="$tos_root/modules/robot/fanuc/cpp"
            local fanuc_temp="$temp_build_dir/fanuc_cpp"
            
            log_info "Copying C++ source to local build directory: $temp_build_dir"
            mkdir -p "$fanuc_temp"
            cp -r "$fanuc_source"/* "$fanuc_temp/"
            
            cd "$fanuc_temp"
            
            if [[ -d build ]]; then
                rm -rf build
            fi
            
            mkdir build
            cd build
            cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
                  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
                  -DPython3_EXECUTABLE=$(which python) \
                  ..
            make -j$(nproc)
            
            # Copy built artifacts back to original location
            log_info "Copying built artifacts back to: $fanuc_source"
            if [[ -d "$fanuc_source/build" ]]; then
                rm -rf "$fanuc_source/build"
            fi
            cp -r . "$fanuc_source/build/"
            
            # Clean up temporary directory
            cd "$tos_root"
            rm -rf "$temp_build_dir"
            
        else
            # Build directly in place (local file system)
            cd "$tos_root/modules/robot/fanuc/cpp"
            
            if [[ -d build ]]; then
                rm -rf build
            fi
            
            mkdir build
            cd build
            cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
                  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
                  -DPython3_EXECUTABLE=$(which python) \
                  ..
            make -j$(nproc)
            
            cd "$tos_root"
        fi
        
        log_success "FANUC control loop module built with conda environment tools"
    else
        log_warning "FANUC C++ module directory not found at: $tos_root/modules/robot/fanuc/cpp, skipping..."
    fi
}

# Setup TOS application configuration
setup_tos_config() {
    log_info "Setting up TOS application configuration..."
    
    # Get the absolute path to the TOS application root
    # Check if we're in a subdirectory and need to go up
    local current_dir="$(pwd)"
    local tos_root="$current_dir"
    
    # If we're in a subdirectory, try to find the TOS root
    while [[ ! -f "$tos_root/applications/tos_ui/main.py" && ! -f "$tos_root/config/config.yaml" ]]; do
        local parent_dir="$(dirname "$tos_root")"
        if [[ "$parent_dir" == "$tos_root" ]]; then
            # We've reached the filesystem root without finding TOS files
            break
        fi
        tos_root="$parent_dir"
    done
    
    # Verify we found the right directory by checking for key files
    if [[ ! -f "$tos_root/applications/tos_ui/main.py" ]] || [[ ! -f "$tos_root/config/config.yaml" ]]; then
        log_error "Cannot find TOS application root directory!"
        log_error "Current directory: $current_dir"
        log_error "Searched up to: $tos_root"
        log_error "Please ensure you're running this script from within the TOS application directory tree."
        exit 1
    fi
    
    # Change to TOS root directory
    cd "$tos_root"
    
    # Create necessary directories
    mkdir -p logs
    # mkdir -p data
    
    # Set permissions for log directory
    chmod 755 logs
    
    log_success "TOS configuration setup complete"
}

# Setup systemd services (optional)
setup_systemd_services() {
    local setup_id="${SELECTED_SETUP#setup_}"
    
    # Customize the prompt based on the role
    if [[ "$IS_UI_HOST" == "true" ]]; then
        echo ""
        log_info "Setup $setup_id is configured as a UI Host"
        log_info "UI Hosts typically start the robot controller manually from the web interface"
        read -p "Do you still want to setup systemd auto-start service? (y/N): " -n 1 -r
    else
        echo ""
        log_info "Setup $setup_id is configured as a Robot Controller Only"
        log_info "Robot-only setups typically use systemd for automatic startup"
        read -p "Do you want to setup systemd services for auto-start? (Y/n): " -n 1 -r
    fi
    
    echo
    
    # Check the response based on the role
    if [[ "$IS_UI_HOST" == "true" ]]; then
        # For UI hosts, default is No, so only proceed if explicitly Yes
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping systemd service setup - you can start the robot controller manually"
            return
        fi
    else
        # For robot-only hosts, default is Yes, so proceed unless explicitly No
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            log_info "Skipping systemd service setup - you can start the robot controller manually"
            return
        fi
    fi
    
    log_info "Setting up systemd services for setup $setup_id..."
    
    # Get the current working directory (TOS root)
    local tos_root="$(pwd)"
    
    # Create TOS service file
    cat > /tmp/tos-robot-controller.service << EOF
[Unit]
Description=TOS Robot Controller (Setup $setup_id)
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$tos_root
Environment=PATH=$CONDA_BASE_PATH/envs/TOS/bin:$PATH
Environment=RABBITMQ_DEFAULT_USER=admin
Environment=RABBITMQ_DEFAULT_PASS=admin
ExecStart=$CONDA_BASE_PATH/envs/TOS/bin/python applications/robot_controller/main.py --setup $setup_id
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/tos-robot-controller.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tos-robot-controller.service
    
    log_success "Systemd service created and enabled for setup $setup_id"
    log_info "Start with: sudo systemctl start tos-robot-controller"
    log_info "Check status with: sudo systemctl status tos-robot-controller"
}

# Create desktop launcher
create_desktop_launcher() {
    log_info "Setting up desktop launchers..."
    
    local desktop_dir="$HOME/.local/share/applications"
    local icons_dir="$HOME/.local/share/icons"
    
    # Get the absolute path to the TOS application root
    # Use the current working directory since that's where the install script should be run from
    local tos_root="$(pwd)"
    
    # Verify we're in the right directory by checking for key files
    if [[ ! -f "applications/tos_ui/main.py" ]] || [[ ! -f "config/config.yaml" ]]; then
        log_error "This script must be run from the TOS application root directory!"
        log_error "Current directory: $tos_root"
        log_error "Please cd to the TOS application directory and run the script from there."
        exit 1
    fi
    
    log_info "TOS application root directory: $tos_root"
    
    # Create directories if they don't exist
    mkdir -p "$desktop_dir"
    mkdir -p "$icons_dir"
    
    # Use the existing TOS logo
    local tos_logo_path="$tos_root/applications/tos_ui/static/tos_logo.jpg"
    local icon_path="$icons_dir/tos-logo.jpg"
    
    if [ -f "$tos_logo_path" ]; then
        # Copy the existing TOS logo
        cp "$tos_logo_path" "$icon_path"
        log_success "Using existing TOS logo: $tos_logo_path"
    else
        # Try alternative logo locations
        if [ -f "$tos_root/applications/tos_ui/static/breda_robotics_logo.png" ]; then
            cp "$tos_root/applications/tos_ui/static/breda_robotics_logo.png" "$icons_dir/tos-logo.png"
            icon_path="$icons_dir/tos-logo.png"
            log_info "Using alternative logo: breda_robotics_logo.png"
        elif [ -f "/usr/share/pixmaps/applications-system.png" ]; then
            cp "/usr/share/pixmaps/applications-system.png" "$icons_dir/tos-app.png"
            icon_path="$icons_dir/tos-app.png"
            log_info "Using system fallback icon"
        else
            icon_path="applications-system"
            log_warning "No suitable icon found, using system default"
        fi
    fi
    
    # Copy and customize the existing TOS UI desktop file
    if [ -f "$tos_root/applications/tos_ui/TOS-UI.desktop" ]; then
        log_info "Found existing TOS UI desktop file"
        # Copy the existing desktop file
        cp "$tos_root/applications/tos_ui/TOS-UI.desktop" "$desktop_dir/tos-ui.desktop"
        
        # Update paths to be absolute using the current working directory
        sed -i "s|TOS_APP_ROOT|$tos_root|g" "$desktop_dir/tos-ui.desktop"
        
        # Also update the tos_ui.sh script to use the detected conda path
        if [ -f "$tos_root/applications/tos_ui/tos_ui.sh" ]; then
            # Create a backup of the original
            cp "$tos_root/applications/tos_ui/tos_ui.sh" "$tos_root/applications/tos_ui/tos_ui.sh.backup"
            
            # Update the conda path in tos_ui.sh to use the detected conda installation
            sed -i "s|source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"|source \"$CONDA_PROFILE_PATH\"|g" "$tos_root/applications/tos_ui/tos_ui.sh"
            
            log_info "Updated tos_ui.sh to use detected conda installation: $CONDA_TYPE"
        fi
        
        log_success "TOS UI desktop launcher installed"
    else
        log_warning "TOS UI desktop file not found at $tos_root/applications/tos_ui/TOS-UI.desktop, creating fallback"
        # Create fallback TOS UI launcher
        cat > "$desktop_dir/tos-ui.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=TOS UI
Comment=TOS Robotic Application - User Interface
Exec=bash -c "cd '$tos_root' && source '$CONDA_PROFILE_PATH' && conda activate TOS && python applications/tos_ui/main.py"
Icon=$icon_path
Terminal=true
StartupNotify=true
Categories=Engineering;Science;Education;
Keywords=robot;UI;interface;TOS;automation;
StartupWMClass=tos-ui
EOF
    fi
    
    # Make scripts executable
    # Only chmod tos_ui.sh if it exists
    if [ -f "$tos_root/applications/tos_ui/tos_ui.sh" ]; then
        chmod +x "$tos_root/applications/tos_ui/tos_ui.sh"
    fi
    
    chmod +x "$desktop_dir/tos-ui.desktop"
    
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$desktop_dir" 2>/dev/null || true
    fi
    
    log_success "Desktop launcher setup complete"
    log_info "TOS UI desktop launcher uses absolute paths from: $tos_root"
    log_info "TOS UI script updated to use $CONDA_TYPE conda installation"
    log_info "TOS UI application should appear in your system menu under 'Engineering'"
    log_info "You can now launch TOS UI directly from the desktop menu!"
    
    if [ -f "$icon_path" ]; then
        log_success "TOS icon created: $icon_path"
    else
        log_warning "Could not create TOS icon. Using system default: $icon_path"
    fi
}

# Create requirements.txt for future reference
# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Activate conda environment using detected path
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"
    
    # Test imports
    local test_script="/tmp/test_tos_imports.py"
    cat > "$test_script" << 'EOF'
import sys
import numpy as np
import yaml
import pika
import h5py
import cv2
import torch
import ruckig
import pyrealsense2
print("All core imports successful!")
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
EOF
    
    if python "$test_script"; then
        log_success "All Python imports working correctly!"
    else
        log_error "Some Python imports failed"
    fi
    
    rm "$test_script"
    
    # Test RabbitMQ
    if sudo systemctl is-active --quiet rabbitmq-server 2>/dev/null; then
        log_success "RabbitMQ service is running"
    elif pgrep -f "rabbitmq-server" > /dev/null; then
        log_success "RabbitMQ process is running"
    else
        log_warning "RabbitMQ is not running"
        log_info "Start with: sudo systemctl start rabbitmq-server"
    fi
    
    # Test Ruckig C++ library
    if ldconfig -p | grep -q ruckig; then
        log_success "Ruckig C++ library is installed"
    else
        log_warning "Ruckig C++ library may not be properly installed"
    fi
}

# Set capabilities on Python executable for process priority control
set_python_capabilities() {
    log_info "Setting Python capabilities for process priority control..."
    
    # Activate conda environment to get the correct Python executable
    source "$CONDA_PROFILE_PATH"
    conda activate "${ENV_NAME}"
    
    # Get the Python executable path from the activated environment
    local python_executable=$(which python)
    local python3_executable=$(which python3)
    
    log_info "Python executable path: $python_executable"
    log_info "Python3 executable path: $python3_executable"
    
    # Function to set capabilities on a Python executable
    set_cap_on_executable() {
        local executable_path="$1"
        local executable_name="$2"
        
        if [[ -L "$executable_path" ]]; then
            # If it's a symlink, resolve to the actual file
            local real_path=$(readlink -f "$executable_path")
            log_info "$executable_name is a symlink pointing to: $real_path"
            
            # Set capabilities on the real executable
            if sudo setcap cap_sys_nice+ep "$real_path" 2>/dev/null; then
                log_success "Successfully set capabilities on $real_path"
                
                # Verify capabilities were set
                local cap_result=$(getcap "$real_path" 2>/dev/null)
                if [[ -n "$cap_result" ]]; then
                    log_info "Capabilities verified: $cap_result"
                else
                    log_warning "Could not verify capabilities on $real_path"
                fi
            else
                log_error "Failed to set capabilities on $real_path"
                log_info "This may cause permission errors when setting process priority"
            fi
        elif [[ -f "$executable_path" ]]; then
            # If it's a regular file, set capabilities directly
            if sudo setcap cap_sys_nice+ep "$executable_path" 2>/dev/null; then
                log_success "Successfully set capabilities on $executable_path"
                
                # Verify capabilities were set
                local cap_result=$(getcap "$executable_path" 2>/dev/null)
                if [[ -n "$cap_result" ]]; then
                    log_info "Capabilities verified: $cap_result"
                else
                    log_warning "Could not verify capabilities on $executable_path"
                fi
            else
                log_error "Failed to set capabilities on $executable_path"
                log_info "This may cause permission errors when setting process priority"
            fi
        else
            log_warning "$executable_name not found or not accessible: $executable_path"
        fi
    }
    
    # Set capabilities on both python and python3 executables
    if [[ -n "$python_executable" ]]; then
        set_cap_on_executable "$python_executable" "python"
    fi
    
    if [[ -n "$python3_executable" && "$python3_executable" != "$python_executable" ]]; then
        set_cap_on_executable "$python3_executable" "python3"
    fi
    
    echo ""
    log_info "TROUBLESHOOTING NOTES for Python capabilities:"
    echo "If you encounter 'Permission denied' errors when setting process priority:"
    echo "1. Check if capabilities are set:"
    echo "   getcap \$(which python)"
    echo "   getcap \$(which python3)"
    echo ""
    echo "2. If capabilities are missing, manually set them:"
    echo "   # First, find the real Python executable (not symlink):"
    echo "   ls -la \$(which python3)"
    echo "   # Then set capabilities on the real file:"
    echo "   sudo setcap cap_sys_nice+ep /path/to/real/python/executable"
    echo ""
    echo "3. Common conda/miniconda paths:"
    echo "   - Miniconda: ~/miniconda3/envs/${ENV_NAME}/bin/python3.x"
    echo "   - Anaconda: ~/anaconda3/envs/${ENV_NAME}/bin/python3.x"
    echo "   - System conda: /opt/conda/envs/${ENV_NAME}/bin/python3.x"
    echo ""
    echo "4. Verify capabilities after setting:"
    echo "   getcap /path/to/python/executable"
    echo "   # Should show: python3.x = cap_sys_nice+ep"
    echo ""
    echo "5. If you update/reinstall conda or Python, you'll need to reapply capabilities"
    echo ""
}

# Main installation function
main() {
    echo "==============================================="
    echo "    TOS Application Installer"
    echo "==============================================="
    echo ""
    
    log_info "Starting TOS installation..."
    
    # Check prerequisites
    log_info "Checking root..."
    check_root
    log_info "Checking Linux distribution..."
    check_linux_distro
    
    # Configuration phase
    log_info "=== CONFIGURATION PHASE ==="
    read_tos_config
    parse_config
    validate_config
    choose_setup
    update_config_file
    
    echo ""
    log_info "=== INSTALLATION PHASE ==="
    echo "Installing for setup: $SELECTED_SETUP"
    echo "Role: $([ "$IS_UI_HOST" == "true" ] && echo "UI Host + Robot Controller" || echo "Robot Controller Only")"
    echo "RabbitMQ Host: $RABBITMQ_HOST_IP"
    echo ""
    
    # Main installation steps
    log_info "Running update system..."
    update_system
    log_info "Installing system dependencies..."
    install_system_deps
    log_info "Installing Miniconda..."
    install_miniconda
    log_info "Setting up conda environment..."
    setup_conda_env
    log_info "Installing Python packages..."
    install_python_packages
    log_info "Installing Ruckig trajectory generation library..."
    install_ruckig
    log_info "Installing RabbitMQ (conditional)..."
    install_rabbitmq
    log_info "Installing Intel RealSense SDK..."
    install_realsense_sdk
    log_info "Building TOS C++ modules..."
    build_cpp_modules
    log_info "Setting up TOS application configuration..."
    setup_tos_config
    
    # Role-specific setup
    log_info "Setting up systemd services..."
    setup_systemd_services
    create_desktop_launcher
    
    # Verification
    verify_installation
    
    # Set Python capabilities for process priority control
    set_python_capabilities
    
    echo ""
    echo "==============================================="
    log_success "TOS installation completed!"
    
    # Final conda configuration cleanup
    log_info "Finalizing conda configuration..."
    
    # Ensure conda base auto-activation is disabled
    conda config --set auto_activate_base false 2>/dev/null || true
    
    # Deactivate any active conda environment to demonstrate the change
    if [[ "$CONDA_DEFAULT_ENV" != "base" ]] && [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        conda deactivate 2>/dev/null || true
    fi
    
    echo ""
    log_info "=== POST-INSTALLATION SUMMARY ==="
    echo ""
    log_info "Setup Configuration:"
    echo "  Selected Setup: $SELECTED_SETUP"
    echo "  Role: $([ "$IS_UI_HOST" == "true" ] && echo "UI Host + Robot Controller" || echo "Robot Controller Only")"
    echo "  RabbitMQ Host: $RABBITMQ_HOST_IP"
    echo "  Auto-start Controller: $([ "$IS_UI_HOST" != "true" ] && echo "Yes (via systemd)" || echo "No (manual start)")"
    echo ""
    
    if [[ "$IS_UI_HOST" == "true" ]]; then
        log_info "UI Host Setup - Available Components:"
        echo "  ✓ TOS UI Web Interface"
        echo "  ✓ RabbitMQ Message Broker"
        echo "  ✓ Robot Controller"
        echo "  ✓ Desktop Launcher"
        echo ""
        echo "Next steps:"
        echo "1. Launch TOS UI from your desktop menu (Engineering category)"
        echo "2. Or manually: python applications/tos_ui/main.py"
        echo "3. Access RabbitMQ Management: http://localhost:15672 (admin/admin)"
    else
        log_info "Robot Controller Setup - Available Components:"
        echo "  ✓ Robot Controller (auto-started via systemd)"
        echo "  ✓ Connects to RabbitMQ at: $RABBITMQ_HOST_IP"
        echo ""
        echo "Next steps:"
        echo "1. Robot controller should be running automatically"
        echo "2. Check status: sudo systemctl status tos-robot-controller"
        echo "3. View logs: journalctl -u tos-robot-controller -f"
        echo "4. Manual start: python applications/robot_controller/main.py"
    fi
    
    echo ""
    echo "==============================================="
    echo ""
    echo "Conda Installation: $CONDA_TYPE at $CONDA_BASE_PATH"
    echo "Environment: $ENV_NAME"
    echo "Config File Updated: $CONFIG_FILE"
    echo ""
    echo "Python Capabilities:"
    echo "   - Process priority control capabilities have been set on Python executables"
    echo "   - This allows the robot interface to set high priority (-10 nice value)"
    echo "   - If you encounter permission errors, see troubleshooting notes above"
    echo ""
    echo "Note: Interbotix and ROS modules were excluded as requested."
    echo "Install them separately if needed for your robot setup."
    echo ""
    log_info "Installation log saved to: /tmp/tos_install.log"
}

# Run main function and log output
main 2>&1 | tee /tmp/tos_install.log
