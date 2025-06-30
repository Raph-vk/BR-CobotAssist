# TOS Application Installation Guide

This guide provides comprehensive instructions for installing and setting up the TOS (Teach-Operate-Simulate) robotic application environment.

## Quick Start

```bash
# 1. Clone the repository (if not already done)
git clone <repository-url>
cd tos_app

# 2. Run the main installer (excludes Interbotix and ROS as requested)
./install_tos.sh

# 3. Optional: Setup development environment
./setup_dev_env.sh

# 4. Activate the environment
./activate_tos.sh

# 5. Run the application
python applications/robot_controller/main.py
```

## What Gets Installed

### System Dependencies
- Build tools (cmake, gcc, g++)
- Python 3.8 development headers
- Essential libraries (Eigen3, OpenSSL, USB, etc.)
- Intel RealSense SDK system packages

### Python Environment
- Miniconda Python 3.8 environment named "TOS"
- Core scientific packages (NumPy, SciPy, Matplotlib, Pandas)
- Computer vision (OpenCV, Pillow)
- PyTorch (CPU version by default)
- Intel RealSense Python SDK
- Data handling (HDF5, YAML)
- Message queue (Pika for RabbitMQ)
- Web framework (Flask)

### C++ Libraries
- Ruckig trajectory generation library
- pybind11 for Python-C++ bindings

### Services
- RabbitMQ message broker
- Optional systemd services for auto-startup

### Development Tools (optional)
- Jupyter Lab/Notebook
- Code formatting (Black, isort)
- Linting (flake8, pylint)
- Testing framework (pytest)
- Documentation tools (Sphinx, MkDocs)
- Pre-commit hooks
- VS Code extensions

## Prerequisites

- Ubuntu 20.04 LTS or later (other distributions may work but are untested)
- At least 4GB RAM
- 10GB free disk space
- Internet connection for downloads
- Sudo privileges for system package installation

## Manual Installation Steps

If you prefer to install components manually:

### 1. System Packages
```bash
sudo apt update
sudo apt install -y build-essential cmake git wget curl \
    python3-dev python3-pip python3-venv \
    libeigen3-dev libssl-dev libusb-1.0-0-dev \
    libudev-dev libgtk-3-dev libglfw3-dev \
    libgl1-mesa-dev libglu1-mesa-dev
```

### 2. Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n TOS python=3.8
conda activate TOS
```

### 3. Python Packages
```bash
pip install -r requirements.txt
```

### 4. Ruckig
```bash
git clone https://github.com/pantor/ruckig.git
cd ruckig
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MODULE=ON ..
make -j$(nproc)
sudo make install
sudo ldconfig
pip install ruckig
```

### 5. RabbitMQ
```bash
sudo apt install -y rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
```

## Configuration

### 1. TOS Configuration
Edit `config/config.yaml` to match your hardware setup:

```yaml
hardware:
  robot:
    brand: "dummy"  # Change to "fanuc" for real robot
    # ... other robot settings
  camera:
    brand: "inteld405"  # Or your camera type
  teachbot:
    brand: "dummy"  # Change to "interbotix" if using real teachbot
```

### 2. RabbitMQ Configuration
The installer creates a default admin user (admin/admin). For production:

```bash
sudo rabbitmqctl delete_user admin
sudo rabbitmqctl add_user your_user your_password
sudo rabbitmqctl set_user_tags your_user administrator
sudo rabbitmqctl set_permissions -p / your_user ".*" ".*" ".*"
```

## Running TOS

### Method 1: Direct Execution
```bash
./activate_tos.sh
python applications/robot_controller/main.py
```

### Method 2: Systemd Service (if configured)
```bash
sudo systemctl start tos-app
sudo systemctl status tos-app
```

### Method 3: Development Mode
```bash
./run_dev_server.sh
```

## Hardware-Specific Setup

### Intel RealSense Cameras
The installer sets up the Intel RealSense SDK. To test:
```bash
realsense-viewer  # GUI tool
rs-enumerate-devices  # List connected devices
```

### FANUC Robot
For real FANUC robot connection:
1. Set `robot.brand: "fanuc"` in config
2. Configure network settings in config
3. Ensure robot and PC are on same network
4. Build C++ control modules: `cd modules/robot/fanuc/cpp && mkdir build && cd build && cmake .. && make`

### Interbotix Teachbot
For real Interbotix teachbot (not installed by this script):
1. Install Interbotix packages manually
2. Set `teachbot.brand: "interbotix"` in config
3. Configure robot-specific settings

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   sudo chmod +x install_tos.sh
   ./install_tos.sh
   ```

2. **Conda Not Found**
   ```bash
   source ~/.bashrc
   # or
   export PATH="$HOME/miniconda3/bin:$PATH"
   ```

3. **RabbitMQ Connection Failed**
   ```bash
   sudo systemctl status rabbitmq-server
   sudo systemctl restart rabbitmq-server
   ```

4. **Python Import Errors**
   ```bash
   ./activate_tos.sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **C++ Build Errors**
   - Ensure cmake and build-essential are installed
   - Check that ruckig library is properly installed: `ldconfig -p | grep ruckig`

### Log Files
- Installation log: `/tmp/tos_install.log`
- Application logs: `logs/` directory
- RabbitMQ logs: `/var/log/rabbitmq/`

### Testing Installation
```bash
./activate_tos.sh
python -c "import numpy, torch, ruckig, cv2, yaml, pika; print('All imports successful!')"
```

## Uninstalling

To remove TOS:

```bash
# Remove conda environment
conda env remove -n TOS

# Remove RabbitMQ (optional)
sudo apt remove --purge rabbitmq-server

# Remove systemd service
sudo systemctl stop tos-app
sudo systemctl disable tos-app
sudo rm /etc/systemd/system/tos-app.service

# Remove installed libraries (optional)
sudo rm -rf /usr/local/lib/libruckig*
sudo rm -rf /usr/local/include/ruckig/
```

## Support

For issues and support:
1. Check the troubleshooting section above
2. Review log files
3. Ensure all prerequisites are met
4. Try running the verification script: `./verify_installation.sh`

## Development

See `setup_dev_env.sh` for additional development tools and workflows.
