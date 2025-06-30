
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

git clone https://github.com/pantor/ruckig.git
cd ruckig
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

sudo make install

sudo ldconfig


# install pybind
cd "$SCRIPT_DIR"
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j4
sudo make install
ls /usr/local/include/pybind11/gil.h
