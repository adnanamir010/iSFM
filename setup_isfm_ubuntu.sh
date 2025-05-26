#!/bin/bash

# iSFM Ubuntu Development Environment Setup Script
# Hybrid Structure-from-Motion with Points, Lines, and Vanishing Points
# For CUDA 12.6, Ubuntu 20.04/22.04

set -e  # Exit on any error

# Configuration
PROJECT_PATH="${1:-$HOME/Projects/iSFM}"
CONDA_ENV_NAME="isfm"
CUDA_VERSION="12.6"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'  
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_step() {
    echo -e "${GREEN}=== $1 ===${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Check if running with sudo
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should NOT be run with sudo/root privileges"
        log_info "Run as regular user. Sudo will be prompted when needed."
        exit 1
    fi
}

# System detection
detect_system() {
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        OS_NAME=$NAME
        OS_VERSION=$VERSION_ID
        log_info "Detected: $OS_NAME $OS_VERSION"
    else
        log_error "Cannot detect Ubuntu version"
        exit 1
    fi
}

# Check CUDA installation
check_cuda() {
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        log_info "CUDA version detected: $CUDA_VER"
        return 0
    else
        log_warning "CUDA not found. Please install CUDA $CUDA_VERSION first."
        log_info "Visit: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

# Update system packages
update_system() {
    log_step "Updating System Packages"
    sudo apt update
    sudo apt upgrade -y
}

# Install system dependencies
install_system_deps() {
    log_step "Installing System Dependencies"
    
    # Essential build tools
    sudo apt install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        unzip \
        pkg-config \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Development libraries
    sudo apt install -y \
        libeigen3-dev \
        libopencv-dev \
        libopencv-contrib-dev \
        libceres-dev \
        libgtest-dev \
        libgmock-dev \
        libomp-dev \
        libfmt-dev \
        libspdlog-dev
    
    # Python development
    sudo apt install -y \
        python3-dev \
        python3-pip \
        python3-venv
    
    # Optional: PCL for point cloud processing
    sudo apt install -y libpcl-dev
    
    log_info "System dependencies installed successfully"
}

# Install Miniconda
install_miniconda() {
    log_step "Installing Miniconda"
    
    CONDA_DIR="$HOME/miniconda3"
    
    if [[ -d "$CONDA_DIR" ]] || command -v conda &> /dev/null; then
        log_info "Conda already installed"
        return 0
    fi
    
    log_info "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    log_info "Installing Miniconda..."
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    
    # Initialize conda
    "$CONDA_DIR/bin/conda" init bash
    source "$HOME/.bashrc"
    
    # Add to current session
    export PATH="$CONDA_DIR/bin:$PATH"
    
    log_info "Miniconda installed successfully"
    rm /tmp/miniconda.sh
}

# Create project directory
setup_project_dir() {
    log_step "Setting up Project Directory"
    
    if [[ ! -d "$PROJECT_PATH" ]]; then
        log_info "Creating project directory: $PROJECT_PATH"
        mkdir -p "$PROJECT_PATH"
    fi
    
    cd "$PROJECT_PATH"
    log_info "Working in: $(pwd)"
}

# Create conda environment
create_conda_env() {
    log_step "Creating iSFM Conda Environment"
    
    # Create environment.yml
    cat > environment.yml << 'EOF'
name: isfm
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - opencv
  - pillow
  - pyyaml
  - tqdm
  # PyTorch with CUDA
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1
  # Scientific computing
  - scikit-learn
  - scikit-image
  - pandas
  # 3D processing
  - open3d
  # Development tools
  - pytest
  - jupyter
  - ipython
  # Build tools
  - cmake
  - ninja
  - pybind11
  - pip
  - pip:
    - onnxruntime-gpu
    - kornia
    - einops
    - rich
    - typer
    - loguru
    - hydra-core
    - wandb
EOF
    
    log_info "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    
    log_info "Conda environment 'isfm' created successfully"
}

# Setup project structure
setup_project_structure() {
    log_step "Creating iSFM Project Structure"
    
    # Main directories
    mkdir -p isfm/{core,features,geometry,optimization,mapping,registration,utils,models,visualization}
    mkdir -p cpp/{src/{core,features,geometry,optimization,mapping,registration,utils},include/isfm,bindings}
    mkdir -p tests/{unit,integration,data}
    mkdir -p examples/{basic_sfm,hybrid_reconstruction,line_detection,evaluation}
    mkdir -p data/{models/{deeplsd,gluestick},datasets/sample,results}
    mkdir -p docs/{api,tutorials,paper_notes}
    mkdir -p scripts/{download_models,benchmarks}
    
    log_info "Project structure created"
}

# Create configuration files
create_config_files() {
    log_step "Creating Configuration Files"
    
    # CMakeLists.txt
    cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(iSFM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA support
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# OpenMP for parallelization
find_package(OpenMP REQUIRED)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -O3)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")  # RTX 4070
endif()

# Include directories
include_directories(cpp/include)

# Add subdirectories
add_subdirectory(cpp/src)
add_subdirectory(cpp/bindings)
add_subdirectory(tests)
EOF

    # setup.py
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "isfm._core",
        [
            "cpp/bindings/pybind_main.cpp",
            # Add more binding files as needed
        ],
        include_dirs=[
            "cpp/include",
        ],
        libraries=["opencv_core", "opencv_imgproc", "ceres"],
        cxx_std=17,
    ),
]

setup(
    name="isfm",
    version="0.1.0",
    author="Your Name",
    description="Incremental Structure-from-Motion with Hybrid Features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "open3d>=0.16.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
EOF

    # pyproject.toml
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "pybind11>=2.10.0"]
build-backend = "setuptools.build_meta"

[project]
name = "isfm"
version = "0.1.0"
description = "Incremental Structure-from-Motion with Hybrid Features"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.9"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
EOF

    log_info "Configuration files created"
}

# Create build scripts
create_build_scripts() {
    log_step "Creating Build Scripts"
    
    # build.sh
    cat > build.sh << 'EOF'
#!/bin/bash

echo "Building iSFM System..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isfm

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -G Ninja

# Build
ninja

# Install Python package in development mode
cd ..
pip install -e .

echo
echo "Build completed successfully!"
echo "To test: python -c \"import isfm; print('iSFM imported successfully!')\""
EOF

    chmod +x build.sh
    
    log_info "Build scripts created"
}

# Setup Python package
setup_python_package() {
    log_step "Setting up Python Package"
    
    # Main __init__.py
    cat > isfm/__init__.py << 'EOF'
"""
iSFM: Incremental Structure-from-Motion with Hybrid Features

A robust SfM system that leverages points, lines, and vanishing points
for improved reconstruction in challenging scenarios.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import core
from . import features
from . import geometry
from . import optimization
from . import mapping
from . import registration
from . import utils
from . import visualization

# Try to import C++ extensions
try:
    from ._core import *
except ImportError:
    import warnings
    warnings.warn("C++ extensions not available. Run './build.sh' to build them.")

__all__ = [
    "core",
    "features",
    "geometry", 
    "optimization",
    "mapping",
    "registration",
    "utils",
    "visualization",
]
EOF

    # Create empty __init__.py files for submodules
    for module in core features geometry optimization mapping registration utils models visualization; do
        echo "# $module module" > "isfm/$module/__init__.py"
    done
    
    log_info "Python package structure created"
}

# Create verification script
create_verify_script() {
    log_step "Creating Verification Script"
    
    cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""iSFM Environment Verification Script"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    return True

def check_package(package_name, optional=False):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        status = "○" if optional else "✗"
        print(f"{status} {package_name}: not installed")
        return optional

def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✗ CUDA not available")
        return cuda_available  
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_opencv_cuda():
    try:
        import cv2
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"✓ OpenCV CUDA devices: {cuda_devices}")
        return cuda_devices > 0
    except:
        print("✗ OpenCV CUDA not available")
        return False

def check_system_libs():
    """Check system-level dependencies"""
    libs = ['eigen3', 'opencv4', 'ceres']
    for lib in libs:
        try:
            result = subprocess.run(['pkg-config', '--exists', lib], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = subprocess.run(['pkg-config', '--modversion', lib],
                                       capture_output=True, text=True)
                print(f"✓ {lib}: {version.stdout.strip()}")
            else:
                print(f"✗ {lib}: not found")
        except FileNotFoundError:
            print("○ pkg-config not available for system lib check")
            break

def main():
    print("=== iSFM Environment Verification ===\n")
    
    # Check Python
    check_python()
    print()
    
    # Check required packages
    print("Required packages:")
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'cv2', 'torch',
        'torchvision', 'open3d', 'sklearn', 'PIL'
    ]
    
    all_required = True
    for pkg in required_packages:
        if not check_package(pkg):
            all_required = False
    
    print("\nOptional packages:")
    optional_packages = ['onnxruntime', 'kornia', 'wandb', 'jupyter']
    for pkg in optional_packages:
        check_package(pkg, optional=True)
    
    print("\nSystem libraries:")
    check_system_libs()
    
    print("\nHardware acceleration:")
    cuda_ok = check_cuda()
    opencv_cuda_ok = check_opencv_cuda()
    
    print("\nProject structure:")
    project_dirs = ['isfm', 'cpp', 'tests', 'examples', 'data']
    for directory in project_dirs:
        if Path(directory).exists():
            print(f"✓ {directory}/ directory exists")
        else:
            print(f"✗ {directory}/ directory missing")
    
    print("\n=== Summary ===")
    if all_required and cuda_ok:
        print("✓ Environment ready for iSFM development!")
    elif all_required:
        print("○ Environment ready, but CUDA support issues detected")
    else:
        print("✗ Environment setup incomplete")
        print("Run the setup script or install missing packages")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x verify_setup.py
    log_info "Verification script created"
}

# Create README
create_readme() {
    cat > README.md << 'EOF'
# iSFM: Incremental Structure-from-Motion with Hybrid Features

A robust Structure-from-Motion system that leverages points, lines, and vanishing points for improved reconstruction in challenging scenarios.

## Features

- **Hybrid Features**: Points, lines, and vanishing points
- **Deep Learning Integration**: DeepLSD line detection, GlueStick matching  
- **Uncertainty Modeling**: Analytical uncertainty propagation for 3D lines
- **GPU Acceleration**: CUDA support for modern GPUs
- **Incremental Mapping**: Robust camera registration and triangulation

## Setup

### Ubuntu 20.04/22.04
```bash
curl -O https://raw.githubusercontent.com/yourusername/isfm/main/setup_isfm_ubuntu.sh
chmod +x setup_isfm_ubuntu.sh
./setup_isfm_ubuntu.sh [project_path]
```

### Usage

1. **Verify installation**:
   ```bash
   conda activate isfm
   python verify_setup.py
   ```

2. **Build C++ components**:
   ```bash
   ./build.sh
   ```

3. **Run example**:
   ```python
   import isfm
   # Your SfM pipeline here
   ```

## Project Structure

- `isfm/` - Python package
- `cpp/` - C++ implementation
- `tests/` - Unit and integration tests  
- `examples/` - Usage examples
- `data/` - Models and datasets
- `docs/` - Documentation

## Requirements

- Ubuntu 20.04/22.04
- CUDA 12.6+ 
- Python 3.11
- GCC 9+

## License

MIT License
EOF

    log_info "README created"
}

# Main execution
main() {
    log_step "iSFM Development Environment Setup for Ubuntu"
    log_info "Project Path: $PROJECT_PATH"
    log_info "CUDA Version: $CUDA_VERSION"
    log_info "Python Version: $PYTHON_VERSION"
    
    # Preliminary checks
    check_sudo
    detect_system
    
    # Check CUDA (optional but recommended)
    if ! check_cuda; then
        log_warning "Continuing without CUDA verification..."
    fi
    
    # Setup steps
    update_system
    install_system_deps
    install_miniconda
    setup_project_dir
    create_conda_env
    setup_project_structure
    create_config_files
    create_build_scripts
    setup_python_package
    create_verify_script
    create_readme
    
    log_step "Setup Complete!"
    
    echo -e "${GREEN}"
    cat << EOF

=== iSFM SETUP SUMMARY ===
✓ System dependencies installed
✓ Miniconda with isfm environment  
✓ Project structure created
✓ Configuration files generated
✓ Build scripts ready

PROJECT PATH: $PROJECT_PATH

NEXT STEPS:
1. Restart terminal or run: source ~/.bashrc
2. Navigate to project: cd "$PROJECT_PATH"
3. Activate environment: conda activate isfm
4. Verify setup: python verify_setup.py
5. Build C++ components: ./build.sh
6. Start development!

QUICK COMMANDS:
conda activate isfm
./build.sh
python verify_setup.py

EOF
    echo -e "${NC}"
    
    log_warning "Please restart your terminal or run 'source ~/.bashrc' to load environment changes."
}

# Run main function
main "$@"
EOF