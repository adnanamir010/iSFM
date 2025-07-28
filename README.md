# iSFM: Robust Incremental Structure-from-Motion

![Build Status](https://img.shields.io/badge/build-passing-green) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python Version](https://img.shields.io/badge/python-3.10+-blue) ![C++ Version](https://img.shields.io/badge/c%2B%2B-17-purple) ![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)

This repository contains the development of **iSFM**, a hybrid, incremental Structure-from-Motion (SfM) system. The goal is to create a robust 3D reconstruction pipeline that leverages not only traditional point features but also lines and vanishing points to achieve superior performance, especially in challenging, weakly-textured indoor environments.

This project is an implementation and exploration of the concepts presented in the paper **"Robust Incremental Structure-from-Motion with Hybrid Features"** by Liu et al.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ System Requirements](#️-system-requirements)
- [🚀 Installation (Ubuntu)](#-installation-ubuntu)
- [👨‍💻 Usage Example](#-usage-example)
- [🔧 Development Workflow](#-development-workflow)
- [📚 Full Documentation](#-full-documentation)
- [🗺️ Project Roadmap](#️-project-roadmap)
- [📜 License](#-license)

---

## ✨ Features

- **High-Performance C++ Core:** All core algorithms are written in C++17 for maximum speed.
- **User-Friendly Python API:** A clean Python interface provided via `pybind11` for rapid prototyping and testing.
- **Advanced Camera Models:** Supports 10 COLMAP-compatible camera models with complex distortion.
- **GPU-Accelerated Image Pipeline:** Features a highly efficient `Image` class with:
  - CUDA-accelerated operations (resizing, undistortion, color conversion).
  - Lazy loading for handling large datasets.
  - Built-in image pyramid generation.
  - Thread-safe design for parallel processing.
- **Batch Processing:** Utilities for loading and processing entire directories of images efficiently.

---

## 🛠️ System Requirements

This project is developed and tested on **Ubuntu 22.04 LTS**.

- **Operating System:** Ubuntu 22.04 LTS
- **GPU:** NVIDIA GPU with CUDA support (developed on RTX 4070)
- **CUDA:** 12.4+
- **Compiler:** GCC/G++ 11.4.0+
- **Build System:** CMake 3.20+ & Make/Ninja
- **Dependencies:** OpenCV 4.13+ (with CUDA), Eigen3, Ceres, Exiv2
- **Python:** Python 3.10+

---

## 🚀 Installation (Ubuntu)

Follow these steps to build the project from source.

### 1. Install Dependencies

First, install the required system libraries.

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libeigen3-dev libceres-dev libexiv2-dev libgtest-dev
```

**Note:** You must have a version of **OpenCV with CUDA support** installed. The system-provided version may not be sufficient. It is recommended to build OpenCV from source if you encounter issues.

### 2\. Clone the Repository

```bash
git clone [https://github.com/adnanamir010/iSFM.git](https://github.com/adnanamir010/iSFM.git)
cd iSFM
```

### 3\. Configure and Build

Use CMake to configure the project and then build it.

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake (make sure to point to your Python executable)
cmake .. -DPYTHON_EXECUTABLE=$(which python3)

# Compile the code (use -j to specify the number of parallel jobs)
make -j$(nproc)
```

### 4\. Set Up Environment

For the Python interpreter to find the compiled module, you need to add the `build` directory to your `PYTHONPATH`.

```bash
# Add this line to your ~/.bashrc or ~/.zshrc file
export PYTHONPATH=$HOME/iSFM/build:$PYTHONPATH

# Then, source the file to apply the changes
source ~/.bashrc
```

---

## 👨‍💻 Usage Example

Once installed, you can use the `hybrid_sfm` package in your Python scripts.

```python
import hybrid_sfm
import numpy as np

# Load an image with options (e.g., build a pyramid on load)
flags = hybrid_sfm.Image.LoadFlags.BUILD_PYRAMID
img = hybrid_sfm.Image("path/to/your/image.jpg", flags)

print(f"Loaded {img.name} ({img.width}x{img.height})")
print(f"Pyramid has {img.pyramid.num_levels} levels.")

# Check for CUDA and perform a GPU-accelerated operation
if hybrid_sfm.has_cuda():
    print("CUDA is available. Performing GPU resize.")
    # The 'resize' method modifies the image in-place
    img.resize(width=640, height=480, use_gpu=True)
    print(f"Image resized to: {img.width}x{img.height}")
else:
    print("CUDA not available, using CPU.")

# Convert to a NumPy array for further processing or display
numpy_image = img.to_numpy()
```

---

## 🔧 Development Workflow

After making any changes to the C++ source code (`.h` or `.cpp` files), you must recompile to make the changes available in Python.

```bash
# Navigate to the build directory
cd ~/iSFM/build

# Re-run make
make -j$(nproc)
```

---

## 📚 Full Documentation

For a complete API reference, architectural overview, and more detailed examples, please see the full project documentation.

➡️ **[Read the Full Documentation](DOCUMENTATION.md)**

---

## 🗺️ Project Roadmap

This project is being developed over a 35-day schedule.

- [x] **Day 1: Project Structure & Build System**
- [x] **Day 2: Camera Models & Calibration**
- [x] **Day 3: Image Data Pipeline & GPU Acceleration**
- [ ] **Day 4: Point Feature Detection (SIFT)**
- [ ] **Day 5: Point Feature Matching**
- [ ] **Day 6: Geometric Verification (Essential Matrix)**
- [ ] **Day 7: Basic SfM Pipeline Integration**
- ... and more\!

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
