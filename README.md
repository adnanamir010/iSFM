# iSFM
### Robust Incremental Structure-from-Motion

![Build Status](https://img.shields.io/badge/build-passing-green)![License](https://img.shields.io/badge/license-MIT-blue)![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)![C++ Version](https://img.shields.io/badge/c++-17-purple)

This repository contains the development of a hybrid, incremental Structure-from-Motion (SfM) system. The goal is to create a robust 3D reconstruction pipeline that leverages not only traditional point features but also lines and vanishing points to achieve superior performance, especially in challenging, weakly-textured indoor environments.

This project is an implementation and exploration of the concepts presented in the paper **"Robust Incremental Structure-from-Motion with Hybrid Features"** by Liu et al.

---

## 📋 Table of Contents

- [iSFM](#isfm)
    - [Robust Incremental Structure-from-Motion](#robust-incremental-structure-from-motion)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🛠️ System Requirements](#️-system-requirements)
  - [🚀 Installation](#-installation)
      - [1. Clone the Repository](#1-clone-the-repository)
      - [2. Install C++ Dependencies](#2-install-c-dependencies)
      - [3. Configure and Build C++ Code](#3-configure-and-build-c-code)
      - [4. Prepare for Python Installation](#4-prepare-for-python-installation)
      - [5. Install the Python Package](#5-install-the-python-package)
  - [👨‍💻 Usage](#-usage)
  - [🔧 Development Workflow](#-development-workflow)
  - [🗺️ Project Roadmap](#️-project-roadmap)
  - [📜 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **Hybrid Feature Integration:** Natively handles points, lines, and vanishing points (VPs) for a richer scene representation.
- **Incremental Processing:** Builds the 3D reconstruction frame-by-frame, suitable for real-time applications.
- **Robust Estimation:** Designed to overcome the limitations of traditional point-based SfM in poorly-textured scenes.
- **Uncertainty Propagation:** Models and propagates uncertainty for more reliable tracking and mapping.
- **C++/Python Architecture:** Combines high-performance C++ for core algorithms with a user-friendly Python API.
- **CUDA Accelerated:** Leverages NVIDIA GPUs for performance-critical operations.

---

## 🛠️ System Requirements

This project is developed and tested on the following environment. While it may work on other systems, this is the supported configuration.

- **Operating System:** Windows 11
- **GPU:** NVIDIA RTX series (developed on RTX 4070)
- **CUDA:** CUDA Toolkit 12.6
- **Compiler:** Visual Studio 2022 Build Tools (with C++17 support)
- **Build System:** CMake 3.20+ & Ninja
- **Package Manager:** Git, vcpkg
- **Python:** Python 3.11+

---

## 🚀 Installation

The project uses a manual build process to ensure all C++ dependencies are correctly compiled and linked.

**Important:** All commands must be run from the **x64 Native Tools Command Prompt for VS 2022**.

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/iSFM.git
cd iSFM
```

#### 2. Install C++ Dependencies
This step uses `vcpkg` to automatically download and build all required C++ libraries (OpenCV+CUDA, Ceres, Eigen, etc.). This is a one-time setup and may take a significant amount of time (1-2 hours).
```bash
# This reads the vcpkg.json file and builds everything
vcpkg install
```

#### 3. Configure and Build C++ Code
This step configures the project with CMake and compiles the core C++ library and Python bindings using Ninja.
```bash
# Create and enter a build directory
mkdir build
cd build

# Configure the project
cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile the code
ninja
```

#### 4. Prepare for Python Installation
The compiled Python module (`.pyd` file) must be copied into the Python source directory before installation.
```bash
# From inside the 'build' directory:
# This copies and renames the compiled library to the correct location for packaging.
copy python\hybrid_sfm\hybrid_sfm._core.cp312-win_amd64.pyd ..\python\hybrid_sfm\_core.pyd /Y
```

#### 5. Install the Python Package
Finally, install the project as a Python package. This command is fast as it doesn't recompile any C++ code.
```bash
# Go back to the project's root directory
cd ..

# Install the package
pip install .
```

---

## 👨‍💻 Usage

Once installed, you can use the `hybrid_sfm` package in your Python scripts. The core data structures are implemented in C++ for performance.

Note that Eigen-based vectors like `Point2D` are handled via NumPy for safety and performance.

```python
import numpy as np
import hybrid_sfm

# --- Create core objects ---
camera = hybrid_sfm.Camera()
camera.fx = 800.0

image = hybrid_sfm.Image()
image.id = 1

# --- Use NumPy for points ---
# This is the safe way to interact with Eigen types in C++
numpy_point = np.array([1024.5, 768.0], dtype=np.float64)

# Assign the NumPy array to a C++ member.
# pybind11 handles the conversion automatically.
observation = hybrid_sfm.Observation()
observation.feature_id = 42
observation.point = numpy_point

print("Created Observation object successfully!")
print(f"Retrieved point from C++ object: {observation.point}")
```

---

## 🔧 Development Workflow

After making any changes to the C++ source code (`.h` or `.cpp` files), you must recompile to make the changes available in Python. A utility script is provided for this.

1. Open the **x64 Native Tools Command Prompt for VS 2022**.
2. Navigate to the project's root directory (`E:\dev\iSFM`).
3. Run the recompile script:
   ```bash
   recompile.bat
   ```
This script will automatically run `ninja`, copy the new `.pyd` file, and update the `pip` installation.

---

## 🗺️ Project Roadmap

This project is being developed over a 35-day schedule.

-   [x] **Week 1: Foundation & Basic Infrastructure (Completed)**
    -   [x] Day 1: Project Structure & Build System
    -   [ ] Day 2: Camera Models & Calibration
    -   [ ] Day 3: Image Data Pipeline
    -   [ ] Day 4: Point Feature Detection (SIFT)
    -   [ ] Day 5: Point Feature Matching
    -   [ ] Day 6: Geometric Verification (Essential Matrix)
    -   [ ] Day 7: Basic SfM Pipeline Integration

-   [ ] **Week 2: Line Feature Integration**
    -   [ ] Day 8-14: Implement line detection, matching, triangulation, and verification.

-   [ ] **Week 3: Bundle Adjustment & Optimization**
    -   [ ] Day 15-21: Design and implement hybrid bundle adjustment with uncertainty.

-   [ ] **Week 4: Incremental Mapping System**
    -   [ ] Day 22-27: Build the full incremental mapping pipeline.

-   [ ] **Week 5 & 6: Advanced Features, Testing, and Optimization**
    -   [ ] Day 28-35: Implement robust registration, visualization, testing, and performance tuning.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work is heavily inspired by and based on the fantastic research from the authors of the following paper:

-   Liu, S., Gao, Y., Zhang, T., Pautrat, R., Schönberger, J. L., Larsson, V., & Pollefeys, M. (2023). **Robust Incremental Structure-from-Motion with Hybrid Features.** In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.