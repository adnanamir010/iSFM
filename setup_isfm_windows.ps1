# iSFM Windows Setup Script - OpenCV 4.10.0 + CUDA 12.6
# Run as Administrator

param(
    [string]$ProjectPath = "E:\backup\Desktop\College\NEU\Individual projects\iSFM"
)

function Write-Step($message) {
    Write-Host "=== $message ===" -ForegroundColor Green
}

function Write-Info($message) {
    Write-Host "[INFO] $message" -ForegroundColor Cyan
}

function Write-Warning($message) {
    Write-Host "[WARNING] $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

# Check admin privileges
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator!"
    exit 1
}

Write-Step "iSFM Setup for Windows with OpenCV 4.10.0 + CUDA 12.6"
Write-Info "Project Path: $ProjectPath"
Write-Info "Target: RTX 4070 with CUDA 12.6 + cuDNN 8.9.7"
Write-Info "OpenCV: 4.10.0 (CUDA 12.6 Compatible)"
Write-Warning "Total time: 1-2 hours"

# Create project directory
if (-not (Test-Path $ProjectPath)) {
    New-Item -ItemType Directory -Path $ProjectPath -Force | Out-Null
}
Set-Location $ProjectPath

Write-Step "Phase 1: Installing Development Tools"

# Install chocolatey if needed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    refreshenv
}

# Install tools
$tools = @("git", "cmake", "ninja", "7zip")
foreach ($tool in $tools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Info "Installing $tool..."
        choco install $tool -y
    } else {
        Write-Info "OK $tool already installed"
    }
}

Write-Step "Phase 2: Installing Visual Studio Build Tools"

# Check for existing VS installations
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasBuildTools = $false

if (Test-Path $vsWhere) {
    try {
        $vsInstalls = & $vsWhere -products * -format json | ConvertFrom-Json
        foreach ($install in $vsInstalls) {
            if ($install.productId -match "BuildTools") {
                Write-Info "OK Visual Studio Build Tools found"
                $hasBuildTools = $true
                break
            }
        }
    } catch {
        Write-Warning "Could not check VS installations"
    }
}

if (-not $hasBuildTools) {
    Write-Info "Installing Visual Studio Build Tools 2022..."
    Write-Warning "This takes 20-30 minutes"
    
    $vsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    Invoke-WebRequest -Uri $vsUrl -OutFile "vs_buildtools.exe"
    
    $vsArgs = @("--quiet", "--wait", "--add", "Microsoft.VisualStudio.Workload.VCTools")
    Start-Process -FilePath "vs_buildtools.exe" -ArgumentList $vsArgs -Wait
    Remove-Item "vs_buildtools.exe"
    Write-Info "OK Visual Studio Build Tools installed"
}

Write-Step "Phase 3: Installing Miniconda"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Miniconda..."
    $condaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri $condaUrl -OutFile "miniconda.exe"
    Start-Process -FilePath "miniconda.exe" -ArgumentList "/S", "/AddToPath=1" -Wait
    Remove-Item "miniconda.exe"
    
    # Add to current session
    $condaPath = "$env:USERPROFILE\miniconda3"
    $env:PATH = "$condaPath;$condaPath\Scripts;" + $env:PATH
    Write-Info "OK Miniconda installed"
} else {
    Write-Info "OK Conda already installed"
}

Write-Step "Phase 4: Setting up vcpkg"

$vcpkgDir = "C:\vcpkg"
if (-not (Test-Path $vcpkgDir)) {
    Write-Info "Installing vcpkg..."
    git clone https://github.com/Microsoft/vcpkg.git $vcpkgDir
    Set-Location $vcpkgDir
    .\bootstrap-vcpkg.bat
    [Environment]::SetEnvironmentVariable("VCPKG_ROOT", $vcpkgDir, "Machine")
    $env:VCPKG_ROOT = $vcpkgDir
    Write-Info "OK vcpkg installed"
} else {
    Write-Info "OK vcpkg already exists"
    Set-Location $vcpkgDir
}

Write-Info "Installing C++ packages..."
$packages = @("eigen3:x64-windows", "ceres[suitesparse]:x64-windows", "pybind11:x64-windows")
foreach ($pkg in $packages) {
    Write-Info "Installing $pkg..."
    .\vcpkg.exe install $pkg --triplet x64-windows
}
.\vcpkg.exe integrate install

Set-Location $ProjectPath

Write-Step "Phase 5: Building OpenCV 4.10.0 with CUDA 12.6"

$opencvBuildDir = "C:\opencv_4_10_cuda_build"
$opencvInstallDir = "C:\opencv_4_10_cuda"

Write-Info "Building OpenCV 4.10.0 with CUDA 12.6 (30-60 minutes)"
Write-Warning "Perfect time for a coffee break!"

if (-not (Test-Path $opencvBuildDir)) {
    New-Item -ItemType Directory -Path $opencvBuildDir -Force | Out-Null
}
Set-Location $opencvBuildDir

# Download OpenCV 4.10.0
if (-not (Test-Path "opencv")) {
    Write-Info "Downloading OpenCV 4.10.0..."
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git
}

if (-not (Test-Path "opencv_contrib")) {
    Write-Info "Downloading OpenCV Contrib 4.10.0..."
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git
}

# Find CUDA and fix path format for CMake
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (-not (Test-Path $cudaPath)) {
    # Try alternative paths
    $alternatePaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", 
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
    )
    
    $cudaFound = $false
    foreach ($altPath in $alternatePaths) {
        if (Test-Path $altPath) {
            $cudaPath = $altPath
            $cudaFound = $true
            Write-Info "Found CUDA at: $cudaPath"
            break
        }
    }
    
    if (-not $cudaFound) {
        Write-Error "CUDA not found in expected locations"
        Write-Info "Please install CUDA 12.x first"
        exit 1
    }
}

# Convert Windows path to CMake-friendly format
$cudaPathCMake = $cudaPath.Replace('\', '/')
Write-Info "Using CUDA path: $cudaPathCMake"

# Clean any previous build attempts to avoid cached conflicts
if (Test-Path "build_cuda") {
    Write-Info "Cleaning previous build directory..."
    Remove-Item -Recurse -Force "build_cuda"
}

Write-Info "Configuring OpenCV 4.10.0 with CUDA 12.6 (disabling TFLite to avoid flatbuffers)..."
$cmakeArgs = @(
    "-B", "build_cuda",
    "-S", "opencv", 
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$opencvInstallDir",
    "-DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules",
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake",
    
    # CUDA 12.6 Configuration
    "-DWITH_CUDA=ON",
    "-DCUDA_TOOLKIT_ROOT_DIR=$cudaPathCMake",
    "-DWITH_CUBLAS=ON",
    "-DWITH_CUFFT=ON",
    "-DWITH_CUDNN=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=89", # RTX 4070 specific
    "-DCUDA_FAST_MATH=ON",
    
    # DISABLE: TensorFlow Lite (avoids flatbuffers conflicts)
    "-DWITH_FLATBUFFERS=OFF",
    "-DBUILD_FLATBUFFERS=OFF",
    "-DBUILD_TFLITE=OFF",
    "-DOPENCV_DNN_TFLITE=OFF",
    "-DWITH_TENSORFLOW=OFF",
    
    # OpenCV 4.10.0 Core Features
    "-DOPENCV_DNN_CUDA=ON",
    "-DBUILD_opencv_world=ON",
    "-DBUILD_SHARED_LIBS=ON",
    "-DOPENCV_ENABLE_NONFREE=ON",
    
    # DNN backends that work without flatbuffers
    "-DWITH_PROTOBUF=ON",
    "-DOPENCV_DNN_OPENCL=ON",
    "-DOPENCV_DNN_OPENCV=ON",
    
    # Threading: Use OpenMP (no TBB issues)
    "-DWITH_TBB=OFF",
    "-DBUILD_TBB=OFF",
    "-DWITH_OPENMP=ON",
    
    # Performance optimizations
    "-DWITH_IPP=ON",
    "-DWITH_EIGEN=ON",
    "-DWITH_LAPACK=ON",
    
    # Disable problematic modules
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_apps=OFF",
    
    # Python and CUDA modules (perfect for SfM)
    "-DBUILD_opencv_python3=ON",
    "-DBUILD_opencv_cudaimgproc=ON",
    "-DBUILD_opencv_cudaarithm=ON",
    "-DBUILD_opencv_cudafeatures2d=ON",
    "-DBUILD_opencv_cudastereo=ON",
    "-DBUILD_opencv_cudaoptflow=ON",
    
    # Classical computer vision modules
    "-DBUILD_opencv_features2d=ON",
    "-DBUILD_opencv_calib3d=ON",
    "-DBUILD_opencv_imgproc=ON",
    "-DBUILD_opencv_core=ON",
    
    # Stability fixes
    "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=OFF",
    "-DOPENCV_FORCE_3RDPARTY_BUILD=ON",
    
    # Suppress warnings
    "-Wno-dev"
)

cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenCV 4.10.0 CMake configuration failed"
    exit 1
}

Write-Info "Building OpenCV 4.10.0..."
cmake --build build_cuda --config Release --parallel --target INSTALL

if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenCV 4.10.0 build failed"
    exit 1
}

Write-Info "OK OpenCV 4.10.0 with CUDA 12.6 built successfully!"

# Add to PATH
$opencvBinDir = Join-Path $opencvInstallDir "x64\vc17\bin"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$opencvBinDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$opencvBinDir", "User")
}

Set-Location $ProjectPath

Write-Step "Phase 6: Creating Python Environment"

# Updated environment for OpenCV 4.10.0
$envContent = "name: isfm`n"
$envContent += "channels:`n"
$envContent += "  - pytorch`n"
$envContent += "  - nvidia`n" 
$envContent += "  - conda-forge`n"
$envContent += "dependencies:`n"
$envContent += "  - python=3.11`n"
$envContent += "  - numpy>=1.21.0`n"
$envContent += "  - scipy`n"
$envContent += "  - matplotlib`n"
$envContent += "  - pytorch`n"
$envContent += "  - torchvision`n"
$envContent += "  - pytorch-cuda=12.1`n"
$envContent += "  - open3d`n"
$envContent += "  - pytest`n"
$envContent += "  - pybind11`n"
$envContent += "  - pip`n"
$envContent += "  - pip:`n"
$envContent += "    - onnxruntime-gpu`n"

Set-Content -Path "environment.yml" -Value $envContent -Encoding UTF8

Write-Info "Creating conda environment..."
conda env create -f environment.yml

Write-Step "Phase 7: Creating Project Structure"

$dirs = @(
    "isfm", "isfm\core", "isfm\features", "isfm\geometry", "isfm\optimization",
    "isfm\mapping", "isfm\utils", "cpp", "cpp\src", "cpp\include", "cpp\bindings",
    "tests", "examples", "data", "docs"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Step "Phase 8: Creating Configuration Files"

# Updated CMakeLists.txt for OpenCV 4.10.0
$cmakeContent = "cmake_minimum_required(VERSION 3.20)`n"
$cmakeContent += "project(iSFM VERSION 1.0.0 LANGUAGES CXX)`n"
$cmakeContent += "`n"
$cmakeContent += "set(CMAKE_CXX_STANDARD 17)`n"
$cmakeContent += "`n"
$cmakeContent += "# Find OpenCV 4.10.0 with CUDA support`n"
$cmakeContent += "find_package(OpenCV REQUIRED)`n"
$cmakeContent += "find_package(Eigen3 REQUIRED)`n"
$cmakeContent += "find_package(Ceres REQUIRED)`n"
$cmakeContent += "find_package(pybind11 REQUIRED)`n"
$cmakeContent += "`n"
$cmakeContent += "# Check for CUDA support in OpenCV`n"
$cmakeContent += "if(OpenCV_CUDA_VERSION)`n"
$cmakeContent += "    message(STATUS `"OpenCV built with CUDA `${OpenCV_CUDA_VERSION}`")`n"
$cmakeContent += "else()`n"
$cmakeContent += "    message(WARNING `"OpenCV built without CUDA support`")`n"
$cmakeContent += "endif()`n"
$cmakeContent += "`n"
$cmakeContent += "include_directories(cpp/include)`n"
$cmakeContent += "`n"
$cmakeContent += "file(GLOB_RECURSE SOURCES `"cpp/src/*.cpp`")`n"
$cmakeContent += "add_library(isfm_core STATIC `${SOURCES})`n"
$cmakeContent += "target_link_libraries(isfm_core `${OpenCV_LIBRARIES} Ceres::ceres Eigen3::Eigen)`n"
$cmakeContent += "`n"
$cmakeContent += "pybind11_add_module(isfm_pybind cpp/bindings/main.cpp)`n"
$cmakeContent += "target_link_libraries(isfm_pybind PRIVATE isfm_core)`n"

Set-Content -Path "CMakeLists.txt" -Value $cmakeContent -Encoding UTF8

# Create setup.py
$setupContent = "from setuptools import setup, find_packages`n"
$setupContent += "setup(`n"
$setupContent += "    name='isfm',`n"
$setupContent += "    version='0.1.0',`n"
$setupContent += "    packages=find_packages(),`n"
$setupContent += "    python_requires='>=3.9'`n"
$setupContent += ")`n"

Set-Content -Path "setup.py" -Value $setupContent -Encoding UTF8

# Create build.bat
$buildContent = "@echo off`n"
$buildContent += "echo Building iSFM with OpenCV 4.10.0...`n"
$buildContent += "call conda activate isfm`n"
$buildContent += "if not exist build mkdir build`n"
$buildContent += "cd build`n"
$buildContent += "cmake .. -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake`n"
$buildContent += "cmake --build . --config Release`n"
$buildContent += "cd ..`n"
$buildContent += "pip install -e .`n"
$buildContent += "echo Build complete!`n"

Set-Content -Path "build.bat" -Value $buildContent -Encoding UTF8

Write-Step "Phase 9: Creating Verification Script"

# Updated verification for OpenCV 4.10.0
$verifyContent = @"
import sys
import importlib

def check_package(name):
    try:
        mod = importlib.import_module(name)
        print(f'OK {name}')
        return True
    except ImportError:
        print(f'MISSING {name}')
        return False

print('=== iSFM Environment Verification (OpenCV 4.10.0) ===')
print(f'Python: {sys.version}')
print()

packages = ['numpy', 'scipy', 'matplotlib', 'cv2', 'torch', 'open3d']
all_good = all(check_package(pkg) for pkg in packages)

print()
try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f'OpenCV CUDA: {cuda_devices} devices')
    if cuda_devices > 0:
        print('OpenCV CUDA: ENABLED [OK]')
    else:
        print('OpenCV CUDA: DISABLED [X]')
except Exception as e:
    print(f'OpenCV CUDA: Error - {e}')

try:
    import torch
    if torch.cuda.is_available():
        print(f'PyTorch CUDA: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('PyTorch CUDA: not available')
except:
    print('PyTorch CUDA: error')

print()
if all_good:
    print('Environment ready!')
else:
    print('Missing packages')
"@

Set-Content -Path "verify_setup.py" -Value $verifyContent -Encoding UTF8

Write-Step "Setup Complete!"

Write-Host ""
Write-Host "=== iSFM SETUP FINISHED (OpenCV 4.10.0 + CUDA 12.6) ===" -ForegroundColor Green
Write-Host ""
Write-Host "INSTALLED:" -ForegroundColor Yellow
Write-Host "- Visual Studio Build Tools 2022" -ForegroundColor Green
Write-Host "- Miniconda with isfm environment" -ForegroundColor Green
Write-Host "- PyTorch with CUDA 12.1" -ForegroundColor Green
Write-Host "- OpenCV 4.10.0 with CUDA 12.6" -ForegroundColor Green
Write-Host "- C++ libraries (Eigen, Ceres)" -ForegroundColor Green
Write-Host "- OpenMP threading" -ForegroundColor Green
Write-Host "- Complete project structure" -ForegroundColor Green
Write-Host ""
Write-Host "OPENCV FEATURES FOR SfM:" -ForegroundColor Yellow
Write-Host "- CUDA-accelerated feature detection (SIFT)" -ForegroundColor Cyan
Write-Host "- CUDA-accelerated feature matching" -ForegroundColor Cyan
Write-Host "- CUDA-accelerated stereo vision" -ForegroundColor Cyan
Write-Host "- Classical computer vision modules" -ForegroundColor Cyan
Write-Host "- DNN with ONNX, Caffe support (no TFLite)" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE:" -ForegroundColor Yellow
Write-Host "- TensorFlow Lite disabled (avoids flatbuffers conflicts)" -ForegroundColor Gray
Write-Host "- Perfect for Structure-from-Motion workflows!" -ForegroundColor Cyan
Write-Host "- Your main.py and utils.py will work perfectly" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Restart PowerShell" -ForegroundColor Cyan
Write-Host "2. cd `"$ProjectPath`"" -ForegroundColor Cyan
Write-Host "3. conda activate isfm" -ForegroundColor Cyan
Write-Host "4. python verify_setup.py" -ForegroundColor Cyan
Write-Host "5. Copy your main.py and utils.py here" -ForegroundColor Cyan
Write-Host "6. python main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "For C++ building: .\build.bat" -ForegroundColor Yellow
Write-Host "Your RTX 4070 is ready for CUDA-accelerated SfM!" -ForegroundColor Green