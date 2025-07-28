# iSFM Project Documentation

This document provides a detailed overview of the iSFM (incremental Structure-from-Motion) project's architecture, core components, and Python API.

## 1. Project Philosophy

The iSFM project is built on a hybrid C++/Python architecture:

- **C++ Core:** All performance-critical algorithms, data structures, and computations are implemented in C++. This provides the speed necessary for complex computer vision tasks.
- **Python Bindings:** A clean, user-friendly Python API is exposed using `pybind11`. This allows for rapid prototyping, testing, and integration with the broader Python data science ecosystem (e.g., NumPy, Matplotlib).

## 2. Core C++ Data Structures

The foundation of the library resides in the `src/core/types` directory.

### `Camera` & `CameraPose`

- **Headers:** `src/core/types/camera.h`
- **Description:** These classes represent the intrinsic (`Camera`) and extrinsic (`CameraPose`) properties of a camera. The system supports 10 different COLMAP-compatible camera models, handles complex lens distortions, and uses Eigen Quaternions for robust rotation representation.
- **Key Functionality:**
  - Projects 3D world points into 2D image coordinates (`worldToImage`).
  - Unprojects 2D image points back into the 3D world (`imageToWorld`).
  - Handles complex lens distortions (radial, tangential, fisheye).
  - Can be serialized to and from the COLMAP file format.

### `Image`

- **Header:** `src/core/types/image.h`
- **Description:** A sophisticated image container designed for high-performance computer vision.
- **Key Features:**
  - **Lazy Loading:** Can be initialized with a path without immediately loading pixel data into memory, which is efficient for large datasets. Data is loaded on first access.
  - **GPU Acceleration:** Seamlessly manages image data on both CPU (`cv::Mat`) and GPU (`cv::cuda::GpuMat`). Operations can be explicitly run on the GPU for significant speedups.
  - **Image Pyramids:** Built-in support for generating multi-scale image pyramids, a crucial component for many feature detection and matching algorithms.
  - **Metadata:** Automatically extracts and stores EXIF metadata using `Exiv2`.
  - **Thread-Safe:** Uses mutexes to protect internal data, allowing for safe use in multi-threaded applications.

## 3. Python API Reference (`hybrid_sfm` module)

The following classes and functions are available in the `hybrid_sfm` Python module.

### `hybrid_sfm.Camera`

Represents camera intrinsics.

**Example:**

```python
import hybrid_sfm
import numpy as np

# Create a simple pinhole camera for a 1920x1080 image
# The constructor automatically sets reasonable default focal length and principal point
camera = hybrid_sfm.Camera(hybrid_sfm.Camera.Model.PINHOLE, 1920, 1080)

# Get the intrinsic matrix K
K = camera.get_K()
print("Intrinsic Matrix K:\n", K)

# Project a 3D point into the camera
point_3d = np.array([1.0, 0.5, 10.0])
point_2d = camera.world_to_image(point_3d)
print(f"Projected {point_3d} -> {point_2d}")
```

### `hybrid_sfm.CameraPose`

Represents camera extrinsics (position and orientation).

**Example:**

```python
# Create an identity pose (no rotation, no translation)
pose = hybrid_sfm.CameraPose()

# Get the 4x4 transformation matrix
transform_matrix = pose.get_transform_matrix()
print("Transform Matrix:\n", transform_matrix)
```

### `hybrid_sfm.Image`

The main class for handling image data.

**Loading an Image:**

```python
# Standard loading
img = hybrid_sfm.Image("path/to/image.jpg")

# Lazy loading (pixel data is not read until first access)
img_lazy = hybrid_sfm.Image()
img_lazy.load_lazy("path/to/image.jpg")

# Create from a NumPy array
numpy_array = np.zeros((100, 200, 3), dtype=np.uint8)
img_from_np = hybrid_sfm.Image.from_numpy(numpy_array, "my_image")
```

**Accessing Properties:**

```python
print(f"Dimensions: {img.width}x{img.height}")
print(f"Is on GPU: {img.is_on_gpu()}")
print(f"Memory usage: {hybrid_sfm.format_memory_size(img.memory_usage)}")

# Access metadata (requires EXTRACT_METADATA flag on load)
img_meta = hybrid_sfm.Image("path/to/real_photo.jpg", hybrid_sfm.Image.LoadFlags.EXTRACT_METADATA)
print(f"Camera Make: {img_meta.metadata.camera_make}")
```

**Image Processing & GPU Operations:**

```python
# Upload to GPU (if CUDA is available)
if hybrid_sfm.has_cuda():
    img.upload_to_gpu()

# Get a grayscale version (runs on GPU if available and requested)
gray_np_array = img.get_gray(use_gpu=True)

# Resize the image (returns a new NumPy array)
resized_np_array = img.get_resized(width=320, height=240, use_gpu=True)

# Resize the image in-place
img.resize(width=320, height=240, use_gpu=True)

# Undistort the image using a camera model
undistorted_np_array = img.get_undistorted(camera, use_gpu=True)
```

**Image Pyramids:**

```python
# Build a 4-level pyramid (on GPU if available)
img.build_pyramid(levels=4, use_gpu=True)

# Access a level of the pyramid as a NumPy array
level_2_np = img.pyramid.get_level(2)
print(f"Pyramid level 2 shape: {level_2_np.shape}")
```

### Batch Processing Functions

These functions are designed to efficiently process many images at once, often using multiple threads.

```python
# Load all images from a directory, resizing them to a max dimension of 1600px
options = hybrid_sfm.LoadOptions()
options.max_dimension = 1600
options.num_threads = 4 # Use 4 threads for loading
images = hybrid_sfm.load_images_from_directory("path/to/dataset", options)

# Resize a list of images in-place
hybrid_sfm.batch_resize(images, max_dimension=800, use_gpu=True)

# Undistort a list of images in-place
hybrid_sfm.batch_undistort(images, camera, use_gpu=True)

# Normalize a list of images in-place
hybrid_sfm.batch_normalize(images, mean=128.0, std=64.0, use_gpu=False)
```

## 4\. Building and Development

The project uses CMake to manage the build process.

- **C++ Sources:** Located in `src/`.
- **Python Bindings:** Located in `bindings/`. Any new C++ function or class that needs to be exposed to Python must be added to a file in this directory.
- **Build Command:** After making C++ changes, run `make` (or `ninja`) in the `build` directory to recompile.

See the [`README.md`](README.md) for detailed, step-by-step build instructions.
