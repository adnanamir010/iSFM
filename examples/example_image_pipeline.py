#!/usr/bin/env python3
"""
Example script demonstrating the Day 3 Image Data Pipeline
Shows GPU acceleration, pyramid generation, and batch processing
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Add build directory to path
sys.path.insert(0, '../build')
import hybrid_sfm


def demo_basic_loading():
    """Demonstrate basic image loading and operations"""
    print("\n=== Basic Image Loading ===")
    
    # Load an image
    img_path = "sample_images/indoor_scene.jpg"
    
    # Standard loading
    img = hybrid_sfm.Image(img_path)
    print(f"Loaded image: {img.name}")
    print(f"Dimensions: {img.width} x {img.height} x {img.channels}")
    print(f"Memory usage: {hybrid_sfm.format_memory_size(img.memory_usage)}")
    
    # Lazy loading
    img_lazy = hybrid_sfm.Image()
    img_lazy.load_lazy(img_path)
    print(f"\nLazy loaded: {img_lazy.path}")
    print(f"Accessing dimensions triggers load: {img_lazy.width} x {img_lazy.height}")
    
    # Load with options
    flags = hybrid_sfm.ImageLoadFlags.EXTRACT_METADATA | hybrid_sfm.ImageLoadFlags.BUILD_PYRAMID
    img_full = hybrid_sfm.Image(img_path, flags)
    print(f"\nLoaded with metadata and pyramid")
    print(f"Camera: {img_full.metadata.camera_make} {img_full.metadata.camera_model}")
    print(f"Pyramid levels: {img_full.pyramid.num_levels}")


def demo_gpu_operations():
    """Demonstrate GPU-accelerated operations"""
    print("\n=== GPU Operations ===")
    
    if not hybrid_sfm.has_cuda():
        print("CUDA not available, skipping GPU demo")
        return
    
    print(f"CUDA devices: {hybrid_sfm.get_cuda_device_count()}")
    
    # Create a large test image
    large_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    img = hybrid_sfm.Image.from_numpy(large_img, "large_test")
    
    # Time CPU vs GPU operations
    # CPU grayscale
    start = time.time()
    gray_cpu = img.get_gray(use_gpu=False)
    cpu_time = time.time() - start
    
    # GPU grayscale (including upload)
    start = time.time()
    img.upload_to_gpu()
    gray_gpu = img.get_gray(use_gpu=True)
    gpu_time = time.time() - start
    
    print(f"Grayscale conversion (2048x2048):")
    print(f"  CPU time: {cpu_time*1000:.2f} ms")
    print(f"  GPU time: {gpu_time*1000:.2f} ms (including upload)")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    
    # GPU pyramid generation
    start = time.time()
    img.build_pyramid(5, use_gpu=True)
    pyramid_time = time.time() - start
    print(f"\nGPU pyramid generation (5 levels): {pyramid_time*1000:.2f} ms")
    
    # Memory usage
    print(f"\nTotal GPU memory usage: {hybrid_sfm.format_memory_size(hybrid_sfm.Image.get_total_gpu_memory_usage())}")


def demo_pyramid_visualization():
    """Visualize image pyramid"""
    print("\n=== Image Pyramid Visualization ===")
    
    # Load or create test image
    img_path = "sample_images/indoor_scene.jpg"
    if not Path(img_path).exists():
        # Create synthetic image with patterns
        img_data = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add checkerboard pattern
        for i in range(0, 512, 32):
            for j in range(0, 512, 32):
                if (i//32 + j//32) % 2 == 0:
                    img_data[i:i+32, j:j+32] = [255, 255, 255]
        img = hybrid_sfm.Image.from_numpy(img_data, "checkerboard")
    else:
        img = hybrid_sfm.Image(img_path)
    
    # Build pyramid
    img.build_pyramid(4, use_gpu=hybrid_sfm.has_cuda())
    
    # Visualize pyramid levels
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(4):
        level = img.get_pyramid_level(i)
        axes[i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Level {i}: {level.shape[1]}x{level.shape[0]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('pyramid_visualization.png')
    print("Saved pyramid visualization to pyramid_visualization.png")


def demo_batch_processing():
    """Demonstrate batch image processing"""
    print("\n=== Batch Processing ===")
    
    # Create test images
    test_dir = Path("test_batch_images")
    test_dir.mkdir(exist_ok=True)
    
    print("Creating test images...")
    for i in range(10):
        img = np.random.randint(0, 255, (480 + i*50, 640 + i*50, 3), dtype=np.uint8)
        cv2.imwrite(str(test_dir / f"image_{i:03d}.jpg"), img)
    
    # Load batch with options
    options = hybrid_sfm.LoadOptions()
    options.max_dimension = 800
    options.num_threads = 4
    options.flags = hybrid_sfm.ImageLoadFlags.BUILD_PYRAMID
    
    # Progress callback
    loaded = []
    def progress(path, success):
        loaded.append((Path(path).name, success))
        print(f"  Loaded {Path(path).name}: {'✓' if success else '✗'}")
    
    options.progress_callback = progress
    
    print("\nLoading images...")
    start = time.time()
    images = hybrid_sfm.load_images_from_directory(str(test_dir), options)
    load_time = time.time() - start
    
    print(f"\nLoaded {len(images)} images in {load_time:.2f}s")
    print(f"Average: {load_time/len(images)*1000:.1f} ms per image")
    
    # Batch resize
    print("\nBatch resizing to max dimension 400...")
    start = time.time()
    hybrid_sfm.batch_resize(images, 400, use_gpu=hybrid_sfm.has_cuda())
    resize_time = time.time() - start
    print(f"Resized in {resize_time:.2f}s")
    
    # Check memory usage
    total_memory = sum(img.memory_usage for img in images)
    print(f"\nTotal memory usage: {hybrid_sfm.format_memory_size(total_memory)}")
    
    # Memory optimization
    print("\nOptimizing memory layout...")
    hybrid_sfm.optimize_memory_layout(images)
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)


def demo_image_sequence():
    """Demonstrate image sequence handling"""
    print("\n=== Image Sequence ===")
    
    # Create test sequence
    seq_dir = Path("test_sequence")
    seq_dir.mkdir(exist_ok=True)
    
    print("Creating image sequence...")
    for i in range(30):
        # Create images with moving square
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(320 + 200 * np.sin(i * 0.2))
        y = int(240 + 100 * np.cos(i * 0.2))
        cv2.rectangle(img, (x-50, y-50), (x+50, y+50), (0, 255, 0), -1)
        cv2.imwrite(str(seq_dir / f"frame_{i:04d}.jpg"), img)
    
    # Load sequence
    seq = hybrid_sfm.ImageSequence(str(seq_dir))
    print(f"Loaded sequence with {seq.frame_count} frames")
    print(f"FPS: {seq.fps}")
    
    # Preload some frames
    print("\nPreloading frames 0-10...")
    seq.preload_frames(0, 10)
    
    # Process sequence
    print("\nProcessing sequence...")
    motion_x = []
    motion_y = []
    
    seq.reset()
    prev_frame = None
    
    while seq.has_next():
        frame = seq.get_next_frame()
        if frame is None:
            break
        
        # Simple motion detection (find green square)
        frame_np = frame.to_numpy()
        green_mask = cv2.inRange(frame_np, (0, 250, 0), (0, 255, 0))
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                motion_x.append(cx)
                motion_y.append(cy)
    
    # Plot motion
    if motion_x:
        plt.figure(figsize=(10, 6))
        plt.plot(motion_x, motion_y, 'b-', linewidth=2)
        plt.scatter(motion_x[0], motion_y[0], c='g', s=100, label='Start')
        plt.scatter(motion_x[-1], motion_y[-1], c='r', s=100, label='End')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Object Motion Path')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('motion_path.png')
        print("Saved motion path to motion_path.png")
    
    # Clean up
    seq.clear_cache()
    import shutil
    shutil.rmtree(seq_dir)


def demo_undistortion():
    """Demonstrate image undistortion"""
    print("\n=== Image Undistortion ===")
    
    # Create test image with grid pattern
    img_data = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw grid
    for i in range(0, 640, 40):
        cv2.line(img_data, (i, 0), (i, 480), (255, 255, 255), 1)
    for i in range(0, 480, 40):
        cv2.line(img_data, (0, i), (640, i), (255, 255, 255), 1)
    
    img = hybrid_sfm.Image.from_numpy(img_data, "grid")
    
    # Create camera with significant distortion
    camera = hybrid_sfm.Camera(
        hybrid_sfm.Camera.Model.OPENCV,
        640, 480,
        [400.0, 400.0, 320.0, 240.0, 0.2, -0.1, 0.02, -0.01]
    )
    
    # Apply undistortion
    print("Applying undistortion...")
    undistorted = img.get_undistorted(camera, use_gpu=hybrid_sfm.has_cuda())
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(img.to_numpy())
    ax1.set_title('Original (with distortion model)')
    ax1.axis('off')
    
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('undistortion_demo.png')
    print("Saved undistortion demo to undistortion_demo.png")


def demo_metadata_and_stats():
    """Demonstrate metadata extraction and image statistics"""
    print("\n=== Metadata and Statistics ===")
    
    # Create test image
    img_data = np.random.normal(128, 32, (480, 640, 3)).astype(np.uint8)
    img = hybrid_sfm.Image.from_numpy(img_data, "random_normal")
    
    # Compute statistics
    stats = hybrid_sfm.compute_image_stats(img, use_gpu=False)
    
    print("Image Statistics:")
    print(f"  Mean: {stats['mean'][:3]}")  # RGB channels
    print(f"  Std Dev: {stats['stddev'][:3]}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.plot(stats['histogram'], 'b-', linewidth=2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.title('Image Histogram')
    plt.grid(True)
    plt.savefig('histogram.png')
    print("\nSaved histogram to histogram.png")
    
    # Normalize image
    normalized = hybrid_sfm.normalize_image(img_data, target_mean=128.0, target_std=32.0)
    print(f"\nNormalized image shape: {normalized.shape}")


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Hybrid SfM - Day 3: Image Data Pipeline Demo")
    print("=" * 60)
    
    # Create sample images directory
    Path("sample_images").mkdir(exist_ok=True)
    
    # Run demos
    try:
        demo_basic_loading()
        demo_gpu_operations()
        demo_pyramid_visualization()
        demo_batch_processing()
        demo_image_sequence()
        demo_undistortion()
        demo_metadata_and_stats()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()