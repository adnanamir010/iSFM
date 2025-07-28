#!/usr/bin/env python3
"""
Python unit tests for the enhanced Image class
"""

import unittest
import numpy as np
import cv2
import tempfile
import shutil
import os
from pathlib import Path

# Import the hybrid_sfm module (assuming it's properly built and installed)
import hybrid_sfm


class TestImage(unittest.TestCase):
    """Test cases for Image class functionality"""
    
    def setUp(self):
        """Create test images and directories"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_image_path = os.path.join(self.test_dir, "test.jpg")
        cv2.imwrite(self.test_image_path, self.test_image)
        
        # Create multiple test images for batch operations
        self.num_test_images = 5
        for i in range(self.num_test_images):
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            path = os.path.join(self.test_dir, f"image_{i:03d}.jpg")
            cv2.imwrite(path, img)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    def test_basic_loading(self):
        """Test basic image loading"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        self.assertEqual(img.width, 640)
        self.assertEqual(img.height, 480)
        self.assertEqual(img.channels, 3)
        self.assertIsNotNone(img.name)
        self.assertEqual(img.path, self.test_image_path)
    
    def test_lazy_loading(self):
        """Test lazy loading functionality"""
        img = hybrid_sfm.Image()
        # Note: The C++ enum is accessed via the class
        flags = hybrid_sfm.Image.LoadFlags.LAZY
        
        success = img.load(self.test_image_path, flags)
        self.assertTrue(success)
        
        # Data should be loaded on first access
        self.assertEqual(img.width, 640)
        self.assertEqual(img.height, 480)
    
    def test_numpy_interface(self):
        """Test numpy array conversion"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        # Get as numpy array
        arr = img.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (480, 640, 3))
        
        # Create from numpy
        new_img = hybrid_sfm.Image.from_numpy(arr, "from_numpy")
        self.assertEqual(new_img.width, 640)
        self.assertEqual(new_img.height, 480)
        self.assertEqual(new_img.name, "from_numpy")
    
    def test_pyramid_generation(self):
        """Test image pyramid generation"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        # Build pyramid
        num_levels = 4
        img.build_pyramid(num_levels, use_gpu=False)
        
        pyramid = img.pyramid
        self.assertEqual(pyramid.num_levels, num_levels)
        
        # Check level dimensions
        for i in range(num_levels):
            level = pyramid.get_level(i)
            expected_width = 640 // (2**i)
            expected_height = 480 // (2**i)
            self.assertEqual(level.shape[1], expected_width)
            self.assertEqual(level.shape[0], expected_height)
    
    @unittest.skipIf(not hybrid_sfm.has_cuda(), "CUDA not available")
    def test_gpu_operations(self):
        """Test GPU operations if CUDA is available"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        # Upload to GPU
        img.upload_to_gpu()
        self.assertTrue(img.is_on_gpu())
        
        # GPU grayscale conversion
        gray = img.get_gray(use_gpu=True)
        self.assertEqual(gray.ndim, 2)
        
        # GPU resize
        resized = img.get_resized(320, 240, use_gpu=True)
        self.assertEqual(resized.shape[:2], (240, 320))
    
    def test_memory_management(self):
        """Test memory usage tracking"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        initial_memory = img.memory_usage
        self.assertGreater(initial_memory, 0)
        
        if hybrid_sfm.has_cuda():
            # GPU upload should increase memory
            img.upload_to_gpu()
            gpu_memory = img.memory_usage
            self.assertGreater(gpu_memory, initial_memory)
            
            # Release GPU memory
            img.release_gpu_memory()
            after_release = img.memory_usage
            self.assertLess(after_release, gpu_memory)
    
    def test_undistortion(self):
        """Test image undistortion"""
        camera = hybrid_sfm.Camera(
            hybrid_sfm.Camera.Model.OPENCV,
            640, 480,
            [500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, 0.01]
        )
        
        img = hybrid_sfm.Image(self.test_image_path)
        undistorted = img.get_undistorted(camera, use_gpu=False)
        
        self.assertEqual(undistorted.shape, (480, 640, 3))
    
    def test_metadata(self):
        """Test metadata extraction"""
        img = hybrid_sfm.Image()
        flags = hybrid_sfm.Image.LoadFlags.EXTRACT_METADATA
        img.load(self.test_image_path, flags)
        
        metadata = img.metadata
        self.assertIsInstance(metadata.camera_make, str)
    
    def test_image_stats(self):
        """Test image statistics computation"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        stats = hybrid_sfm.compute_image_stats(img, use_gpu=False)
        
        self.assertIn('mean', stats)
        self.assertIn('stddev', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('histogram', stats)
        
        self.assertGreaterEqual(stats['min'], 0)
        self.assertLessEqual(stats['max'], 255)
    
    def test_batch_loading(self):
        """Test batch image loading"""
        options = hybrid_sfm.LoadOptions()
        options.num_threads = 2
        
        images = hybrid_sfm.load_images_from_directory(
            self.test_dir, options
        )
        
        self.assertEqual(len(images), self.num_test_images + 1)
        
        for img in images:
            self.assertIsNotNone(img)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)
    
    def test_image_sequence(self):
        """Test image sequence handling"""
        seq = hybrid_sfm.ImageSequence(self.test_dir)
        
        self.assertEqual(seq.frame_count, self.num_test_images + 1)
        self.assertTrue(seq.has_next())
        
        count = 0
        while seq.has_next():
            frame = seq.get_next_frame()
            self.assertIsNotNone(frame)
            count += 1
        
        self.assertEqual(count, self.num_test_images + 1)
        seq.reset()
        self.assertEqual(seq.current_index, 0)
    
    def test_format_detection(self):
        """Test image format detection"""
        self.assertEqual(hybrid_sfm.detect_image_format("test.jpg"), hybrid_sfm.ImageFormat.JPEG)
        self.assertEqual(hybrid_sfm.detect_image_format("test.png"), hybrid_sfm.ImageFormat.PNG)
        self.assertEqual(hybrid_sfm.detect_image_format("test.xyz"), hybrid_sfm.ImageFormat.UNKNOWN)
    
    def test_pfm_format(self):
        """Test PFM (Portable Float Map) format"""
        float_img = np.random.rand(100, 100).astype(np.float32)
        pfm_path = os.path.join(self.test_dir, "test.pfm")
        
        success = hybrid_sfm.save_pfm(pfm_path, float_img)
        self.assertTrue(success)
        
        loaded = hybrid_sfm.load_pfm(pfm_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, float_img.shape)
        self.assertEqual(loaded.dtype, np.float32)
    
    def test_batch_operations(self):
        """Test batch image operations"""
        images = hybrid_sfm.load_images_from_directory(self.test_dir)
        
        # Batch resize
        hybrid_sfm.batch_resize(images, max_dimension=160, use_gpu=False)
        
        for img in images:
            self.assertLessEqual(img.width, 160)
            self.assertLessEqual(img.height, 120) # 320x240 -> 160x120 or 640x480 -> 160x120
        
        # Batch normalization (check that it runs without error)
        try:
            hybrid_sfm.batch_normalize(images, mean=128.0, std=64.0, use_gpu=False)
        except Exception as e:
            self.fail(f"batch_normalize raised an exception: {e}")

    def test_batch_undistort(self):
        """Test batch image undistortion"""
        images = hybrid_sfm.load_images_from_directory(self.test_dir)
        camera = hybrid_sfm.Camera(
            hybrid_sfm.Camera.Model.OPENCV,
            320, 240, # For the smaller images in the batch
            [200.0, 200.0, 160.0, 120.0, 0.1, -0.05, 0.01, 0.01]
        )
        
        # Find a small image to test against
        small_image = next((img for img in images if img.width == 320), None)
        self.assertIsNotNone(small_image)
        
        original_data = small_image.to_numpy().copy()

        # Apply batch undistort
        hybrid_sfm.batch_undistort(images, camera, use_gpu=False)

        # Check that the data has changed
        undistorted_data = small_image.to_numpy()
        self.assertFalse(np.array_equal(original_data, undistorted_data))
        self.assertEqual(original_data.shape, undistorted_data.shape)

    def test_color_conversions(self):
        """Test color space conversions"""
        img = hybrid_sfm.Image(self.test_image_path)
        
        gray = img.get_gray(use_gpu=False)
        self.assertEqual(len(gray.shape), 2)
        
        bgr = img.to_numpy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        img_rgb = hybrid_sfm.Image.from_numpy(rgb, "rgb_image")
        self.assertEqual(img_rgb.width, img.width)
    
    def test_memory_limits(self):
        """Test memory limit enforcement"""
        hybrid_sfm.set_max_memory_usage(1024 * 1024)  # 1MB
        
        paths = [self.test_image_path] * 10
        options = hybrid_sfm.LoadOptions()
        options.skip_corrupted = True
        
        images = hybrid_sfm.load_images_batch(paths, options)
        
        self.assertLessEqual(len(images), len(paths))
        hybrid_sfm.set_max_memory_usage(4 * 1024 * 1024 * 1024)  # 4GB
    
    def test_thread_safety(self):
        """Test thread-safe operations"""
        import threading
        
        img = hybrid_sfm.Image(self.test_image_path)
        errors = []
        
        def worker(img, errors):
            try:
                _ = img.get_gray(use_gpu=False)
                _ = img.get_resized(160, 120, use_gpu=False)
                _ = hybrid_sfm.compute_image_stats(img)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker, args=(img, errors))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)


class TestImagePerformance(unittest.TestCase):
    """Performance tests for image operations"""
    
    def setUp(self):
        """Create larger test images for performance testing"""
        self.test_dir = tempfile.mkdtemp()
        
        self.large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        self.large_image_path = os.path.join(self.test_dir, "large.jpg")
        cv2.imwrite(self.large_image_path, self.large_image)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not hybrid_sfm.has_cuda(), "CUDA not available")
    def test_gpu_speedup(self):
        """Test GPU acceleration speedup"""
        img = hybrid_sfm.Image(self.large_image_path)
        
        import time
        
        start = time.time()
        _ = img.get_gray(use_gpu=False)
        cpu_time = time.time() - start
        
        start = time.time()
        img.upload_to_gpu()
        _ = img.get_gray(use_gpu=True)
        gpu_time = time.time() - start
        
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    
    def test_pyramid_performance(self):
        """Test pyramid generation performance"""
        img = hybrid_sfm.Image(self.large_image_path)
        
        import time
        
        start = time.time()
        img.build_pyramid(5, use_gpu=hybrid_sfm.has_cuda())
        pyramid_time = time.time() - start
        
        print(f"Pyramid generation time: {pyramid_time:.4f}s")
        self.assertLess(pyramid_time, 1.0)


if __name__ == '__main__':
    unittest.main()
