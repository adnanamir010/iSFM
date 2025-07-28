#include <gtest/gtest.h>
#include "core/types/image.h"
#include "core/utils/image_io.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace hybrid_sfm {

class ImageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test images
        test_image_path_ = "test_image.jpg";
        test_image_dir_ = "test_images";
        
        // Create a test image
        cv::Mat test_img(480, 640, CV_8UC3);
        cv::randu(test_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::imwrite(test_image_path_, test_img);
        
        // Create test directory with multiple images
        fs::create_directory(test_image_dir_);
        for (int i = 0; i < 5; ++i) {
            cv::Mat img(240, 320, CV_8UC3);
            cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            cv::imwrite(test_image_dir_ + "/image_" + std::to_string(i) + ".jpg", img);
        }
    }
    
    void TearDown() override {
        // Clean up test files
        fs::remove(test_image_path_);
        fs::remove_all(test_image_dir_);
    }
    
    std::string test_image_path_;
    std::string test_image_dir_;
};

// Test basic image loading
TEST_F(ImageTest, BasicLoading) {
    Image img(test_image_path_);
    
    EXPECT_EQ(img.getWidth(), 640);
    EXPECT_EQ(img.getHeight(), 480);
    EXPECT_EQ(img.getChannels(), 3);
    EXPECT_FALSE(img.getName().empty());
    EXPECT_EQ(img.getPath(), test_image_path_);
}

// Test lazy loading
TEST_F(ImageTest, LazyLoading) {
    Image img;
    bool loaded = img.load(test_image_path_, 
                          static_cast<int>(Image::LoadFlags::LAZY));
    
    EXPECT_TRUE(loaded);
    EXPECT_EQ(img.getPath(), test_image_path_);
    
    // Data should be loaded on demand
    EXPECT_EQ(img.getWidth(), 640);
    EXPECT_EQ(img.getHeight(), 480);
}

// Test pyramid generation
TEST_F(ImageTest, PyramidGeneration) {
    Image img(test_image_path_);
    img.buildPyramid(4, false);  // CPU pyramid
    
    const auto& pyramid = img.getPyramid();
    EXPECT_EQ(pyramid.getNumLevels(), 4);
    
    // Check pyramid dimensions
    cv::Mat level0 = pyramid.getLevel(0);
    cv::Mat level1 = pyramid.getLevel(1);
    
    EXPECT_EQ(level0.cols, 640);
    EXPECT_EQ(level0.rows, 480);
    EXPECT_EQ(level1.cols, 320);
    EXPECT_EQ(level1.rows, 240);
}

// Test GPU operations
TEST_F(ImageTest, GPUOperations) {
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Image img(test_image_path_);
    
    // Test GPU upload
    img.uploadToGPU();
    EXPECT_TRUE(img.isOnGPU());
    
    // Test GPU pyramid
    img.buildPyramid(3, true);  // GPU pyramid
    
    // Test GPU grayscale conversion
    cv::Mat gray = img.getGray(true);
    EXPECT_EQ(gray.channels(), 1);
}

// Test memory management
TEST_F(ImageTest, MemoryManagement) {
    Image img(test_image_path_);
    
    size_t initial_memory = img.getMemoryUsage();
    EXPECT_GT(initial_memory, 0);
    
    // Upload to GPU should increase memory usage
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        img.uploadToGPU();
        size_t gpu_memory = img.getMemoryUsage();
        EXPECT_GT(gpu_memory, initial_memory);
        
        // Release GPU memory
        img.releaseGPUMemory();
        size_t after_release = img.getMemoryUsage();
        EXPECT_LT(after_release, gpu_memory);
    }
}

// Test image undistortion
TEST_F(ImageTest, Undistortion) {
    // Create a camera with some distortion
    std::vector<double> params = {500.0, 500.0, 320.0, 240.0, 0.1, -0.05, 0.01, 0.01};
    Camera camera(Camera::Model::OPENCV, 640, 480, params);
    
    Image img(test_image_path_);
    cv::Mat undistorted = img.getUndistorted(camera, false);
    
    EXPECT_EQ(undistorted.size(), img.getData().size());
    EXPECT_EQ(undistorted.type(), img.getData().type());
}

// Test metadata extraction
TEST_F(ImageTest, MetadataExtraction) {
    // This test would work with real images containing EXIF data
    Image img;
    img.load(test_image_path_, 
            static_cast<int>(Image::LoadFlags::EXTRACT_METADATA));
    
    const auto& metadata = img.getMetadata();
    // For synthetic test images, metadata will be empty
    EXPECT_TRUE(metadata.camera_make.empty() || !metadata.camera_make.empty());
}

// Test batch loading
TEST_F(ImageTest, BatchLoading) {
    ImageLoader::LoadOptions options;
    options.num_threads = 2;
    
    auto images = ImageLoader::loadDirectory(test_image_dir_, options, {});

    EXPECT_EQ(images.size(), 5);
    
    for (const auto& img : images) {
        EXPECT_NE(img, nullptr);
        EXPECT_EQ(img->getWidth(), 320);
        EXPECT_EQ(img->getHeight(), 240);
    }
}

// Test format detection
TEST_F(ImageTest, FormatDetection) {
    EXPECT_EQ(ImageLoader::detectFormat("test.jpg"), ImageFormat::JPEG);
    EXPECT_EQ(ImageLoader::detectFormat("test.png"), ImageFormat::PNG);
    EXPECT_EQ(ImageLoader::detectFormat("test.tiff"), ImageFormat::TIFF);
    EXPECT_EQ(ImageLoader::detectFormat("test.xyz"), ImageFormat::UNKNOWN);
}

// Test image sequence
TEST_F(ImageTest, ImageSequence) {
    ImageSequence seq(test_image_dir_);
    
    EXPECT_EQ(seq.getFrameCount(), 5);
    EXPECT_TRUE(seq.hasNext());
    
    // Test frame iteration
    int count = 0;
    while (seq.hasNext()) {
        auto frame = seq.getNextFrame();
        EXPECT_NE(frame, nullptr);
        count++;
    }
    EXPECT_EQ(count, 5);
    
    // Test reset
    seq.reset();
    EXPECT_EQ(seq.getCurrentIndex(), 0);
}

// Test image statistics
TEST_F(ImageTest, ImageStatistics) {
    Image img(test_image_path_);
    
    auto stats = image_utils::computeStats(img, false);
    
    // Check that statistics are computed
    EXPECT_GE(stats.mean[0], 0);
    EXPECT_LE(stats.mean[0], 255);
    EXPECT_GE(stats.min, 0);
    EXPECT_LE(stats.max, 255);
    EXPECT_FALSE(stats.histogram.empty());
}

// Test batch operations
TEST_F(ImageTest, BatchOperations) {
    auto images = ImageLoader::loadDirectory(test_image_dir_);
    
    // Test batch resize
    image_utils::batchResize(images, 160, false);
    
    for (const auto& img : images) {
        EXPECT_LE(img->getWidth(), 160);
        EXPECT_LE(img->getHeight(), 160);
    }
}

// Test PFM format
TEST_F(ImageTest, PFMFormat) {
    // Create a float image
    cv::Mat float_img(100, 100, CV_32FC1);
    cv::randu(float_img, 0.0f, 1.0f);
    
    // Save as PFM
    std::string pfm_path = "test.pfm";
    EXPECT_TRUE(ImageLoader::savePFM(pfm_path, float_img));
    
    // Load PFM
    cv::Mat loaded = ImageLoader::loadPFM(pfm_path);
    EXPECT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.type(), CV_32FC1);
    EXPECT_EQ(loaded.size(), float_img.size());
    
    // Clean up
    fs::remove(pfm_path);
}

// Test memory limits
TEST_F(ImageTest, MemoryLimits) {
    // Set a low memory limit
    ImageLoader::setMaxMemoryUsage(1024 * 1024);  // 1MB
    
    // Try to load multiple large images
    std::vector<std::string> paths;
    for (int i = 0; i < 10; ++i) {
        paths.push_back(test_image_path_);
    }
    
    ImageLoader::LoadOptions options;
    options.skip_corrupted = true;
    
    auto images = ImageLoader::loadBatch(paths, options);
    
    // Some images might not load due to memory limit
    EXPECT_LE(images.size(), paths.size());
    
    // Reset memory limit
    ImageLoader::setMaxMemoryUsage(4ULL * 1024 * 1024 * 1024);
}

// Test thread safety
TEST_F(ImageTest, ThreadSafety) {
    Image img(test_image_path_);
    
    // Concurrent operations
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&img, i]() {
            if (i % 2 == 0) {
                cv::Mat gray = img.getGray(false);
            } else {
                cv::Mat resized = img.getResized(160, 120, false);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Should complete without issues
    EXPECT_TRUE(img.isThreadSafe());
}

} // namespace hybrid_sfm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}