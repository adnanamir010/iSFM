#pragma once

#include "common.h"
#include "camera.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

namespace hybrid_sfm {

// Forward declarations
class ImageMetadata;
class ImagePyramid;

// Image metadata container
class ImageMetadata {
public:
    // Camera info
    std::string camera_make;
    std::string camera_model;
    
    // EXIF data
    double focal_length_mm = -1.0;
    double exposure_time = -1.0;
    double iso_speed = -1.0;
    double aperture = -1.0;
    
    // Timestamps
    int64_t capture_time = -1;
    
    // GPS data (if available)
    bool has_gps = false;
    double latitude = 0.0;
    double longitude = 0.0;
    double altitude = 0.0;
    
    // Parsed from EXIF
    void parseFromExif(const std::string& path);
};

// GPU memory manager for images
class GPUImageMemory {
public:
    GPUImageMemory();
    ~GPUImageMemory();
    
    // Upload/download
    void upload(const cv::Mat& cpu_img);
    cv::Mat download() const;
    
    // Get GPU mat
    const cv::cuda::GpuMat& getGpuMat() const { return gpu_data_; }
    cv::cuda::GpuMat& getGpuMat() { return gpu_data_; }
    
    // Memory info
    size_t getMemoryUsage() const;
    bool isOnGPU() const { return !gpu_data_.empty(); }
    
private:
    cv::cuda::GpuMat gpu_data_;
    mutable std::mutex mutex_;
};

// Image pyramid for multi-scale processing
class ImagePyramid {
public:
    ImagePyramid() = default;
    
    // Build pyramid
    void build(const cv::Mat& base_image, int num_levels, bool use_gpu = true);
    void buildGPU(const cv::cuda::GpuMat& base_image, int num_levels);
    
    // Access levels
    cv::Mat getLevel(int level) const;
    cv::cuda::GpuMat getGPULevel(int level) const;
    
    // Properties
    int getNumLevels() const { return static_cast<int>(cpu_pyramid_.size()); }
    double getScaleFactor() const { return scale_factor_; }
    
    // Memory management
    void releaseGPU();
    void releaseCPU();
    
private:
    std::vector<cv::Mat> cpu_pyramid_;
    std::vector<cv::cuda::GpuMat> gpu_pyramid_;
    double scale_factor_ = 0.5;  // Default: halve resolution each level
    mutable std::mutex mutex_;
};

// Enhanced Image class with GPU support and pyramids
class Image {
public:
    enum class LoadFlags {
        LAZY = 1 << 0,          // Don't load pixel data immediately
        LOAD_GPU = 1 << 1,      // Load directly to GPU
        BUILD_PYRAMID = 1 << 2,  // Build pyramid on load
        EXTRACT_METADATA = 1 << 3 // Extract EXIF metadata
    };
    friend class GPUImageMemory;
    
    // Constructors
    Image();
    explicit Image(const std::string& path, int flags = 0);
    Image(const cv::Mat& data, const std::string& name = "");
    
    // Disable copy, enable move
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;
    
    // Loading
    bool load(const std::string& path, int flags = 0);
    bool loadLazy(const std::string& path);
    void ensureLoaded() const;
    
    // Save
    bool save(const std::string& path) const;
    
    // GPU operations
    void uploadToGPU();
    void downloadFromGPU();
    bool isOnGPU() const { return gpu_memory_ && gpu_memory_->isOnGPU(); }
    cv::cuda::GpuMat getGpuMat();
    
    // Pyramid operations
    void buildPyramid(int levels, bool use_gpu = true);
    const ImagePyramid& getPyramid() const { return pyramid_; }
    cv::Mat getPyramidLevel(int level) const;
    
    // Undistortion
    cv::Mat getUndistorted(const Camera& camera, bool use_gpu = true) const;
    void undistort(const Camera& camera, bool use_gpu = true);
    
    // Basic operations
    cv::Mat getGray(bool use_gpu = false) const;
    cv::Mat getResized(int width, int height, bool use_gpu = false) const;
    
    // Getters
    const cv::Mat& getData() const;
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;
    const std::string& getName() const { return name_; }
    const std::string& getPath() const { return path_; }
    size_t getId() const { return id_; }
    const ImageMetadata& getMetadata() const { return metadata_; }
    
    // Setters
    void setName(const std::string& name) { name_ = name; }
    
    // Memory management
    size_t getMemoryUsage() const;
    void releaseGPUMemory();
    static size_t getTotalGPUMemoryUsage();
    
    // Thread safety
    bool isThreadSafe() const { return true; }  // All operations are mutex-protected
    
private:
    // Core data
    mutable cv::Mat data_;  // Mutable for lazy loading
    std::string name_;
    std::string path_;
    size_t id_;
    
    // GPU support
    std::unique_ptr<GPUImageMemory> gpu_memory_;
    
    // Pyramid
    ImagePyramid pyramid_;
    
    // Metadata
    ImageMetadata metadata_;
    
    // Lazy loading
    mutable bool is_loaded_ = true;
    mutable std::mutex load_mutex_;
    
    // Undistortion cache
    mutable cv::Mat undistorted_cache_;
    mutable int undistort_camera_model_ = -1;
        
    // Global state
    static std::atomic<size_t> next_id_;
    static std::atomic<size_t> total_gpu_memory_;
    
    // Helper functions
    void loadMetadata();
};

// Inline flag operators
inline Image::LoadFlags operator|(Image::LoadFlags a, Image::LoadFlags b) {
    return static_cast<Image::LoadFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline bool operator&(int flags, Image::LoadFlags check) {
    return (flags & static_cast<int>(check)) != 0;
}

} // namespace hybrid_sfm