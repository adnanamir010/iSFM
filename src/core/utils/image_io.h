#pragma once

#include "core/types/image.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <future>

namespace hybrid_sfm {

// Image format detection
enum class ImageFormat {
    UNKNOWN,
    JPEG,
    PNG,
    TIFF,
    BMP,
    WEBP,
    RAW,    // Various RAW formats
    PFM,    // Portable float map
    EXR     // OpenEXR
};

// Image loader with format detection and batch processing
class ImageLoader {
public:
    struct LoadOptions {
        int flags = 0;  // Image::LoadFlags
        int max_dimension = -1;  // Resize if larger
        int num_threads = -1;  // -1 for auto
        bool skip_corrupted = true;
        std::function<void(const std::string&, bool)> progress_callback;
    };
    
    // Single image loading
    static std::unique_ptr<Image> load(const std::string& path, 
                                    const LoadOptions& options);
    static std::unique_ptr<Image> load(const std::string& path) {
        return load(path, LoadOptions());
    }

    // Batch loading
    static std::vector<std::unique_ptr<Image>> loadBatch(
        const std::vector<std::string>& paths,
        const LoadOptions& options);
    static std::vector<std::unique_ptr<Image>> loadBatch(
        const std::vector<std::string>& paths) {
        return loadBatch(paths, LoadOptions());
    }

    // Directory loading
    static std::vector<std::unique_ptr<Image>> loadDirectory(
        const std::string& directory,
        const LoadOptions& options,
        const std::vector<std::string>& extensions);
    static std::vector<std::unique_ptr<Image>> loadDirectory(
        const std::string& directory) {
        return loadDirectory(directory, LoadOptions(), {});
    }
        
    // Format detection
    static ImageFormat detectFormat(const std::string& path);
    static bool isImageFile(const std::string& path);
    static std::vector<std::string> getSupportedExtensions();
    
    // RAW format support
    static cv::Mat loadRAW(const std::string& path);
    
    // HDR format support
    static cv::Mat loadPFM(const std::string& path);
    static bool savePFM(const std::string& path, const cv::Mat& image);
    
    // Memory management
    static void setMaxMemoryUsage(size_t max_bytes);
    static size_t getCurrentMemoryUsage();
    
    static std::vector<std::string> listImageFiles(
        const std::string& directory,
        const std::vector<std::string>& extensions);


private:    
    static std::unique_ptr<Image> loadSingle(
        const std::string& path,
        const LoadOptions& options);
};

// Image sequence handler for video-like data
class ImageSequence {
public:
    ImageSequence(const std::string& pattern_or_directory);
    
    // Frame access
    std::unique_ptr<Image> getFrame(int index) const;
    std::unique_ptr<Image> getNextFrame();
    
    // Properties
    int getFrameCount() const { return static_cast<int>(image_paths_.size()); }
    int getCurrentIndex() const { return current_index_; }
    double getFPS() const { return fps_; }
    
    // Iteration
    void reset() { current_index_ = 0; }
    bool hasNext() const { return current_index_ < getFrameCount(); }
    
    // Preloading for performance
    void preloadFrames(int start, int count);
    void clearCache();
    
private:
    std::vector<std::string> image_paths_;
    int current_index_ = 0;
    double fps_ = 30.0;  // Default FPS
    
    // Cache for preloaded frames
    mutable std::unordered_map<int, std::unique_ptr<Image>> cache_;
    mutable std::mutex cache_mutex_;
    
    void parsePattern(const std::string& pattern);
};

// Utility functions
namespace image_utils {
    // Image statistics
    struct ImageStats {
        cv::Scalar mean;
        cv::Scalar stddev;
        double min, max;
        cv::Mat histogram;
    };
    
    ImageStats computeStats(const Image& image, bool use_gpu = false);
    
    // Image preprocessing
    cv::Mat normalizeImage(const cv::Mat& image, double target_mean = 128.0, 
                          double target_std = 64.0);
    
    // Batch operations
    void batchResize(std::vector<std::unique_ptr<Image>>& images, 
                    int max_dimension, bool use_gpu = true);
    
    void batchUndistort(std::vector<std::unique_ptr<Image>>& images,
                       const Camera& camera, bool use_gpu = true);
    
    // Memory utilities
    std::string formatMemorySize(size_t bytes);
    void optimizeMemoryLayout(std::vector<std::unique_ptr<Image>>& images);
}

} // namespace hybrid_sfm