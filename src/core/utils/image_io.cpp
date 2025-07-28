#include "core/utils/image_io.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <algorithm>
#include <execution>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <regex>

namespace fs = std::filesystem;

namespace hybrid_sfm {

// Static memory limit
static std::atomic<size_t> max_memory_usage{4ULL * 1024 * 1024 * 1024};  // 4GB default
static std::atomic<size_t> current_memory_usage{0};

// ===================== ImageLoader Implementation =====================

std::unique_ptr<Image> ImageLoader::load(const std::string& path, 
                                       const LoadOptions& options) {
    return loadSingle(path, options);
}

std::vector<std::unique_ptr<Image>> ImageLoader::loadBatch(
    const std::vector<std::string>& paths,
    const LoadOptions& options) {
    
    int num_threads = options.num_threads;
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    std::vector<std::unique_ptr<Image>> images;
    images.resize(paths.size());
    
    // Parallel loading
    std::vector<std::future<std::unique_ptr<Image>>> futures;
    
    for (size_t i = 0; i < paths.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            auto img = loadSingle(paths[i], options);
            
            // Progress callback
            if (options.progress_callback) {
                options.progress_callback(paths[i], img != nullptr);
            }
            
            return img;
        }));
        
        // Limit concurrent threads
        if (futures.size() >= static_cast<size_t>(num_threads)) {
            for (size_t j = 0; j < futures.size(); ++j) {
                images[i - futures.size() + j + 1] = futures[j].get();
            }
            futures.clear();
        }
    }
    
    // Get remaining results
    for (size_t j = 0; j < futures.size(); ++j) {
        images[paths.size() - futures.size() + j] = futures[j].get();
    }
    
    // Remove failed loads if requested
    if (options.skip_corrupted) {
        images.erase(
            std::remove_if(images.begin(), images.end(),
                         [](const std::unique_ptr<Image>& img) { 
                             return img == nullptr; 
                         }),
            images.end()
        );
    }
    
    return images;
}

std::vector<std::unique_ptr<Image>> ImageLoader::loadDirectory(
    const std::string& directory,
    const LoadOptions& options,
    const std::vector<std::string>& extensions) {
    
    auto paths = listImageFiles(directory, extensions);
    return loadBatch(paths, options);
}

ImageFormat ImageLoader::detectFormat(const std::string& path) {
    // Get extension
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Check common formats
    if (ext == ".jpg" || ext == ".jpeg") return ImageFormat::JPEG;
    if (ext == ".png") return ImageFormat::PNG;
    if (ext == ".tiff" || ext == ".tif") return ImageFormat::TIFF;
    if (ext == ".bmp") return ImageFormat::BMP;
    if (ext == ".webp") return ImageFormat::WEBP;
    if (ext == ".pfm") return ImageFormat::PFM;
    if (ext == ".exr") return ImageFormat::EXR;
    
    // Check RAW formats
    static const std::vector<std::string> raw_exts = {
        ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"
    };
    if (std::find(raw_exts.begin(), raw_exts.end(), ext) != raw_exts.end()) {
        return ImageFormat::RAW;
    }
    
    // Try to read file header for format detection
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return ImageFormat::UNKNOWN;
    
    unsigned char header[12];
    file.read(reinterpret_cast<char*>(header), 12);
    
    // JPEG
    if (header[0] == 0xFF && header[1] == 0xD8) return ImageFormat::JPEG;
    
    // PNG
    if (header[0] == 0x89 && header[1] == 'P' && header[2] == 'N' && header[3] == 'G') 
        return ImageFormat::PNG;
    
    // TIFF
    if ((header[0] == 'I' && header[1] == 'I') || (header[0] == 'M' && header[1] == 'M'))
        return ImageFormat::TIFF;
    
    // BMP
    if (header[0] == 'B' && header[1] == 'M') return ImageFormat::BMP;
    
    return ImageFormat::UNKNOWN;
}

bool ImageLoader::isImageFile(const std::string& path) {
    return detectFormat(path) != ImageFormat::UNKNOWN;
}

std::vector<std::string> ImageLoader::getSupportedExtensions() {
    return {
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
        ".pfm", ".exr", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"
    };
}

cv::Mat ImageLoader::loadRAW(const std::string& path) {
    // Simplified RAW loading - in practice, you'd use libraw
    std::cerr << "RAW loading not fully implemented. Using OpenCV fallback." << std::endl;
    return cv::imread(path, cv::IMREAD_UNCHANGED);
}

cv::Mat ImageLoader::loadPFM(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return cv::Mat();
    
    std::string header;
    int width, height;
    float scale;
    
    // Read header
    file >> header >> width >> height >> scale;
    file.get(); // Skip newline
    
    if (header != "PF" && header != "Pf") {
        return cv::Mat();
    }
    
    int channels = (header == "PF") ? 3 : 1;
    cv::Mat image(height, width, channels == 3 ? CV_32FC3 : CV_32FC1);
    
    // Read pixel data (bottom-up if scale < 0)
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            if (channels == 3) {
                float rgb[3];
                file.read(reinterpret_cast<char*>(rgb), sizeof(rgb));
                cv::Vec3f& pixel = image.at<cv::Vec3f>(scale < 0 ? y : height - 1 - y, x);
                pixel[0] = rgb[2] * std::abs(scale);  // BGR order
                pixel[1] = rgb[1] * std::abs(scale);
                pixel[2] = rgb[0] * std::abs(scale);
            } else {
                float gray;
                file.read(reinterpret_cast<char*>(&gray), sizeof(gray));
                image.at<float>(scale < 0 ? y : height - 1 - y, x) = gray * std::abs(scale);
            }
        }
    }
    
    return image;
}

bool ImageLoader::savePFM(const std::string& path, const cv::Mat& image) {
    if (image.type() != CV_32FC1 && image.type() != CV_32FC3) {
        return false;
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Write header
    file << (image.channels() == 3 ? "PF" : "Pf") << "\n";
    file << image.cols << " " << image.rows << "\n";
    file << "-1.0\n";  // Little endian, scale = 1.0
    
    // Write pixel data (bottom-up)
    for (int y = image.rows - 1; y >= 0; y--) {
        for (int x = 0; x < image.cols; x++) {
            if (image.channels() == 3) {
                cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);
                float rgb[3] = {pixel[2], pixel[1], pixel[0]};  // RGB order
                file.write(reinterpret_cast<char*>(rgb), sizeof(rgb));
            } else {
                float gray = image.at<float>(y, x);
                file.write(reinterpret_cast<char*>(&gray), sizeof(gray));
            }
        }
    }
    
    return true;
}

void ImageLoader::setMaxMemoryUsage(size_t max_bytes) {
    max_memory_usage = max_bytes;
}

size_t ImageLoader::getCurrentMemoryUsage() {
    return current_memory_usage.load();
}

std::vector<std::string> ImageLoader::listImageFiles(
    const std::string& directory,
    const std::vector<std::string>& extensions) {
    
    std::vector<std::string> image_files;
    
    auto exts = extensions;
    if (exts.empty()) {
        exts = getSupportedExtensions();
    }
    
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                image_files.push_back(entry.path().string());
            }
        }
    }
    
    // Sort for consistent ordering
    std::sort(image_files.begin(), image_files.end());
    
    return image_files;
}

std::unique_ptr<Image> ImageLoader::loadSingle(
    const std::string& path,
    const LoadOptions& options) {
    
    try {
        // Check memory limit
        if (current_memory_usage.load() > max_memory_usage.load()) {
            std::cerr << "Memory limit exceeded, cannot load: " << path << std::endl;
            return nullptr;
        }
        
        auto format = detectFormat(path);
        
        // Handle special formats
        cv::Mat data;
        if (format == ImageFormat::RAW) {
            data = loadRAW(path);
        } else if (format == ImageFormat::PFM) {
            data = loadPFM(path);
        } else {
            // Use OpenCV for standard formats
            data = cv::imread(path, cv::IMREAD_UNCHANGED);
        }
        
        if (data.empty()) {
            return nullptr;
        }
        
        // Resize if needed
        if (options.max_dimension > 0) {
            int max_dim = std::max(data.cols, data.rows);
            if (max_dim > options.max_dimension) {
                double scale = static_cast<double>(options.max_dimension) / max_dim;
                cv::resize(data, data, cv::Size(), scale, scale);
            }
        }
        
        // Create image object
        auto image = std::make_unique<Image>(data, fs::path(path).filename().string());
        
        // Apply load flags
        if (options.flags & static_cast<int>(Image::LoadFlags::EXTRACT_METADATA)) {
            // Metadata is extracted in Image constructor
        }
        
        if (options.flags & static_cast<int>(Image::LoadFlags::LOAD_GPU)) {
            image->uploadToGPU();
        }
        
        if (options.flags & static_cast<int>(Image::LoadFlags::BUILD_PYRAMID)) {
            image->buildPyramid(4, true);
        }
        
        // Update memory usage
        current_memory_usage += image->getMemoryUsage();
        
        return image;
        
    } catch (const std::exception& e) {
        if (!options.skip_corrupted) {
            throw;
        }
        std::cerr << "Failed to load image " << path << ": " << e.what() << std::endl;
        return nullptr;
    }
}

// ===================== ImageSequence Implementation =====================

ImageSequence::ImageSequence(const std::string& pattern_or_directory) {
    if (fs::is_directory(pattern_or_directory)) {
        image_paths_ = ImageLoader::listImageFiles(pattern_or_directory, {});
    } else {
        parsePattern(pattern_or_directory);
    }
}

std::unique_ptr<Image> ImageSequence::getFrame(int index) const {
    if (index < 0 || index >= getFrameCount()) {
        return nullptr;
    }
    
    // Can't copy Image, so always load from file
    return ImageLoader::load(image_paths_[index]);
}

std::unique_ptr<Image> ImageSequence::getNextFrame() {
    if (!hasNext()) return nullptr;
    return getFrame(current_index_++);
}

void ImageSequence::preloadFrames(int start, int count) {
    std::vector<std::future<void>> futures;
    
    for (int i = start; i < start + count && i < getFrameCount(); ++i) {
        futures.push_back(std::async(std::launch::async, [this, i]() {
            auto image = ImageLoader::load(image_paths_[i]);
            if (image) {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                cache_[i] = std::move(image);
            }
        }));
    }
    
    // Wait for all to complete
    for (auto& future : futures) {
        future.wait();
    }
}

void ImageSequence::clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

void ImageSequence::parsePattern(const std::string& pattern) {
    // Simple pattern parsing (e.g., "image_%04d.jpg")
    fs::path dir = fs::path(pattern).parent_path();
    std::string filename_pattern = fs::path(pattern).filename().string();
    
    // Replace %d patterns with regex
    std::string regex_pattern = filename_pattern;
    size_t pos = 0;
    while ((pos = regex_pattern.find("%", pos)) != std::string::npos) {
        if (pos + 1 < regex_pattern.size() && regex_pattern[pos + 1] == 'd') {
            regex_pattern.replace(pos, 2, "\\d+");
        } else if (pos + 2 < regex_pattern.size()) {
            // Handle %04d type patterns
            regex_pattern.replace(pos, 4, "\\d+");
        }
        pos++;
    }
    
    // Find matching files
    std::regex pattern_regex(regex_pattern);
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, pattern_regex)) {
                image_paths_.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(image_paths_.begin(), image_paths_.end());
}

// ===================== Utility Functions =====================

namespace image_utils {

ImageStats computeStats(const Image& image, bool use_gpu) {
    (void)use_gpu;  // Supress unused parameter warning
    ImageStats stats;
    cv::Mat data = image.getData();
    
    // Convert to grayscale for histogram
    cv::Mat gray;
    if (data.channels() > 1) {
        cv::cvtColor(data, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = data;
    }
    
    // Basic statistics
    cv::meanStdDev(data, stats.mean, stats.stddev);
    cv::minMaxLoc(gray, &stats.min, &stats.max);
    
    // Histogram
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), stats.histogram, 1, &histSize, &histRange);
    
    return stats;
}

cv::Mat normalizeImage(const cv::Mat& image, double target_mean, double target_std) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(normalized, mean, stddev);
    
    // Normalize each channel
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    
    for (size_t i = 0; i < channels.size(); ++i) {
        channels[i] = (channels[i] - mean[i]) / stddev[i] * target_std + target_mean;
    }
    
    cv::merge(channels, normalized);
    
    // Convert back to original type
    normalized.convertTo(normalized, image.type());
    
    return normalized;
}

void batchResize(std::vector<std::unique_ptr<Image>>& images, 
                int max_dimension, bool use_gpu) {
    
    std::for_each(std::execution::par_unseq, 
                  images.begin(), images.end(),
                  [max_dimension, use_gpu](std::unique_ptr<Image>& img) {
        if (!img) return;
        
        int max_dim = std::max(img->getWidth(), img->getHeight());
        if (max_dim > max_dimension) {
            double scale = static_cast<double>(max_dimension) / max_dim;
            int new_width = static_cast<int>(img->getWidth() * scale);
            int new_height = static_cast<int>(img->getHeight() * scale);
            
            cv::Mat resized = img->getResized(new_width, new_height, use_gpu);
            img = std::make_unique<Image>(resized, img->getName());        }
    });
}

void batchUndistort(std::vector<std::unique_ptr<Image>>& images,
                   const Camera& camera, bool use_gpu) {
    
    std::for_each(std::execution::par_unseq,
                  images.begin(), images.end(),
                  [&camera, use_gpu](std::unique_ptr<Image>& img) {
        if (!img) return;
        img->undistort(camera, use_gpu);
    });
}

std::string formatMemorySize(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

void optimizeMemoryLayout(std::vector<std::unique_ptr<Image>>& images) {
    // Sort images by size to improve cache locality
    std::sort(images.begin(), images.end(),
              [](const std::unique_ptr<Image>& a, const std::unique_ptr<Image>& b) {
        if (!a) return false;
        if (!b) return true;
        return a->getMemoryUsage() < b->getMemoryUsage();
    });
    
    // Release GPU memory for images not recently used
    size_t total_memory = Image::getTotalGPUMemoryUsage();
    size_t target_memory = max_memory_usage.load() / 2;  // Keep GPU usage under 50%
    
    if (total_memory > target_memory) {
        // Release GPU memory from largest images first
        for (auto it = images.rbegin(); it != images.rend() && total_memory > target_memory; ++it) {
            if (*it && (*it)->isOnGPU()) {
                size_t img_memory = (*it)->getMemoryUsage();
                (*it)->releaseGPUMemory();
                total_memory -= img_memory;
            }
        }
    }
}

} // namespace image_utils

} // namespace hybrid_sfm