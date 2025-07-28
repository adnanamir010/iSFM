#include "core/types/image.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <exiv2/exiv2.hpp>
#include <iostream>
#include <fstream>

namespace hybrid_sfm {

// Static member initialization
std::atomic<size_t> Image::next_id_{0};
std::atomic<size_t> Image::total_gpu_memory_{0};

// ===================== ImageMetadata Implementation =====================

void ImageMetadata::parseFromExif(const std::string& path) {
    try {
        std::unique_ptr<Exiv2::Image> image = Exiv2::ImageFactory::open(path);
        if (!image.get()) return;
        
        image->readMetadata();
        Exiv2::ExifData& exifData = image->exifData();
        
        if (exifData.empty()) return;
        
        // Camera info
        auto make = exifData.findKey(Exiv2::ExifKey("Exif.Image.Make"));
        if (make != exifData.end()) {
            camera_make = make->toString();
        }
        
        auto model = exifData.findKey(Exiv2::ExifKey("Exif.Image.Model"));
        if (model != exifData.end()) {
            camera_model = model->toString();
        }
        
        // Focal length
        auto focal = exifData.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength"));
        if (focal != exifData.end()) {
            focal_length_mm = focal->toFloat();
        }
        
        // Exposure time
        auto exposure = exifData.findKey(Exiv2::ExifKey("Exif.Photo.ExposureTime"));
        if (exposure != exifData.end()) {
            exposure_time = exposure->toFloat();
        }
        
        // ISO
        auto iso = exifData.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings"));
        if (iso != exifData.end()) {
            iso_speed = iso->toFloat();
        }
        
        // Aperture
        auto aperture = exifData.findKey(Exiv2::ExifKey("Exif.Photo.FNumber"));
        if (aperture != exifData.end()) {
            this->aperture = aperture->toFloat();
        }
        
        // GPS data
        auto lat = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"));
        auto lon = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude"));
        if (lat != exifData.end() && lon != exifData.end()) {
            has_gps = true;
            // Parse GPS coordinates (simplified - full implementation would handle DMS)
            latitude = lat->toFloat();
            longitude = lon->toFloat();
        }
        
    } catch (Exiv2::Error& e) {
        std::cerr << "EXIF parsing error: " << e.what() << std::endl;
    }
}

// ===================== GPUImageMemory Implementation =====================

GPUImageMemory::GPUImageMemory() {}

GPUImageMemory::~GPUImageMemory() {
    if (!gpu_data_.empty()) {
        size_t mem_usage = (gpu_data_.rows * gpu_data_.cols) * gpu_data_.elemSize();
        Image::total_gpu_memory_ -= mem_usage;
    }
}

void GPUImageMemory::upload(const cv::Mat& cpu_img) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Release old memory if exists
    if (!gpu_data_.empty()) {
        size_t old_mem = (gpu_data_.rows * gpu_data_.cols) * gpu_data_.elemSize();
        Image::total_gpu_memory_ -= old_mem;
    }
    
    // Upload new data
    gpu_data_.upload(cpu_img);
    
    // Track memory usage
    size_t new_mem = (gpu_data_.rows * gpu_data_.cols) * gpu_data_.elemSize();
    Image::total_gpu_memory_ += new_mem;
}

cv::Mat GPUImageMemory::download() const {
    std::lock_guard<std::mutex> lock(mutex_);
    cv::Mat result;
    if (!gpu_data_.empty()) {
        gpu_data_.download(result);
    }
    return result;
}

size_t GPUImageMemory::getMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpu_data_.empty()) return 0;
    return (gpu_data_.rows * gpu_data_.cols) * gpu_data_.elemSize();
}

// ===================== ImagePyramid Implementation =====================

void ImagePyramid::build(const cv::Mat& base_image, int num_levels, bool use_gpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    cpu_pyramid_.clear();
    gpu_pyramid_.clear();
    
    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::GpuMat gpu_base;
        gpu_base.upload(base_image);
        buildGPU(gpu_base, num_levels);
        
        // Also build CPU pyramid for compatibility
        cpu_pyramid_.reserve(num_levels);
        for (const auto& gpu_level : gpu_pyramid_) {
            cv::Mat cpu_level;
            gpu_level.download(cpu_level);
            cpu_pyramid_.push_back(cpu_level);
        }
    } else {
        // CPU-only pyramid
        cpu_pyramid_.reserve(num_levels);
        cpu_pyramid_.push_back(base_image);
        
        cv::Mat current = base_image;
        for (int i = 1; i < num_levels; ++i) {
            cv::Mat next;
            cv::pyrDown(current, next);
            cpu_pyramid_.push_back(next);
            current = next;
        }
    }
}

void ImagePyramid::buildGPU(const cv::cuda::GpuMat& base_image, int num_levels) {
    gpu_pyramid_.clear();
    gpu_pyramid_.reserve(num_levels);
    gpu_pyramid_.push_back(base_image);
    
    cv::cuda::GpuMat current = base_image;
    for (int i = 1; i < num_levels; ++i) {
        cv::cuda::GpuMat next;
        cv::cuda::pyrDown(current, next);
        gpu_pyramid_.push_back(next);
        current = next;
    }
}

cv::Mat ImagePyramid::getLevel(int level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < 0 || level >= static_cast<int>(cpu_pyramid_.size())) {
        return cv::Mat();
    }
    return cpu_pyramid_[level];
}

cv::cuda::GpuMat ImagePyramid::getGPULevel(int level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < 0 || level >= static_cast<int>(gpu_pyramid_.size())) {
        return cv::cuda::GpuMat();
    }
    return gpu_pyramid_[level];
}

void ImagePyramid::releaseGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    gpu_pyramid_.clear();
}

void ImagePyramid::releaseCPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_pyramid_.clear();
}

// ===================== Image Implementation =====================

Image::Image() : id_(next_id_++) {}

Image::Image(const std::string& path, int flags) : id_(next_id_++) {
    load(path, flags);
}

Image::Image(const cv::Mat& data, const std::string& name) 
    : data_(data), name_(name), id_(next_id_++), is_loaded_(true) {}

bool Image::load(const std::string& path, int flags) {
    std::lock_guard<std::mutex> lock(load_mutex_);
    
    path_ = path;
    if (name_.empty()) {
        // Extract filename from path
        size_t pos = path.find_last_of("/\\");
        name_ = (pos != std::string::npos) ? path.substr(pos + 1) : path;
    }
    
    // Handle lazy loading
    if (flags & LoadFlags::LAZY) {
        is_loaded_ = false;
        // Still extract metadata if requested
        if (flags & LoadFlags::EXTRACT_METADATA) {
            loadMetadata();
        }
        return true;  // Assume file exists
    }
    
    // Load image
    data_ = cv::imread(path, cv::IMREAD_COLOR);
    if (data_.empty()) {
        return false;
    }
    is_loaded_ = true;
    
    // Extract metadata
    if (flags & LoadFlags::EXTRACT_METADATA) {
        loadMetadata();
    }
    
    // Upload to GPU if requested
    if (flags & LoadFlags::LOAD_GPU) {
        uploadToGPU();
    }
    
    // Build pyramid if requested
    if (flags & LoadFlags::BUILD_PYRAMID) {
        buildPyramid(4, (flags & LoadFlags::LOAD_GPU) != 0);
    }
    
    return true;
}

bool Image::loadLazy(const std::string& path) {
    return load(path, static_cast<int>(LoadFlags::LAZY | LoadFlags::EXTRACT_METADATA));
}

void Image::ensureLoaded() const {
    if (is_loaded_) return;
    
    std::lock_guard<std::mutex> lock(load_mutex_);
    if (is_loaded_) return;  // Double-check
    
    data_ = cv::imread(path_, cv::IMREAD_COLOR);
    if (!data_.empty()) {
        is_loaded_ = true;
    }
}

bool Image::save(const std::string& path) const {
    ensureLoaded();
    return cv::imwrite(path, data_);
}

void Image::uploadToGPU() {
    ensureLoaded();
    if (!gpu_memory_) {
        gpu_memory_ = std::make_unique<GPUImageMemory>();
    }
    gpu_memory_->upload(data_);
}

void Image::downloadFromGPU() {
    if (gpu_memory_ && gpu_memory_->isOnGPU()) {
        data_ = gpu_memory_->download();
        is_loaded_ = true;
    }
}

cv::cuda::GpuMat Image::getGpuMat() {
    ensureLoaded();
    if (!gpu_memory_) {
        gpu_memory_ = std::make_unique<GPUImageMemory>();
    }
    if (!gpu_memory_->isOnGPU()) {
        gpu_memory_->upload(data_);
    }
    return gpu_memory_->getGpuMat();
}

void Image::buildPyramid(int levels, bool use_gpu) {
    ensureLoaded();
    pyramid_.build(data_, levels, use_gpu);
}

cv::Mat Image::getPyramidLevel(int level) const {
    return pyramid_.getLevel(level);
}

cv::Mat Image::getUndistorted(const Camera& camera, bool use_gpu) const {
    ensureLoaded();
    
    // Check cache
    if (!undistorted_cache_.empty() && undistort_camera_model_ == static_cast<int>(camera.getModel())) {
        return undistorted_cache_;
    }
    
    cv::Mat undistorted;
    
    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        // GPU undistortion
        cv::cuda::GpuMat gpu_src, gpu_dst;
        gpu_src.upload(data_);
        
        // Get undistortion maps
        cv::Mat map1, map2;
        // Convert Eigen matrix to cv::Mat
        Eigen::Matrix3d K_eigen = camera.getK();
        cv::Mat K(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                K.at<double>(i, j) = K_eigen(i, j);
            }
        }
        cv::Mat dist_coeffs = cv::Mat::zeros(camera.getParams().size(), 1, CV_64F);
        for (size_t i = 0; i < camera.getParams().size(); ++i) {
            dist_coeffs.at<double>(i) = camera.getParams()[i];
        }
        
        cv::initUndistortRectifyMap(K, dist_coeffs, cv::Mat(), K,
                                   cv::Size(data_.cols, data_.rows),
                                   CV_32FC1, map1, map2);
        
        cv::cuda::GpuMat gpu_map1, gpu_map2;
        gpu_map1.upload(map1);
        gpu_map2.upload(map2);
        
        cv::cuda::remap(gpu_src, gpu_dst, gpu_map1, gpu_map2, cv::INTER_LINEAR);
        gpu_dst.download(undistorted);
    } else {
        // CPU undistortion
        // Convert Eigen matrix to cv::Mat
        Eigen::Matrix3d K_eigen = camera.getK();
        cv::Mat K(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                K.at<double>(i, j) = K_eigen(i, j);
            }
        }        cv::Mat dist_coeffs = cv::Mat::zeros(camera.getParams().size(), 1, CV_64F);
        for (size_t i = 0; i < camera.getParams().size(); ++i) {
            dist_coeffs.at<double>(i) = camera.getParams()[i];
        }
        
        cv::undistort(data_, undistorted, K, dist_coeffs);
    }
    
    // Update cache
    undistorted_cache_ = undistorted;
    undistort_camera_model_ = static_cast<int>(camera.getModel());

    return undistorted;
}

void Image::undistort(const Camera& camera, bool use_gpu) {
    data_ = getUndistorted(camera, use_gpu);
    undistorted_cache_.release();  // Clear cache since main data is now undistorted
}

cv::Mat Image::getGray(bool use_gpu) const {
    ensureLoaded();
    
    if (data_.channels() == 1) {
        return data_;
    }
    
    cv::Mat gray;
    
    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::GpuMat gpu_src, gpu_gray;
        gpu_src.upload(data_);
        cv::cuda::cvtColor(gpu_src, gpu_gray, cv::COLOR_BGR2GRAY);
        gpu_gray.download(gray);
    } else {
        cv::cvtColor(data_, gray, cv::COLOR_BGR2GRAY);
    }
    
    return gray;
}

cv::Mat Image::getResized(int width, int height, bool use_gpu) const {
    ensureLoaded();
    
    cv::Mat resized;
    
    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::GpuMat gpu_src, gpu_resized;
        gpu_src.upload(data_);
        cv::cuda::resize(gpu_src, gpu_resized, cv::Size(width, height));
        gpu_resized.download(resized);
    } else {
        cv::resize(data_, resized, cv::Size(width, height));
    }
    
    return resized;
}

const cv::Mat& Image::getData() const {
    ensureLoaded();
    return data_;
}

int Image::getWidth() const {
    ensureLoaded();
    return data_.cols;
}

int Image::getHeight() const {
    ensureLoaded();
    return data_.rows;
}

int Image::getChannels() const {
    ensureLoaded();
    return data_.channels();
}

size_t Image::getMemoryUsage() const {
    size_t total = 0;
    
    // CPU memory
    if (!data_.empty()) {
        total += data_.total() * data_.elemSize();
    }
    
    // GPU memory
    if (gpu_memory_) {
        total += gpu_memory_->getMemoryUsage();
    }
    
    // Pyramid memory
    for (int i = 0; i < pyramid_.getNumLevels(); ++i) {
        cv::Mat level = pyramid_.getLevel(i);
        if (!level.empty()) {
            total += level.total() * level.elemSize();
        }
    }
    
    // Cache memory
    if (!undistorted_cache_.empty()) {
        total += undistorted_cache_.total() * undistorted_cache_.elemSize();
    }
    
    return total;
}

void Image::releaseGPUMemory() {
    if (gpu_memory_) {
        gpu_memory_.reset();
    }
    pyramid_.releaseGPU();
}

size_t Image::getTotalGPUMemoryUsage() {
    return total_gpu_memory_.load();
}

void Image::loadMetadata() {
    metadata_.parseFromExif(path_);
}

} // namespace hybrid_sfm