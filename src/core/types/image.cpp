#include "core/types/image.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace hybrid_sfm {

size_t Image::next_id_ = 0;

Image::Image() : id_(next_id_++) {}

Image::Image(const std::string& path) : id_(next_id_++) {
    load(path);
}

Image::Image(const cv::Mat& data, const std::string& name) 
    : data_(data), name_(name), id_(next_id_++) {}

bool Image::load(const std::string& path) {
    data_ = cv::imread(path, cv::IMREAD_COLOR);
    if (data_.empty()) {
        return false;
    }
    path_ = path;
    if (name_.empty()) {
        // Extract filename from path
        size_t pos = path.find_last_of("/\\");
        name_ = (pos != std::string::npos) ? path.substr(pos + 1) : path;
    }
    return true;
}

bool Image::save(const std::string& path) const {
    return cv::imwrite(path, data_);
}

void Image::buildPyramid(int levels) {
    pyramid_.clear();
    pyramid_.reserve(levels);
    pyramid_.push_back(data_);
    
    cv::Mat current = data_;
    for (int i = 1; i < levels; ++i) {
        cv::Mat next;
        cv::pyrDown(current, next);
        pyramid_.push_back(next);
        current = next;
    }
}

cv::Mat Image::getPyramidLevel(int level) const {
    if (level < 0 || level >= static_cast<int>(pyramid_.size())) {
        return cv::Mat();
    }
    return pyramid_[level];
}

cv::Mat Image::getGray() const {
    if (data_.channels() == 1) {
        return data_;
    }
    cv::Mat gray;
    cv::cvtColor(data_, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

} // namespace hybrid_sfm