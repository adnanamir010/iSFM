#pragma once

#include "common.h"
#include <opencv2/core.hpp>
#include <string>

namespace hybrid_sfm {

class Image {
public:
    Image();
    Image(const std::string& path);
    Image(const cv::Mat& data, const std::string& name = "");
    
    // Load/Save
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    
    // Image pyramid
    void buildPyramid(int levels);
    cv::Mat getPyramidLevel(int level) const;
    
    // Getters
    const cv::Mat& getData() const { return data_; }
    int getWidth() const { return data_.cols; }
    int getHeight() const { return data_.rows; }
    const std::string& getName() const { return name_; }
    const std::string& getPath() const { return path_; }
    size_t getId() const { return id_; }
    
    // Setters
    void setId(size_t id) { id_ = id; }
    void setName(const std::string& name) { name_ = name; }
    
    // Utilities
    cv::Mat getGray() const;
    cv::Mat getUndistorted(const Camera& camera) const;

private:
    cv::Mat data_;
    std::vector<cv::Mat> pyramid_;
    std::string name_;
    std::string path_;
    size_t id_;
    
    static size_t next_id_;
};

} // namespace hybrid_sfm