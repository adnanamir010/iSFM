#include "core/types/camera.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>

namespace hybrid_sfm {

// Define static model information
const std::map<Camera::Model, Camera::ModelInfo> Camera::model_info_ = {
    {Model::SIMPLE_PINHOLE, {"SIMPLE_PINHOLE", 3, {"f", "cx", "cy"}}},
    {Model::PINHOLE, {"PINHOLE", 4, {"fx", "fy", "cx", "cy"}}},
    {Model::SIMPLE_RADIAL, {"SIMPLE_RADIAL", 4, {"f", "cx", "cy", "k"}}},
    {Model::RADIAL, {"RADIAL", 5, {"f", "cx", "cy", "k1", "k2"}}},
    {Model::OPENCV, {"OPENCV", 8, {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"}}},
    {Model::OPENCV_FISHEYE, {"OPENCV_FISHEYE", 8, {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"}}},
    {Model::FULL_OPENCV, {"FULL_OPENCV", 12, {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"}}},
    {Model::SIMPLE_RADIAL_FISHEYE, {"SIMPLE_RADIAL_FISHEYE", 4, {"f", "cx", "cy", "k"}}},
    {Model::RADIAL_FISHEYE, {"RADIAL_FISHEYE", 5, {"f", "cx", "cy", "k1", "k2"}}},
    {Model::THIN_PRISM_FISHEYE, {"THIN_PRISM_FISHEYE", 12, {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"}}}
};

Camera::Camera() : model_(Model::PINHOLE), width_(0), height_(0) {
    params_ = {1.0, 1.0, 0.0, 0.0}; // Default pinhole parameters
}

Camera::Camera(Model model, int width, int height) 
    : model_(model), width_(width), height_(height) {
    const auto& info = getModelInfo(model);
    params_.resize(info.num_params);
    
    // Set default parameters with proper casting
    double focal = static_cast<double>(std::max(width, height));
    double cx = width / 2.0;
    double cy = height / 2.0;
    
    switch (model) {
        case Model::SIMPLE_PINHOLE:
            params_ = {focal, cx, cy};
            break;
        case Model::PINHOLE:
            params_ = {focal, focal, cx, cy};
            break;
        case Model::SIMPLE_RADIAL:
            params_ = {focal, cx, cy, 0.0};
            break;
        case Model::RADIAL:
            params_ = {focal, cx, cy, 0.0, 0.0};
            break;
        case Model::OPENCV:
            params_ = {focal, focal, cx, cy, 0.0, 0.0, 0.0, 0.0};
            break;
        default:
            std::fill(params_.begin(), params_.end(), 0.0);
            break;
    }
}

Camera::Camera(Model model, int width, int height, const std::vector<double>& params)
    : model_(model), width_(width), height_(height), params_(params) {
    const auto& info = getModelInfo(model);
    if (params_.size() != static_cast<size_t>(info.num_params)) {
        throw std::invalid_argument("Invalid number of parameters for camera model");
    }
}

Point2D Camera::worldToImage(const Point3D& point3D) const {
    // Project to normalized image coordinates
    const double z = point3D.z();
    if (z <= 0) {
        return Point2D(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());
    }
    
    Point2D normalized(point3D.x() / z, point3D.y() / z);
    
    // Apply distortion
    Point2D distorted = distortNormalized(normalized);
    
    // Convert to image coordinates
    return normalizedToImage(distorted);
}

Point3D Camera::imageToWorld(const Point2D& point2D, double depth) const {
    // Convert to normalized coordinates
    Point2D normalized = imageToNormalized(point2D);
    
    // Remove distortion
    Point2D undistorted = undistortNormalized(normalized);
    
    return Point3D(undistorted.x() * depth, undistorted.y() * depth, depth);
}

Point2D Camera::imageToNormalized(const Point2D& point2D) const {
    double fx = getFocalLengthX();
    double fy = getFocalLengthY();
    double cx = getPrincipalPointX();
    double cy = getPrincipalPointY();
    
    return Point2D((point2D.x() - cx) / fx, (point2D.y() - cy) / fy);
}

Point2D Camera::normalizedToImage(const Point2D& normalized) const {
    double fx = getFocalLengthX();
    double fy = getFocalLengthY();
    double cx = getPrincipalPointX();
    double cy = getPrincipalPointY();
    
    return Point2D(normalized.x() * fx + cx, normalized.y() * fy + cy);
}

double Camera::getFocalLengthX() const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
        case Model::SIMPLE_RADIAL_FISHEYE:
        case Model::RADIAL_FISHEYE:
            return params_[0];
        default:
            return params_[0];
    }
}

double Camera::getFocalLengthY() const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
        case Model::SIMPLE_RADIAL_FISHEYE:
        case Model::RADIAL_FISHEYE:
            return params_[0];  // Same as fx for these models
        default:
            return params_[1];
    }
}

double Camera::getPrincipalPointX() const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
        case Model::SIMPLE_RADIAL_FISHEYE:
        case Model::RADIAL_FISHEYE:
            return params_[1];
        default:
            return params_[2];
    }
}

double Camera::getPrincipalPointY() const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
        case Model::SIMPLE_RADIAL_FISHEYE:
        case Model::RADIAL_FISHEYE:
            return params_[2];
        default:
            return params_[3];
    }
}

Eigen::Matrix3d Camera::getK() const {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = getFocalLengthX();
    K(1, 1) = getFocalLengthY();
    K(0, 2) = getPrincipalPointX();
    K(1, 2) = getPrincipalPointY();
    return K;
}

void Camera::setParams(const std::vector<double>& params) {
    const auto& info = getModelInfo(model_);
    if (params.size() != static_cast<size_t>(info.num_params)) {
        throw std::invalid_argument("Invalid number of parameters for camera model");
    }
    params_ = params;
}

Point2D Camera::distortNormalized(const Point2D& normalized) const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::PINHOLE:
            return normalized;  // No distortion
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
            return distortRadial(normalized);
        case Model::OPENCV:
        case Model::FULL_OPENCV:
            return distortOpenCV(normalized);
        case Model::OPENCV_FISHEYE:
            return distortOpenCVFisheye(normalized);
        case Model::SIMPLE_RADIAL_FISHEYE:
        case Model::RADIAL_FISHEYE:
            return distortRadialFisheye(normalized);
        case Model::THIN_PRISM_FISHEYE:
            return distortThinPrismFisheye(normalized);
        default:
            return normalized;
    }
}

Point2D Camera::undistortNormalized(const Point2D& distorted) const {
    switch (model_) {
        case Model::SIMPLE_PINHOLE:
        case Model::PINHOLE:
            return distorted;  // No distortion
        case Model::SIMPLE_RADIAL:
        case Model::RADIAL:
            return undistortRadial(distorted);
        case Model::OPENCV:
        case Model::FULL_OPENCV:
            return undistortOpenCV(distorted);
        default:
            // Use iterative method for complex models
            return undistortIterative(distorted);
    }
}

Point2D Camera::distortRadial(const Point2D& normalized) const {
    const double r2 = normalized.squaredNorm();
    double radial = 1.0;
    
    if (model_ == Model::SIMPLE_RADIAL) {
        radial += params_[3] * r2;  // k
    } else if (model_ == Model::RADIAL) {
        const double r4 = r2 * r2;
        radial += params_[3] * r2 + params_[4] * r4;  // k1, k2
    }
    
    return normalized * radial;
}

Point2D Camera::distortOpenCV(const Point2D& normalized) const {
    const double x = normalized.x();
    const double y = normalized.y();
    const double r2 = x * x + y * y;
    
    // Get distortion parameters based on model
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;
    
    if (model_ == Model::OPENCV) {
        k1 = params_[4];
        k2 = params_[5];
        p1 = params_[6];
        p2 = params_[7];
    } else if (model_ == Model::FULL_OPENCV) {
        k1 = params_[4];
        k2 = params_[5];
        p1 = params_[6];
        p2 = params_[7];
        k3 = params_[8];
    }
    
    // Radial distortion
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    
    // Tangential distortion
    const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    
    return Point2D(x * radial + dx, y * radial + dy);
}

Point2D Camera::undistortIterative(const Point2D& distorted) const {
    // Newton-Raphson iteration
    Point2D undistorted = distorted;
    
    for (int i = 0; i < 10; ++i) {
        const Point2D distorted_est = distortNormalized(undistorted);
        const Point2D error = distorted - distorted_est;
        
        if (error.squaredNorm() < 1e-10) {
            break;
        }
        
        undistorted += error;
    }
    
    return undistorted;
}

// COLMAP I/O
Camera Camera::fromCOLMAP(const std::string& line) {
    std::istringstream iss(line);
    int camera_id, width, height;
    std::string model_name;
    
    iss >> camera_id >> model_name >> width >> height;
    
    Model model = modelFromName(model_name);
    const auto& info = getModelInfo(model);
    
    std::vector<double> params(info.num_params);
    for (int i = 0; i < info.num_params; ++i) {
        iss >> params[i];
    }
    
    return Camera(model, width, height, params);
}

std::string Camera::toCOLMAP() const {
    std::ostringstream oss;
    const auto& info = getModelInfo(model_);
    
    oss << info.name << " " << width_ << " " << height_;
    
    for (double param : params_) {
        oss << " " << std::setprecision(10) << param;
    }
    
    return oss.str();
}

const Camera::ModelInfo& Camera::getModelInfo(Model model) {
    return model_info_.at(model);
}

Camera::Model Camera::modelFromName(const std::string& name) {
    for (const auto& [model, info] : model_info_) {
        if (info.name == name) {
            return model;
        }
    }
    throw std::invalid_argument("Unknown camera model: " + name);
}

Point2D Camera::distortOpenCVFisheye(const Point2D& normalized) const {
    // Placeholder for OpenCV fisheye distortion
    // TODO: Implement proper fisheye distortion
    return normalized;
}

Point2D Camera::distortRadialFisheye(const Point2D& normalized) const {
    // Placeholder for radial fisheye distortion
    // TODO: Implement proper radial fisheye distortion
    return normalized;
}

Point2D Camera::distortThinPrismFisheye(const Point2D& normalized) const {
    // Placeholder for thin prism fisheye distortion
    // TODO: Implement proper thin prism fisheye distortion
    return normalized;
}

Point2D Camera::undistortRadial(const Point2D& distorted) const {
    // Simple iterative undistortion for radial model
    return undistortIterative(distorted);
}

Point2D Camera::undistortOpenCV(const Point2D& distorted) const {
    // Simple iterative undistortion for OpenCV model
    return undistortIterative(distorted);
}

CameraPose::CameraPose() : q_(Eigen::Quaterniond::Identity()), t_(Eigen::Vector3d::Zero()) {}

CameraPose::CameraPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) 
    : q_(q.normalized()), t_(t) {}

CameraPose::CameraPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
    : q_(R), t_(t) {
    q_.normalize();
}

Point3D CameraPose::worldToCamera(const Point3D& point) const {
    return q_ * point + t_;
}

Point3D CameraPose::cameraToWorld(const Point3D& point) const {
    return q_.inverse() * (point - t_);
}

Eigen::Matrix4d CameraPose::getTransformMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q_.toRotationMatrix();
    T.block<3,1>(0,3) = t_;
    return T;
}

Eigen::Vector3d CameraPose::getCenter() const {
    // Camera center = -R^T * t = -q^(-1) * t
    return -(q_.inverse() * t_);
}

void CameraPose::setRotationMatrix(const Eigen::Matrix3d& R) {
    q_ = Eigen::Quaterniond(R);
    q_.normalize();
}

CameraPose CameraPose::inverse() const {
    Eigen::Quaterniond q_inv = q_.inverse();
    Eigen::Vector3d t_inv = -(q_inv * t_);
    return CameraPose(q_inv, t_inv);
}

CameraPose CameraPose::operator*(const CameraPose& other) const {
    return CameraPose(q_ * other.q_, q_ * other.t_ + t_);
}
} // namespace hybrid_sfm