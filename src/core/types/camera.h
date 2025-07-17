#pragma once

#include "common.h"
#include <Eigen/Geometry>
#include <map>

namespace hybrid_sfm {

// Camera pose (extrinsics) - from Day 1
class CameraPose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    CameraPose();
    CameraPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    CameraPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    
    // Transform points
    Point3D worldToCamera(const Point3D& point) const;
    Point3D cameraToWorld(const Point3D& point) const;
    
    // Getters
    const Eigen::Quaterniond& getQuaternion() const { return q_; }
    const Eigen::Vector3d& getTranslation() const { return t_; }
    Eigen::Matrix3d getRotationMatrix() const { return q_.toRotationMatrix(); }
    Eigen::Matrix4d getTransformMatrix() const;
    Eigen::Vector3d getCenter() const;
    
    // Setters
    void setQuaternion(const Eigen::Quaterniond& q) { q_ = q.normalized(); }
    void setTranslation(const Eigen::Vector3d& t) { t_ = t; }
    void setRotationMatrix(const Eigen::Matrix3d& R);
    
    // Operators
    CameraPose inverse() const;
    CameraPose operator*(const CameraPose& other) const;

private:
    Eigen::Quaterniond q_;  // Rotation (quaternion)
    Eigen::Vector3d t_;     // Translation
};

// Camera intrinsics - Day 2
class Camera {
public:
    // Camera models (COLMAP compatible)
    enum class Model {
        SIMPLE_PINHOLE,         // f, cx, cy
        PINHOLE,               // fx, fy, cx, cy
        SIMPLE_RADIAL,         // f, cx, cy, k
        RADIAL,                // f, cx, cy, k1, k2
        OPENCV,                // fx, fy, cx, cy, k1, k2, p1, p2
        OPENCV_FISHEYE,        // fx, fy, cx, cy, k1, k2, k3, k4
        FULL_OPENCV,           // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        SIMPLE_RADIAL_FISHEYE, // f, cx, cy, k
        RADIAL_FISHEYE,        // f, cx, cy, k1, k2
        THIN_PRISM_FISHEYE    // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
    };

    // Model info
    struct ModelInfo {
        std::string name;
        int num_params;
        std::vector<std::string> param_names;
    };

    Camera();
    Camera(Model model, int width, int height);
    Camera(Model model, int width, int height, const std::vector<double>& params);
    
    // Projection/Unprojection
    Point2D worldToImage(const Point3D& point3D) const;
    Point3D imageToWorld(const Point2D& point2D, double depth = 1.0) const;
    
    // Normalized coordinates (remove principal point and focal length)
    Point2D imageToNormalized(const Point2D& point2D) const;
    Point2D normalizedToImage(const Point2D& normalized) const;
    
    // Distortion
    Point2D distortNormalized(const Point2D& normalized) const;
    Point2D undistortNormalized(const Point2D& distorted) const;
    
    // Parameter access
    double getFocalLengthX() const;
    double getFocalLengthY() const;
    double getPrincipalPointX() const;
    double getPrincipalPointY() const;
    std::vector<double> getParams() const { return params_; }
    void setParams(const std::vector<double>& params);
    
    // Model info
    Model getModel() const { return model_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    static const ModelInfo& getModelInfo(Model model);
    static Model modelFromName(const std::string& name);
    
    // COLMAP I/O
    static Camera fromCOLMAP(const std::string& line);
    std::string toCOLMAP() const;
    
    // Intrinsic matrix
    Eigen::Matrix3d getK() const;

private:
    Model model_;
    int width_, height_;
    std::vector<double> params_;
    
    static const std::map<Model, ModelInfo> model_info_;
    
    // Helper functions for different distortion models
    Point2D distortRadial(const Point2D& normalized) const;
    Point2D distortOpenCV(const Point2D& normalized) const;
    Point2D distortOpenCVFisheye(const Point2D& normalized) const;
    Point2D distortRadialFisheye(const Point2D& normalized) const;
    Point2D distortThinPrismFisheye(const Point2D& normalized) const;
    
    Point2D undistortRadial(const Point2D& distorted) const;
    Point2D undistortOpenCV(const Point2D& distorted) const;
    Point2D undistortIterative(const Point2D& distorted) const;
};

} // namespace hybrid_sfm