#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "features.h"

namespace hybrid_sfm {

class Camera {
public:
    Camera() = default;

    // Camera Intrinsics
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    std::vector<double> distortion_params;

    // Camera Extrinsics (Pose)
    // We store rotation as a quaternion and translation as a vector
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    
    ImageID image_id = 0;
};

} // namespace hybrid_sfm