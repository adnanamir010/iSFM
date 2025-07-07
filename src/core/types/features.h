#pragma once

#include <Eigen/Core>
#include <vector>

namespace hybrid_sfm {
    
// Use unsigned integers for unique identifiers
using FeatureID = size_t;
using ImageID = size_t;
using TrackID = size_t;

// Define 2D and 3D points using Eigen for easy math operations
using Point2D = Eigen::Vector2d;
using Point3D = Eigen::Vector3d;

// A simple struct to hold a 2D feature observation in an image
struct Observation {
    Point2D point;
    FeatureID feature_id;
};

} // namespace hybrid_sfm