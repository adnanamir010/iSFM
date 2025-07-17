#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>

namespace hybrid_sfm {

// Type aliases
using Point2D = Eigen::Vector2d;
using Point3D = Eigen::Vector3d;
using Point2f = Eigen::Vector2f;
using Point3f = Eigen::Vector3f;

// Forward declarations
class Camera;
class Image;
class Feature2D;
class Feature3D;

} // namespace hybrid_sfm