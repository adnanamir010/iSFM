#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "core/types/camera.h"

namespace hybrid_sfm {

// COLMAP file readers
class COLMAPReader {
public:
    struct CameraData {
        int camera_id;
        Camera camera;
    };
    
    struct ImageData {
        int image_id;
        CameraPose pose;
        int camera_id;
        std::string name;
        std::vector<Point2D> points2D;
        std::vector<int> point3D_ids;
    };
    
    struct Point3DData {
        int point3D_id;
        Point3D xyz;
        Eigen::Vector3i rgb;
        double error;
        std::vector<std::pair<int, int>> track; // image_id, point2D_idx
    };
    
    // Read cameras.txt
    static std::unordered_map<int, CameraData> readCameras(const std::string& path);
    
    // Read images.txt
    static std::unordered_map<int, ImageData> readImages(const std::string& path);
    
    // Read points3D.txt
    static std::unordered_map<int, Point3DData> readPoints3D(const std::string& path);
};

} // namespace hybrid_sfm