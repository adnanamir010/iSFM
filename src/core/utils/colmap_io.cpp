#include "core/utils/colmap_io.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace hybrid_sfm {

std::unordered_map<int, COLMAPReader::CameraData> 
COLMAPReader::readCameras(const std::string& path) {
    std::unordered_map<int, CameraData> cameras;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open cameras file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        int camera_id;
        iss >> camera_id;
        
        // Parse the rest of the line as COLMAP camera
        std::string remaining;
        std::getline(iss, remaining);
        Camera camera = Camera::fromCOLMAP(remaining);
        
        cameras[camera_id] = {camera_id, camera};
    }
    
    return cameras;
}

std::unordered_map<int, COLMAPReader::ImageData>
COLMAPReader::readImages(const std::string& path) {
    std::unordered_map<int, ImageData> images;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open images file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        ImageData data;
        
        // Read IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        double qw, qx, qy, qz, tx, ty, tz;
        iss >> data.image_id >> qw >> qx >> qy >> qz 
            >> tx >> ty >> tz >> data.camera_id >> data.name;
        
        // Create pose from quaternion and translation
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);
        data.pose = CameraPose(q, t);
        
        // Read the next line with 2D points
        if (std::getline(file, line)) {
            std::istringstream point_stream(line);
            double x, y;
            int point3D_id;
            
            while (point_stream >> x >> y >> point3D_id) {
                data.points2D.emplace_back(x, y);
                data.point3D_ids.push_back(point3D_id);
            }
        }
        
        images[data.image_id] = data;
    }
    
    return images;
}

} // namespace hybrid_sfm