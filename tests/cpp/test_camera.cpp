#include <gtest/gtest.h>
#include "core/types/camera.h"
#include <random>

using namespace hybrid_sfm;

class CameraTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test cameras
        simple_pinhole_ = Camera(Camera::Model::SIMPLE_PINHOLE, 640, 480);
        pinhole_ = Camera(Camera::Model::PINHOLE, 640, 480);
        
        // Camera with distortion
        std::vector<double> radial_params = {500.0, 320.0, 240.0, -0.1, 0.05};
        radial_ = Camera(Camera::Model::RADIAL, 640, 480, radial_params);
    }
    
    Camera simple_pinhole_;
    Camera pinhole_;
    Camera radial_;
};

TEST_F(CameraTest, ProjectUnprojectSimplePinhole) {
    Point3D point3d(1.0, 2.0, 5.0);
    Point2D point2d = simple_pinhole_.worldToImage(point3d);
    
    // Check projection
    // For SIMPLE_PINHOLE: f=640, cx=320, cy=240
    // Normalized: (1/5, 2/5) = (0.2, 0.4)
    // Image: (0.2*640 + 320, 0.4*640 + 240) = (448, 496)
    EXPECT_NEAR(point2d.x(), 448.0, 1e-6);
    EXPECT_NEAR(point2d.y(), 496.0, 1e-6);
    
    // Check unprojection
    Point3D unprojected = simple_pinhole_.imageToWorld(point2d, 5.0);
    EXPECT_NEAR(unprojected.x(), point3d.x(), 1e-6);
    EXPECT_NEAR(unprojected.y(), point3d.y(), 1e-6);
    EXPECT_NEAR(unprojected.z(), point3d.z(), 1e-6);
}

TEST_F(CameraTest, DistortionRadial) {
    Point3D point3d(1.0, 2.0, 10.0);
    Point2D point2d = radial_.worldToImage(point3d);
    
    // Create simple camera with same focal length for comparison
    std::vector<double> simple_params = {500.0, 320.0, 240.0};
    Camera simple(Camera::Model::SIMPLE_PINHOLE, 640, 480, simple_params);
    Point2D simple_point = simple.worldToImage(point3d);
    
    // With distortion, the point should be different from simple projection
    EXPECT_GT((point2d - simple_point).norm(), 0.1);
    
    // Unprojection should recover the original point
    Point3D unprojected = radial_.imageToWorld(point2d, 10.0);
    EXPECT_NEAR(unprojected.x(), point3d.x(), 1e-4);
    EXPECT_NEAR(unprojected.y(), point3d.y(), 1e-4);
    EXPECT_NEAR(unprojected.z(), point3d.z(), 1e-4);
}

TEST_F(CameraTest, COLMAPSerialization) {
    std::string colmap_str = pinhole_.toCOLMAP();
    
    // Should contain model name and parameters
    EXPECT_TRUE(colmap_str.find("PINHOLE") != std::string::npos);
    EXPECT_TRUE(colmap_str.find("640") != std::string::npos);
    EXPECT_TRUE(colmap_str.find("480") != std::string::npos);
}

TEST_F(CameraTest, IntrinsicMatrix) {
    Eigen::Matrix3d K = pinhole_.getK();
    
    EXPECT_EQ(K(0, 0), pinhole_.getFocalLengthX());
    EXPECT_EQ(K(1, 1), pinhole_.getFocalLengthY());
    EXPECT_EQ(K(0, 2), pinhole_.getPrincipalPointX());
    EXPECT_EQ(K(1, 2), pinhole_.getPrincipalPointY());
    EXPECT_EQ(K(2, 2), 1.0);
}

TEST_F(CameraTest, ModelTypes) {
    // Test different camera models
    Camera opencv(Camera::Model::OPENCV, 640, 480);
    EXPECT_EQ(opencv.getParams().size(), 8);
    
    Camera fisheye(Camera::Model::OPENCV_FISHEYE, 640, 480);
    EXPECT_EQ(fisheye.getParams().size(), 8);
    
    // Test model name lookup
    Camera::Model model = Camera::modelFromName("RADIAL");
    EXPECT_EQ(model, Camera::Model::RADIAL);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}