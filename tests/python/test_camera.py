#!/usr/bin/env python3
import sys
sys.path.insert(0, 'python')

import numpy as np
import hybrid_sfm as sfm

def test_camera_models():
    """Test different camera models."""
    print("Testing camera models...")
    
    # Test PINHOLE model
    camera_pinhole = sfm.Camera(sfm.CameraModel.PINHOLE, 640, 480)
    print(f"✅ Created PINHOLE camera: {camera_pinhole.get_width()}x{camera_pinhole.get_height()}")
    
    # Test projection/unprojection
    point_3d = np.array([1.0, 2.0, 5.0])
    point_2d = camera_pinhole.world_to_image(point_3d)
    print(f"✅ Projected 3D {point_3d} -> 2D {point_2d}")
    
    point_3d_recovered = camera_pinhole.image_to_world(point_2d, 5.0)
    print(f"✅ Unprojected 2D {point_2d} -> 3D {point_3d_recovered}")
    
    # Check error
    error = np.linalg.norm(point_3d - point_3d_recovered)
    assert error < 1e-6, f"Projection error too large: {error}"
    print("✅ Projection/unprojection cycle accurate")
    
    # Test intrinsic matrix
    K = camera_pinhole.get_K()
    print(f"✅ Intrinsic matrix K:\n{K}")
    
    # Test COLMAP serialization
    colmap_str = camera_pinhole.to_COLMAP()
    print(f"✅ COLMAP format: {colmap_str}")

def test_camera_pose():
    """Test camera pose operations."""
    print("\nTesting camera pose...")
    
    # Create poses
    pose1 = sfm.CameraPose()
    R = np.eye(3)
    R[0,0] = 0; R[0,1] = -1
    R[1,0] = 1; R[1,1] = 0  # 90 degree rotation
    t = np.array([1.0, 2.0, 3.0])
    pose2 = sfm.CameraPose(R, t)
    
    # Test pose composition
    pose3 = pose1 * pose2
    print("✅ Pose composition working")
    
    # Test inverse
    pose_inv = pose2.inverse()
    identity = pose2 * pose_inv
    print("✅ Pose inverse working")
    
    # Test world/camera transforms
    point_world = np.array([1.0, 0.0, 0.0])
    point_camera = pose2.world_to_camera(point_world)
    point_world_recovered = pose2.camera_to_world(point_camera)
    
    error = np.linalg.norm(point_world - point_world_recovered)
    assert error < 1e-6, f"Transform error too large: {error}"
    print("✅ World/camera transforms accurate")

def test_distortion():
    """Test camera distortion models."""
    print("\nTesting distortion models...")
    
    # Create camera with radial distortion
    params = [500.0, 320.0, 240.0, -0.1, 0.05]  # f, cx, cy, k1, k2
    camera_radial = sfm.Camera(sfm.CameraModel.RADIAL, 640, 480)
    camera_radial.set_params(params)
    
    # Test that distortion affects projection
    point_3d = np.array([1.0, 1.0, 10.0])
    
    # Compare with no distortion
    camera_simple = sfm.Camera(sfm.CameraModel.SIMPLE_PINHOLE, 640, 480)
    point_2d_simple = camera_simple.world_to_image(point_3d)
    point_2d_radial = camera_radial.world_to_image(point_3d)
    
    diff = np.linalg.norm(point_2d_simple - point_2d_radial)
    assert diff > 0.1, "Distortion should have effect"
    print(f"✅ Distortion effect: {diff:.2f} pixels")
    
    # Test unprojection accuracy
    point_3d_recovered = camera_radial.image_to_world(point_2d_radial, 10.0)
    error = np.linalg.norm(point_3d - point_3d_recovered)
    assert error < 1e-4, f"Unprojection error too large: {error}"
    print("✅ Distortion unprojection accurate")

if __name__ == "__main__":
    print("Day 2: Camera Models & Calibration Tests\n")
    
    test_camera_models()
    test_camera_pose()
    test_distortion()
    
    print("\n🎉 Day 2 Complete! Camera models and calibration working!")