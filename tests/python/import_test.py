#!/usr/bin/env python3
import sys
sys.path.insert(0, 'python')

try:
    import hybrid_sfm as sfm
    print(f"✅ Successfully imported hybrid_sfm version {sfm.__version__}")
    
    # Test basic functionality
    import numpy as np
    K = np.eye(3)
    K[0,0] = K[1,1] = 500  # focal length
    K[0,2] = 320  # cx
    K[1,2] = 240  # cy
    
    camera = sfm.Camera(sfm.CameraModel.PINHOLE, K)
    print(f"✅ Created camera with focal length: {camera.get_focal_length()}")
    
    # Test projection
    point3d = np.array([1.0, 2.0, 5.0])
    point2d = camera.project(point3d)
    print(f"✅ Projected 3D point {point3d} to 2D: {point2d}")
    
    # Test camera pose
    pose = sfm.CameraPose()
    print(f"✅ Created identity camera pose")
    
    # Test image
    img = sfm.Image()
    print(f"✅ Created empty image")
    
    print("\n🎉 Day 1 Complete! All basic types working!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    raise