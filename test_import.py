#!/usr/bin/env python3
import sys
sys.path.insert(0, 'python')

try:
    import hybrid_sfm as sfm
    print(f"✅ Successfully imported hybrid_sfm version {sfm.__version__}")
    
    # Test Day 2 functionality
    import numpy as np
    
    # Create camera with new API
    camera = sfm.Camera(sfm.CameraModel.PINHOLE, 640, 480)
    print(f"✅ Created PINHOLE camera: {camera.get_width()}x{camera.get_height()}")
    print(f"✅ Focal lengths: fx={camera.get_focal_length_x()}, fy={camera.get_focal_length_y()}")
    
    # Test projection with new method names
    point3d = np.array([1.0, 2.0, 5.0])
    point2d = camera.world_to_image(point3d)
    print(f"✅ Projected 3D {point3d} to 2D: {point2d}")
    
    # Test camera pose
    pose = sfm.CameraPose()
    print(f"✅ Created identity camera pose")
    
    print("\n🎉 Day 2 imports working!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    raise