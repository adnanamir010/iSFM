#!/bin/bash
# quick_rebuild.sh - Rebuild and test Python bindings

set -e

echo "🔧 Rebuilding hybrid_sfm with fixed bindings..."

# Clean and reconfigure
cd ~/iSFM/build
rm -f hybrid_sfm*.so
rm -f CMakeCache.txt

# Reconfigure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPYTHON_EXECUTABLE=$(which python3) \
         -DCMAKE_CUDA_ARCHITECTURES=89

# Build the Python module
make hybrid_sfm -j$(nproc)

# Check if the module was built
echo -e "\n📦 Checking for Python module..."
if [ -f hybrid_sfm*.so ]; then
    echo "✅ Found: $(ls hybrid_sfm*.so)"
else
    echo "❌ Python module not found!"
    exit 1
fi

# Test import
cd ~/iSFM
export PYTHONPATH=$PWD/build:$PYTHONPATH

echo -e "\n🧪 Testing Python import..."
python3 -c "
import sys
sys.path.insert(0, 'build')
try:
    import hybrid_sfm
    print('✅ Import successful!')
    print(f'  Camera: {hasattr(hybrid_sfm, \"Camera\")}')
    print(f'  CameraPose: {hasattr(hybrid_sfm, \"CameraPose\")}')
    print(f'  Image: {hasattr(hybrid_sfm, \"Image\")}')
    
    # Quick test
    cam = hybrid_sfm.Camera()
    print('✅ Created Camera object successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo -e "\n📝 Next steps:"
echo "  1. python3 tests/python/test_camera.py"
echo "  2. python3 tests/python/test_image.py"