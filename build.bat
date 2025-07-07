@echo off
echo Building iSFM with OpenCV 4.10.0...
call conda activate isfm
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build . --config Release
cd ..
pip install -e .
echo Build complete!

