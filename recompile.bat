@echo off
echo ===================================================
echo   HYBRID SFM - C++ RECOMPILATION SCRIPT
echo ===================================================

echo.
echo [Step 1/4] Entering build directory...
cd build
if %errorlevel% neq 0 (
    echo [ERROR] 'build' directory not found. Please run the full CMake setup first.
    goto :eof
)

echo.
echo [Step 2/4] Compiling C++ code with Ninja...
ninja
if %errorlevel% neq 0 (
    echo [ERROR] C++ compilation failed. Please check the errors above.
    goto :eof
)
echo [SUCCESS] C++ compilation complete.

echo.
echo [Step 3/4] Copying compiled library to Python source tree...
REM This copies the .pyd file and renames it to _core.pyd, overwriting the old one.
copy /Y hybrid_sfm._core.cp312-win_amd64.pyd ..\python\hybrid_sfm\_core.pyd
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy the compiled .pyd file.
    goto :eof
)
echo [SUCCESS] Library copied.

echo.
echo [Step 4/4] Updating the Python package...
cd ..
pip install .
if %errorlevel% neq 0 (
    echo [ERROR] Final pip installation failed.
    goto :eof
)

echo.
echo ===================================================
echo   RECOMPILATION COMPLETE AND PACKAGE UPDATED!
echo ===================================================