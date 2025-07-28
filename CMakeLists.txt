cmake_minimum_required(VERSION 3.20)
project(hybrid_sfm VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find packages
find_package(OpenCV 4.13 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(EXIV2 REQUIRED exiv2)

# CUDA settings for RTX 4070
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${EXIV2_INCLUDE_DIRS}
)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES
    src/core/types/*.cpp
    src/core/utils/*.cpp
)

# Collect CUDA sources
file(GLOB CUDA_SOURCES
    src/core/cuda/*.cu
)

# Create core library
add_library(hybrid_sfm_core SHARED ${CORE_SOURCES} ${CUDA_SOURCES})

target_compile_definitions(hybrid_sfm_core PUBLIC HYBRID_SFM_USE_CUDA)

target_link_libraries(hybrid_sfm_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${EXIV2_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cuda_driver
)

# Python bindings
pybind11_add_module(hybrid_sfm  # Changed from pyhybrid_sfm
    bindings/pybind_main.cpp
    bindings/types_binding.cpp
    bindings/image_bindings.cpp
)

target_link_libraries(hybrid_sfm PRIVATE hybrid_sfm_core)

# Set properties for Python module - OUTPUT TO BUILD DIRECTORY
set_target_properties(hybrid_sfm PROPERTIES
    # Remove OUTPUT_NAME since module name is already hybrid_sfm
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}  # Changed to build directory
)

# Copy Python __init__.py to build directory for testing
add_custom_command(TARGET hybrid_sfm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/python/hybrid_sfm/__init__.py
    ${CMAKE_BINARY_DIR}/__init__.py
)

# Optional: Create a symlink in the source tree for development
# (This allows importing from the source tree during development)
add_custom_command(TARGET hybrid_sfm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_BINARY_DIR}/hybrid_sfm${Python3_EXT_SUFFIX}
    ${CMAKE_SOURCE_DIR}/python/hybrid_sfm/hybrid_sfm${Python3_EXT_SUFFIX}
    COMMENT "Creating symlink for development"
)

# Installation rules
install(TARGETS hybrid_sfm_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install Python module
install(TARGETS hybrid_sfm
    LIBRARY DESTINATION ${Python3_SITELIB}
)

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest REQUIRED)

# Add test executable
add_executable(test_camera tests/cpp/test_camera.cpp)
target_link_libraries(test_camera
    hybrid_sfm_core
    GTest::GTest
    GTest::Main
)

# Register test
add_test(NAME test_camera COMMAND test_camera)

# Add test executable for images
add_executable(test_image tests/cpp/test_image.cpp)
target_link_libraries(test_image
    hybrid_sfm_core
    GTest::GTest
    GTest::Main
)

# Register test
add_test(NAME test_image COMMAND test_image)

# Add Python tests
add_test(NAME test_python_camera 
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_camera.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_python_camera PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
)

add_test(NAME test_python_image 
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_image.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_python_image PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  Eigen version: ${EIGEN3_VERSION}")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python module output: ${CMAKE_BINARY_DIR}")
message(STATUS "")