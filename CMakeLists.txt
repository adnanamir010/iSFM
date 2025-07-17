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

# CUDA settings for RTX 4070
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES 
    src/core/types/*.cpp
    src/core/utils/*.cpp
)

# Create core library
add_library(hybrid_sfm_core SHARED ${CORE_SOURCES})
target_link_libraries(hybrid_sfm_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# Python bindings
pybind11_add_module(pyhybrid_sfm 
    bindings/pybind_main.cpp
    bindings/types_binding.cpp
)
target_link_libraries(pyhybrid_sfm PRIVATE hybrid_sfm_core)

# Set properties for Python module
set_target_properties(pyhybrid_sfm PROPERTIES
    OUTPUT_NAME hybrid_sfm
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/hybrid_sfm
)

# Installation rules
install(TARGETS hybrid_sfm_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
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