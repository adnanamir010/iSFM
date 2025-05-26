cmake_minimum_required(VERSION 3.15)
project(hybrid_sfm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)
find_package(GTest REQUIRED)

# OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(your_target PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(bindings)
add_subdirectory(tests)