cmake_minimum_required(VERSION 3.20)
project(iSFM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV 4.10.0 with CUDA support
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)


# --- Core C++ Library ---
# This library will contain all our core SfM logic.
# Today, it's just the data types, but we will add more to it.
add_library(hybrid_sfm_core INTERFACE)

# Add our source directory to the include path
target_include_directories(hybrid_sfm_core INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Link the core library to its dependencies
target_link_libraries(hybrid_sfm_core INTERFACE 
    Eigen3::Eigen 
    ${OpenCV_LIBRARIES}
    Ceres::ceres
)

# --- Python Module ---
# This creates the Python module (_core.pyd or _core.so)
# The name "hybrid_sfm._core" means it will be a submodule named _core
pybind11_add_module(hybrid_sfm._core
    python/bindings.cpp
)

# Link the python module to our core library and its dependencies
target_link_libraries(hybrid_sfm._core PRIVATE 
    hybrid_sfm_core
)

# Add include directory for python bindings
target_include_directories(hybrid_sfm._core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
