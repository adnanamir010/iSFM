#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/types/camera.h"
#include "core/types/image.h"

namespace py = pybind11;

namespace hybrid_sfm {

void bind_types(py::module& m) {
    // Camera class
    py::class_<Camera> camera_class(m, "Camera");
    
    // Camera Model enum - bind as nested enum
    py::enum_<Camera::Model>(camera_class, "Model")
        .value("SIMPLE_PINHOLE", Camera::Model::SIMPLE_PINHOLE)
        .value("PINHOLE", Camera::Model::PINHOLE)
        .value("SIMPLE_RADIAL", Camera::Model::SIMPLE_RADIAL)
        .value("RADIAL", Camera::Model::RADIAL)
        .value("OPENCV", Camera::Model::OPENCV)
        .value("OPENCV_FISHEYE", Camera::Model::OPENCV_FISHEYE)
        .value("FULL_OPENCV", Camera::Model::FULL_OPENCV)
        .value("SIMPLE_RADIAL_FISHEYE", Camera::Model::SIMPLE_RADIAL_FISHEYE)
        .value("RADIAL_FISHEYE", Camera::Model::RADIAL_FISHEYE)
        .value("THIN_PRISM_FISHEYE", Camera::Model::THIN_PRISM_FISHEYE);
    
    // Camera class methods
    camera_class
        .def(py::init<>())
        .def(py::init<Camera::Model, int, int>())
        .def(py::init<Camera::Model, int, int, const std::vector<double>&>())
        .def("world_to_image", &Camera::worldToImage, "Project 3D point to 2D image")
        .def("image_to_world", &Camera::imageToWorld, "Unproject 2D image point to 3D", 
             py::arg("point2D"), py::arg("depth") = 1.0)
        .def("get_model", &Camera::getModel)
        .def("get_width", &Camera::getWidth)
        .def("get_height", &Camera::getHeight)
        .def("get_K", &Camera::getK)
        .def("get_focal_length_x", &Camera::getFocalLengthX)
        .def("get_focal_length_y", &Camera::getFocalLengthY)
        .def("get_principal_point_x", &Camera::getPrincipalPointX)
        .def("get_principal_point_y", &Camera::getPrincipalPointY)
        .def("get_params", &Camera::getParams)
        .def("set_params", &Camera::setParams)
        .def("to_COLMAP", &Camera::toCOLMAP)
        .def_static("from_COLMAP", &Camera::fromCOLMAP);
    
    // CameraPose class
    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def(py::init<const Eigen::Quaterniond&, const Eigen::Vector3d&>())
        .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&>())
        .def("world_to_camera", &CameraPose::worldToCamera)
        .def("camera_to_world", &CameraPose::cameraToWorld)
        .def("get_quaternion", &CameraPose::getQuaternion)
        .def("get_translation", &CameraPose::getTranslation)
        .def("get_rotation_matrix", &CameraPose::getRotationMatrix)
        .def("get_transform_matrix", &CameraPose::getTransformMatrix)
        .def("get_center", &CameraPose::getCenter)
        .def("inverse", &CameraPose::inverse)
        .def("__mul__", &CameraPose::operator*);
    
    // Also export enums at module level for backwards compatibility
    m.attr("CameraModel") = camera_class.attr("Model");
}

} // namespace hybrid_sfm