#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "core/types/features.h"
#include "core/types/camera.h"
#include "core/types/image.h"

namespace py = pybind11;
using namespace hybrid_sfm;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Core C++ components for the Hybrid SfM system";

    // We DO NOT bind Point2D or Point3D.
    // NumPy will be used on the Python side, and pybind11
    // will automatically convert them to Eigen types.

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        // This is the key: the `point` member is an Eigen::Vector2d.
        // pybind11 will automatically handle the numpy->Eigen conversion here.
        .def_readwrite("point", &Observation::point)
        .def_readwrite("feature_id", &Observation::feature_id);

    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("fx", &Camera::fx)
        // ... (rest of bindings are the same)
        .def_readwrite("fy", &Camera::fy)
        .def_readwrite("cx", &Camera::cx)
        .def_readwrite("cy", &Camera::cy)
        .def_readwrite("distortion_params", &Camera::distortion_params)
        .def_readwrite("rotation", &Camera::rotation)
        .def_readwrite("translation", &Camera::translation)
        .def_readwrite("image_id", &Camera::image_id);

    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def_readwrite("id", &Image::id)
        .def_readwrite("path", &Image::path)
        .def_readwrite("observations", &Image::observations);
}