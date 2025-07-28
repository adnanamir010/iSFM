#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hybrid_sfm {
    void bind_types(py::module& m);
    void bind_image(py::module& m);
}

PYBIND11_MODULE(hybrid_sfm, m) {
    m.doc() = "Hybrid Structure-from-Motion Python bindings";
    
    // Core types
    hybrid_sfm::bind_types(m);
    
    // Image processing
    hybrid_sfm::bind_image(m);
}