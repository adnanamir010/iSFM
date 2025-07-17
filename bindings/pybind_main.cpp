#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for binding functions
void bind_types(py::module& m);

PYBIND11_MODULE(hybrid_sfm, m) {
    m.doc() = "Hybrid Structure-from-Motion Python bindings";
    
    // Bind types
    bind_types(m);
}