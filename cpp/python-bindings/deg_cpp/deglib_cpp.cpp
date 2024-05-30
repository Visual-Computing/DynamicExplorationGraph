//
// Created by Bruno Schilling on 29.05.24.
//
#include <filesystem>
#include <pybind11/pybind11.h>

#include <deglib.h>

namespace py = pybind11;

bool avx_usable() {
    #if defined(USE_AVX)
        return true;
    #else
        return false;
    #endif
}

bool sse_usable() {
    #if defined(USE_SSE)
        return true;
    #else
        return false;
    #endif
}

PYBIND11_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    m.def("avx_usable", &avx_usable, "Returns whether AVX instructions are available");
    m.def("sse_usable", &sse_usable, "Returns whether SSE instructions are available");

	// distances
	py::enum_<deglib::Metric>(m, "Metric")
		.value("L2", deglib::Metric::L2)
		.value("InnerProduct", deglib::Metric::InnerProduct);

	py::class_<deglib::FloatSpace>(m, "FloatSpace")
		.def(py::init<const size_t, const deglib::Metric>());

	// read only graph
	py::class_<deglib::graph::ReadOnlyGraph>(m, "ReadOnlyGraph")
		.def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
		.def("size", &deglib::graph::ReadOnlyGraph::size);

	m.def("load_readonly_graph", &deglib::graph::load_readonly_graph);
}
