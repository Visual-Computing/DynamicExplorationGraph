//
// Python bindings for Dynamic Exploration Graph using nanobind
//

#include "bindings_common.h"

NB_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    deglib_py::bind_cpu_and_distances(m);
    deglib_py::bind_optimization(m);
    deglib_py::bind_transformation(m);
    deglib_py::bind_search(m);
    deglib_py::bind_graph_and_builder(m);
    deglib_py::bind_tuning(m);
}
