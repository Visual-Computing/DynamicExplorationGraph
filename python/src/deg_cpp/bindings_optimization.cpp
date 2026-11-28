#include "bindings_common.h"


namespace deglib_py {
void bind_optimization(nb::module_& m) {
    nb::module_ optimization_module = m.def_submodule("optimization", "Graph optimization and layout utilities");
    optimization_module.def(
        "presort", &presort_wrapper, nb::arg("vectors"), nb::arg("space"), nb::arg("radius_decay") = 0.9f, nb::arg("threads") = 0,
        nb::arg("callback") = nb::none(), nb::arg("seed") = 42, "Perform 1D pre-sorting of dataset feature vectors using FLAS"
    );
    optimization_module.def(
        "prune_non_mrng_edges",
        [](deglib::DynamicExplorationGraph& graph, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), threads);
        },
        nb::arg("graph"), nb::arg("threads") = 0, "Prune edges that violate the MRNG condition"
    );
    optimization_module.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, threads);
        },
        nb::arg("graph"), nb::arg("prune_worst"), nb::arg("threads") = 0, "Prune the worst edges of each vertex"
    );

    // transformation submodule

}
}  // namespace deglib_py
