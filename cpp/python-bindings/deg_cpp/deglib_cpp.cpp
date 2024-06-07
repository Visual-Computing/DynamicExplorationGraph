//
// Created by Bruno Schilling on 29.05.24.
//
#define PYBIND11_DETAILED_ERROR_MESSAGES  // TODO: remove

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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


deglib::search::ResultSet read_only_graph_search_wrapper(
    const deglib::graph::ReadOnlyGraph &graph, const std::vector<uint32_t> &entry_vertex_indices,
    const py::array_t<float, py::array::c_style> query, const float eps, const uint32_t k,
    const uint32_t max_distance_computation_count)
{
  py::buffer_info query_info = query.request();
  return graph.search(entry_vertex_indices, static_cast<std::byte *>(query_info.ptr), eps, k,
                      max_distance_computation_count);
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
      .def(py::init<const size_t, const deglib::Metric>())
      .def("dim", &deglib::FloatSpace::dim);

  py::class_<deglib::search::ObjectDistance>(m, "ObjectDistance")
      .def("get_internal_index", &deglib::search::ObjectDistance::getInternalIndex);

  py::class_<deglib::search::ResultSet>(m, "ResultSet")
      .def("top", &deglib::search::ResultSet::top)
      .def("pop", &deglib::search::ResultSet::pop)
      .def("size", [](const deglib::search::ResultSet& rs) { return rs.size(); }) // TODO: why can't I bind function reference directly?
      .def("empty", [](const deglib::search::ResultSet& rs) { return rs.empty(); });

  // TODO: SpaceInterface in c++ is general over datatype, here we use float always
  py::class_<deglib::SpaceInterface<float>>(m, "SpaceInterface")
      .def("dim", &deglib::SpaceInterface<float>::dim);

  // read only graph
  py::class_<deglib::graph::ReadOnlyGraph>(m, "ReadOnlyGraph")
      .def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
      .def("size", &deglib::graph::ReadOnlyGraph::size)
      .def("get_feature_space",
           [](const deglib::graph::ReadOnlyGraph &g) -> const deglib::SpaceInterface<float> & { return g.getFeatureSpace(); },
           py::return_value_policy::reference)
      .def("get_feature_vector",
           [](const deglib::graph::ReadOnlyGraph &g, const uint32_t internal_idx) {
             return py::memoryview::from_buffer(
                 g.getFeatureVector(internal_idx),
                 sizeof(float), "f", {static_cast<ssize_t>(g.getFeatureSpace().dim())}, {sizeof(float)});
           }, py::return_value_policy::reference
      )
      .def("get_internal_index", &deglib::graph::ReadOnlyGraph::getInternalIndex)
      .def("search", &read_only_graph_search_wrapper)
      .def("get_entry_vertex_indices", &deglib::graph::ReadOnlyGraph::getEntryVertexIndices)
      .def("get_external_label", &deglib::graph::ReadOnlyGraph::getExternalLabel);

  m.def("load_readonly_graph", &deglib::graph::load_readonly_graph);

  // repository
  py::class_<deglib::StaticFeatureRepository>(m, "StaticFeatureRepository")
      .def("get_feature",
           [](const deglib::StaticFeatureRepository &fr, const uint32_t vertex_id) {
             return py::memoryview::from_buffer(
                 fr.getFeature(vertex_id),
                 sizeof(float), "f", {static_cast<ssize_t>(fr.dims())}, {sizeof(float)});
           }, py::return_value_policy::reference
      )
      .def("size", &deglib::StaticFeatureRepository::size)
      .def("dims", &deglib::StaticFeatureRepository::dims);

  m.def("load_static_repository", &deglib::load_static_repository);
}
