//
// Created by Bruno Schilling on 29.05.24.
//

// #define PYBIND11_DETAILED_ERROR_MESSAGES

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

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


template<typename G>
deglib::search::ResultSet graph_search_wrapper(
    const G& graph, const std::vector<uint32_t> &entry_vertex_indices,
    const py::array_t<float, py::array::c_style> query, const float eps, const uint32_t k,
    const uint32_t max_distance_computation_count)
{
  py::buffer_info query_info = query.request();
  return graph.search(entry_vertex_indices, static_cast<std::byte *>(query_info.ptr), eps, k,
                      max_distance_computation_count);
}

void test_callback(std::function<void(deglib::builder::BuilderStatus&)> callback, int start, int end) {
  deglib::builder::BuilderStatus bs {};
  for (int i = start; i < end; i++) {
    bs.added = i;
    callback(bs);
  }
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
      .def("dim", &deglib::FloatSpace::dim)
      .def("metric", &deglib::FloatSpace::metric)
      .def("get_data_size", &deglib::FloatSpace::get_data_size);

  py::class_<deglib::search::ObjectDistance>(m, "ObjectDistance")
    .def(py::init<const uint32_t, const float>())
    .def("get_internal_index", &deglib::search::ObjectDistance::getInternalIndex)
    .def("get_distance", &deglib::search::ObjectDistance::getDistance)
    .def("__eq__", &deglib::search::ObjectDistance::operator==)
    .def("__lt__", &deglib::search::ObjectDistance::operator<)
    .def("__gt__", &deglib::search::ObjectDistance::operator>);

  py::class_<deglib::search::ResultSet>(m, "ResultSet")
    .def("top", &deglib::search::ResultSet::top)
    .def("pop", &deglib::search::ResultSet::pop)
    .def("size", [](const deglib::search::ResultSet& rs) { return rs.size(); })
    .def("empty", [](const deglib::search::ResultSet& rs) { return rs.empty(); })
    .def("__getitem__", [](const deglib::search::ResultSet& rs, std::size_t index) { return rs[index]; })
  ;

  // TODO: SpaceInterface in c++ is general over datatype, here we use float always
  py::class_<deglib::SpaceInterface<float>>(m, "SpaceInterface")
      .def("dim", &deglib::SpaceInterface<float>::dim);

  // graphs
  py::class_<deglib::search::SearchGraph>(m, "SearchGraph");


  // read only graph
  py::class_<deglib::graph::ReadOnlyGraph, deglib::search::SearchGraph>(m, "ReadOnlyGraph")
      .def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
      .def("size", &deglib::graph::ReadOnlyGraph::size)
      .def("get_feature_space",
           [](const deglib::graph::ReadOnlyGraph &g) -> const deglib::SpaceInterface<float> & { return g.getFeatureSpace(); },
           py::return_value_policy::reference)
      .def("get_feature_vector",
           [](const deglib::graph::ReadOnlyGraph &g, const uint32_t internal_idx) {
             return py::memoryview::from_buffer(
                 g.getFeatureVector(internal_idx),
                 sizeof(float), "f", {g.getFeatureSpace().dim()}, {sizeof(float)});
           }, py::return_value_policy::reference
      )
      .def("get_internal_index", &deglib::graph::ReadOnlyGraph::getInternalIndex)
      .def("search", &graph_search_wrapper<deglib::graph::ReadOnlyGraph>)
      .def("explore", &deglib::graph::ReadOnlyGraph::explore)
      .def("has_path", &deglib::graph::ReadOnlyGraph::hasPath)
      .def("get_entry_vertex_indices", &deglib::graph::ReadOnlyGraph::getEntryVertexIndices)
      .def("get_edges_per_vertex", &deglib::graph::ReadOnlyGraph::getEdgesPerVertex)
      .def("get_neighbor_indices",
           [](const deglib::graph::ReadOnlyGraph &g, const uint32_t internal_idx) {
             return py::memoryview::from_buffer(
                 g.getNeighborIndices(internal_idx),
                 sizeof(uint32_t), "I", {g.getEdgesPerVertex()}, {sizeof(uint32_t)});
           }, py::return_value_policy::reference
      )
      .def("has_vertex", &deglib::graph::ReadOnlyGraph::hasVertex)
      .def("has_edge", &deglib::graph::ReadOnlyGraph::hasEdge)
      .def("get_external_label", &deglib::graph::ReadOnlyGraph::getExternalLabel);

  m.def("load_readonly_graph", &deglib::graph::load_readonly_graph);
  
  // mutable graph
  py::class_<deglib::graph::MutableGraph, deglib::search::SearchGraph>(m, "MutableGraph");

  // size bounded graph
  py::class_<deglib::graph::SizeBoundedGraph, deglib::graph::MutableGraph>(m, "SizeBoundedGraph")
    .def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
    .def("size", &deglib::graph::SizeBoundedGraph::size)
    .def("get_feature_space",
         [](const deglib::graph::SizeBoundedGraph &g) -> const deglib::SpaceInterface<float> & { return g.getFeatureSpace(); },
         py::return_value_policy::reference)
    .def("get_feature_vector",
         [](const deglib::graph::SizeBoundedGraph &g, const uint32_t internal_idx) {
           return py::memoryview::from_buffer(
               g.getFeatureVector(internal_idx),
               sizeof(float), "f", {g.getFeatureSpace().dim()}, {sizeof(float)});
         }, py::return_value_policy::reference
    )
    .def("get_internal_index", &deglib::graph::SizeBoundedGraph::getInternalIndex)
    .def("has_path", &deglib::graph::SizeBoundedGraph::hasPath)
    .def("get_entry_vertex_indices", &deglib::graph::SizeBoundedGraph::getEntryVertexIndices)
    .def("get_external_label", &deglib::graph::SizeBoundedGraph::getExternalLabel)
    .def("get_edges_per_vertex", &deglib::graph::SizeBoundedGraph::getEdgesPerVertex)
    .def("save_graph", &deglib::graph::SizeBoundedGraph::saveGraph)
    .def("add_vertex", [] (deglib::graph::SizeBoundedGraph& g, const uint32_t external_label, const py::array_t<float, py::array::c_style> feature_vector) -> uint32_t {
        const py::buffer_info feature_info = feature_vector.request();
        const std::byte* ptr = static_cast<std::byte*>(feature_info.ptr);
        // only allow one dimensional arrays
        assert((feature_info.ndim == 1) && std::format("Expected feature to have only one dimension, got {}\n", feature_info.ndim));
        return g.addVertex(external_label, ptr);
    })
    .def("remove_vertex", &deglib::graph::SizeBoundedGraph::removeVertex)
    .def("change_edge", &deglib::graph::SizeBoundedGraph::changeEdge)
    .def("change_edges", [] (deglib::graph::SizeBoundedGraph& g, const uint32_t internal_index, const py::array_t<uint32_t, py::array::c_style> neighbor_indices, const py::array_t<float, py::array::c_style> neighbor_weights) {
      const py::buffer_info neighbor_info = neighbor_indices.request();
      const uint32_t* neighbor_ptr = static_cast<uint32_t*>(neighbor_info.ptr);
      // only allow one dimensional arrays
      assert((neighbor_info.ndim == 1) && std::format("Expected neighbor_indices to have only one dimension, got {}\n", neighbor_info.ndim));

      const py::buffer_info weight_info = neighbor_weights.request();
      const float* weight_ptr = static_cast<float*>(weight_info.ptr);
      // only allow one dimensional arrays
      assert((weight_info.ndim == 1) && std::format("Expected neighbor_weights to have only one dimension, got {}\n", weight_info.ndim));

      g.changeEdges(internal_index, neighbor_ptr, weight_ptr);
    })
    .def("get_neighbor_weights",
         [](const deglib::graph::SizeBoundedGraph &g, const uint32_t internal_idx) {
           return py::memoryview::from_buffer(
               g.getNeighborWeights(internal_idx),
               sizeof(float), "f", {g.getEdgesPerVertex()}, {sizeof(float)});
         }, py::return_value_policy::reference
    )
    .def("get_edge_weight", &deglib::graph::SizeBoundedGraph::getEdgeWeight)
    .def("get_neighbor_indices",
      [](const deglib::graph::SizeBoundedGraph &g, const uint32_t internal_idx) {
        return py::memoryview::from_buffer(
            g.getNeighborIndices(internal_idx),
            sizeof(uint32_t), "I", {g.getEdgesPerVertex()}, {sizeof(uint32_t)});
      }, py::return_value_policy::reference
    )
    .def("has_vertex", &deglib::graph::SizeBoundedGraph::hasVertex)
    .def("has_edge", &deglib::graph::SizeBoundedGraph::hasEdge)
    .def("search", &graph_search_wrapper<deglib::graph::SizeBoundedGraph>)
    .def("explore", &deglib::graph::SizeBoundedGraph::explore);

  // repository
  py::class_<deglib::StaticFeatureRepository>(m, "StaticFeatureRepository")
    .def("get_feature",
         [](const deglib::StaticFeatureRepository &fr, const uint32_t vertex_id) {
           return py::memoryview::from_buffer(
               fr.getFeature(vertex_id),
               sizeof(float), "f", {fr.dims()}, {sizeof(float)});
         }, py::return_value_policy::reference
    )
    .def("size", &deglib::StaticFeatureRepository::size)
    .def("dims", &deglib::StaticFeatureRepository::dims)
    .def("clear", &deglib::StaticFeatureRepository::clear);

  m.def("load_static_repository", &deglib::load_static_repository);

  // random mt19937
  py::class_<std::mt19937>(m, "Mt19937")
      .def(py::init<std::uint_fast32_t>());

  // even regular builder
  py::class_<deglib::builder::EvenRegularGraphBuilder>(m, "EvenRegularGraphBuilder")
    .def(py::init<deglib::graph::MutableGraph&, std::mt19937&, const uint8_t, const float, const uint8_t, const float, const uint8_t, const uint32_t, const uint32_t>())
    .def("add_entry", [] (deglib::builder::EvenRegularGraphBuilder& builder, const uint32_t label, py::array_t<float, py::array::c_style> feature) {
      // request buffer info
      const py::buffer_info feature_info = feature.request();
      const std::byte* ptr = static_cast<std::byte*>(feature_info.ptr);
      // only allow one dimensional arrays
      assert((feature_info.ndim == 1) && std::format("Expected feature to have only one dimension, got {}\n", feature_info.ndim));
      // copy to vector
      std::vector<std::byte> feature_vec(ptr, ptr + feature_info.itemsize * feature_info.shape[0]);
      builder.addEntry(label, std::move(feature_vec));
    })
    .def("remove_entry", &deglib::builder::EvenRegularGraphBuilder::removeEntry)
    .def("build", [] (deglib::builder::EvenRegularGraphBuilder& builder, std::function<void(deglib::builder::BuilderStatus&)> callback, const bool infinite) -> deglib::graph::MutableGraph& {
      return builder.build(callback, infinite);
    })
    .def("build_silent", [] (deglib::builder::EvenRegularGraphBuilder& builder, const bool infinite) -> deglib::graph::MutableGraph& {
      return builder.build([] (deglib::builder::BuilderStatus&) {}, infinite);
    })
    .def("stop", &deglib::builder::EvenRegularGraphBuilder::stop);

  m.def("calc_avg_edge_weight", &deglib::analysis::calc_avg_edge_weight);
  m.def("calc_edge_weight_histogram", &deglib::analysis::calc_edge_weight_histogram);
  m.def("check_graph_weights", &deglib::analysis::check_graph_weights);
  m.def("check_graph_regularity", &deglib::analysis::check_graph_regularity);
  m.def("check_graph_connectivity", &deglib::analysis::check_graph_connectivity);
  m.def("calc_non_rng_edges", &deglib::analysis::calc_non_rng_edges);

  py::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
    .def_readwrite("step", &deglib::builder::BuilderStatus::step)
    .def_readwrite("added", &deglib::builder::BuilderStatus::added)
    .def_readwrite("deleted", &deglib::builder::BuilderStatus::deleted)
    .def_readwrite("improved", &deglib::builder::BuilderStatus::improved)
    .def_readwrite("tries", &deglib::builder::BuilderStatus::tries);

  // tests TODO: remove
  m.def("test_take_graph", [] (deglib::graph::MutableGraph& g) { std::cout << g.size() << std::endl; });
  m.def("test_callback", &test_callback);
  m.def("print_py_buffer", [](const py::array buffer) {
    const py::buffer_info info = buffer.request();
    std::cout << "format: " << info.format << std::endl;
  });
}
