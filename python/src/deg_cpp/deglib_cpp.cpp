//
// Created by Bruno Schilling on 29.05.24.
//

// #define PYBIND11_DETAILED_ERROR_MESSAGES
#include <algorithm>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <deglib.h>


namespace py = pybind11;
constexpr int MAX_TRIES_SAME_RESULT_SIZE = 10;


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


// Multithreaded executor
// The helper function copied from https://github.com/nmslib/hnswlib/blob/master/examples/cpp/example_mt_search.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline size_t parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
  size_t all_num_results = std::numeric_limits<size_t>::max();
  if (numThreads <= 0) {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      size_t num_results = fn(id, 0);
      if (all_num_results == std::numeric_limits<size_t>::max()) {
        all_num_results = num_results;
      } else if (all_num_results != num_results) {
        return std::numeric_limits<size_t>::max(); // return error
      }
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);
    std::vector<size_t> all_results(numThreads);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if (id >= end) {
            break;
          }

          size_t num_results;
          try {
            num_results = fn(id, threadId);
          } catch (...) {
            std::unique_lock<std::mutex> lastExceptLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
          all_results[threadId] = num_results;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
    for (size_t num_results : all_results) {
      if (num_results == std::numeric_limits<size_t>::max()) {
        all_num_results = std::numeric_limits<size_t>::max(); // return error
        break;
      }
      if (all_num_results == std::numeric_limits<size_t>::max()) {
        all_num_results = num_results;
      } else if (all_num_results != num_results) {
        all_num_results = std::numeric_limits<size_t>::max(); // return error
        break;
      }
    }
  }
  return all_num_results;
}

template<typename G>
size_t search_one_query(const G& graph, size_t query_index, const py::buffer_info& query_info, const std::vector<uint32_t> &entry_vertex_indices, uint32_t k, const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count, const float eps, uint32_t* const result_indices_ptr, float* const result_distances_ptr) {
  std::byte* query_ptr = static_cast<std::byte *>(query_info.ptr) + query_info.strides[0] * query_index;
  deglib::search::ResultSet result = graph.search(entry_vertex_indices, query_ptr, eps, k, filter, max_distance_computation_count);

  assert((void(std::format("Expected result should have k={} entries, but got {} entries.\n", k, result.size())), (k == result.size())));

  uint32_t k_index = result.size()-1;  // start by last index to reverse result order
  size_t num_results = result.size();
  while (!result.empty()) {
    // location in result buffer
    const uint32_t offset = k_index + query_index * k;
    uint32_t* indices_target_ptr = static_cast<uint32_t*>(result_indices_ptr) + offset;
    float* distances_target_ptr = static_cast<float*>(result_distances_ptr) + offset;

    // get best result
    deglib::search::ObjectDistance next_result = result.top();
    *indices_target_ptr = graph.getExternalLabel(next_result.getInternalIndex());
    *distances_target_ptr = next_result.getDistance();

    result.pop();
    k_index--;
  }
  return num_results;
}

template<typename G>
size_t search_batch_of_queries(const G& graph, size_t batch_index, size_t batch_size, const py::buffer_info& query_info, const std::vector<uint32_t> &entry_vertex_indices, uint32_t k, const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count, const float eps, uint32_t* const result_indices_ptr, float* const result_distances_ptr) {
  size_t num_queries = query_info.shape[0];
  auto upper_bound = std::min(num_queries, (batch_index+1)*batch_size);
  size_t all_num_results = std::numeric_limits<size_t>::max();

  // repeat until all searches return same number of results
  for (int try_counter = 0; try_counter < MAX_TRIES_SAME_RESULT_SIZE; try_counter++) {
    for (size_t query_index = batch_index*batch_size; query_index < upper_bound; query_index++) {
      size_t num_results = search_one_query(
        graph, query_index, query_info, entry_vertex_indices, k, filter, max_distance_computation_count, eps,
        result_indices_ptr, result_distances_ptr
      );
      // check if all have same number of results
      if (all_num_results == std::numeric_limits<size_t>::max()) {
        all_num_results = num_results;
      } else if (all_num_results != num_results) {
        // retry! this case should be extremely rare. Only if the graph is updated and searched simultaneously
        all_num_results = std::numeric_limits<size_t>::max();
        break;
      }
    }
    if (all_num_results != std::numeric_limits<size_t>::max()) {
      break;
    }
  }
  if (all_num_results == std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Got queries with different number of results, after 10 tries. This should not happen.");
  }
  return all_num_results;
}

template<typename G>
std::tuple<py::array_t<uint32_t>, py::array_t<float>, size_t> graph_search_wrapper(
    const G& graph, const std::vector<uint32_t> &entry_vertex_indices,
    const py::array query, const float eps, const uint32_t k,
    const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count, const uint32_t threads,
    const uint32_t batch_size
) {
  py::buffer_info query_info = query.request();

  assert((void(std::format("Expected query to have two dimensions, got {}\n", query_info.ndim)), (query_info.ndim == 2)));

  uint32_t n_queries = query_info.shape[0];

  py::array_t<uint32_t> result_indices({n_queries, k});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t*>(result_indices_info.ptr);

  py::array_t<float> result_distances({n_queries, k});
  py::buffer_info result_distances_info = result_distances.request();
  auto result_distances_ptr = static_cast<float*>(result_distances_info.ptr);

  py::gil_scoped_release release; // release the gil

  size_t all_num_results = std::numeric_limits<size_t>::max();
  if (threads == 1) {
    for (int try_counter = 0; try_counter < MAX_TRIES_SAME_RESULT_SIZE; try_counter++) {
      for (uint32_t query_index = 0; query_index < query_info.shape[0]; query_index++) {
        size_t num_results = search_one_query(
          graph, query_index, query_info, entry_vertex_indices, k, filter, max_distance_computation_count, eps,
          result_indices_ptr, result_distances_ptr
        );
        if (all_num_results == std::numeric_limits<size_t>::max()) {
          all_num_results = num_results;
        } else if (all_num_results != num_results) {
          all_num_results = std::numeric_limits<size_t>::max();
          break;
        }
      }
      if (all_num_results != std::numeric_limits<size_t>::max()) {
        break;
      }
    }
  } else {
    size_t n_batches = (n_queries / batch_size) + ((n_queries % batch_size != 0) ? 1 : 0);  // +1, if n_queries % batch_size != 0
    for (int i = 0; i < MAX_TRIES_SAME_RESULT_SIZE; i++) {
      all_num_results = parallel_for(0, n_batches, threads, [&] (size_t batch_index, size_t thread_id) {
        // search_one_query(graph, query_index, query_info, entry_vertex_indices, k, filter, max_distance_computation_count, eps, result_indices_ptr, result_distances_ptr);
        return search_batch_of_queries(graph, batch_index, batch_size, query_info, entry_vertex_indices, k, filter, max_distance_computation_count, eps, result_indices_ptr, result_distances_ptr);
      });
      if (all_num_results != std::numeric_limits<size_t>::max()) {
        break;
      }
    }
  }
  if (all_num_results == std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Got queries with different number of results, after 10 tries. This should not happen.");
  }

  return {result_indices, result_distances, all_num_results};
}

deglib::graph::ReadOnlyGraph read_only_graph_from_search_graph(deglib::search::SearchGraph& search_graph, const uint32_t max_vertex_count, const deglib::FloatSpace& feature_space, const uint8_t edges_per_vertex) {
  return {max_vertex_count, edges_per_vertex, feature_space, search_graph};
}

PYBIND11_MODULE(deglib_cpp, m) {
  m.doc() = "Python bindings for Dynamic Exploration Graph";

  m.def("avx_usable", &avx_usable, "Returns whether AVX instructions are available");
  m.def("sse_usable", &sse_usable, "Returns whether SSE instructions are available");

  // distances
  py::enum_<deglib::Metric>(m, "Metric")
      .value("L2", deglib::Metric::L2)
      .value("L2_Uint8", deglib::Metric::L2_Uint8)
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

  py::class_<deglib::graph::Filter>(m, "Filter")
      .def(py::init<const int*, size_t, size_t, size_t>());

  m.def("create_filter", [](py::array_t<int, py::array::c_style> valid_labels, size_t max_value, size_t max_label_count) {
    const py::buffer_info labels_info = valid_labels.request();
    const int* ptr = static_cast<int*>(labels_info.ptr);
    // only allow one dimensional arrays
    assert((void(std::format("Expected feature to have only one dimension, got {}\n", labels_info.ndim)), (labels_info.ndim == 1)));

    size_t size = labels_info.shape[0];
    return new deglib::graph::Filter(ptr, size, max_value, max_label_count);
  });

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
             const bool uint8_metric = g.getFeatureSpace().metric() == deglib::Metric::L2_Uint8;
             const char* format_descriptor = uint8_metric ? "B" : "f";
             const size_t item_size = uint8_metric ? sizeof(uint8_t) : sizeof(float);
             return py::memoryview::from_buffer(
                 g.getFeatureVector(internal_idx),
                 item_size, format_descriptor, {g.getFeatureSpace().dim()}, {item_size});
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

  m.def("read_only_graph_from_graph", &read_only_graph_from_search_graph);

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
           const bool uint8_metric = g.getFeatureSpace().metric() == deglib::Metric::L2_Uint8;
           const char* format_descriptor = uint8_metric ? "B" : "f";
           const size_t item_size = uint8_metric ? sizeof(uint8_t) : sizeof(float);
           return py::memoryview::from_buffer(
               g.getFeatureVector(internal_idx),
               item_size, format_descriptor, {g.getFeatureSpace().dim()}, {item_size});
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
        assert((void(std::format("Expected feature to have only one dimension, got {}\n", feature_info.ndim)), (feature_info.ndim == 1)));
        return g.addVertex(external_label, ptr);
    })
    .def("remove_vertex", &deglib::graph::SizeBoundedGraph::removeVertex)
    .def("change_edge", &deglib::graph::SizeBoundedGraph::changeEdge)
    .def("change_edges", [] (deglib::graph::SizeBoundedGraph& g, const uint32_t internal_index, const py::array_t<uint32_t, py::array::c_style> neighbor_indices, const py::array_t<float, py::array::c_style> neighbor_weights) {
      const py::buffer_info neighbor_info = neighbor_indices.request();
      const uint32_t* neighbor_ptr = static_cast<uint32_t*>(neighbor_info.ptr);
      // only allow one dimensional arrays
      assert((void(std::format("Expected neighbor_indices to have only one dimension, got {}\n", neighbor_info.ndim)), (neighbor_info.ndim == 1)));

      const py::buffer_info weight_info = neighbor_weights.request();
      const float* weight_ptr = static_cast<float*>(weight_info.ptr);
      // only allow one dimensional arrays
      assert((void(std::format("Expected neighbor_weights to have only one dimension, got {}\n", weight_info.ndim)), (weight_info.ndim == 1)));

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
  py::enum_<deglib::builder::LID>(m, "LID")
    .value("Unknown", deglib::builder::LID::Unknown)
    .value("High", deglib::builder::LID::High)
    .value("Low", deglib::builder::LID::Low);

  py::class_<deglib::builder::EvenRegularGraphBuilder>(m, "EvenRegularGraphBuilder")
    .def(py::init<deglib::graph::MutableGraph&, std::mt19937&, const deglib::builder::LID, const uint8_t, const float, const uint8_t, const float, const uint8_t, const uint32_t, const uint32_t>())
    .def("add_entry", [] (deglib::builder::EvenRegularGraphBuilder& builder, const py::array_t<uint32_t, py::array::c_style>& label, const py::array& feature) {
      // label buffer
      const auto label_access = label.unchecked<1>();

      // feature buffer
      const py::buffer_info feature_info = feature.request();
      const std::byte* feature_ptr = static_cast<std::byte*>(feature_info.ptr);
      // only allow two dimensional array
      assert((void(std::format("Expected feature to have two dimensions, got {}\n", feature_info.ndim)), (feature_info.ndim == 2)));

      py::gil_scoped_release release; // release the gil

      // add entries
      const size_t feature_len = feature_info.itemsize * feature_info.shape[1];
      for (uint32_t i = 0; i < feature_info.shape[0]; i++) {
        // copy to vector
        std::vector<std::byte> feature_vec(
          feature_ptr + (feature_len*i),
          feature_ptr + (feature_len*(i+1))
        );
        const uint32_t current_label = label_access(i);
        builder.addEntry(current_label, std::move(feature_vec));
      }
    })
    .def("remove_entry", &deglib::builder::EvenRegularGraphBuilder::removeEntry)
    .def("get_num_new_entries", &deglib::builder::EvenRegularGraphBuilder::getNumNewEntries)
    .def("get_num_remove_entries", &deglib::builder::EvenRegularGraphBuilder::getNumRemoveEntries)
    .def("set_thread_count", &deglib::builder::EvenRegularGraphBuilder::setThreadCount)
    .def("set_batch_size", &deglib::builder::EvenRegularGraphBuilder::setBatchSize)
    .def("build", [] (deglib::builder::EvenRegularGraphBuilder& builder, std::function<void(deglib::builder::BuilderStatus&)> callback, const bool infinite) -> deglib::graph::MutableGraph& {
      return builder.build(callback, infinite);
    })
    .def("build_silent", [] (deglib::builder::EvenRegularGraphBuilder& builder, const bool infinite) -> deglib::graph::MutableGraph& {
      py::gil_scoped_release release;
      return builder.build([] (deglib::builder::BuilderStatus&) {}, infinite);
    })
    .def("stop", &deglib::builder::EvenRegularGraphBuilder::stop);

  m.def("calc_avg_edge_weight", &deglib::analysis::calc_avg_edge_weight);
  m.def("calc_edge_weight_histogram", &deglib::analysis::calc_edge_weight_histogram);
  m.def("check_graph_weights", &deglib::analysis::check_graph_weights);
  m.def("check_graph_regularity", &deglib::analysis::check_graph_regularity);
  m.def("check_graph_connectivity", &deglib::analysis::check_graph_connectivity);
  m.def("calc_non_rng_edges", &deglib::analysis::calc_non_rng_edges);
  m.def("remove_non_mrng_edges", &deglib::builder::remove_non_mrng_edges);

  py::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
    .def_readwrite("step", &deglib::builder::BuilderStatus::step)
    .def_readwrite("added", &deglib::builder::BuilderStatus::added)
    .def_readwrite("deleted", &deglib::builder::BuilderStatus::deleted)
    .def_readwrite("improved", &deglib::builder::BuilderStatus::improved)
    .def_readwrite("tries", &deglib::builder::BuilderStatus::tries);
}
