//
// Created by Bruno Schilling on 29.05.24.
//

// #define PYBIND11_DETAILED_ERROR_MESSAGES
#include <algorithm>
#include <limits>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <deglib/deglib.h>
#include <deglib/distance/fp16.h>
#include <deglib/optimization/pruning.h>
#include <deglib/optimization/quantization/evp_quantize.h>

namespace py = pybind11;

template <typename G>
std::tuple<py::array_t<uint32_t>, py::array_t<float>>
graph_search_batch_wrapper(const G &graph, const py::array query,
                           const float eps, const uint32_t k,
                           const deglib::graph::Filter *filter,
                           const uint32_t max_distance_computation_count,
                           const uint32_t threads) {
  py::buffer_info query_info = query.request();

  if (query_info.ndim != 2) {
    throw std::invalid_argument(std::format(
        "Expected query to have two dimensions, got {}", query_info.ndim));
  }
  const size_t req_bytes = graph.getFeatureSpace().get_data_size();
  const size_t query_stride_bytes =
      size_t(query_info.shape[1]) * query_info.itemsize;
  if (query_stride_bytes != req_bytes) {
    throw std::invalid_argument(
        std::format("Query row size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    query_stride_bytes, req_bytes));
  }

  const uint32_t n_queries = static_cast<uint32_t>(query_info.shape[0]);

  py::array_t<uint32_t> result_indices({n_queries, k});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t *>(result_indices_info.ptr);
  std::fill_n(result_indices_ptr, n_queries * k, 0);

  py::array_t<float> result_distances({n_queries, k});
  py::buffer_info result_distances_info = result_distances.request();
  auto result_distances_ptr = static_cast<float *>(result_distances_info.ptr);
  std::fill_n(result_distances_ptr, n_queries * k, std::numeric_limits<float>::quiet_NaN());

  py::gil_scoped_release release; // release the gil

  auto search_range = [&](size_t begin, size_t end) {
    for (size_t q = begin; q < end; ++q) {
      const std::byte *query_ptr = static_cast<const std::byte *>(query_info.ptr) +
                                   query_info.strides[0] * q;
      const size_t query_bytes = query_info.shape[1] * query_info.itemsize;
      std::span<const std::byte> query_span(query_ptr, query_bytes);

      deglib::search::ResultSet result =
          graph.search(query_span, k, eps, filter, max_distance_computation_count);

      // Limit results to at most k elements
      while (result.size() > k) {
        result.pop();
      }

      const size_t count = result.size();
      const size_t base_offset = q * k;

      // Extract elements from heap in descending order (furthest to closest)
      for (size_t i = count; i > 0; --i) {
        deglib::search::ObjectDistance next_result = result.top();
        result_indices_ptr[base_offset + i - 1] =
            graph.getExternalLabel(next_result.getInternalIndex());
        result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
        result.pop();
      }
    }
  };

  if (threads <= 1) {
    search_range(0, n_queries);
  } else {
    deglib::concurrent::parallel_batch_for(
        0, n_queries, threads, [&](size_t begin, size_t end, size_t thread_id) {
          search_range(begin, end);
        });
  }

  return std::make_tuple(result_indices, result_distances);
}

template <typename G>
deglib::search::ResultSet
graph_search_single_wrapper(const G &graph, const py::array query,
                            const float eps, const uint32_t k,
                            const deglib::graph::Filter *filter,
                            const uint32_t max_distance_computation_count) {
  py::buffer_info query_info = query.request();
  const size_t req_bytes = graph.getFeatureSpace().get_data_size();
  const size_t query_bytes = query_info.size * query_info.itemsize;
  if (query_bytes != req_bytes) {
    throw std::invalid_argument(
        std::format("Query vector size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    query_bytes, req_bytes));
  }
  const std::byte *query_ptr = static_cast<const std::byte *>(query_info.ptr);
  std::span<const std::byte> query_span(query_ptr, query_bytes);
  return graph.search(query_span, k, eps, filter,
                      max_distance_computation_count);
}

template <typename G>
std::tuple<py::array_t<uint32_t>, py::array_t<float>> graph_explore_wrapper(
    const G &graph,
    const py::array_t<uint32_t, py::array::c_style> entry_vertex_indices,
    const uint32_t k, const bool include_entry,
    const uint32_t max_distance_computation_count, const uint32_t threads) {
  py::buffer_info entry_info = entry_vertex_indices.request();
  if (entry_info.ndim != 1) {
    throw std::invalid_argument(
        std::format("Expected entry_vertex_indices to have 1 dimension, got {}",
                    entry_info.ndim));
  }

  const uint32_t *entry_ptr = static_cast<const uint32_t *>(entry_info.ptr);
  const uint32_t n_queries = entry_info.shape[0];

  py::array_t<uint32_t> result_indices({n_queries, k});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t *>(result_indices_info.ptr);
  std::fill_n(result_indices_ptr, n_queries * k, 0);

  py::array_t<float> result_distances({n_queries, k});
  py::buffer_info result_distances_info = result_distances.request();
  auto result_distances_ptr = static_cast<float *>(result_distances_info.ptr);
  std::fill_n(result_distances_ptr, n_queries * k, std::numeric_limits<float>::quiet_NaN());

  py::gil_scoped_release release; // release the gil

  auto explore_range = [&](size_t begin, size_t end) {
    for (size_t q = begin; q < end; ++q) {
      uint32_t entry_idx = entry_ptr[q];
      deglib::search::ResultSet result = graph.explore(
          entry_idx, k, max_distance_computation_count, 0.0f, include_entry);

      // Limit results to at most k elements
      while (result.size() > k) {
        result.pop();
      }

      const size_t count = result.size();
      const size_t base_offset = q * k;

      // Extract elements from heap in descending order (furthest to closest)
      for (size_t i = count; i > 0; --i) {
        deglib::search::ObjectDistance next_result = result.top();
        result_indices_ptr[base_offset + i - 1] =
            graph.getExternalLabel(next_result.getInternalIndex());
        result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
        result.pop();
      }
    }
  };

  if (threads <= 1) {
    explore_range(0, n_queries);
  } else {
    deglib::concurrent::parallel_batch_for(
        0, n_queries, threads, [&](size_t begin, size_t end, size_t thread_id) {
          explore_range(begin, end);
        });
  }

  return std::make_tuple(result_indices, result_distances);
}

float float_space_compute_distance(const deglib::FloatSpace &space,
                                   py::array vec1, py::array vec2) {
  auto buf1 = vec1.request();
  auto buf2 = vec2.request();
  const size_t req_bytes = space.get_data_size();
  if (size_t(buf1.size) * buf1.itemsize != req_bytes) {
    throw std::invalid_argument(
        std::format("Vector vec1 size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    size_t(buf1.size) * buf1.itemsize, req_bytes));
  }
  if (size_t(buf2.size) * buf2.itemsize != req_bytes) {
    throw std::invalid_argument(
        std::format("Vector vec2 size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    size_t(buf2.size) * buf2.itemsize, req_bytes));
  }
  return deglib::distances::compute_distance(space, buf1.ptr, buf2.ptr);
}

py::array_t<float>
float_space_compute_distances(const deglib::FloatSpace &space, py::array query,
                              py::array targets) {
  auto q_buf = query.request();
  auto t_buf = targets.request();
  const size_t req_bytes = space.get_data_size();

  if (size_t(q_buf.size) * q_buf.itemsize != req_bytes) {
    throw std::invalid_argument(
        std::format("Query vector size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    size_t(q_buf.size) * q_buf.itemsize, req_bytes));
  }

  size_t num_targets = 0;
  size_t target_stride_bytes = 0;
  if (t_buf.ndim == 1) {
    num_targets = 1;
    target_stride_bytes = size_t(t_buf.size) * t_buf.itemsize;
  } else if (t_buf.ndim == 2) {
    num_targets = t_buf.shape[0];
    target_stride_bytes = size_t(t_buf.shape[1]) * t_buf.itemsize;
  } else {
    throw std::invalid_argument("Targets array must be 1D or 2D NumPy array");
  }

  if (target_stride_bytes != req_bytes) {
    throw std::invalid_argument(
        std::format("Target vector size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    target_stride_bytes, req_bytes));
  }

  auto result = py::array_t<float>(num_targets);
  auto r_buf = result.request();

  deglib::distances::compute_distances(space, q_buf.ptr, 1, t_buf.ptr,
                                       num_targets,
                                       static_cast<float *>(r_buf.ptr));

  return result;
}

py::array_t<uint32_t>
float_space_rerank(const deglib::FloatSpace &space, py::array queries,
                   py::array_t<uint32_t> candidate_indices,
                   py::object base_vectors = py::none(), size_t k_top = 0,
                   size_t num_threads = 0) {
  auto q_buf = queries.request();
  if (q_buf.ndim != 2) {
    throw std::invalid_argument("queries must be a 2D array");
  }
  const size_t req_bytes = space.get_data_size();
  const size_t query_stride_bytes = size_t(q_buf.shape[1]) * q_buf.itemsize;
  if (query_stride_bytes != req_bytes) {
    throw std::invalid_argument(
        std::format("Query row size in bytes ({}) does not match required "
                    "feature space data size ({})",
                    query_stride_bytes, req_bytes));
  }
  size_t n_queries = q_buf.shape[0];

  auto cand_buf = candidate_indices.request();
  if (cand_buf.ndim != 2) {
    throw std::invalid_argument("candidate_indices must be a 2D array");
  }
  size_t n_cand_rows = cand_buf.shape[0];
  size_t evp_k = cand_buf.shape[1];

  if (n_cand_rows != n_queries) {
    throw std::invalid_argument(
        "Number of candidate rows must match number of queries");
  }

  if (k_top == 0 || k_top > evp_k) {
    k_top = evp_k;
  }

  bool has_base_vectors = !base_vectors.is_none();
  py::buffer_info base_buf;
  if (has_base_vectors) {
    base_buf = py::cast<py::array>(base_vectors).request();
    if (base_buf.ndim != 2) {
      throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const size_t base_stride_bytes =
        size_t(base_buf.shape[1]) * base_buf.itemsize;
    if (base_stride_bytes != req_bytes) {
      throw std::invalid_argument(
          std::format("Base vector row size in bytes ({}) does not match "
                      "required feature space data size ({})",
                      base_stride_bytes, req_bytes));
    }
  }

  const void *base_ptr = has_base_vectors ? base_buf.ptr : nullptr;
  size_t num_base_vectors = has_base_vectors ? base_buf.shape[0] : 0;

  auto result_indices = py::array_t<uint32_t>({n_queries, k_top});
  auto res_buf = result_indices.request();

  deglib::search::rerank(
      space, q_buf.ptr, n_queries, base_ptr, num_base_vectors,
      static_cast<const uint32_t *>(cand_buf.ptr), evp_k, k_top, num_threads,
      static_cast<uint32_t *>(res_buf.ptr));

  return result_indices;
}

// ============================================================================
// EVP Quantization Bindings
// ============================================================================

py::array_t<uint8_t> quantize_batch_wrapper(py::array vectors,
                                            uint32_t non_zeros,
                                            size_t num_threads) {
  py::buffer_info buf = vectors.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D array");
  }
  size_t count = buf.shape[0];
  uint32_t dim = static_cast<uint32_t>(buf.shape[1]);
  if (dim % 8 != 0) {
    throw std::invalid_argument(std::format(
        "Vector dimension ({}) must be a multiple of 8 for EVP quantization",
        dim));
  }
  size_t evp_bytes = 2 * (dim / 8);

  std::vector<std::byte> result;
  if (buf.itemsize == 2 &&
      (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
    result = deglib::optimization::quantize_evp_batch(
        static_cast<const uint16_t *>(buf.ptr), count, dim, non_zeros,
        num_threads);
  } else if (buf.itemsize == 4 &&
             (buf.format == "f" || buf.format == "float")) {
    result = deglib::optimization::quantize_evp_batch(
        static_cast<const float *>(buf.ptr), count, dim, non_zeros,
        num_threads);
  } else {
    throw std::invalid_argument(
        std::format("vectors must be float32 (format 'f') or FP16/uint16 "
                    "(format 'e'/'H'), got format '{}' with itemsize {}",
                    buf.format, buf.itemsize));
  }

  py::array_t<uint8_t> output({count, evp_bytes});
  py::buffer_info out_buf = output.request();
  std::memcpy(out_buf.ptr, result.data(), count * evp_bytes);
  return output;
}

// ============================================================================
// FP16 Conversion Bindings
// ============================================================================

py::array_t<uint16_t>
floats_to_fp16_wrapper(py::array_t<float, py::array::c_style> floats) {
  py::buffer_info buf = floats.request();
  if (buf.itemsize != sizeof(float)) {
    throw std::invalid_argument(
        "Input array must contain 32-bit floats (itemsize == 4)");
  }
  size_t count = buf.size;

  py::array_t<uint16_t> output(count);
  py::buffer_info out_buf = output.request();
  deglib::distances::fp16::floats_to_fp16(static_cast<const float *>(buf.ptr),
                                          static_cast<uint16_t *>(out_buf.ptr),
                                          count);
  return output;
}

py::array_t<float>
fp16_to_floats_wrapper(py::array_t<uint16_t, py::array::c_style> fp16_vals) {
  py::buffer_info buf = fp16_vals.request();
  if (buf.itemsize != sizeof(uint16_t)) {
    throw std::invalid_argument(
        "Input array must contain 16-bit uint16/FP16 values (itemsize == 2)");
  }
  size_t count = buf.size;

  py::array_t<float> output(count);
  py::buffer_info out_buf = output.request();
  deglib::distances::fp16::fp16_to_floats(
      static_cast<const uint16_t *>(buf.ptr), static_cast<float *>(out_buf.ptr),
      count);
  return output;
}

deglib::graph::ReadOnlyGraph read_only_graph_from_search_graph(
    deglib::search::SearchGraph &search_graph, const uint32_t max_vertex_count,
    const deglib::FloatSpace &feature_space, const uint8_t edges_per_vertex) {
  return {max_vertex_count, edges_per_vertex, feature_space, search_graph};
}

PYBIND11_MODULE(deglib_cpp, m) {
  m.doc() = "Python bindings for Dynamic Exploration Graph";

  m.def("avx_usable", &deglib::cpu::has_avx2,
        "Returns whether AVX2 instructions are available");
  m.def("avx512_usable", &deglib::cpu::has_avx512,
        "Returns whether AVX512 instructions are available");

  // quantization
  m.def("quantize_batch", &quantize_batch_wrapper,
        "Quantize float32 or float16/uint16 vectors to byte-packed EVP format");
  m.def("floats_to_fp16", &floats_to_fp16_wrapper,
        "Convert float32 array to FP16 (uint16_t)");
  m.def("fp16_to_floats", &fp16_to_floats_wrapper,
        "Convert FP16 (uint16_t) array to float32");

  // distances
  py::enum_<deglib::cpu::InstructionSet>(m, "InstructionSet")
      .value("Auto", deglib::cpu::InstructionSet::Auto)
      .value("Scalar", deglib::cpu::InstructionSet::Scalar)
      .value("AVX2", deglib::cpu::InstructionSet::AVX2)
      .value("AVX512", deglib::cpu::InstructionSet::AVX512);

  py::enum_<deglib::MetricType>(m, "Metric")
      .value("FP32_L2", deglib::MetricType::FP32_L2)
      .value("FP32_InnerProduct", deglib::MetricType::FP32_InnerProduct)
      .value("Uint8_L2", deglib::MetricType::Uint8_L2)
      .value("FP16_InnerProduct", deglib::MetricType::FP16_InnerProduct)
      .value("EVP_InnerProduct", deglib::MetricType::EVP_InnerProduct);

  py::class_<deglib::FloatSpace>(m, "FloatSpace")
      .def(py::init([](const size_t dim, const deglib::MetricType metric,
                       const deglib::cpu::InstructionSet instruction) {
             return deglib::FloatSpace(dim, deglib::Metric(metric),
                                       instruction);
           }),
           py::arg("dim"), py::arg("metric") = deglib::MetricType::FP32_L2,
           py::arg("instruction") = deglib::cpu::InstructionSet::Auto)
      .def("dim", &deglib::FloatSpace::dim)
      .def("metric",
           [](const deglib::FloatSpace &fs) { return fs.metric().value; })
      .def("get_data_size", &deglib::FloatSpace::get_data_size)
      .def("get_instruction", &deglib::FloatSpace::get_instruction)
      .def("compute_distance", &float_space_compute_distance, py::arg("vec1"),
           py::arg("vec2"))
      .def("compute_distances", &float_space_compute_distances,
           py::arg("query"), py::arg("targets"))
      .def("rerank", &float_space_rerank, py::arg("queries"),
           py::arg("candidate_indices"), py::arg("base_vectors") = py::none(),
           py::arg("k_top") = 0, py::arg("num_threads") = 0);

  py::class_<deglib::search::ObjectDistance>(m, "ObjectDistance")
      .def(py::init<const uint32_t, const float>())
      .def("get_internal_index",
           &deglib::search::ObjectDistance::getInternalIndex)
      .def("get_distance", &deglib::search::ObjectDistance::getDistance)
      .def("__eq__", &deglib::search::ObjectDistance::operator==)
      .def("__lt__", &deglib::search::ObjectDistance::operator<)
      .def("__gt__", &deglib::search::ObjectDistance::operator>);

  py::class_<deglib::search::ResultSet>(m, "ResultSet")
      .def("top", &deglib::search::ResultSet::top)
      .def("pop", &deglib::search::ResultSet::pop)
      .def("size",
           [](const deglib::search::ResultSet &rs) { return rs.size(); })
      .def("empty",
           [](const deglib::search::ResultSet &rs) { return rs.empty(); })
      .def("__getitem__", [](const deglib::search::ResultSet &rs,
                             std::size_t index) { return rs[index]; });

  py::class_<deglib::graph::Filter>(m, "Filter")
      .def(py::init<const int *, size_t, size_t, size_t>());

  m.def("create_filter", [](py::array_t<int, py::array::c_style> valid_labels,
                            size_t max_value, size_t max_label_count) {
    const py::buffer_info labels_info = valid_labels.request();
    const int *ptr = static_cast<int *>(labels_info.ptr);
    // only allow one dimensional arrays
    assert((void(std::format(
                "Expected feature to have only one dimension, got {}\n",
                labels_info.ndim)),
            (labels_info.ndim == 1)));

    size_t size = labels_info.shape[0];
    return new deglib::graph::Filter(ptr, size, max_value, max_label_count);
  });

  // graphs
  py::class_<deglib::search::SearchGraph>(m, "SearchGraph");

  // read only graph
  py::class_<deglib::graph::ReadOnlyGraph, deglib::search::SearchGraph>(
      m, "ReadOnlyGraph")
      .def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
      .def("size", &deglib::graph::ReadOnlyGraph::size)
      .def(
          "get_feature_space",
          [](const deglib::graph::ReadOnlyGraph &g)
              -> const deglib::FloatSpace & { return g.getFeatureSpace(); },
          py::return_value_policy::reference)
      .def(
          "get_feature_vector",
          [](const deglib::graph::ReadOnlyGraph &g,
             const uint32_t internal_idx) {
            const auto metric = g.getFeatureSpace().metric();
            const bool is_uint8 = metric == deglib::Metric::Uint8_L2;
            const bool is_evp = metric == deglib::Metric::EVP_InnerProduct;
            const bool is_fp16 = metric == deglib::Metric::FP16_InnerProduct;
            const char *format_descriptor =
                (is_uint8 || is_evp) ? "B" : (is_fp16 ? "H" : "f");
            const size_t item_size =
                (is_uint8 || is_evp)
                    ? sizeof(uint8_t)
                    : (is_fp16 ? sizeof(uint16_t) : sizeof(float));
            const size_t data_size = g.getFeatureSpace().get_data_size();
            return py::memoryview::from_buffer(
                g.getFeatureVector(internal_idx), item_size, format_descriptor,
                {data_size / item_size}, {item_size});
          },
          py::return_value_policy::reference)
      .def("get_internal_index",
           &deglib::graph::ReadOnlyGraph::getInternalIndex)
      .def("search", &graph_search_single_wrapper<deglib::graph::ReadOnlyGraph>,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0)
      .def("search_batch",
           &graph_search_batch_wrapper<deglib::graph::ReadOnlyGraph>,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0,
           py::arg("threads") = 1)
      .def(
          "explore",
          [](const deglib::graph::ReadOnlyGraph &g, uint32_t entry_vertex_index,
             uint32_t k, bool include_entry,
             uint32_t max_distance_computation_count) {
            return g.explore(entry_vertex_index, k,
                             max_distance_computation_count, 0.0f,
                             include_entry);
          },
          py::arg("entry_vertex_index"), py::arg("k"),
          py::arg("include_entry") = true,
          py::arg("max_distance_computation_count") = 0)
      .def(
          "explore_batch", &graph_explore_wrapper<deglib::graph::ReadOnlyGraph>,
          py::arg("entry_vertex_indices"), py::arg("k"),
          py::arg("include_entry") = true,
          py::arg("max_distance_computation_count") = 0, py::arg("threads") = 1)
      .def("has_path", &deglib::graph::ReadOnlyGraph::hasPath)
      .def("get_entry_vertex_indices",
           &deglib::graph::ReadOnlyGraph::getEntryVertexIndices)
      .def("get_edges_per_vertex",
           &deglib::graph::ReadOnlyGraph::getEdgesPerVertex)
      .def(
          "get_neighbor_indices",
          [](const deglib::graph::ReadOnlyGraph &g,
             const uint32_t internal_idx) {
            return py::memoryview::from_buffer(
                g.getNeighborIndices(internal_idx), sizeof(uint32_t), "I",
                {g.getEdgesPerVertex()}, {sizeof(uint32_t)});
          },
          py::return_value_policy::reference)
      .def("has_vertex", &deglib::graph::ReadOnlyGraph::hasVertex)
      .def("has_edge", &deglib::graph::ReadOnlyGraph::hasEdge)
      .def("get_external_label",
           &deglib::graph::ReadOnlyGraph::getExternalLabel);

  m.def("read_only_graph_from_graph", &read_only_graph_from_search_graph);

  m.def("load_readonly_graph", &deglib::graph::load_readonly_graph);

  // mutable graph
  py::class_<deglib::graph::MutableGraph, deglib::search::SearchGraph>(
      m, "MutableGraph");

  // size bounded graph
  py::class_<deglib::graph::SizeBoundedGraph, deglib::graph::MutableGraph>(
      m, "SizeBoundedGraph")
      .def(py::init<const uint32_t, const uint8_t, const deglib::FloatSpace>())
      .def("size", &deglib::graph::SizeBoundedGraph::size)
      .def(
          "get_feature_space",
          [](const deglib::graph::SizeBoundedGraph &g)
              -> const deglib::FloatSpace & { return g.getFeatureSpace(); },
          py::return_value_policy::reference)
      .def(
          "get_feature_vector",
          [](const deglib::graph::SizeBoundedGraph &g,
             const uint32_t internal_idx) {
            const auto metric = g.getFeatureSpace().metric();
            const bool is_uint8 = metric == deglib::Metric::Uint8_L2;
            const bool is_evp = metric == deglib::Metric::EVP_InnerProduct;
            const bool is_fp16 = metric == deglib::Metric::FP16_InnerProduct;
            const char *format_descriptor =
                (is_uint8 || is_evp) ? "B" : (is_fp16 ? "H" : "f");
            const size_t item_size =
                (is_uint8 || is_evp)
                    ? sizeof(uint8_t)
                    : (is_fp16 ? sizeof(uint16_t) : sizeof(float));
            const size_t data_size = g.getFeatureSpace().get_data_size();
            return py::memoryview::from_buffer(
                g.getFeatureVector(internal_idx), item_size, format_descriptor,
                {data_size / item_size}, {item_size});
          },
          py::return_value_policy::reference)
      .def("get_internal_index",
           &deglib::graph::SizeBoundedGraph::getInternalIndex)
      .def("has_path", &deglib::graph::SizeBoundedGraph::hasPath)
      .def("get_entry_vertex_indices",
           &deglib::graph::SizeBoundedGraph::getEntryVertexIndices)
      .def("get_external_label",
           &deglib::graph::SizeBoundedGraph::getExternalLabel)
      .def("get_edges_per_vertex",
           &deglib::graph::SizeBoundedGraph::getEdgesPerVertex)
      .def("save_graph", &deglib::graph::SizeBoundedGraph::saveGraph)
      .def("add_vertex",
           [](deglib::graph::SizeBoundedGraph &g, const uint32_t external_label,
              const py::array feature_vector) -> uint32_t {
             const py::buffer_info feature_info = feature_vector.request();
             const std::byte *ptr = static_cast<std::byte *>(feature_info.ptr);
             // only allow one dimensional arrays
             assert(
                 (void(std::format(
                      "Expected feature to have only one dimension, got {}\n",
                      feature_info.ndim)),
                  (feature_info.ndim == 1)));
             return g.addVertex(external_label, ptr);
           })
      .def("remove_vertex", &deglib::graph::SizeBoundedGraph::removeVertex)
      .def("change_edge", &deglib::graph::SizeBoundedGraph::changeEdge)
      .def("change_edges",
           [](deglib::graph::SizeBoundedGraph &g, const uint32_t internal_index,
              const py::array_t<uint32_t, py::array::c_style> neighbor_indices,
              const py::array_t<float, py::array::c_style> neighbor_weights) {
             const py::buffer_info neighbor_info = neighbor_indices.request();
             const uint32_t *neighbor_ptr =
                 static_cast<uint32_t *>(neighbor_info.ptr);
             // only allow one dimensional arrays
             assert((void(std::format("Expected neighbor_indices to have only "
                                      "one dimension, got {}\n",
                                      neighbor_info.ndim)),
                     (neighbor_info.ndim == 1)));

             const py::buffer_info weight_info = neighbor_weights.request();
             const float *weight_ptr = static_cast<float *>(weight_info.ptr);
             // only allow one dimensional arrays
             assert((void(std::format("Expected neighbor_weights to have only "
                                      "one dimension, got {}\n",
                                      weight_info.ndim)),
                     (weight_info.ndim == 1)));

             g.changeEdges(internal_index, neighbor_ptr, weight_ptr);
           })
      .def(
          "get_neighbor_weights",
          [](const deglib::graph::SizeBoundedGraph &g,
             const uint32_t internal_idx) {
            return py::memoryview::from_buffer(
                g.getNeighborWeights(internal_idx), sizeof(float), "f",
                {g.getEdgesPerVertex()}, {sizeof(float)});
          },
          py::return_value_policy::reference)
      .def("get_edge_weight", &deglib::graph::SizeBoundedGraph::getEdgeWeight)
      .def(
          "get_neighbor_indices",
          [](const deglib::graph::SizeBoundedGraph &g,
             const uint32_t internal_idx) {
            return py::memoryview::from_buffer(
                g.getNeighborIndices(internal_idx), sizeof(uint32_t), "I",
                {g.getEdgesPerVertex()}, {sizeof(uint32_t)});
          },
          py::return_value_policy::reference)
      .def("has_vertex", &deglib::graph::SizeBoundedGraph::hasVertex)
      .def("has_edge", &deglib::graph::SizeBoundedGraph::hasEdge)
      .def("search",
           &graph_search_single_wrapper<deglib::graph::SizeBoundedGraph>,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0)
      .def("search_batch",
           &graph_search_batch_wrapper<deglib::graph::SizeBoundedGraph>,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0,
           py::arg("threads") = 1)
      .def(
          "explore",
          [](const deglib::graph::SizeBoundedGraph &g,
             uint32_t entry_vertex_index, uint32_t k, bool include_entry,
             uint32_t max_distance_computation_count) {
            return g.explore(entry_vertex_index, k,
                             max_distance_computation_count, 0.0f,
                             include_entry);
          },
          py::arg("entry_vertex_index"), py::arg("k"),
          py::arg("include_entry") = true,
          py::arg("max_distance_computation_count") = 0)
      .def("explore_batch",
           &graph_explore_wrapper<deglib::graph::SizeBoundedGraph>,
           py::arg("entry_vertex_indices"), py::arg("k"),
           py::arg("include_entry") = true,
           py::arg("max_distance_computation_count") = 0,
           py::arg("threads") = 1);

  // repository
  py::class_<deglib::StaticFeatureRepository>(m, "StaticFeatureRepository")
      .def(
          "get_feature",
          [](const deglib::StaticFeatureRepository &fr,
             const uint32_t vertex_id) {
            return py::memoryview::from_buffer(fr.getFeature(vertex_id),
                                               sizeof(float), "f", {fr.dims()},
                                               {sizeof(float)});
          },
          py::return_value_policy::reference)
      .def("size", &deglib::StaticFeatureRepository::size)
      .def("dims", &deglib::StaticFeatureRepository::dims)
      .def("clear", &deglib::StaticFeatureRepository::clear);

  m.def("load_static_repository", &deglib::load_static_repository);

  // random mt19937
  py::class_<std::mt19937>(m, "Mt19937").def(py::init<std::uint_fast32_t>());

  // even regular builder
  py::enum_<deglib::builder::OptimizationTarget>(m, "OptimizationTarget")
      .value("StreamingData",
             deglib::builder::OptimizationTarget::StreamingData)
      .value("HighLID", deglib::builder::OptimizationTarget::HighLID)
      .value("LowLID", deglib::builder::OptimizationTarget::LowLID);

  py::class_<deglib::builder::EvenRegularGraphBuilder>(
      m, "EvenRegularGraphBuilder")
      .def(py::init<deglib::graph::MutableGraph &, std::mt19937 &,
                    const deglib::builder::OptimizationTarget, const uint8_t,
                    const float, const uint8_t, const float, const uint8_t,
                    const uint32_t, const uint32_t>())
      .def("add_entry",
           [](deglib::builder::EvenRegularGraphBuilder &builder,
              const py::array_t<uint32_t, py::array::c_style> &label,
              const py::array &feature) {
             // label buffer
             const auto label_access = label.unchecked<1>();

             // feature buffer
             const py::buffer_info feature_info = feature.request();
             const std::byte *feature_ptr =
                 static_cast<std::byte *>(feature_info.ptr);
             // only allow two dimensional array
             assert((void(std::format(
                         "Expected feature to have two dimensions, got {}\n",
                         feature_info.ndim)),
                     (feature_info.ndim == 2)));

             py::gil_scoped_release release; // release the gil

             // add entries
             const size_t feature_len =
                 feature_info.itemsize * feature_info.shape[1];
             for (uint32_t i = 0; i < feature_info.shape[0]; i++) {
               // copy to vector
               std::vector<std::byte> feature_vec(
                   feature_ptr + (feature_len * i),
                   feature_ptr + (feature_len * (i + 1)));
               const uint32_t current_label = label_access(i);
               builder.addEntry(current_label, std::move(feature_vec));
             }
           })
      .def("remove_entry",
           &deglib::builder::EvenRegularGraphBuilder::removeEntry)
      .def("get_num_new_entries",
           &deglib::builder::EvenRegularGraphBuilder::getNumNewEntries)
      .def("get_num_remove_entries",
           &deglib::builder::EvenRegularGraphBuilder::getNumRemoveEntries)
      .def("set_thread_count",
           &deglib::builder::EvenRegularGraphBuilder::setThreadCount)
      .def("set_batch_size",
           &deglib::builder::EvenRegularGraphBuilder::setBatchSize)
      .def("get_batch_size",
           &deglib::builder::EvenRegularGraphBuilder::getBatchSize)

      .def("build",
           [](deglib::builder::EvenRegularGraphBuilder &builder,
              std::function<void(deglib::builder::BuilderStatus &)> callback,
              const bool infinite) -> deglib::graph::MutableGraph & {
             return builder.build(callback, infinite);
           })
      .def("build_silent",
           [](deglib::builder::EvenRegularGraphBuilder &builder,
              const bool infinite) -> deglib::graph::MutableGraph & {
             py::gil_scoped_release release;
             return builder.build([](deglib::builder::BuilderStatus &) {},
                                  infinite);
           })
      .def("stop", &deglib::builder::EvenRegularGraphBuilder::stop);

  m.def("calc_avg_edge_weight", &deglib::analysis::calc_avg_edge_weight);
  m.def("calc_edge_weight_histogram",
        &deglib::analysis::calc_edge_weight_histogram);
  m.def("check_graph_weights", &deglib::analysis::check_graph_weights);
  m.def("check_graph_regularity", &deglib::analysis::check_graph_regularity);
  m.def("check_graph_connectivity",
        &deglib::analysis::check_graph_connectivity);
  m.def("calc_non_rng_edges", &deglib::analysis::calc_non_rng_edges);
  m.def("remove_non_mrng_edges", &deglib::optimization::remove_non_mrng_edges,
        py::arg("graph"), py::arg("num_threads") = 0);
  m.def("prune_worst_edges", &deglib::optimization::pruning::prune_worst_edges,
        py::arg("graph"), py::arg("prune_worst"), py::arg("num_threads") = 0);

  py::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
      .def_readwrite("step", &deglib::builder::BuilderStatus::step)
      .def_readwrite("added", &deglib::builder::BuilderStatus::added)
      .def_readwrite("deleted", &deglib::builder::BuilderStatus::deleted)
      .def_readwrite("improved", &deglib::builder::BuilderStatus::improved)
      .def_readwrite("tries", &deglib::builder::BuilderStatus::tries);
}
