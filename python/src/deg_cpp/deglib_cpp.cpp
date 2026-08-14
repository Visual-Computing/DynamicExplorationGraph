//
// Created by Bruno Schilling on 29.05.24.
//

// #define PYBIND11_DETAILED_ERROR_MESSAGES
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <deglib/deglib.h>
#include <deglib/distance/fp16.h>
#include <deglib/optimization.h>
#include <deglib/optimization/pruning.h>
#include <deglib/optimization/quantization/evp_quantize.h>
#include <deglib/optimization/transform.h>

namespace py = pybind11;

template <typename G>
std::tuple<py::array_t<uint32_t>, py::array_t<float>>
graph_search_batch_wrapper(const G &graph, const py::array query,
                           const float eps, const uint32_t k,
                           const deglib::search::Filter *filter,
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
            graph.getExternalLabel(next_result.getIdentifier());
        result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
        result.pop();
      }
    }
  };

  if (threads <= 1) {
    search_range(0, n_queries);
  } else {
    const size_t chunk_size = std::clamp(
        (n_queries + static_cast<size_t>(threads) * 8 - 1) / (static_cast<size_t>(threads) * 8),
        size_t{1}, size_t{8196});
    const size_t num_chunks = (n_queries + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, threads,
        [&](size_t chunk_id, size_t) {
          size_t start = chunk_id * chunk_size;
          size_t end = std::min(start + chunk_size, static_cast<size_t>(n_queries));
          search_range(start, end);
        });
  }

  return std::make_tuple(result_indices, result_distances);
}

template <typename G>
py::object
graph_search_single_wrapper(const G &graph, const py::array query,
                            const float eps, const uint32_t k,
                            const deglib::search::Filter *filter,
                            const uint32_t max_distance_computation_count,
                            const bool return_distances = true,
                            const bool unsorted = false) {
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

  deglib::search::ResultSet result =
      graph.search(query_span, k, eps, filter,
                   max_distance_computation_count);

  while (result.size() > k) {
    result.pop();
  }

  const size_t count = result.size();
  py::array_t<uint32_t> result_indices({static_cast<py::ssize_t>(count)});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t *>(result_indices_info.ptr);

  py::array_t<float> result_distances;
  float *result_distances_ptr = nullptr;
  if (return_distances) {
    result_distances = py::array_t<float>({static_cast<py::ssize_t>(count)});
    result_distances_ptr = static_cast<float *>(result_distances.request().ptr);
  }

  if (unsorted) {
    for (size_t i = 0; i < count; ++i) {
      if constexpr (std::is_same_v<G, deglib::DynamicExplorationGraph>) {
        result_indices_ptr[i] = result[i].getIdentifier();
      } else {
        result_indices_ptr[i] = graph.getExternalLabel(result[i].getIdentifier());
      }
      if (result_distances_ptr) {
        result_distances_ptr[i] = result[i].getDistance();
      }
    }
  } else {
    for (size_t i = count; i > 0; --i) {
      deglib::search::ObjectDistance next_result = result.top();
      if constexpr (std::is_same_v<G, deglib::DynamicExplorationGraph>) {
        result_indices_ptr[i - 1] = next_result.getIdentifier();
      } else {
        result_indices_ptr[i - 1] = graph.getExternalLabel(next_result.getIdentifier());
      }
      if (result_distances_ptr) {
        result_distances_ptr[i - 1] = next_result.getDistance();
      }
      result.pop();
    }
  }

  if (return_distances) {
    return py::make_tuple(result_indices, result_distances);
  } else {
    return result_indices;
  }
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
          entry_idx, k, include_entry, max_distance_computation_count);

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
            graph.getExternalLabel(next_result.getIdentifier());
        result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
        result.pop();
      }
    }
  };

  if (threads <= 1) {
    explore_range(0, n_queries);
  } else {
    const size_t chunk_size = std::clamp(
        (n_queries + static_cast<size_t>(threads) * 8 - 1) / (static_cast<size_t>(threads) * 8),
        size_t{1}, size_t{8196});
    const size_t num_chunks = (n_queries + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, threads,
        [&](size_t chunk_id, size_t) {
          size_t start = chunk_id * chunk_size;
          size_t end = std::min(start + chunk_size, static_cast<size_t>(n_queries));
          explore_range(start, end);
        });
  }

  return std::make_tuple(result_indices, result_distances);
}

// Batch search wrapper for DynamicExplorationGraph — search() already returns external labels
py::object
dynamic_exploration_graph_search_batch_wrapper(const deglib::DynamicExplorationGraph &graph,
                                               const py::array query,
                                               const float eps, const uint32_t k,
                                               const deglib::search::Filter *filter,
                                               const uint32_t max_distance_computation_count,
                                               const uint32_t threads,
                                               const bool return_distances = true,
                                               const bool unsorted = false) {
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
  std::fill_n(result_indices_ptr, n_queries * k, std::numeric_limits<uint32_t>::max());

  py::array_t<float> result_distances;
  float *result_distances_ptr = nullptr;
  if (return_distances) {
    result_distances = py::array_t<float>({n_queries, k});
    result_distances_ptr = static_cast<float *>(result_distances.request().ptr);
    std::fill_n(result_distances_ptr, n_queries * k, std::numeric_limits<float>::quiet_NaN());
  }

  {
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

        if (unsorted) {
          for (size_t idx = 0; idx < count; ++idx) {
            result_indices_ptr[base_offset + idx] = result[idx].getIdentifier();
            if (result_distances_ptr) {
              result_distances_ptr[base_offset + idx] = result[idx].getDistance();
            }
          }
        } else {
          // Extract elements from heap in descending order (furthest to closest)
          for (size_t i = count; i > 0; --i) {
            deglib::search::ObjectDistance next_result = result.top();
            result_indices_ptr[base_offset + i - 1] = next_result.getIdentifier();
            if (result_distances_ptr) {
              result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
            }
            result.pop();
          }
        }
      }
    };

    if (threads <= 1) {
      search_range(0, n_queries);
    } else {
      const size_t chunk_size = std::clamp(
          (n_queries + static_cast<size_t>(threads) * 8 - 1) / (static_cast<size_t>(threads) * 8),
          size_t{1}, size_t{8196});
      const size_t num_chunks = (n_queries + chunk_size - 1) / chunk_size;

      deglib::concurrent::parallel_for(
          static_cast<size_t>(0), num_chunks, threads,
          [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, static_cast<size_t>(n_queries));
            search_range(start, end);
          });
    }
  }

  if (return_distances) {
    return py::make_tuple(result_indices, result_distances);
  } else {
    return result_indices;
  }
}

template <typename G>
py::object graph_explore_single_wrapper(
    const G &graph,
    const uint32_t entry_external_label,
    const uint32_t k,
    const uint32_t max_distance_computation_count = 0,
    const float eps = 0.0f,
    const bool include_entry = true,
    const deglib::search::Filter *filter = nullptr,
    const bool return_distances = true,
    const bool unsorted = false) {
  deglib::search::ResultSet result = graph.explore(
      entry_external_label, k, max_distance_computation_count, eps, include_entry, filter);

  while (result.size() > k) {
    result.pop();
  }

  const size_t count = result.size();
  py::array_t<uint32_t> result_indices({static_cast<py::ssize_t>(count)});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t *>(result_indices_info.ptr);

  py::array_t<float> result_distances;
  float *result_distances_ptr = nullptr;
  if (return_distances) {
    result_distances = py::array_t<float>({static_cast<py::ssize_t>(count)});
    result_distances_ptr = static_cast<float *>(result_distances.request().ptr);
  }

  if (unsorted) {
    for (size_t i = 0; i < count; ++i) {
      result_indices_ptr[i] = result[i].getIdentifier();
      if (result_distances_ptr) {
        result_distances_ptr[i] = result[i].getDistance();
      }
    }
  } else {
    for (size_t i = count; i > 0; --i) {
      deglib::search::ObjectDistance next_result = result.top();
      result_indices_ptr[i - 1] = next_result.getIdentifier();
      if (result_distances_ptr) {
        result_distances_ptr[i - 1] = next_result.getDistance();
      }
      result.pop();
    }
  }

  if (return_distances) {
    return py::make_tuple(result_indices, result_distances);
  } else {
    return result_indices;
  }
}

template <typename G>
std::tuple<py::array_t<uint32_t>, py::array_t<float>>
graph_has_path_wrapper(const G &graph,
                       const std::vector<uint32_t> &entry_vertex_indices,
                       const uint32_t to_vertex, const float eps,
                       const uint32_t k) {
  std::vector<deglib::search::ObjectDistance> path =
      graph.hasPath(entry_vertex_indices, to_vertex, eps, k);

  const size_t count = path.size();
  py::array_t<uint32_t> result_indices({static_cast<py::ssize_t>(count)});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr =
      static_cast<uint32_t *>(result_indices_info.ptr);

  py::array_t<float> result_distances({static_cast<py::ssize_t>(count)});
  py::buffer_info result_distances_info = result_distances.request();
  auto result_distances_ptr =
      static_cast<float *>(result_distances_info.ptr);

  for (size_t i = 0; i < count; ++i) {
    result_indices_ptr[i] =
        graph.getExternalLabel(path[i].getIdentifier());
    result_distances_ptr[i] = path[i].getDistance();
  }

  return std::make_tuple(result_indices, result_distances);
}
py::object
dynamic_exploration_graph_explore_batch_wrapper(
   const deglib::DynamicExplorationGraph &graph,
   const py::array_t<uint32_t, py::array::c_style> entry_external_labels,
   const uint32_t k, const uint32_t max_distance_computation_count,
   const float eps, const bool include_entry,
   const deglib::search::Filter *filter, const uint32_t threads,
   const bool return_distances = true, const bool unsorted = false) {
  py::buffer_info entry_info = entry_external_labels.request();
  if (entry_info.ndim != 1) {
    throw std::invalid_argument(
        std::format("Expected entry_external_labels to have 1 dimension, got {}",
                    entry_info.ndim));
  }

  const uint32_t *entry_ptr = static_cast<const uint32_t *>(entry_info.ptr);
  const uint32_t n_queries = entry_info.shape[0];

  py::array_t<uint32_t> result_indices({n_queries, k});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr = static_cast<uint32_t *>(result_indices_info.ptr);
  std::fill_n(result_indices_ptr, n_queries * k, std::numeric_limits<uint32_t>::max());

  py::array_t<float> result_distances;
  float *result_distances_ptr = nullptr;
  if (return_distances) {
    result_distances = py::array_t<float>({n_queries, k});
    result_distances_ptr = static_cast<float *>(result_distances.request().ptr);
    std::fill_n(result_distances_ptr, n_queries * k, std::numeric_limits<float>::quiet_NaN());
  }

  {
    py::gil_scoped_release release; // release the gil

    auto explore_range = [&](size_t begin, size_t end) {
      for (size_t q = begin; q < end; ++q) {
        uint32_t entry_label = entry_ptr[q];
        deglib::search::ResultSet result = graph.explore(
            entry_label, k, max_distance_computation_count, eps, include_entry, filter);

        // Limit results to at most k elements
        while (result.size() > k) {
          result.pop();
        }

        const size_t count = result.size();
        const size_t base_offset = q * k;

        if (unsorted) {
          for (size_t idx = 0; idx < count; ++idx) {
            result_indices_ptr[base_offset + idx] = result[idx].getIdentifier();
            if (result_distances_ptr) {
              result_distances_ptr[base_offset + idx] = result[idx].getDistance();
            }
          }
        } else {
          // Extract elements from heap in descending order (furthest to closest)
          for (size_t i = count; i > 0; --i) {
            deglib::search::ObjectDistance next_result = result.top();
            result_indices_ptr[base_offset + i - 1] = next_result.getIdentifier();
            if (result_distances_ptr) {
              result_distances_ptr[base_offset + i - 1] = next_result.getDistance();
            }
            result.pop();
          }
        }
      }
    };

    if (threads <= 1) {
      explore_range(0, n_queries);
    } else {
      const size_t chunk_size = std::clamp(
          (n_queries + static_cast<size_t>(threads) * 8 - 1) / (static_cast<size_t>(threads) * 8),
          size_t{1}, size_t{8196});
      const size_t num_chunks = (n_queries + chunk_size - 1) / chunk_size;

      deglib::concurrent::parallel_for(
          static_cast<size_t>(0), num_chunks, threads,
          [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, static_cast<size_t>(n_queries));
            explore_range(start, end);
          });
    }
  }

  if (return_distances) {
    return py::make_tuple(result_indices, result_distances);
  } else {
    return result_indices;
  }
}

std::tuple<py::array_t<uint32_t>, py::array_t<float>>
dynamic_exploration_graph_explore_single_wrapper(
    const deglib::DynamicExplorationGraph &graph,
    const uint32_t entry_external_label,
    const uint32_t k,
    const uint32_t max_distance_computation_count, const float eps,
    const bool include_entry, const deglib::search::Filter *filter) {
  deglib::search::ResultSet result = graph.explore(
      entry_external_label, k, max_distance_computation_count, eps,
      include_entry, filter);

  py::array_t<uint32_t> result_indices({static_cast<py::ssize_t>(k)});
  py::buffer_info result_indices_info = result_indices.request();
  auto result_indices_ptr =
      static_cast<uint32_t *>(result_indices_info.ptr);
  std::fill_n(result_indices_ptr, k, 0);

  py::array_t<float> result_distances({static_cast<py::ssize_t>(k)});
  py::buffer_info result_distances_info = result_distances.request();
  auto result_distances_ptr =
      static_cast<float *>(result_distances_info.ptr);
  std::fill_n(result_distances_ptr, k,
              std::numeric_limits<float>::quiet_NaN());

  while (result.size() > k) {
    result.pop();
  }

  const size_t count = result.size();
  for (size_t i = count; i > 0; --i) {
    deglib::search::ObjectDistance next_result = result.top();
    result_indices_ptr[i - 1] = next_result.getIdentifier();
    result_distances_ptr[i - 1] = next_result.getDistance();
    result.pop();
  }

  return std::make_tuple(result_indices, result_distances);
}


float float_space_compute_distance(const deglib::distances::FloatSpace &space,
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
float_space_compute_distances(const deglib::distances::FloatSpace &space, py::array query,
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

py::object
float_space_rerank(const deglib::distances::FloatSpace &space, py::array queries,
                   py::array_t<uint32_t> candidate_indices,
                   py::object base_vectors = py::none(), size_t k_top = 0,
                   size_t num_threads = 0, bool return_distances = false,
                   bool unsorted = false) {
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

  auto results = deglib::search::rerank(
      space, q_buf.ptr, n_queries, base_ptr, num_base_vectors,
      static_cast<const uint32_t *>(cand_buf.ptr), evp_k, k_top, num_threads);

  auto result_indices = py::array_t<uint32_t>({n_queries, k_top});
  auto res_indices_buf = result_indices.request();
  uint32_t *ind_ptr = static_cast<uint32_t *>(res_indices_buf.ptr);

  py::array_t<float> result_distances;
  float *dist_ptr = nullptr;
  if (return_distances) {
    result_distances = py::array_t<float>({n_queries, k_top});
    dist_ptr = static_cast<float *>(result_distances.request().ptr);
  }

  // search query has producted a ResultSet (a heap)
  for (size_t i = 0; i < n_queries; ++i) {
    auto &res_set = results[i];
    if (!unsorted) {
      res_set.sort();
    }
    size_t actual_k = res_set.size();
    uint32_t *row_ind = ind_ptr + i * k_top;
    float *row_dist = dist_ptr ? (dist_ptr + i * k_top) : nullptr;

    for (size_t k = 0; k < actual_k; ++k) {
      row_ind[k] = res_set[k].getIdentifier();
      if (row_dist) {
        row_dist[k] = res_set[k].getDistance();
      }
    }
    for (size_t k = actual_k; k < k_top; ++k) {
      row_ind[k] = std::numeric_limits<uint32_t>::max();
      if (row_dist) {
        row_dist[k] = std::numeric_limits<float>::max();
      }
    }
  }

  if (return_distances) {
    return py::make_tuple(result_indices, result_distances);
  }
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

deglib::DynamicExplorationGraph load_readonly_graph_wrapper(const char* path) {
  auto graph = deglib::graph::load_readonly_graph(path);
  auto* heap_graph = new deglib::graph::ReadOnlyGraph(std::move(graph));
  return deglib::DynamicExplorationGraph(*heap_graph);
}

std::tuple<py::array_t<float>, float>
mips_l2_transform_wrapper(py::array_t<float, py::array::c_style> vectors) {
  py::buffer_info buf = vectors.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D float32 array");
  }
  size_t count = buf.shape[0];
  size_t dim = buf.shape[1];

  py::array_t<float> output({count, dim + 1});
  py::buffer_info out_buf = output.request();

  float max_norm = 0.0f;
  {
    py::gil_scoped_release release;
    max_norm = deglib::optimization::mips_l2_transform(
        static_cast<const float*>(buf.ptr), count, dim,
        static_cast<float*>(out_buf.ptr)
    );
  }
  return std::make_tuple(output, max_norm);
}

py::array_t<float>
mips_l2_transform_query_wrapper(py::array_t<float, py::array::c_style> queries) {
  py::buffer_info buf = queries.request();
  if (buf.ndim != 1 && buf.ndim != 2) {
    throw std::invalid_argument("queries must be a 1D or 2D float32 array");
  }
  bool is_1d = (buf.ndim == 1);
  size_t count = is_1d ? 1 : buf.shape[0];
  size_t dim = is_1d ? buf.shape[0] : buf.shape[1];

  py::array_t<float> output = is_1d ? py::array_t<float>(dim + 1) : py::array_t<float>({count, dim + 1});
  py::buffer_info out_buf = output.request();

  {
    py::gil_scoped_release release;
    deglib::optimization::mips_l2_transform_query(
        static_cast<const float*>(buf.ptr), count, dim,
        static_cast<float*>(out_buf.ptr)
    );
  }
  return output;
}

py::array_t<uint32_t>
presort_wrapper(py::array_t<float, py::array::c_style> vectors,
                const deglib::distances::FloatSpace &space,
                float radius_decay = 0.9f,
                size_t threads = 0,
                py::object callback = py::none()) {
  py::buffer_info buf = vectors.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("vectors must be a 2D float32 array");
  }
  size_t count = buf.shape[0];
  size_t dim = buf.shape[1];
  if (count > 0 && dim != space.dim()) {
    throw std::invalid_argument(std::format("Vector dimension ({}) does not match FloatSpace dimension ({})", dim, space.dim()));
  }

  std::function<bool(float)> cb = nullptr;
  if (!callback.is_none()) {
    cb = [callback](float progress) -> bool {
      py::gil_scoped_acquire acquire;
      try {
        py::object res = callback(progress);
        if (!res.is_none() && py::isinstance<py::bool_>(res)) {
          return res.cast<bool>();
        }
      } catch (const std::exception& e) {
        // Ignore callback exceptions during sorting loop
      }
      return false;
    };
  }

  std::vector<uint32_t> sorted_indices;
  {
    py::gil_scoped_release release;
    sorted_indices = deglib::optimization::presort(
        static_cast<const float*>(buf.ptr), count, space,
        radius_decay, threads, cb
    );
  }

  py::array_t<uint32_t> output(count);
  py::buffer_info out_buf = output.request();
  std::memcpy(out_buf.ptr, sorted_indices.data(), count * sizeof(uint32_t));
  return output;
}

deglib::DynamicExplorationGraph read_only_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt) {

  const void* feat_ptr = nullptr;
  py::buffer_info feat_info;

  if (custom_features.has_value() && !custom_features->is_none()) {
    feat_info = custom_features->request();
    feat_ptr = feat_info.ptr;
    if (custom_feature_space.has_value()) {
      size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
      size_t actual_bytes = feat_info.size * feat_info.itemsize;
      if (actual_bytes != expected_bytes) {
        throw std::invalid_argument(std::format(
            "custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)",
            actual_bytes, expected_bytes));
      }
    }
  }

  deglib::graph::ReadOnlyGraph read_only = custom_feature_space.has_value()
      ? deglib::graph::convert_to_readonly_graph(graph.internal(), custom_feature_space.value(), feat_ptr)
      : deglib::graph::convert_to_readonly_graph(graph.internal());

  auto* heap_graph = new deglib::graph::ReadOnlyGraph(std::move(read_only));
  return deglib::DynamicExplorationGraph(*heap_graph);
}

deglib::DynamicExplorationGraph create_size_bounded_graph(
    const uint32_t max_vertex_count, const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace &feature_space) {
  auto* graph = new deglib::graph::SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space);
  return deglib::DynamicExplorationGraph(*graph);
}

// ============================================================================
// Analysis Wrapper Functions
// ============================================================================

float calc_avg_edge_weight_wrapper(const deglib::DynamicExplorationGraph &graph, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_avg_edge_weight");
    }
    return deglib::analysis::calc_avg_edge_weight(static_cast<const deglib::graph::MutableGraph &>(graph.internal()), scale);
}

std::vector<float> calc_edge_weight_histogram_wrapper(const deglib::DynamicExplorationGraph &graph, const bool sorted, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_edge_weight_histogram");
    }
    return deglib::analysis::calc_edge_weight_histogram(static_cast<const deglib::graph::MutableGraph &>(graph.internal()), sorted, scale);
}

bool check_graph_weights_wrapper(const deglib::DynamicExplorationGraph &graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for check_graph_weights");
    }
    return deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph &>(graph.internal()));
}

uint32_t calc_non_rng_edges_wrapper(const deglib::DynamicExplorationGraph &graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_non_rng_edges");
    }
    return deglib::analysis::calc_non_rng_edges(static_cast<const deglib::graph::MutableGraph &>(graph.internal()));
}

bool check_graph_regularity_wrapper(const deglib::DynamicExplorationGraph &graph, const uint32_t expected_vertices, const bool check_back_link) {
    return deglib::analysis::check_graph_regularity(graph.internal(), expected_vertices, check_back_link);
}

bool check_graph_connectivity_wrapper(const deglib::DynamicExplorationGraph &graph) {
    return deglib::analysis::check_graph_connectivity(graph.internal());
}

uint32_t calc_search_reachability_wrapper(const deglib::DynamicExplorationGraph &graph) {
    return deglib::analysis::calc_search_reachability(graph.internal());
}

float calc_exploration_reach_wrapper(const deglib::DynamicExplorationGraph &graph) {
    return deglib::analysis::calc_exploration_reach(graph.internal());
}

deglib::analysis::GraphStats analyze_graph_wrapper(const deglib::DynamicExplorationGraph &graph) {
    return deglib::analysis::analyze_graph(graph.internal());
}

// ============================================================================

PYBIND11_MODULE(deglib_cpp, m) {
  m.doc() = "Python bindings for Dynamic Exploration Graph";

  m.def("avx_usable", &deglib::cpu::has_avx2,
        "Returns whether AVX2 instructions are available");
  m.def("avx512_usable", &deglib::cpu::has_avx512,
        "Returns whether AVX512 instructions are available");

  // cpu submodule
  py::module_ cpu_module = m.def_submodule("cpu", "CPU feature detection and InstructionSet");
  py::enum_<deglib::cpu::InstructionSet>(cpu_module, "InstructionSet")
      .value("Auto", deglib::cpu::InstructionSet::Auto)
      .value("Scalar", deglib::cpu::InstructionSet::Scalar)
      .value("AVX2", deglib::cpu::InstructionSet::AVX2)
      .value("AVX512", deglib::cpu::InstructionSet::AVX512);

  cpu_module.def("has_avx2", &deglib::cpu::has_avx2,
        "Returns whether AVX2 instructions are available");
  cpu_module.def("has_avx512", &deglib::cpu::has_avx512,
        "Returns whether AVX512 instructions are available");

  // distances submodule
  py::module_ distances_module = m.def_submodule("distances", "Distance metrics and feature spaces");
  py::enum_<deglib::distances::MetricType>(distances_module, "Metric")
      .value("FP32_L2", deglib::distances::MetricType::FP32_L2)
      .value("FP32_InnerProduct", deglib::distances::MetricType::FP32_InnerProduct)
      .value("Uint8_L2", deglib::distances::MetricType::Uint8_L2)
      .value("FP16_InnerProduct", deglib::distances::MetricType::FP16_InnerProduct)
      .value("EVP_InnerProduct", deglib::distances::MetricType::EVP_InnerProduct);

  py::class_<deglib::distances::FloatSpace>(distances_module, "FloatSpace")
      .def(py::init([](const size_t dim, const deglib::distances::MetricType metric,
                       const deglib::cpu::InstructionSet instruction) {
              return deglib::distances::FloatSpace(dim, deglib::distances::Metric(metric),
                                        instruction);
            }),
            py::arg("dim"), py::arg("metric") = deglib::distances::MetricType::FP32_L2,
            py::arg("instruction") = deglib::cpu::InstructionSet::Auto)
      .def("dim", &deglib::distances::FloatSpace::dim)
      .def("metric",
           [](const deglib::distances::FloatSpace &fs) { return fs.metric().value; })
      .def("get_data_size", &deglib::distances::FloatSpace::get_data_size)
      .def("get_instruction", &deglib::distances::FloatSpace::get_instruction)
      .def("compute_distance", &float_space_compute_distance, py::arg("vec1"),
           py::arg("vec2"))
      .def("compute_distances", &float_space_compute_distances,
           py::arg("query"), py::arg("targets"))
      .def("rerank", &float_space_rerank, py::arg("queries"),
           py::arg("candidate_indices"), py::arg("base_vectors") = py::none(),
           py::arg("k_top") = 0, py::arg("num_threads") = 0,
           py::arg("return_distances") = true,
           py::arg("unsorted") = false);

  // quantization functions in distances submodule
  distances_module.def("quantize_batch", &quantize_batch_wrapper,
        "Quantize float32 or float16/uint16 vectors to byte-packed EVP format");
  distances_module.def("floats_to_fp16", &floats_to_fp16_wrapper,
        "Convert float32 array to FP16 (uint16_t)");
  distances_module.def("fp16_to_floats", &fp16_to_floats_wrapper,
        "Convert FP16 (uint16_t) array to float32");

  // search submodule
  py::module_ search_module = m.def_submodule("search", "Search utilities including Filter");
  py::class_<deglib::search::Filter>(search_module, "Filter")
      .def(py::init<const int *, size_t, size_t, size_t>());

  search_module.def("create_filter", [](py::array_t<int, py::array::c_style> valid_labels,
                            size_t max_value, size_t max_label_count) {
    const py::buffer_info labels_info = valid_labels.request();
    const int *ptr = static_cast<int *>(labels_info.ptr);
    if (labels_info.ndim != 1) {
      throw std::invalid_argument(std::format(
          "Expected feature to have only one dimension, got {}",
          labels_info.ndim));
    }

    size_t size = labels_info.shape[0];
    return new deglib::search::Filter(ptr, size, max_value, max_label_count);
  });

  // graphs
  py::class_<deglib::DynamicExplorationGraph>(m, "DynamicExplorationGraph")
      .def(py::init<deglib::graph::InternalGraph &>())
      .def("size", &deglib::DynamicExplorationGraph::size)
      .def("get_edges_per_vertex", &deglib::DynamicExplorationGraph::getEdgesPerVertex)
      .def("get_feature_space",
           [](const deglib::DynamicExplorationGraph &g)
               -> const deglib::distances::FloatSpace & { return g.getFeatureSpace(); },
           py::return_value_policy::reference)
      .def("has_vertex", &deglib::DynamicExplorationGraph::hasVertex)
      .def("get_neighbors", &deglib::DynamicExplorationGraph::getNeighbors)
      .def("is_mutable", &deglib::DynamicExplorationGraph::isMutable)
      .def("to_readonly", &deglib::DynamicExplorationGraph::to_readonly)
      .def("search", &graph_search_single_wrapper<deglib::DynamicExplorationGraph>,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0,
           py::arg("return_distances") = true,
           py::arg("unsorted") = false)
      .def("search_batch",
           &dynamic_exploration_graph_search_batch_wrapper,
           py::arg("query"), py::arg("eps"), py::arg("k"),
           py::arg("filter") = nullptr,
           py::arg("max_distance_computation_count") = 0,
           py::arg("threads") = 1,
           py::arg("return_distances") = true,
           py::arg("unsorted") = false)
      .def("explore",
           &graph_explore_single_wrapper<deglib::DynamicExplorationGraph>,
           py::arg("entry_external_label"), py::arg("k"),
           py::arg("max_distance_computation_count") = 0,
           py::arg("eps") = 0.0f, py::arg("include_entry") = true,
           py::arg("filter") = nullptr,
           py::arg("return_distances") = true,
           py::arg("unsorted") = false)
        .def("explore_batch",
             &dynamic_exploration_graph_explore_batch_wrapper,
             py::arg("entry_external_labels"), py::arg("k"),
             py::arg("max_distance_computation_count") = 0,
             py::arg("eps") = 0.0f, py::arg("include_entry") = true,
             py::arg("filter") = nullptr, py::arg("threads") = 1,
             py::arg("return_distances") = true,
             py::arg("unsorted") = false)
      .def("save_graph", [](deglib::DynamicExplorationGraph &g, const char* path) {
          g.saveGraph(path);
      }, py::arg("path"));

  m.def("load_readonly_graph", &load_readonly_graph_wrapper);
  m.def("read_only_graph_from_graph", &read_only_graph_from_graph_wrapper,
        py::arg("graph"), py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none());
  m.def("presort", &presort_wrapper,
        py::arg("vectors"), py::arg("space"),
        py::arg("radius_decay") = 0.9f, py::arg("threads") = 0,
        py::arg("callback") = py::none());
  m.def("mips_l2_transform", &mips_l2_transform_wrapper, py::arg("vectors"));
  m.def("mips_l2_transform_query", &mips_l2_transform_query_wrapper, py::arg("queries"));

  m.def("create_size_bounded_graph", &create_size_bounded_graph,
        py::arg("max_vertex_count"), py::arg("edges_per_vertex"),
        py::arg("feature_space"));

  // GraphBuilder (renamed from EvenRegularGraphBuilder)
  py::enum_<deglib::builder::OptimizationTarget>(m, "OptimizationTarget")
      .value("StreamingData",
             deglib::builder::OptimizationTarget::StreamingData)
      .value("HighLID", deglib::builder::OptimizationTarget::HighLID)
      .value("LowLID", deglib::builder::OptimizationTarget::LowLID);

  py::class_<deglib::builder::EvenRegularGraphBuilder>(
      m, "GraphBuilder")
      .def(py::init([](deglib::DynamicExplorationGraph &graph, std::optional<uint32_t> seed,
                       const deglib::builder::OptimizationTarget optimization_target,
                       const uint8_t extend_k, const float extend_eps,
                       const uint8_t improve_k, const float improve_eps,
                       const uint8_t max_path_length, const uint32_t swap_tries, const uint32_t additional_swap_tries) {
          if (!graph.isMutable()) {
              throw std::runtime_error("Graph must be mutable to use GraphBuilder");
          }
          uint32_t seed_val = seed.value_or(5489);
          auto* rng = new std::mt19937(seed_val);
          return new deglib::builder::EvenRegularGraphBuilder(
              static_cast<deglib::graph::MutableGraph &>(graph.internal()), *rng, optimization_target,
              extend_k, extend_eps, improve_k, improve_eps, max_path_length, swap_tries, additional_swap_tries);
      }))
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
             if (feature_info.ndim != 2) {
               throw std::invalid_argument(std::format(
                   "Expected feature to have two dimensions, got {}",
                   feature_info.ndim));
             }

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
              const bool infinite) {
             builder.build(callback, infinite);
           })
      .def("build_silent",
           [](deglib::builder::EvenRegularGraphBuilder &builder,
              const bool infinite) {
             py::gil_scoped_release release;
             builder.build([](deglib::builder::BuilderStatus &) {},
                                  infinite);
           })
      .def("stop", &deglib::builder::EvenRegularGraphBuilder::stop);
  m.def("calc_avg_edge_weight", &calc_avg_edge_weight_wrapper, py::arg("graph"), py::arg("scale") = 1);
  m.def("calc_edge_weight_histogram", &calc_edge_weight_histogram_wrapper, py::arg("graph"), py::arg("sort"), py::arg("scale") = 1);
  m.def("check_graph_weights", &check_graph_weights_wrapper, py::arg("graph"));
  m.def("check_graph_regularity", &check_graph_regularity_wrapper, py::arg("graph"), py::arg("expected_vertices"), py::arg("check_back_link") = false);
  m.def("check_graph_connectivity", &check_graph_connectivity_wrapper, py::arg("graph"));
  m.def("calc_non_rng_edges", &calc_non_rng_edges_wrapper, py::arg("graph"));
  m.def("calc_search_reachability", &calc_search_reachability_wrapper, py::arg("graph"));
  m.def("calc_exploration_reach", &calc_exploration_reach_wrapper, py::arg("graph"));
 py::class_<deglib::analysis::GraphStats>(m, "GraphStats")
     .def_readonly("vertex_count", &deglib::analysis::GraphStats::vertex_count)
     .def_readonly("edge_count", &deglib::analysis::GraphStats::edge_count)
     .def_readonly("feature_dims", &deglib::analysis::GraphStats::feature_dims)
     .def_readonly("edges_per_vertex", &deglib::analysis::GraphStats::edges_per_vertex)
     .def_readonly("avg_out_degree", &deglib::analysis::GraphStats::avg_out_degree)
     .def_readonly("min_out_degree", &deglib::analysis::GraphStats::min_out_degree)
     .def_readonly("max_out_degree", &deglib::analysis::GraphStats::max_out_degree)
     .def_readonly("avg_in_degree", &deglib::analysis::GraphStats::avg_in_degree)
     .def_readonly("min_in_degree", &deglib::analysis::GraphStats::min_in_degree)
     .def_readonly("max_in_degree", &deglib::analysis::GraphStats::max_in_degree)
     .def_readonly("source_vertices", &deglib::analysis::GraphStats::source_vertices)
     .def_readonly("search_reachability", &deglib::analysis::GraphStats::search_reachability)
     .def_readonly("exploration_reachability", &deglib::analysis::GraphStats::exploration_reachability)
     .def_readonly("memory_bytes", &deglib::analysis::GraphStats::memory_bytes);
 m.def("analyze_graph", &analyze_graph_wrapper, py::arg("graph"));
  m.def("prune_non_mrng_edges", [](deglib::DynamicExplorationGraph &graph, const size_t num_threads) {
      if (!graph.isMutable()) {
          throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
      }
      deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph &>(graph.internal()), num_threads);
  }, py::arg("graph"), py::arg("num_threads") = 0);
  m.def("prune_worst_edges", [](deglib::DynamicExplorationGraph &graph, const uint8_t prune_worst, const size_t num_threads) {
      if (!graph.isMutable()) {
          throw std::runtime_error("Graph must be mutable to prune worst edges");
      }
      deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph &>(graph.internal()), prune_worst, num_threads);
  }, py::arg("graph"), py::arg("prune_worst"), py::arg("num_threads") = 0);

  py::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
      .def_readwrite("step", &deglib::builder::BuilderStatus::step)
      .def_readwrite("added", &deglib::builder::BuilderStatus::added)
      .def_readwrite("deleted", &deglib::builder::BuilderStatus::deleted)
      .def_readwrite("improved", &deglib::builder::BuilderStatus::improved)
      .def_readwrite("tries", &deglib::builder::BuilderStatus::tries);
}

