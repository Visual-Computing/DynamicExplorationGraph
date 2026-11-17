//
// Created by Bruno Schilling on 29.05.24.
//

#include <deglib/deglib.h>
#include <deglib/concurrent.h>
#include <deglib/distance/fp16.h>
#include <deglib/optimization.h>
#include <deglib/optimization/pruning.h>
#include <deglib/transformation.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

template <typename G>
py::object graph_search_single_wrapper(
    const G& graph,
    const py::array query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const bool return_distances = true,
    const bool unsorted = false
) {
    auto c_query = py::array::ensure(query, py::array::c_style);
    if (!c_query) {
        throw std::invalid_argument("search: query must be convertible to a C-contiguous array");
    }
    py::buffer_info query_info = c_query.request();
    if constexpr (std::is_same_v<G, deglib::DynamicExplorationGraph>) {
        if (graph.hasTransformations()) {
            const size_t in_dim = graph.getInputDim();
            if (query_info.size != in_dim) {
                throw std::invalid_argument(std::format("Query vector size ({}) does not match transformations input dim ({})", query_info.size, in_dim));
            }
            const bool is_fp16 = (query_info.format == "e" || (query_info.itemsize == 2 && query_info.format != "h" && query_info.format != "H"));
            auto perform_search = [&](auto q_span) -> py::object {
                py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(k));
                auto result_indices_ptr = result_indices.mutable_data();

                py::array_t<float> result_distances;
                float* result_distances_ptr = nullptr;
                if (return_distances) {
                    result_distances = py::array_t<float>(static_cast<py::ssize_t>(k));
                    result_distances_ptr = result_distances.mutable_data();
                }

                size_t count = 0;
                if (eps_or_ef >= 1.0f) {
                    const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                    std::vector<deglib::graph::ObjectDistance> result = graph.search_ef(q_span, k, ef, true, filter, max_distance_computation_count);
                    count = std::min(static_cast<size_t>(k), result.size());
                    for (size_t i = 0; i < count; ++i) {
                        result_indices_ptr[i] = result[i].getIdentifier();
                        if (result_distances_ptr) {
                            result_distances_ptr[i] = result[i].getDistance();
                        }
                    }
                } else {
                    deglib::graph::ResultSet result = graph.search(q_span, k, eps_or_ef, filter, max_distance_computation_count);
                    while (result.size() > k) result.pop();
                    count = result.size();
                    if (unsorted) {
                        for (size_t i = 0; i < count; ++i) {
                            result_indices_ptr[i] = result[i].getIdentifier();
                            if (result_distances_ptr) result_distances_ptr[i] = result[i].getDistance();
                        }
                    } else {
                        for (size_t i = count; i > 0; --i) {
                            deglib::search::ObjectDistance next_result = result.top();
                            result_indices_ptr[i - 1] = next_result.getIdentifier();
                            if (result_distances_ptr) result_distances_ptr[i - 1] = next_result.getDistance();
                            result.pop();
                        }
                    }
                }

                for (size_t i = count; i < k; ++i) {
                    result_indices_ptr[i] = std::numeric_limits<uint32_t>::max();
                    if (result_distances_ptr) {
                        result_distances_ptr[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                }

                if (return_distances)
                    return py::make_tuple(result_indices, result_distances);
                else
                    return py::object(result_indices);
            };

            if (is_fp16) {
                return perform_search(std::span<const uint16_t>(static_cast<const uint16_t*>(query_info.ptr), in_dim));
            } else if (query_info.itemsize == sizeof(float) && (query_info.format == "f" || query_info.format == "float")) {
                return perform_search(std::span<const float>(static_cast<const float*>(query_info.ptr), in_dim));
            } else {
                throw std::invalid_argument(std::format("Unsupported query dtype (format: {}, itemsize: {})", query_info.format, query_info.itemsize));
            }
        }
    }
    const size_t req_bytes = graph.getFeatureSpace().get_data_size();
    const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
    const size_t query_bytes = query_info.size * query_info.itemsize;
    if (query_bytes != query_req_bytes && query_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_bytes, query_req_bytes, req_bytes
            )
        );
    }
    const std::byte* query_ptr = static_cast<const std::byte*>(query_info.ptr);
    std::span<const std::byte> query_span(query_ptr, query_bytes);

    py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(k));
    auto result_indices_ptr = result_indices.mutable_data();

    py::array_t<float> result_distances;
    float* result_distances_ptr = nullptr;
    if (return_distances) {
        result_distances = py::array_t<float>(static_cast<py::ssize_t>(k));
        result_distances_ptr = result_distances.mutable_data();
    }

    size_t count = 0;
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        std::vector<deglib::graph::ObjectDistance> result = graph.search_ef(query_span, k, ef, true, filter, max_distance_computation_count);
        count = std::min(static_cast<size_t>(k), result.size());
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
        deglib::graph::ResultSet result = graph.search(query_span, k, eps_or_ef, filter, max_distance_computation_count);
        while (result.size() > k) {
            result.pop();
        }
        count = result.size();
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
    }

    for (size_t i = count; i < k; ++i) {
        result_indices_ptr[i] = std::numeric_limits<uint32_t>::max();
        if (result_distances_ptr) {
            result_distances_ptr[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    if (return_distances) {
        return py::make_tuple(result_indices, result_distances);
    } else {
        return py::object(result_indices);
    }
}

// Batch search wrapper for DynamicExplorationGraph — search() already returns external labels
py::object dynamic_exploration_graph_search_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const py::array query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    auto c_query = py::array::ensure(query, py::array::c_style);
    if (!c_query) {
        throw std::invalid_argument("search_batch: query must be convertible to a C-contiguous array");
    }
    py::buffer_info query_info = c_query.request();

    if (query_info.ndim != 2) {
        throw std::invalid_argument(std::format("Expected query to have two dimensions, got {}", query_info.ndim));
    }

    const uint32_t n_queries = static_cast<uint32_t>(query_info.shape[0]);
    py::array_t<uint32_t> result_indices({n_queries, k});
    auto result_indices_ptr = static_cast<uint32_t*>(result_indices.request().ptr);

    py::array_t<float> result_distances;
    float* result_distances_ptr = nullptr;
    if (return_distances) {
        result_distances = py::array_t<float>({n_queries, k});
        result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
    }

    std::span<uint32_t> out_indices_span(result_indices_ptr, static_cast<size_t>(n_queries) * k);
    std::span<float> out_distances_span = return_distances ? std::span<float>(result_distances_ptr, static_cast<size_t>(n_queries) * k) : std::span<float>{};

    {
        py::gil_scoped_release release;

        if (graph.hasTransformations()) {
            const size_t in_dim = graph.getInputDim();
            if (size_t(query_info.shape[1]) != in_dim) {
                throw std::invalid_argument(std::format("Query row dimension ({}) does not match transformations input dim ({})", query_info.shape[1], in_dim));
            }
            const bool is_fp16 = (query_info.format == "e" || (query_info.itemsize == 2 && query_info.format != "h" && query_info.format != "H"));
            if (is_fp16) {
                std::span<const uint16_t> queries_span(static_cast<const uint16_t*>(query_info.ptr), n_queries * in_dim);
                if (eps_or_ef >= 1.0f) {
                    const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                    graph.search_batch_ef<uint16_t>(
                        queries_span, n_queries, k, ef, out_indices_span, out_distances_span, true, filter, max_distance_computation_count, threads, unsorted
                    );
                } else {
                    graph.search_batch<uint16_t>(
                        queries_span, n_queries, k, out_indices_span, out_distances_span, eps_or_ef, filter, max_distance_computation_count, threads, unsorted
                    );
                }
            } else {
                std::span<const float> queries_span(static_cast<const float*>(query_info.ptr), n_queries * in_dim);
                if (eps_or_ef >= 1.0f) {
                    const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                    graph.search_batch_ef<float>(
                        queries_span, n_queries, k, ef, out_indices_span, out_distances_span, true, filter, max_distance_computation_count, threads, unsorted
                    );
                } else {
                    graph.search_batch<float>(
                        queries_span, n_queries, k, out_indices_span, out_distances_span, eps_or_ef, filter, max_distance_computation_count, threads, unsorted
                    );
                }
            }
        } else {
            const size_t req_bytes = graph.getFeatureSpace().get_data_size();
            const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
            const size_t query_stride_bytes = size_t(query_info.shape[1]) * query_info.itemsize;
            if (query_stride_bytes != query_req_bytes && query_stride_bytes != req_bytes) {
                throw std::invalid_argument(
                    std::format(
                        "Query row size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_stride_bytes, query_req_bytes,
                        req_bytes
                    )
                );
            }
            std::span<const std::byte> queries_span(static_cast<const std::byte*>(query_info.ptr), n_queries * query_stride_bytes);
            if (eps_or_ef >= 1.0f) {
                const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                graph.search_batch_ef<std::byte>(
                    queries_span, n_queries, k, ef, out_indices_span, out_distances_span, true, filter, max_distance_computation_count, threads, unsorted
                );
            } else {
                graph.search_batch<std::byte>(
                    queries_span, n_queries, k, out_indices_span, out_distances_span, eps_or_ef, filter, max_distance_computation_count, threads, unsorted
                );
            }
        }
    }

    if (return_distances) {
        return py::make_tuple(result_indices, result_distances);
    } else {
        return py::object(result_indices);
    }
}

template <typename G>
py::object graph_explore_single_wrapper(
    const G& graph,
    const uint32_t entry_external_label,
    const uint32_t k,
    const uint32_t max_distance_computation_count = 0,
    const float eps_or_ef = 0.0f,
    const bool include_entry = true,
    const deglib::search::Filter* filter = nullptr,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        auto result = graph.explore_ef(entry_external_label, k, ef, max_distance_computation_count, include_entry, filter);
        const size_t count = result.size();
        py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(count));
        auto result_indices_ptr = static_cast<uint32_t*>(result_indices.request().ptr);

        py::array_t<float> result_distances;
        float* result_distances_ptr = nullptr;
        if (return_distances) {
            result_distances = py::array_t<float>(static_cast<py::ssize_t>(count));
            result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
        }

        for (size_t i = 0; i < count; ++i) {
            result_indices_ptr[i] = result[i].getIdentifier();
            if (result_distances_ptr) {
                result_distances_ptr[i] = result[i].getDistance();
            }
        }

        if (return_distances) {
            return py::make_tuple(result_indices, result_distances);
        } else {
            return result_indices;
        }
    }

    deglib::graph::ResultSet result = graph.explore(entry_external_label, k, max_distance_computation_count, eps_or_ef, include_entry, filter);

    while (result.size() > k) {
        result.pop();
    }

    const size_t count = result.size();
    py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(count));
    auto result_indices_ptr = static_cast<uint32_t*>(result_indices.request().ptr);

    py::array_t<float> result_distances;
    float* result_distances_ptr = nullptr;
    if (return_distances) {
        result_distances = py::array_t<float>(static_cast<py::ssize_t>(count));
        result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
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
py::object graph_has_path_wrapper(
    const G& graph,
    const std::vector<uint32_t>& from_external_labels,
    const uint32_t to_external_label,
    const float eps,
    const uint32_t k,
    const bool return_distances = true
) {
    const std::vector<deglib::graph::ObjectDistance> path = graph.hasPath(from_external_labels, to_external_label, eps, k);

    const size_t count = path.size();
    py::array_t<uint32_t> path_indices(static_cast<py::ssize_t>(count));
    auto path_indices_ptr = static_cast<uint32_t*>(path_indices.request().ptr);

    py::array_t<float> path_distances;
    float* path_distances_ptr = nullptr;
    if (return_distances) {
        path_distances = py::array_t<float>(static_cast<py::ssize_t>(count));
        path_distances_ptr = static_cast<float*>(path_distances.request().ptr);
    }

    for (size_t i = 0; i < count; ++i) {
        path_indices_ptr[i] = path[i].getIdentifier();
        if (path_distances_ptr) {
            path_distances_ptr[i] = path[i].getDistance();
        }
    }

    if (return_distances) {
        return py::make_tuple(path_indices, path_distances);
    } else {
        return path_indices;
    }
}

py::object dynamic_exploration_graph_explore_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const py::array_t<uint32_t, py::array::c_style> entry_external_labels,
    const uint32_t k,
    const uint32_t max_distance_computation_count,
    const float eps_or_ef,
    const bool include_entry,
    const deglib::search::Filter* filter,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    py::buffer_info entry_info = entry_external_labels.request();
    if (entry_info.ndim != 1) {
        throw std::invalid_argument(std::format("Expected entry_external_labels to have 1 dimension, got {}", entry_info.ndim));
    }

    const uint32_t* entry_ptr = static_cast<const uint32_t*>(entry_info.ptr);
    const uint32_t n_queries = entry_info.shape[0];

    py::array_t<uint32_t> result_indices({n_queries, k});
    auto result_indices_ptr = static_cast<uint32_t*>(result_indices.request().ptr);

    py::array_t<float> result_distances;
    float* result_distances_ptr = nullptr;
    if (return_distances) {
        result_distances = py::array_t<float>({n_queries, k});
        result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
    }

    std::span<const uint32_t> entries_span(entry_ptr, n_queries);
    std::span<uint32_t> out_indices_span(result_indices_ptr, static_cast<size_t>(n_queries) * k);
    std::span<float> out_distances_span = return_distances ? std::span<float>(result_distances_ptr, static_cast<size_t>(n_queries) * k) : std::span<float>{};

    {
        py::gil_scoped_release release;
        if (eps_or_ef >= 1.0f) {
            const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
            graph.explore_batch_ef(
                entries_span, k, ef, out_indices_span, out_distances_span, max_distance_computation_count, include_entry, filter, threads, unsorted
            );
        } else {
            graph.explore_batch(
                entries_span, k, out_indices_span, out_distances_span, max_distance_computation_count, eps_or_ef, include_entry, filter, threads, unsorted
            );
        }
    }

    if (return_distances) {
        return py::make_tuple(result_indices, result_distances);
    } else {
        return result_indices;
    }
}

float float_space_compute_distance(const deglib::distances::FloatSpace& space, py::array vec1, py::array vec2) {
    auto buf1 = vec1.request();
    auto buf2 = vec2.request();
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();
    if (size_t(buf1.size) * buf1.itemsize != query_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec1 size in bytes ({}) does not match required "
                "query size ({})",
                size_t(buf1.size) * buf1.itemsize, query_bytes
            )
        );
    }
    if (size_t(buf2.size) * buf2.itemsize != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec2 size in bytes ({}) does not match required "
                "feature space data size ({})",
                size_t(buf2.size) * buf2.itemsize, req_bytes
            )
        );
    }
    return deglib::distances::compute_distance(space, buf1.ptr, buf2.ptr);
}

py::array_t<float> float_space_compute_distances(const deglib::distances::FloatSpace& space, py::array query, py::array targets, size_t threads) {
    auto q_buf = query.request();
    auto t_buf = targets.request();
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();

    size_t num_queries = 0;
    size_t query_stride_bytes = 0;
    if (q_buf.ndim == 1) {
        num_queries = 1;
        query_stride_bytes = size_t(q_buf.size) * q_buf.itemsize;
    } else if (q_buf.ndim == 2) {
        num_queries = q_buf.shape[0];
        query_stride_bytes = size_t(q_buf.shape[1]) * q_buf.itemsize;
    } else {
        throw std::invalid_argument("Query array must be 1D or 2D NumPy array");
    }

    if (query_stride_bytes != query_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match required "
                "query size ({})",
                query_stride_bytes, query_bytes
            )
        );
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
            std::format(
                "Target vector size in bytes ({}) does not match required "
                "feature space data size ({})",
                target_stride_bytes, req_bytes
            )
        );
    }

    auto result = py::array_t<float>({num_queries, num_targets});
    auto r_buf = result.request();
    float* out = static_cast<float*>(r_buf.ptr);
    const size_t query_stride = query_stride_bytes;
    const void* qptr = q_buf.ptr;
    const void* tptr = t_buf.ptr;

    if (num_queries == 1) {
        py::gil_scoped_release release;
        deglib::distances::compute_distances(space, qptr, 1, tptr, num_targets, out);
    } else {
        py::gil_scoped_release release;
        deglib::concurrent::parallel_for(0, num_queries, threads, [&](size_t q, size_t) {
            deglib::distances::compute_distances(space, static_cast<const uint8_t*>(qptr) + q * query_stride, 1, tptr, num_targets, out + q * num_targets);
        });
    }

    return result;
}
// ============================================================================
// Standalone Reranker Wrapper
// ============================================================================

// ============================================================================
// Reranker Binding Helpers (thin marshaling around C++ owning Reranker)
// ============================================================================

static std::shared_ptr<deglib::search::Reranker> reranker_from_float32(
    deglib::distances::MetricType metric,
    py::array_t<float, py::array::c_style> data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<py::array_t<uint32_t, py::array::c_style>> row_indices = std::nullopt
) {
    auto buf = data.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const size_t n_rows = static_cast<size_t>(buf.shape[0]);
    const uint32_t dim = static_cast<uint32_t>(buf.shape[1]);
    std::span<const float> data_span(static_cast<const float*>(buf.ptr), n_rows * dim);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value()) {
        auto idx_buf = row_indices->request();
        if (idx_buf.ndim != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(static_cast<const uint32_t*>(idx_buf.ptr), static_cast<size_t>(idx_buf.shape[0]));
    }

    py::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_data(deglib::distances::Metric(metric), data_span, dim, std::move(transformations), threads, idx_span)
    );
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_raw(
    const deglib::distances::FloatSpace& space,
    py::array data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<py::array_t<uint32_t, py::array::c_style>> row_indices = std::nullopt
) {
    auto c_data = py::array::ensure(data, py::array::c_style);
    if (!c_data) {
        throw std::invalid_argument("base_vectors must be convertible to a C-contiguous array");
    }
    auto buf = c_data.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const size_t req_bytes = space.get_data_size();
    const size_t stride_bytes = static_cast<size_t>(buf.shape[1]) * static_cast<size_t>(buf.itemsize);
    if (stride_bytes != req_bytes) {
        throw std::invalid_argument(std::format("Base vector row size ({}) does not match feature space data size ({})", stride_bytes, req_bytes));
    }
    const size_t n_rows = static_cast<size_t>(buf.shape[0]);
    std::span<const std::byte> data_span(static_cast<const std::byte*>(buf.ptr), n_rows * stride_bytes);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value()) {
        auto idx_buf = row_indices->request();
        if (idx_buf.ndim != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(static_cast<const uint32_t*>(idx_buf.ptr), static_cast<size_t>(idx_buf.shape[0]));
    }

    py::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_raw(space, data_span, n_rows, std::move(transformations), threads, idx_span)
    );
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_graph(
    const deglib::DynamicExplorationGraph& graph,
    deglib::distances::MetricType metric,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0
) {
    if (transformations.empty()) {
        const auto target_dt = deglib::distances::Metric(metric).get_data_type();
        const auto graph_dt = graph.getFeatureSpace().metric().get_data_type();
        if (target_dt == deglib::distances::MetricDataType::FP16 &&
            graph_dt == deglib::distances::MetricDataType::FP16 &&
            graph.hasTransformations()) {
            transformations = graph.getTransformations();
        }
    }
    py::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_graph(graph.internal(), deglib::distances::Metric(metric), std::move(transformations), threads)
    );
}

static py::object reranker_rerank_py(
    const deglib::search::Reranker& reranker,
    py::array queries,
    py::array_t<uint32_t, py::array::c_style> candidate_indices,
    size_t k_top = 0,
    size_t num_threads = 0,
    bool return_distances = false,
    bool unsorted = false
) {
    auto c_queries = py::array::ensure(queries, py::array::c_style);
    if (!c_queries) {
        throw std::invalid_argument("queries must be convertible to a C-contiguous array");
    }
    auto q_buf = c_queries.request();
    if (q_buf.ndim != 2) {
        throw std::invalid_argument("queries must be a 2D array");
    }
    const size_t n_queries = static_cast<size_t>(q_buf.shape[0]);

    auto cand_buf = candidate_indices.request();
    if (cand_buf.ndim != 2) {
        throw std::invalid_argument("candidate_indices must be a 2D array");
    }
    if (static_cast<size_t>(cand_buf.shape[0]) != n_queries) {
        throw std::invalid_argument("Number of candidate rows must match number of queries");
    }

    const size_t evp_k = static_cast<size_t>(cand_buf.shape[1]);
    if (k_top == 0 || k_top > evp_k) {
        k_top = evp_k;
    }

    auto result_indices = py::array_t<uint32_t>({n_queries, k_top});
    uint32_t* ind_ptr = static_cast<uint32_t*>(result_indices.request().ptr);

    py::array_t<float> result_distances;
    float* dist_ptr = nullptr;
    if (return_distances) {
        result_distances = py::array_t<float>({n_queries, k_top});
        dist_ptr = static_cast<float*>(result_distances.request().ptr);
    }

    const bool is_f32_queries = py::isinstance<py::array_t<float>>(c_queries);
    const size_t query_row_bytes = static_cast<size_t>(q_buf.shape[1]) * static_cast<size_t>(q_buf.itemsize);
    const size_t req_bytes = reranker.getSpace().get_data_size();
    const auto* cand_ptr = static_cast<const uint32_t*>(cand_buf.ptr);

    if (is_f32_queries && (!reranker.getTransformations().empty() || reranker.getSpace().metric().get_data_type() == deglib::distances::MetricDataType::FP32)) {
        const size_t q_dim = static_cast<size_t>(q_buf.shape[1]);
        const auto* q_ptr = static_cast<const float*>(q_buf.ptr);
        py::gil_scoped_release release;
        deglib::concurrent::parallel_batch_for(
            size_t{0}, n_queries, num_threads,
            [&](size_t begin, size_t end, size_t) {
                for (size_t q = begin; q < end; ++q) {
                    reranker.rerank_f32(
                        std::span<const float>(q_ptr + q * q_dim, q_dim),
                        std::span<const uint32_t>(cand_ptr + q * evp_k, evp_k),
                        static_cast<uint32_t>(k_top),
                        std::span<uint32_t>(ind_ptr + q * k_top, k_top),
                        return_distances ? std::span<float>(dist_ptr + q * k_top, k_top) : std::span<float>{},
                        return_distances,
                        unsorted
                    );
                }
            },
            64
        );
    } else {
        if (query_row_bytes != req_bytes) {
            throw std::invalid_argument(
                std::format("Query row size in bytes ({}) does not match required feature space data size ({})", query_row_bytes, req_bytes)
            );
        }
        py::gil_scoped_release release;
        reranker.rerank(
            std::span<const std::byte>(static_cast<const std::byte*>(q_buf.ptr), n_queries * req_bytes), n_queries,
            std::span<const uint32_t>(cand_ptr, n_queries * evp_k), evp_k, static_cast<uint32_t>(k_top),
            std::span<uint32_t>(ind_ptr, n_queries * k_top), return_distances ? std::span<float>(dist_ptr, n_queries * k_top) : std::span<float>{},
            return_distances, unsorted, num_threads
        );
    }

    if (return_distances) {
        return py::make_tuple(result_indices, result_distances);
    }
    return result_indices;
}

// ============================================================================
// Fast Searcher Wrapper
// ============================================================================

class SearcherPy {
  private:
    std::unique_ptr<deglib::search::Searcher> searcher_;

  public:
    SearcherPy(
        const deglib::DynamicExplorationGraph& graph,
        std::shared_ptr<deglib::search::Reranker> shared_reranker = nullptr,
        bool normalize_query = false
    ) {
        if (shared_reranker) {
            searcher_ = deglib::make_searcher(graph, std::move(shared_reranker));
        } else {
            searcher_ = deglib::make_searcher(graph);
        }
        searcher_->set_normalize_query(normalize_query);
    }

    bool get_normalize_query() const noexcept { return searcher_->normalize_query(); }
    void set_normalize_query(bool val) noexcept { searcher_->set_normalize_query(val); }

    void optimize(uint32_t n_clusters = 256, uint32_t n_iter = 20, size_t sample_size = 0, size_t threads = 0, bool use_routing = true) {
        py::gil_scoped_release release;
        searcher_->optimize(n_clusters, n_iter, sample_size, threads, use_routing);
    }

    py::object search(
        py::array query,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) {
        // Enforce C-contiguous array with zero-copy if already contiguous
        auto c_query = py::array::ensure(query, py::array::c_style);
        if (!c_query) {
            throw std::invalid_argument("search: query must be convertible to a C-contiguous array");
        }
        auto buf = c_query.request();

        // 1D query or 2D single-row/single-column query -> execute fast single-query search
        if (buf.ndim == 1 || (buf.ndim == 2 && (buf.shape[0] == 1 || buf.shape[1] == 1))) {
            py::array_t<uint32_t> result(static_cast<py::ssize_t>(k));
            uint32_t* out_ptr = static_cast<uint32_t*>(result.request().ptr);

            py::array_t<float> dist_result;
            float* dist_ptr = nullptr;
            if (return_distances) {
                dist_result = py::array_t<float>(static_cast<py::ssize_t>(k));
                dist_ptr = static_cast<float*>(dist_result.request().ptr);
            }

            uint32_t found = 0;
            if (eps_or_ef >= 1.0f) {
                const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    found = searcher_->search_ef_f16(
                        static_cast<const uint16_t*>(buf.ptr), k, ef, rerank_factor, out_ptr, dist_ptr, unsorted, max_distance_computation_count
                    );
                } else {
                    found = searcher_->search_ef_f32(
                        static_cast<const float*>(buf.ptr), k, ef, rerank_factor, out_ptr, dist_ptr, unsorted, max_distance_computation_count
                    );
                }
            } else {
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    found = searcher_->search_f16(
                        static_cast<const uint16_t*>(buf.ptr), k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, unsorted, max_distance_computation_count
                    );
                } else {
                    found = searcher_->search_f32(
                        static_cast<const float*>(buf.ptr), k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, unsorted, max_distance_computation_count
                    );
                }
            }

            if (found < k) {
                result.resize({static_cast<py::ssize_t>(found)});
                if (return_distances) {
                    dist_result.resize({static_cast<py::ssize_t>(found)});
                }
            }

            if (return_distances) {
                return py::make_tuple(result, dist_result);
            }
            return result;
        }

        // 2D batch query
        if (buf.ndim == 2) {
            size_t n_queries = buf.shape[0];
            auto result = py::array_t<uint32_t>({n_queries, size_t(k)});
            uint32_t* out_ptr = static_cast<uint32_t*>(result.request().ptr);

            py::array_t<float> dist_result;
            float* dist_ptr = nullptr;
            if (return_distances) {
                dist_result = py::array_t<float>({n_queries, size_t(k)});
                dist_ptr = static_cast<float*>(dist_result.request().ptr);
            }

            {
                py::gil_scoped_release release;
                if (eps_or_ef >= 1.0f) {
                    const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                    if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                        searcher_->search_batch_ef_f16(
                            static_cast<const uint16_t*>(buf.ptr), n_queries, k, ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted,
                            max_distance_computation_count
                        );
                    } else {
                        searcher_->search_batch_ef_f32(
                            static_cast<const float*>(buf.ptr), n_queries, k, ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted,
                            max_distance_computation_count
                        );
                    }
                } else {
                    if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                        searcher_->search_batch_f16(
                            static_cast<const uint16_t*>(buf.ptr), n_queries, k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted,
                            max_distance_computation_count
                        );
                    } else {
                        searcher_->search_batch_f32(
                            static_cast<const float*>(buf.ptr), n_queries, k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted,
                            max_distance_computation_count
                        );
                    }
                }
            }

            if (return_distances) {
                return py::make_tuple(result, dist_result);
            }
            return result;
        }

        throw std::invalid_argument("search query must be 1D vector or 2D batch array");
    }

    py::object search_batch(
        py::array queries,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) {
        return search(queries, k, eps_or_ef, rerank_factor, num_threads, return_distances, unsorted, max_distance_computation_count);
    }
};

// ============================================================================
// ============================================================================
// Scalar Quantization (SQ8 / Int8 / Uint8) Bindings
// ============================================================================

// ============================================================================
// Scalar Quantization Helper
// ============================================================================

inline std::pair<size_t, uint32_t> get_2d_shape(const py::buffer_info& buf) {
    if (buf.ndim != 2) {
        throw std::invalid_argument(std::format("Expected 2D array, got {} dimensions", buf.ndim));
    }
    return {static_cast<size_t>(buf.shape[0]), static_cast<uint32_t>(buf.shape[1])};
}

// ============================================================================
// FP16 Conversion Bindings
// ============================================================================

py::array_t<uint16_t> floats_to_fp16_wrapper(py::array_t<float, py::array::c_style> floats) {
    py::buffer_info buf = floats.request();
    if (buf.itemsize != sizeof(float)) {
        throw std::invalid_argument("Input array must contain 32-bit floats (itemsize == 4)");
    }
    size_t count = buf.size;

    py::array_t<uint16_t> output(count);
    py::buffer_info out_buf = output.request();
    deglib::distances::fp16::floats_to_fp16(static_cast<const float*>(buf.ptr), static_cast<uint16_t*>(out_buf.ptr), count);
    return output;
}

py::array_t<float> fp16_to_floats_wrapper(py::array_t<uint16_t, py::array::c_style> fp16_vals) {
    py::buffer_info buf = fp16_vals.request();
    if (buf.itemsize != sizeof(uint16_t)) {
        throw std::invalid_argument("Input array must contain 16-bit uint16/FP16 values (itemsize == 2)");
    }
    size_t count = buf.size;

    py::array_t<float> output(count);
    py::buffer_info out_buf = output.request();
    deglib::distances::fp16::fp16_to_floats(static_cast<const uint16_t*>(buf.ptr), static_cast<float*>(out_buf.ptr), count);
    return output;
}

deglib::DynamicExplorationGraph load_readonly_graph_wrapper(const char* path) { return deglib::DynamicExplorationGraph::load_readonly(path); }

deglib::DynamicExplorationGraph load_sizebounded_graph_wrapper(const char* path, const uint32_t capacity = 0) {
    return deglib::load_mutable_graph(path, capacity);
}

inline deglib::distances::MetricDataType parse_quant_data_type(const py::object& quant) {
    if (py::isinstance<deglib::distances::Metric>(quant)) {
        return quant.cast<deglib::distances::Metric>().get_data_type();
    }
    if (py::isinstance<py::int_>(quant)) {
        return deglib::distances::Metric(static_cast<deglib::distances::MetricType>(quant.cast<uint8_t>())).get_data_type();
    }
    if (py::isinstance<py::str>(quant)) {
        std::string q_lower = quant.cast<std::string>();
        for (char& c : q_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (q_lower == "uint4") return deglib::distances::MetricDataType::Uint4;
        if (q_lower == "int4") return deglib::distances::MetricDataType::Int4;
        if (q_lower == "uint7") return deglib::distances::MetricDataType::UInt7;
        if (q_lower == "uint8") return deglib::distances::MetricDataType::Uint8;
        if (q_lower == "int8") return deglib::distances::MetricDataType::Int8;
        if (q_lower == "fp16") return deglib::distances::MetricDataType::FP16;
        if (q_lower == "evp") return deglib::distances::MetricDataType::EVP;
        throw std::invalid_argument("Unsupported quantizer data type: " + q_lower);
    }
    throw std::invalid_argument("quant must be a Metric or data type string (uint4, int4, uint7, uint8, int8, fp16, evp)");
}

py::array_t<uint32_t> presort_wrapper(
    py::array_t<float, py::array::c_style> vectors,
    const deglib::distances::FloatSpace& space,
    float radius_decay = 0.9f,
    size_t threads = 0,
    py::object callback = py::none()
) {
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
        sorted_indices = deglib::optimization::presort(static_cast<const float*>(buf.ptr), count, space, radius_decay, threads, cb);
    }

    py::array_t<uint32_t> output(count);
    py::buffer_info out_buf = output.request();
    std::memcpy(out_buf.ptr, sorted_indices.data(), count * sizeof(uint32_t));
    return output;
}

deglib::DynamicExplorationGraph read_only_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> custom_transformations = {}
) {
    const void* feat_ptr = nullptr;
    py::buffer_info feat_info;

    if (custom_features.has_value() && !custom_features->is_none()) {
        feat_info = custom_features->request();
        feat_ptr = feat_info.ptr;
        if (custom_feature_space.has_value()) {
            const size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
            const size_t actual_bytes = feat_info.size * feat_info.itemsize;
            if (actual_bytes != expected_bytes) {
                throw std::invalid_argument(
                    std::format("custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes)
                );
            }
        }
    }

    return graph.to_readonly(custom_feature_space, feat_ptr, std::move(custom_transformations));
}

deglib::DynamicExplorationGraph dynamic_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt,
    const uint32_t chunk_size = 1024,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> custom_transformations = {}
) {
    const void* feat_ptr = nullptr;
    py::buffer_info feat_info;

    if (custom_features.has_value() && !custom_features->is_none()) {
        feat_info = custom_features->request();
        feat_ptr = feat_info.ptr;
        if (custom_feature_space.has_value()) {
            size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
            size_t actual_bytes = feat_info.size * feat_info.itemsize;
            if (actual_bytes != expected_bytes) {
                throw std::invalid_argument(
                    std::format("custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes)
                );
            }
        }
    }

    return graph.to_dynamic(custom_feature_space, feat_ptr, chunk_size, std::move(custom_transformations));
}

deglib::DynamicExplorationGraph load_dynamic_graph_wrapper(const char* path, const uint32_t chunk_size = 1024) {
    return deglib::load_dynamic_graph(path, chunk_size);
}

deglib::DynamicExplorationGraph size_bounded_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt,
    const uint32_t new_max_size = 0,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> custom_transformations = {}
) {
    const void* feat_ptr = nullptr;
    py::buffer_info feat_info;

    if (custom_features.has_value() && !custom_features->is_none()) {
        feat_info = custom_features->request();
        feat_ptr = feat_info.ptr;
        if (custom_feature_space.has_value()) {
            size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
            size_t actual_bytes = feat_info.size * feat_info.itemsize;
            if (actual_bytes != expected_bytes) {
                throw std::invalid_argument(
                    std::format("custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes)
                );
            }
        }
    }

    return graph.to_mutable(custom_feature_space, feat_ptr, new_max_size, std::move(custom_transformations));
}

deglib::DynamicExplorationGraph create_size_bounded_graph(
    const uint32_t max_vertex_count,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    return deglib::create_empty(max_vertex_count, edges_per_vertex, feature_space, std::move(transformations));
}

deglib::DynamicExplorationGraph create_dynamic_graph(
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t chunk_size = 1024,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    return deglib::create_dynamic_empty(edges_per_vertex, feature_space, chunk_size, std::move(transformations));
}

deglib::DynamicExplorationGraph create_random_graph(
    py::array features,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t seed,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    py::buffer_info buf = features.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", buf.ndim));
    }
    const uint32_t vertex_count = static_cast<uint32_t>(buf.shape[0]);
    const size_t req_bytes = feature_space.get_data_size();
    const size_t row_bytes = size_t(buf.shape[1]) * buf.itemsize;
    if (row_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Feature row size in bytes ({}) does not match required feature space "
                "data size ({})",
                row_bytes, req_bytes
            )
        );
    }

    const std::byte* feature_data = static_cast<const std::byte*>(buf.ptr);
    return deglib::create_random_graph(feature_data, vertex_count, edges_per_vertex, feature_space, seed, std::move(transformations));
}

// ============================================================================
// Analysis Wrapper Functions
// ============================================================================

float calc_avg_edge_weight_wrapper(const deglib::DynamicExplorationGraph& graph, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_avg_edge_weight");
    }
    return deglib::analysis::calc_avg_edge_weight(static_cast<const deglib::graph::MutableGraph&>(graph.internal()), scale);
}

std::vector<float> calc_edge_weight_histogram_wrapper(const deglib::DynamicExplorationGraph& graph, const bool sorted, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_edge_weight_histogram");
    }
    return deglib::analysis::calc_edge_weight_histogram(static_cast<const deglib::graph::MutableGraph&>(graph.internal()), sorted, scale);
}

bool check_graph_weights_wrapper(const deglib::DynamicExplorationGraph& graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for check_graph_weights");
    }
    return deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph&>(graph.internal()));
}

uint32_t calc_non_mrng_edges_wrapper(const deglib::DynamicExplorationGraph& graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_non_mrng_edges");
    }
    return deglib::analysis::calc_non_mrng_edges(static_cast<const deglib::graph::MutableGraph&>(graph.internal()));
}

bool check_graph_regularity_wrapper(const deglib::DynamicExplorationGraph& graph, const uint32_t expected_vertices, const bool check_back_link) {
    return deglib::analysis::check_graph_regularity(graph.internal(), expected_vertices, check_back_link);
}

bool check_graph_connectivity_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::check_graph_connectivity(graph.internal()); }

uint32_t calc_search_reachability_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::calc_search_reachability(graph.internal()); }

float calc_exploration_reach_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::calc_exploration_reach(graph.internal()); }

deglib::analysis::GraphStats analyze_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<py::array_t<uint32_t>> base_top = std::nullopt,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    const uint32_t* ptr = nullptr;
    size_t rows = 0;
    size_t k = 0;
    size_t row_stride = 0;

    if (base_top.has_value() && base_top->size() > 0) {
        py::buffer_info buf = base_top->request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("base_top must be a 2D numpy array");
        }
        rows = static_cast<size_t>(buf.shape[0]);
        k = static_cast<size_t>(buf.shape[1]);
        const size_t col_stride = static_cast<size_t>(buf.strides[1]) / sizeof(uint32_t);
        if (col_stride != 1) {
            throw std::invalid_argument("base_top columns must be contiguous");
        }
        row_stride = static_cast<size_t>(buf.strides[0]) / sizeof(uint32_t);
        ptr = static_cast<const uint32_t*>(buf.ptr);
    }

    deglib::analysis::GraphStats result;
    {
        py::gil_scoped_release release;
        result = deglib::analysis::analyze_graph(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
    }
    return result;
}

float calc_graph_quality_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    py::array_t<uint32_t> base_top,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    py::buffer_info buf = base_top.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("base_top must be a 2D numpy array");
    }
    const size_t rows = static_cast<size_t>(buf.shape[0]);
    const size_t k = static_cast<size_t>(buf.shape[1]);
    const size_t col_stride = static_cast<size_t>(buf.strides[1]) / sizeof(uint32_t);
    if (col_stride != 1) {
        throw std::invalid_argument("base_top columns must be contiguous");
    }
    const size_t row_stride = static_cast<size_t>(buf.strides[0]) / sizeof(uint32_t);
    const auto* ptr = static_cast<const uint32_t*>(buf.ptr);

    float result = 0.0f;
    {
        py::gil_scoped_release release;
        result = deglib::analysis::calc_graph_quality(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
    }
    return result;
}

// ============================================================================

PYBIND11_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    // cpu submodule
    py::module_ cpu_module = m.def_submodule("cpu", "CPU feature detection and InstructionSet");
    py::enum_<deglib::cpu::InstructionSet>(cpu_module, "InstructionSet")
        .value("Auto", deglib::cpu::InstructionSet::Auto)
        .value("Scalar", deglib::cpu::InstructionSet::Scalar)
        .value("AVX2", deglib::cpu::InstructionSet::AVX2)
        .value("AVX2_VNNI", deglib::cpu::InstructionSet::AVX2_VNNI)
        .value("AVX512", deglib::cpu::InstructionSet::AVX512)
        .value("AVX512_VNNI", deglib::cpu::InstructionSet::AVX512_VNNI);

    cpu_module.def("has_avx2", &deglib::cpu::has_avx2, "Returns whether AVX2 instructions are available");
    cpu_module.def("has_avx2_vnni", &deglib::cpu::has_avx2_vnni, "Returns whether AVX2-VNNI (256-bit) instructions are available");
    cpu_module.def("has_avx512", &deglib::cpu::has_avx512, "Returns whether AVX512 instructions are available");
    cpu_module.def("has_avx512_vnni", &deglib::cpu::has_avx512_vnni, "Returns whether AVX-512 VNNI instructions are available");
    cpu_module.def("has_avx512_fp16", &deglib::cpu::has_avx512_fp16, "Returns whether AVX-512 FP16 instructions are available");
    cpu_module.def("has_fp16", &deglib::cpu::has_fp16, "Returns whether FP16 instructions are available");

    // distances submodule
    py::module_ distances_module = m.def_submodule("distances", "Distance metrics and feature spaces");
    py::enum_<deglib::distances::MetricType>(distances_module, "Metric")
        .value("FP32_L2", deglib::distances::MetricType::FP32_L2)
        .value("FP32_InnerProduct", deglib::distances::MetricType::FP32_InnerProduct)
        .value("Uint8_L2", deglib::distances::MetricType::Uint8_L2)
        .value("Uint8_InnerProduct", deglib::distances::MetricType::Uint8_InnerProduct)
        .value("FP16_L2", deglib::distances::MetricType::FP16_L2)
        .value("FP16_InnerProduct", deglib::distances::MetricType::FP16_InnerProduct)
        .value("EVP_InnerProduct", deglib::distances::MetricType::EVP_InnerProduct)
        .value("Int8_InnerProduct", deglib::distances::MetricType::Int8_InnerProduct)
        .value("Int8_L2", deglib::distances::MetricType::Int8_L2)
        .value("Uint4_InnerProduct", deglib::distances::MetricType::Uint4_InnerProduct)
        .value("Uint4_L2", deglib::distances::MetricType::Uint4_L2)
        .value("Int4_InnerProduct", deglib::distances::MetricType::Int4_InnerProduct)
        .value("Int4_L2", deglib::distances::MetricType::Int4_L2)
        .value("UInt7_InnerProduct", deglib::distances::MetricType::UInt7_InnerProduct)
        .value("UInt7_L2", deglib::distances::MetricType::UInt7_L2);

    py::class_<deglib::distances::FloatSpace>(distances_module, "FloatSpace")
        .def(
            py::init([](const size_t dim, const deglib::distances::MetricType metric, const deglib::cpu::InstructionSet instruction) {
                return deglib::distances::FloatSpace(dim, deglib::distances::Metric(metric), instruction);
            }),
            py::arg("dim"), py::arg("metric") = deglib::distances::MetricType::FP32_L2, py::arg("instruction") = deglib::cpu::InstructionSet::Auto
        )
        .def("dim", &deglib::distances::FloatSpace::dim)
        .def("metric", [](const deglib::distances::FloatSpace& fs) { return fs.metric().value; })
        .def("get_data_size", &deglib::distances::FloatSpace::get_data_size)
        .def("get_query_data_size", &deglib::distances::FloatSpace::get_query_data_size)
        .def("get_instruction", &deglib::distances::FloatSpace::get_instruction)
        .def("compute_distance", &float_space_compute_distance, py::arg("vec1"), py::arg("vec2"))
        .def("compute_distances", &float_space_compute_distances, py::arg("query"), py::arg("targets"), py::arg("threads") = 0);

    // quantization functions in distances submodule
    distances_module.def("floats_to_fp16", &floats_to_fp16_wrapper, "Convert float32 array to FP16 (uint16_t)");
    distances_module.def("fp16_to_floats", &fp16_to_floats_wrapper, "Convert FP16 (uint16_t) array to float32");

    // optimization submodule
    py::module_ optimization_module = m.def_submodule("optimization", "Graph optimization and layout utilities");
    optimization_module.def(
        "presort", &presort_wrapper, py::arg("vectors"), py::arg("space"), py::arg("radius_decay") = 0.9f, py::arg("threads") = 0,
        py::arg("callback") = py::none(), "Perform 1D pre-sorting of dataset feature vectors using FLAS"
    );
    optimization_module.def(
        "prune_non_mrng_edges",
        [](deglib::DynamicExplorationGraph& graph, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), threads);
        },
        py::arg("graph"), py::arg("threads") = 0, "Prune edges that violate the MRNG condition"
    );
    optimization_module.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, threads);
        },
        py::arg("graph"), py::arg("prune_worst"), py::arg("threads") = 0, "Prune the worst edges of each vertex"
    );

    // transformation submodule
    py::module_ transformation_module = m.def_submodule("transformation", "Feature transformation and quantization");

    auto apply_transformations_array = [](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr,
                                          bool is_database, size_t threads = 0) -> py::array {
        if (transformations.empty()) return in_arr;
        auto c_arr = py::array::ensure(in_arr, py::array::c_style);
        if (!c_arr) {
            throw std::invalid_argument("Input array must be convertible to a C-contiguous array");
        }
        py::buffer_info buf = c_arr.request();
        bool is_1d = (buf.ndim == 1);
        size_t count = is_1d ? 1 : buf.shape[0];
        size_t dim = is_1d ? buf.shape[0] : buf.shape[1];
        if (dim != transformations.front()->input_dim()) {
            throw std::invalid_argument(
                std::format("Input dimension ({}) does not match transformation input_dim ({})", dim, transformations.front()->input_dim())
            );
        }
        const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
        const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
        py::array_t<float, py::array::c_style> f_arr;
        const float* f_ptr = nullptr;
        const uint16_t* u16_ptr = nullptr;
        if (is_database && is_fp16) {
            u16_ptr = static_cast<const uint16_t*>(buf.ptr);
        } else if (is_fp32) {
            f_ptr = static_cast<const float*>(buf.ptr);
        } else {
            f_arr = py::array_t<float, py::array::c_style>(c_arr);
            f_ptr = static_cast<const float*>(f_arr.request().ptr);
        }

        const size_t out_stride = transformations.back()->output_byte_size();
        auto run_transform = [&](std::byte* out_ptr) {
            py::gil_scoped_release release;
            std::span<std::byte> out_span(out_ptr, count * out_stride);
            if (is_database) {
                if (u16_ptr != nullptr) {
                    deglib::transformation::apply_on_database(transformations, std::span<const uint16_t>(u16_ptr, count * dim), out_span, threads);
                } else {
                    deglib::transformation::apply_on_database(transformations, std::span<const float>(f_ptr, count * dim), out_span, threads);
                }
            } else {
                deglib::transformation::apply_on_query(transformations, std::span<const float>(f_ptr, count * dim), out_span);
            }
        };
        const uint32_t tag = transformations.back()->tag();
        if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<int8_t> out(shape);
            run_transform(reinterpret_cast<std::byte*>(out.request().ptr));
            return out;
        } else if (
            tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
            tag == deglib::transformation::TAG_SQI4
        ) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<uint8_t> out(shape);
            run_transform(reinterpret_cast<std::byte*>(out.request().ptr));
            return out;
        } else if (tag == deglib::transformation::TAG_SQ16) {
            std::vector<py::ssize_t> shape =
                is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(transformations.back()->output_dim())}
                      : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(transformations.back()->output_dim())};
            py::array out(py::dtype("float16"), shape);
            run_transform(reinterpret_cast<std::byte*>(out.request().ptr));
            return out;
        } else {
            std::vector<py::ssize_t> shape =
                is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(transformations.back()->output_dim())}
                      : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(transformations.back()->output_dim())};
            py::array_t<float> out(shape);
            run_transform(reinterpret_cast<std::byte*>(out.request().ptr));
            return out;
        }
    };

    py::class_<deglib::transformation::FeatureTransformation, std::shared_ptr<deglib::transformation::FeatureTransformation>>(
        transformation_module, "FeatureTransformation"
    )
        .def_property_readonly("name", [](const deglib::transformation::FeatureTransformation& t) { return std::string(t.name()); })
        .def_property_readonly("tag", &deglib::transformation::FeatureTransformation::tag)
        .def_property_readonly("input_dim", &deglib::transformation::FeatureTransformation::input_dim)
        .def_property_readonly("output_dim", &deglib::transformation::FeatureTransformation::output_dim)
        .def_property_readonly("input_byte_size", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_property_readonly("output_byte_size", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_property_readonly("input_bytes", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_property_readonly("output_bytes", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_property_readonly("is_fitted", &deglib::transformation::FeatureTransformation::is_fitted)
        .def_property_readonly(
            "output_dtype",
            [](const deglib::transformation::FeatureTransformation& t) -> py::object {
                uint32_t tag = t.tag();
                py::module_ np = py::module_::import("numpy");
                if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) return np.attr("int8");
                if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
                    tag == deglib::transformation::TAG_SQI4)
                    return np.attr("uint8");
                if (tag == deglib::transformation::TAG_SQ16) return np.attr("float16");
                return np.attr("float32");
            }
        )
        .def(
            "transform_database",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, py::array in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def(
            "quantize",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, py::array in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def(
            "transform_query",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, py::array in_arr) {
                return apply_transformations_array({std::move(t)}, in_arr, false, 1);
            },
            py::arg("queries")
        )
        .def(
            "dequantize",
            [](const deglib::transformation::FeatureTransformation& t, py::array quantized, size_t threads) -> py::array {
                auto c_arr = py::array::ensure(quantized, py::array::c_style);
                if (!c_arr) {
                    throw std::invalid_argument("dequantize: input must be convertible to a C-contiguous array");
                }
                py::buffer_info buf = c_arr.request();
                const bool is_1d = (buf.ndim == 1);
                if (buf.ndim != 1 && buf.ndim != 2) {
                    throw std::invalid_argument(std::format("Expected 1D or 2D array, got {} dimensions", buf.ndim));
                }
                const size_t count = is_1d ? 1 : static_cast<size_t>(buf.shape[0]);
                const size_t row_bytes =
                    (is_1d ? static_cast<size_t>(buf.shape[0]) : static_cast<size_t>(buf.shape[1])) * static_cast<size_t>(buf.itemsize);
                if (row_bytes != t.output_byte_size()) {
                    throw std::invalid_argument(std::format("quantized row bytes ({}) does not match expected ({})", row_bytes, t.output_byte_size()));
                }
                const uint32_t out_dim = t.output_dim();
                std::vector<py::ssize_t> shape = is_1d
                                                     ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_dim)}
                                                     : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_dim)};
                py::array_t<float> output(shape);
                const std::byte* src = static_cast<const std::byte*>(buf.ptr);
                float* dst = static_cast<float*>(output.request().ptr);
                if (count > 0) {
                    py::gil_scoped_release release;
                    deglib::concurrent::parallel_batch_for(
                        size_t{0}, count, threads,
                        [&](size_t begin, size_t end, size_t) {
                            for (size_t i = begin; i < end; ++i) {
                                t.dequantize({src + i * row_bytes, row_bytes}, {dst + i * out_dim, out_dim});
                            }
                        },
                        16
                    );
                }
                return output;
            },
            py::arg("quantized"), py::arg("threads") = 0,
            "Dequantizes transformed vectors back to approximate float32 vectors."
        )
        .def("serialize", [](const deglib::transformation::FeatureTransformation& t) -> py::bytes {
            std::ostringstream ss(std::ios::binary);
            t.serialize(ss);
            return py::bytes(ss.str());
        });

    py::class_<deglib::transformation::MipsTransform, deglib::transformation::FeatureTransformation, std::shared_ptr<deglib::transformation::MipsTransform>>(
        transformation_module, "MipsTransform"
    )
        .def_property_readonly("max_norm", &deglib::transformation::MipsTransform::max_norm)
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::MipsTransform::deserialize(ss);
        });

    py::class_<deglib::transformation::EvpQuantizer, deglib::transformation::FeatureTransformation, std::shared_ptr<deglib::transformation::EvpQuantizer>>(
        transformation_module, "EvpQuantizer"
    )
        .def_property_readonly("dim", &deglib::transformation::EvpQuantizer::dim, "Dimension of vectors handled by this quantizer")
        .def_property_readonly("non_zeros", &deglib::transformation::EvpQuantizer::non_zeros, "Number of non-zero bits retained per vector")
        .def_property_readonly("is_fitted", &deglib::transformation::EvpQuantizer::is_fitted, "True if non_zeros is calibrated/set and mean is computed")
        .def_property_readonly(
            "mean",
            [](const deglib::transformation::EvpQuantizer& q) -> py::object {
                if (!q.has_mean()) return py::none();
                auto m = q.mean();
                py::array_t<float> arr(static_cast<py::ssize_t>(m.size()));
                std::memcpy(arr.request().ptr, m.data(), m.size() * sizeof(float));
                return arr;
            },
            "Current 1D mean vector subtracted during quantization, or None if not set"
        )
        .def(
            "quantize",
            [](deglib::transformation::EvpQuantizer& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                if (q.dim() != 0 && q.dim() != dim) {
                    throw std::invalid_argument(std::format("EvpQuantizer: vector dim ({}) does not match quantizer dim ({})", dim, q.dim()));
                }

                const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
                const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }

                if (!q.is_fitted()) {
                    throw std::logic_error("EvpQuantizer: quantize called before fit()");
                }

                const size_t evp_bytes = 2 * ((dim + 7) / 8);
                py::array_t<uint8_t> output({count, evp_bytes});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);

                {
                    py::gil_scoped_release release;
                    if (is_fp16) {
                        q.transform_database(
                            std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * evp_bytes),
                            threads
                        );
                    } else {
                        q.transform_database(
                            std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * evp_bytes), threads
                        );
                    }
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::EvpQuantizer::deserialize(ss);
        });

    py::class_<
        deglib::transformation::ScalarQuantizerInt8, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerInt8>>(transformation_module, "ScalarQuantizerInt8")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerInt8::dim)
        .def_property_readonly("abs_max", &deglib::transformation::ScalarQuantizerInt8::abs_max)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerInt8::scale)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerInt8::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerInt8& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                py::array_t<int8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(int8_t)),
                        threads
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(int8_t)),
                        threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerInt8>(deglib::transformation::ScalarQuantizerInt8::deserialize(ss));
        });

    py::class_<
        deglib::transformation::ScalarQuantizerUint8, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerUint8>>(transformation_module, "ScalarQuantizerUint8")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerUint8::dim)
        .def_property_readonly("min_val", &deglib::transformation::ScalarQuantizerUint8::min_val)
        .def_property_readonly("max_val", &deglib::transformation::ScalarQuantizerUint8::max_val)
        .def_property_readonly("dif", &deglib::transformation::ScalarQuantizerUint8::dif)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerUint8::scale)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerUint8::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerUint8& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                py::array_t<uint8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint8_t)),
                        threads
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint8_t)),
                        threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerUint8>(deglib::transformation::ScalarQuantizerUint8::deserialize(ss));
        });

    py::class_<
        deglib::transformation::ScalarQuantizerInt4, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerInt4>>(transformation_module, "ScalarQuantizerInt4")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerInt4::dim)
        .def_property_readonly("abs_max", &deglib::transformation::ScalarQuantizerInt4::abs_max)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerInt4::scale)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerInt4::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerInt4& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                const size_t out_stride = (dim + 1) / 2;
                py::array_t<uint8_t> output({count, out_stride});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * out_stride),
                        threads
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * out_stride), threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerInt4>(deglib::transformation::ScalarQuantizerInt4::deserialize(ss));
        });

    py::class_<
        deglib::transformation::ScalarQuantizerUInt7, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerUInt7>>(transformation_module, "ScalarQuantizerUInt7")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerUInt7::dim)
        .def_property_readonly("min", &deglib::transformation::ScalarQuantizerUInt7::min)
        .def_property_readonly("max", &deglib::transformation::ScalarQuantizerUInt7::max)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerUInt7::scale)
        .def_property_readonly("inv_scale", &deglib::transformation::ScalarQuantizerUInt7::inv_scale)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerUInt7::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerUInt7& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                py::array_t<int8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(int8_t)),
                        threads
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(int8_t)),
                        threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerUInt7>(deglib::transformation::ScalarQuantizerUInt7::deserialize(ss));
        });

    py::class_<
        deglib::transformation::ScalarQuantizerUint4, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerUint4>>(transformation_module, "ScalarQuantizerUint4")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerUint4::dim)
        .def_property_readonly("min_val", &deglib::transformation::ScalarQuantizerUint4::min_val)
        .def_property_readonly("max_val", &deglib::transformation::ScalarQuantizerUint4::max_val)
        .def_property_readonly("dif", &deglib::transformation::ScalarQuantizerUint4::dif)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerUint4::scale)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerUint4::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerUint4& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                const size_t out_stride = (dim + 1) / 2;
                py::array_t<uint8_t> output({count, out_stride});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * out_stride),
                        threads
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * out_stride), threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerUint4>(deglib::transformation::ScalarQuantizerUint4::deserialize(ss));
        });

    py::class_<
        deglib::transformation::ScalarQuantizerFP16, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerFP16>>(transformation_module, "ScalarQuantizerFP16")
        .def_property_readonly("dim", &deglib::transformation::ScalarQuantizerFP16::dim)
        .def_property_readonly("abs_max", &deglib::transformation::ScalarQuantizerFP16::abs_max)
        .def_property_readonly("scale", &deglib::transformation::ScalarQuantizerFP16::scale)
        .def_property_readonly("inv_scale", &deglib::transformation::ScalarQuantizerFP16::inv_scale)
        .def_property_readonly("quantile", &deglib::transformation::ScalarQuantizerFP16::quantile)
        .def_property_readonly("is_fitted", &deglib::transformation::ScalarQuantizerFP16::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerFP16& q, py::array vectors, size_t threads) {
                auto c_vecs = py::array::ensure(vectors, py::array::c_style);
                if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
                py::buffer_info buf = c_vecs.request();
                auto [count, dim] = get_2d_shape(buf);
                py::array output(py::dtype("float16"), std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t))
                    );
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    py::gil_scoped_release release;
                    q.transform_database(
                        std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t)),
                        threads
                    );
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                return output;
            },
            py::arg("vectors"), py::arg("threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return std::shared_ptr<deglib::transformation::ScalarQuantizerFP16>(deglib::transformation::ScalarQuantizerFP16::deserialize(ss));
        });

    py::class_<
        deglib::transformation::L2NormalizeTransform, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::L2NormalizeTransform>>(transformation_module, "L2NormalizeTransform")
        .def_property_readonly("dim", &deglib::transformation::L2NormalizeTransform::dim)
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::L2NormalizeTransform::deserialize(ss);
        });

    auto bind_rotation_common = [](auto& cls) {
        using RotT = typename std::decay_t<decltype(cls)>::type;
        cls.def_property_readonly("dim", &RotT::dim)
            .def_property_readonly("seed", &RotT::seed)
            .def_property_readonly("is_enabled", &RotT::is_enabled)
            .def_property("enabled", &RotT::is_enabled, &RotT::set_enabled)
            .def_property_readonly("crest_factor", &RotT::crest_factor)
            .def_property_readonly("excess_kurtosis", &RotT::excess_kurtosis)
            .def_static("deserialize", [](py::bytes b) {
                std::string s = b;
                std::istringstream ss(s, std::ios::binary);
                return RotT::deserialize(ss);
            });
    };

    auto fwht_cls = py::class_<
                        deglib::transformation::FastHadamardRotation, deglib::transformation::FeatureTransformation,
                        std::shared_ptr<deglib::transformation::FastHadamardRotation>>(transformation_module, "FastHadamardRotation")
                        .def_property_readonly("rounds", &deglib::transformation::FastHadamardRotation::rounds);
    bind_rotation_common(fwht_cls);

    auto make_scalar_quantizer_impl = [](auto type_tag, py::array vectors, float quantile) {
        using QuantT = typename decltype(type_tag)::type;
        auto c_vecs = py::array::ensure(vectors, py::array::c_style);
        if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
        py::buffer_info buf = c_vecs.request();
        auto [count, dim] = get_2d_shape(buf);
        if (count == 0 || dim == 0) {
            throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
        }
        auto q = std::make_shared<QuantT>(dim);
        if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
            py::gil_scoped_release release;
            q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), quantile);
        } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
            py::gil_scoped_release release;
            q->fit(std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), quantile);
        } else {
            throw std::invalid_argument("vectors must be float32 or float16");
        }
        return q;
    };

    transformation_module.def(
        "make_scalar_quantizer_int8",
        [make_scalar_quantizer_impl](py::array vectors, float quantile) {
            return make_scalar_quantizer_impl(std::type_identity<deglib::transformation::ScalarQuantizerInt8>{}, vectors, quantile);
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_uint8",
        [make_scalar_quantizer_impl](py::array vectors, float quantile) {
            return make_scalar_quantizer_impl(std::type_identity<deglib::transformation::ScalarQuantizerUint8>{}, vectors, quantile);
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_int4",
        [make_scalar_quantizer_impl](py::array vectors, float quantile) {
            return make_scalar_quantizer_impl(std::type_identity<deglib::transformation::ScalarQuantizerInt4>{}, vectors, quantile);
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_uint7",
        [make_scalar_quantizer_impl](py::array vectors, float quantile) {
            return make_scalar_quantizer_impl(std::type_identity<deglib::transformation::ScalarQuantizerUInt7>{}, vectors, quantile);
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_uint4",
        [make_scalar_quantizer_impl](py::array vectors, float quantile) {
            return make_scalar_quantizer_impl(std::type_identity<deglib::transformation::ScalarQuantizerUint4>{}, vectors, quantile);
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f,
        "Factory function creating a ScalarQuantizerUint4 calibrated on vectors. If quantile < 0 (default), auto-optimizes."
    );

    transformation_module.def(
        "make_quantizer_pipeline",
        [](py::array vectors, py::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed) {
            auto c_vecs = py::array::ensure(vectors, py::array::c_style);
            if (!c_vecs) {
                throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
            }
            py::buffer_info buf = c_vecs.request();
            auto [count, dim] = get_2d_shape(buf);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_quantizer_pipeline: vectors must be a non-empty 2D array with dim > 0");
            }
            const auto data_type = parse_quant_data_type(quant);

            const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
            const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
            if (!is_fp16 && !is_fp32) {
                py::array_t<float, py::array::c_style> f_arr(c_vecs);
                auto f_buf = f_arr.request();
                py::gil_scoped_release release;
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const float>(static_cast<const float*>(f_buf.ptr), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank,
                    threads, seed
                );
            }

            py::gil_scoped_release release;
            if (is_fp16) {
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank,
                    threads, seed
                );
            }
            return deglib::transformation::make_quantizer_pipeline(
                std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank,
                threads, seed
            );
        },
        py::arg("vectors"), py::arg("quant"), py::arg("quantile") = -1.0f, py::arg("use_rerank") = false, py::arg("threads") = 0, py::arg("seed") = 7,
        "Builds a transformation pipeline [FastHadamardRotation, ScalarQuantizer] or [ScalarQuantizer] based on kurtosis."
    );

    transformation_module.def(
        "make_quantizer_pipeline_from_graph",
        [](const deglib::DynamicExplorationGraph& graph, py::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed) {
            if (graph.size() == 0) {
                throw std::invalid_argument("Cannot calibrate quantizer pipeline from an empty graph");
            }
            const auto& src_space = graph.getFeatureSpace();
            const auto src_dt = src_space.metric().get_data_type();
            if (src_dt != deglib::distances::MetricDataType::FP32 && src_dt != deglib::distances::MetricDataType::FP16) {
                throw std::invalid_argument("Cannot calibrate quantizer pipeline from a non-float graph without external features");
            }
            const auto target_dt = parse_quant_data_type(quant);
            const size_t n = graph.size();
            const uint32_t dim = src_space.dim();
            const size_t max_sample = (target_dt == deglib::distances::MetricDataType::EVP) ? 65536 : 10000;
            const size_t sample_count = std::min(n, max_sample);
            std::vector<float> sample_f32(sample_count * dim);
            const bool src_fp16 = (src_dt == deglib::distances::MetricDataType::FP16);
            const auto& internal = graph.internal();

            std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> pipeline;
            {
                py::gil_scoped_release release;
                deglib::concurrent::parallel_batch_for(
                    size_t{0}, sample_count, threads,
                    [&](size_t begin, size_t end, size_t) {
                        for (size_t s = begin; s < end; ++s) {
                            const uint32_t v_idx = static_cast<uint32_t>((sample_count == n) ? s : (s * n) / sample_count);
                            const std::byte* src = internal.getFeatureVector(v_idx);
                            float* dst = sample_f32.data() + s * dim;
                            if (src_fp16) {
                                deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), dst, dim);
                            } else {
                                std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
                            }
                        }
                    },
                    64
                );
                pipeline = deglib::transformation::make_quantizer_pipeline(
                    std::span<const float>(sample_f32), dim, target_dt, quantile, use_rerank, threads, seed
                );
            }
            return pipeline;
        },
        py::arg("graph"), py::arg("quant"), py::arg("quantile") = -1.0f, py::arg("use_rerank") = false, py::arg("threads") = 0, py::arg("seed") = 7,
        "Calibrate a quantizer pipeline directly from an FP32/FP16 graph's stored vectors."
    );

    transformation_module.def(
        "make_scalar_quantizer_fp16",
        [](py::array vectors, float quantile) {
            auto c_vecs = py::array::ensure(vectors, py::array::c_style);
            if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
            py::buffer_info buf = c_vecs.request();
            auto [count, dim] = get_2d_shape(buf);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
            }
            auto q = std::make_shared<deglib::transformation::ScalarQuantizerFP16>(dim);
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                py::gil_scoped_release release;
                q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim));
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                py::gil_scoped_release release;
                q->fit(std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), quantile);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_evp_quantizer",
        [](py::array vectors, uint32_t non_zeros, size_t threads) {
            auto c_vecs = py::array::ensure(vectors, py::array::c_style);
            if (!c_vecs) throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
            py::buffer_info buf = c_vecs.request();
            auto [count, dim] = get_2d_shape(buf);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
            }
            auto q = std::make_shared<deglib::transformation::EvpQuantizer>(dim, non_zeros);
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                py::gil_scoped_release release;
                q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(buf.ptr), count * dim), threads);
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                py::gil_scoped_release release;
                q->fit(std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), threads);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("non_zeros") = 0, py::arg("threads") = 0,
        "Factory function creating an EvpQuantizer calibrated on vectors."
    );

    transformation_module.def(
        "make_mips_transform",
        [](py::array vectors, float max_norm) {
            auto c_vecs = py::array::ensure(vectors, py::array::c_style);
            if (!c_vecs) {
                throw std::invalid_argument("vectors must be convertible to a C-contiguous array");
            }
            py::buffer_info buf = c_vecs.request();
            auto [count, dim] = get_2d_shape(buf);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_mips_transform: vectors must be a non-empty 2D array with dim > 0");
            }
            auto t = std::make_shared<deglib::transformation::MipsTransform>(dim, max_norm);
            if (max_norm <= 0.0f) {
                const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
                const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
                if (is_fp32) {
                    py::gil_scoped_release release;
                    t->fit(std::span<const float>(static_cast<const float*>(buf.ptr), count * dim));
                } else if (is_fp16) {
                    std::vector<float> fp16_converted(count * dim);
                    py::gil_scoped_release release;
                    deglib::distances::fp16::fp16_to_floats(static_cast<const uint16_t*>(buf.ptr), fp16_converted.data(), count * dim);
                    t->fit(std::span<const float>(fp16_converted.data(), count * dim));
                } else {
                    py::array_t<float, py::array::c_style> f_arr(c_vecs);
                    auto f_buf = f_arr.request();
                    py::gil_scoped_release release;
                    t->fit(std::span<const float>(static_cast<const float*>(f_buf.ptr), count * dim));
                }
            }
            return t;
        },
        py::arg("vectors"), py::arg("max_norm") = 0.0f,
        "Factory function creating a MipsTransform (d -> d+1 spherical lift) fitted on vectors."
    );

    transformation_module.def(
        "make_l2_normalize_transform",
        [](uint32_t dim) {
            if (dim == 0) {
                throw std::invalid_argument("make_l2_normalize_transform: dim must be > 0");
            }
            return std::make_shared<deglib::transformation::L2NormalizeTransform>(dim);
        },
        py::arg("dim"),
        "Factory function creating an L2NormalizeTransform."
    );

    transformation_module.def(
        "make_fast_hadamard_rotation",
        [](py::array_t<float, py::array::c_style> vectors, uint64_t seed, uint32_t rounds, bool auto_detect) {
            py::buffer_info buf = vectors.request();
            auto [count, dim] = get_2d_shape(buf);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_fast_hadamard_rotation: vectors must be a non-empty 2D array with dim > 0");
            }
            auto t = std::make_shared<deglib::transformation::FastHadamardRotation>(dim, seed, rounds);
            {
                py::gil_scoped_release release;
                t->fit(std::span<const float>(static_cast<const float*>(buf.ptr), count * dim), auto_detect);
            }
            return t;
        },
        py::arg("vectors"), py::arg("seed") = 7, py::arg("rounds") = 0, py::arg("auto_detect") = true,
        "Factory function creating a FastHadamardRotation fitted on vectors."
    );

    transformation_module.def(
        "apply_on_database",
        [apply_transformations_array](
            const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr, size_t threads
        ) { return apply_transformations_array(transformations, in_arr, true, threads); },
        py::arg("transformations"), py::arg("vectors"), py::arg("threads") = 0
    );

    transformation_module.def(
        "apply_on_query",
        [apply_transformations_array](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr) {
            return apply_transformations_array(transformations, in_arr, false);
        },
        py::arg("transformations"), py::arg("queries")
    );

    m.attr("FeatureTransformation") = transformation_module.attr("FeatureTransformation");
    m.attr("make_l2_normalize_transform") = transformation_module.attr("make_l2_normalize_transform");
    m.attr("make_fast_hadamard_rotation") = transformation_module.attr("make_fast_hadamard_rotation");
    m.attr("make_mips_transform") = transformation_module.attr("make_mips_transform");
    m.attr("make_quantizer_pipeline") = transformation_module.attr("make_quantizer_pipeline");
    m.attr("make_evp_quantizer") = transformation_module.attr("make_evp_quantizer");
    m.attr("make_scalar_quantizer_int8") = transformation_module.attr("make_scalar_quantizer_int8");
    m.attr("make_scalar_quantizer_uint8") = transformation_module.attr("make_scalar_quantizer_uint8");
    m.attr("make_scalar_quantizer_uint4") = transformation_module.attr("make_scalar_quantizer_uint4");
    m.attr("make_scalar_quantizer_int4") = transformation_module.attr("make_scalar_quantizer_int4");
    m.attr("make_scalar_quantizer_uint7") = transformation_module.attr("make_scalar_quantizer_uint7");
    m.attr("make_scalar_quantizer_fp16") = transformation_module.attr("make_scalar_quantizer_fp16");

    // search submodule
    py::module_ search_module = m.def_submodule("search", "Search utilities including Filter");
    py::class_<deglib::search::Filter>(search_module, "Filter").def(py::init<const int*, size_t, size_t, size_t>());

    search_module.def("create_filter", [](py::array_t<int, py::array::c_style> valid_labels, size_t max_value, size_t max_label_count) {
        const py::buffer_info labels_info = valid_labels.request();
        const int* ptr = static_cast<int*>(labels_info.ptr);
        if (labels_info.ndim != 1) {
            throw std::invalid_argument(std::format("Expected feature to have only one dimension, got {}", labels_info.ndim));
        }

        size_t size = labels_info.shape[0];
        return new deglib::search::Filter(ptr, size, max_value, max_label_count);
    });

    py::class_<deglib::search::Reranker, std::shared_ptr<deglib::search::Reranker>>(search_module, "Reranker", "Owning C++ candidate reranker.")
        .def_static(
            "from_float32", &reranker_from_float32,
            py::arg("metric"), py::arg("data"),
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            py::arg("threads") = 0, py::arg("row_indices") = std::nullopt
        )
        .def_static(
            "from_raw", &reranker_from_raw,
            py::arg("space"), py::arg("data"),
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            py::arg("threads") = 0, py::arg("row_indices") = std::nullopt
        )
        .def_static(
            "from_graph", &reranker_from_graph,
            py::arg("graph"), py::arg("metric"),
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            py::arg("threads") = 0
        )
        .def("get_num_base_vectors", &deglib::search::Reranker::getNumBaseVectors)
        .def("get_dim", [](const deglib::search::Reranker& r) { return r.getSpace().dim(); })
        .def("get_po", &deglib::search::Reranker::getPo, "Prefetch lookahead offset (0 disables rerank prefetching).")
        .def("get_pl", &deglib::search::Reranker::getPl, "Cache lines prefetched per feature vector (0 prefetches the whole vector).")
        .def(
            "optimize",
            [](deglib::search::Reranker& r, size_t sample_count, size_t num_candidates) {
                py::gil_scoped_release release;
                r.optimize(sample_count, num_candidates);
            },
            py::arg("sample_count") = 300, py::arg("num_candidates") = 150,
            "Auto-tune the reranker's prefetch parameters (po, pl) by timing reranking over sampled base vectors."
        )
        .def(
            "rerank", &reranker_rerank_py, py::arg("queries"), py::arg("candidate_indices"), py::arg("k_top") = 0, py::arg("threads") = 0,
            py::arg("return_distances") = false, py::arg("unsorted") = false
        );

    py::class_<SearcherPy>(search_module, "Searcher", "Search pipeline. Results are the graph's labels.")
        .def(
            py::init<const deglib::DynamicExplorationGraph&, std::shared_ptr<deglib::search::Reranker>, bool>(),
            py::arg("graph"), py::arg("reranker") = nullptr, py::arg("normalize_query") = false, py::keep_alive<1, 2>(),
            "Construct a Searcher with optional shared owning Reranker."
        )
        .def_property("normalize_query", &SearcherPy::get_normalize_query, &SearcherPy::set_normalize_query)
        .def(
            "search", &SearcherPy::search, py::arg("query"), py::arg("k") = 10, py::arg("eps_or_ef") = 0.1f, py::arg("rerank_factor") = 1.0f,
            py::arg("threads") = 1, py::arg("return_distances") = false, py::arg("unsorted") = false, py::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors of a single query vector (1D) or batch of query vectors (2D).\n\n"
            "Args:\n"
            "    query: 1D query vector or 2D batch array.\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for batch search (default: 1).\n"
            "    return_distances: If True, returns a tuple of (indices, distances).\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited)."
        )
        .def(
            "search_batch", &SearcherPy::search_batch, py::arg("queries"), py::arg("k") = 10, py::arg("eps_or_ef") = 0.1f, py::arg("rerank_factor") = 1.0f,
            py::arg("threads") = 1, py::arg("return_distances") = false, py::arg("unsorted") = false, py::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors for a batch of query vectors.\n\n"
            "Args:\n"
            "    queries: 2D query array (shape N x Dim).\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for parallel search.\n"
            "    return_distances: If True, returns a tuple of (indices, distances).\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited)."
        )
        .def(
            "optimize", &SearcherPy::optimize, py::arg("n_clusters") = 256, py::arg("n_iter") = 20, py::arg("sample_size") = 0, py::arg("threads") = 0,
            py::arg("use_routing") = true,
            "Select entry vertices via k-means medoids and tune prefetch parameters.\n\nArgs:\n    n_clusters: number of entry vertices to select (maximum "
            "512).\n    n_iter: number of "
            "k-means "
            "iterations.\n    sample_size: vertices sampled for clustering, 0 selects 3% of the graph size.\n    threads: worker threads, 0 selects "
            "a library default.\n    use_routing: build a k-means routing graph for entry selection (used by the QG/query-graph searcher)."
        );

    // graphs
    py::class_<deglib::DynamicExplorationGraph>(m, "DynamicExplorationGraph")
        .def(py::init<deglib::graph::InternalGraph&>())
        .def_static(
            "create_empty",
            [](const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space) {
                return deglib::create_empty(max_vertex_count, edges_per_vertex, feature_space);
            },
            py::arg("max_vertex_count"), py::arg("edges_per_vertex"), py::arg("feature_space")
        )
        .def_static(
            "create_dynamic_empty",
            [](const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t chunk_size) {
                return deglib::create_dynamic_empty(edges_per_vertex, feature_space, chunk_size);
            },
            py::arg("edges_per_vertex"), py::arg("feature_space"), py::arg("chunk_size") = 1024
        )
        .def_static(
            "create_random_graph",
            [](py::array features, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t seed) {
                py::buffer_info buf = features.request();
                if (buf.ndim != 2) {
                    throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", buf.ndim));
                }
                const size_t req_bytes = feature_space.get_data_size();
                const size_t row_bytes = size_t(buf.shape[1]) * buf.itemsize;
                if (row_bytes != req_bytes) {
                    throw std::invalid_argument(
                        std::format(
                            "Feature row size in bytes ({}) does not match required "
                            "feature space data size ({})",
                            row_bytes, req_bytes
                        )
                    );
                }
                return deglib::create_random_graph(
                    static_cast<const std::byte*>(buf.ptr), static_cast<uint32_t>(buf.shape[0]), edges_per_vertex, feature_space, seed
                );
            },
            py::arg("features"), py::arg("edges_per_vertex"), py::arg("feature_space"), py::arg("seed") = 7
        )
        .def("size", &deglib::DynamicExplorationGraph::size)
        .def("get_edges_per_vertex", &deglib::DynamicExplorationGraph::getEdgesPerVertex)
        .def("get_po", &deglib::DynamicExplorationGraph::getPo)
        .def("get_pl", &deglib::DynamicExplorationGraph::getPl)
        .def("get_nl", &deglib::DynamicExplorationGraph::getNl)
        .def(
            "get_feature_space", [](const deglib::DynamicExplorationGraph& g) -> const deglib::distances::FloatSpace& { return g.getFeatureSpace(); },
            py::return_value_policy::reference
        )
        .def("has_vertex", &deglib::DynamicExplorationGraph::hasVertex)
        .def(
            "get_labels",
            [](const deglib::DynamicExplorationGraph& g) {
                auto labels = g.getExternalLabels();
                py::array_t<uint32_t> arr(labels.size());
                std::memcpy(arr.mutable_data(), labels.data(), labels.size() * sizeof(uint32_t));
                return arr;
            },
            "Returns a 1D numpy array of all external labels ordered by their internal graph index (0 .. size()-1)."
        )
        .def("get_neighbors", &deglib::DynamicExplorationGraph::getNeighbors)
        .def(
            "get_neighbors_batch",
            [](const deglib::DynamicExplorationGraph& g, const py::array_t<uint32_t, py::array::c_style> labels, size_t threads) {
                auto info = labels.request();
                if (info.ndim != 1) {
                    throw std::invalid_argument(std::format("Expected labels to have 1 dimension, got {}", info.ndim));
                }
                const uint32_t n_labels = static_cast<uint32_t>(info.shape[0]);
                const uint8_t edges = g.getEdgesPerVertex();
                py::array_t<uint32_t> result({n_labels, static_cast<uint32_t>(edges)});
                auto out_ptr = static_cast<uint32_t*>(result.request().ptr);
                std::span<const uint32_t> labels_span(static_cast<const uint32_t*>(info.ptr), n_labels);
                std::span<uint32_t> out_span(out_ptr, static_cast<size_t>(n_labels) * edges);
                {
                    py::gil_scoped_release release;
                    g.getNeighbors_batch(labels_span, out_span, threads);
                }
                return result;
            },
            py::arg("external_labels"), py::arg("threads") = 1
        )
        .def("is_mutable", &deglib::DynamicExplorationGraph::isMutable)
        .def(
            "to_readonly",
            [](const deglib::DynamicExplorationGraph& g, std::optional<deglib::distances::FloatSpace> custom_feature_space,
               std::optional<py::array> custom_features, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
               size_t num_threads) {
                const void* feat_ptr = nullptr;
                py::buffer_info feat_info;
                if (custom_features.has_value() && !custom_features->is_none()) {
                    feat_info = custom_features->request();
                    feat_ptr = feat_info.ptr;
                    if (custom_feature_space.has_value()) {
                        const size_t expected_bytes = g.size() * custom_feature_space->get_data_size();
                        const size_t actual_bytes = feat_info.size * feat_info.itemsize;
                        if (actual_bytes != expected_bytes) {
                            throw std::invalid_argument(
                                std::format(
                                    "custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes
                                )
                            );
                        }
                    }
                }
                py::gil_scoped_release release;
                return g.to_readonly(custom_feature_space, feat_ptr, std::move(transformations), num_threads);
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none(),
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            py::arg("threads") = 0
        )
        .def("set_transformations", &deglib::DynamicExplorationGraph::setTransformations, py::arg("transformations"))
        .def(
            "to_mutable",
            [](const deglib::DynamicExplorationGraph& g, std::optional<deglib::distances::FloatSpace> custom_feature_space,
               std::optional<py::array> custom_features, uint32_t new_max_size,
               std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations) {
                const void* feat_ptr = nullptr;
                py::buffer_info feat_info;
                if (custom_features.has_value() && !custom_features->is_none()) {
                    feat_info = custom_features->request();
                    feat_ptr = feat_info.ptr;
                    if (custom_feature_space.has_value()) {
                        size_t expected_bytes = g.size() * custom_feature_space->get_data_size();
                        size_t actual_bytes = feat_info.size * feat_info.itemsize;
                        if (actual_bytes != expected_bytes) {
                            throw std::invalid_argument(
                                std::format(
                                    "custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes
                                )
                            );
                        }
                    }
                }
                return g.to_mutable(custom_feature_space, feat_ptr, new_max_size, std::move(transformations));
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none(), py::arg("new_max_size") = 0,
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
        )
        .def(
            "to_dynamic",
            [](const deglib::DynamicExplorationGraph& g, std::optional<deglib::distances::FloatSpace> custom_feature_space,
               std::optional<py::array> custom_features, uint32_t chunk_size,
               std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations) {
                const void* feat_ptr = nullptr;
                py::buffer_info feat_info;
                if (custom_features.has_value() && !custom_features->is_none()) {
                    feat_info = custom_features->request();
                    feat_ptr = feat_info.ptr;
                    if (custom_feature_space.has_value()) {
                        size_t expected_bytes = g.size() * custom_feature_space->get_data_size();
                        size_t actual_bytes = feat_info.size * feat_info.itemsize;
                        if (actual_bytes != expected_bytes) {
                            throw std::invalid_argument(
                                std::format(
                                    "custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes
                                )
                            );
                        }
                    }
                }
                return g.to_dynamic(custom_feature_space, feat_ptr, chunk_size, std::move(transformations));
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none(), py::arg("chunk_size") = 1024,
            py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
        )
        .def(
            "search", &graph_search_single_wrapper<deglib::DynamicExplorationGraph>, py::arg("query"), py::arg("eps_or_ef"), py::arg("k"),
            py::arg("filter") = nullptr, py::arg("max_distance_computation_count") = 0, py::arg("return_distances") = true, py::arg("unsorted") = false
        )
        .def(
            "search_batch", &dynamic_exploration_graph_search_batch_wrapper, py::arg("query"), py::arg("eps_or_ef"), py::arg("k"), py::arg("filter") = nullptr,
            py::arg("max_distance_computation_count") = 0, py::arg("threads") = 1, py::arg("return_distances") = true, py::arg("unsorted") = false
        )
        .def(
            "explore", &graph_explore_single_wrapper<deglib::DynamicExplorationGraph>, py::arg("entry_external_label"), py::arg("k"),
            py::arg("max_distance_computation_count") = 0, py::arg("eps_or_ef") = 0.0f, py::arg("include_entry") = true, py::arg("filter") = nullptr,
            py::arg("return_distances") = true, py::arg("unsorted") = false
        )
        .def(
            "explore_batch", &dynamic_exploration_graph_explore_batch_wrapper, py::arg("entry_external_labels"), py::arg("k"),
            py::arg("max_distance_computation_count") = 0, py::arg("eps_or_ef") = 0.0f, py::arg("include_entry") = true, py::arg("filter") = nullptr,
            py::arg("threads") = 1, py::arg("return_distances") = true, py::arg("unsorted") = false
        )
        .def(
            "has_path", &graph_has_path_wrapper<deglib::DynamicExplorationGraph>, py::arg("from_external_labels"), py::arg("to_external_label"),
            py::arg("eps") = 0.0f, py::arg("k") = 1, py::arg("return_distances") = true
        )
        .def(
            "save_graph", [](deglib::DynamicExplorationGraph& g, const char* path) { g.saveGraph(path); }, py::arg("path")
        )
        .def("has_transformations", &deglib::DynamicExplorationGraph::hasTransformations)
        .def("get_transformations", [](const deglib::DynamicExplorationGraph& g) { return g.getTransformations(); })
        .def("get_entry_vertex_indices", &deglib::DynamicExplorationGraph::getEntryVertexIndices)
        .def("set_entry_vertex_indices", &deglib::DynamicExplorationGraph::setEntryVertexIndices, py::arg("indices"))
        .def(
            "optimize", &deglib::DynamicExplorationGraph::optimize, py::arg("sample_count") = 300, py::arg("median_subset") = 4096,
            "Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.\n\n"
            "Args:\n"
            "    sample_count: number of vertices sampled as queries (capped at the graph size).\n"
            "    median_subset: subset size for computing the graph-median entry point (<= 1 keeps native entries)."
        );

    m.def("load_readonly_graph", &load_readonly_graph_wrapper);
    m.def("load_dynamic_graph", &load_dynamic_graph_wrapper, py::arg("path"), py::arg("chunk_size") = 1024);
    m.def("load_sizebounded_graph", &load_sizebounded_graph_wrapper, py::arg("path"), py::arg("capacity") = 0);
    m.def(
        "read_only_graph_from_graph", &read_only_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none(),
        py::arg("custom_transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "dynamic_graph_from_graph", &dynamic_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none(), py::arg("chunk_size") = 1024,
        py::arg("custom_transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "size_bounded_graph_from_graph", &size_bounded_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none(), py::arg("new_max_size") = 0,
        py::arg("custom_transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "presort", &presort_wrapper, py::arg("vectors"), py::arg("space"), py::arg("radius_decay") = 0.9f, py::arg("threads") = 0,
        py::arg("callback") = py::none()
    );

    m.def(
        "create_size_bounded_graph", &create_size_bounded_graph, py::arg("max_vertex_count"), py::arg("edges_per_vertex"), py::arg("feature_space"),
        py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_dynamic_graph", &create_dynamic_graph, py::arg("edges_per_vertex"), py::arg("feature_space"), py::arg("chunk_size") = 1024,
        py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_random_graph", &create_random_graph, py::arg("features"), py::arg("edges_per_vertex"), py::arg("feature_space"), py::arg("seed") = 7,
        py::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );

    // GraphBuilder (renamed from EvenRegularGraphBuilder)
    py::enum_<deglib::builder::OptimizationTarget>(m, "OptimizationTarget")
        .value("StreamingData", deglib::builder::OptimizationTarget::StreamingData)
        .value("HighLID", deglib::builder::OptimizationTarget::HighLID)
        .value("LowLID", deglib::builder::OptimizationTarget::LowLID);

    py::class_<deglib::builder::EvenRegularGraphBuilder>(m, "GraphBuilder")
        .def(
            py::init([](deglib::DynamicExplorationGraph& graph, std::optional<uint32_t> seed, const deglib::builder::OptimizationTarget optimization_target,
                        const uint8_t extend_k, const float extend_eps, const uint8_t improve_k, const float improve_eps, const uint8_t max_path_length,
                        const uint32_t improve_tries) {
                if (!graph.isMutable()) {
                    throw std::runtime_error("Graph must be mutable to use GraphBuilder");
                }
                uint32_t seed_val = seed.value_or(7);
                auto* rng = new std::mt19937(seed_val);
                return new deglib::builder::EvenRegularGraphBuilder(
                    graph, *rng, optimization_target, extend_k, extend_eps, improve_k, improve_eps, max_path_length, improve_tries
                );
            })
        )
        .def(
            "add_entry",
            [](deglib::builder::EvenRegularGraphBuilder& builder, const py::array_t<uint32_t, py::array::c_style>& label, const py::array& feature) {
                const auto label_access = label.unchecked<1>();
                const py::buffer_info feature_info = feature.request();
                if (feature_info.ndim != 2) {
                    throw std::invalid_argument(std::format("Expected feature to have two dimensions, got {}", feature_info.ndim));
                }

                py::gil_scoped_release release;  // release the gil
                const size_t num_entries = feature_info.shape[0];
                const size_t dim = feature_info.shape[1];

                if (builder.hasTransformations() && feature_info.itemsize == sizeof(float) && (feature_info.format == "f" || feature_info.format == "float")) {
                    const float* f_ptr = static_cast<const float*>(feature_info.ptr);
                    for (uint32_t i = 0; i < num_entries; i++) {
                        const uint32_t current_label = label_access(i);
                        builder.addEntry(current_label, std::span<const float>(f_ptr + i * dim, dim));
                    }
                } else {
                    const std::byte* feature_ptr = static_cast<const std::byte*>(feature_info.ptr);
                    const size_t feature_len = feature_info.itemsize * dim;
                    for (uint32_t i = 0; i < num_entries; i++) {
                        std::vector<std::byte> feature_vec(feature_ptr + (feature_len * i), feature_ptr + (feature_len * (i + 1)));
                        const uint32_t current_label = label_access(i);
                        builder.addEntry(current_label, std::move(feature_vec));
                    }
                }
            }
        )
        .def("remove_entry", &deglib::builder::EvenRegularGraphBuilder::removeEntry)
        .def("get_num_new_entries", &deglib::builder::EvenRegularGraphBuilder::getNumNewEntries)
        .def("get_num_remove_entries", &deglib::builder::EvenRegularGraphBuilder::getNumRemoveEntries)
        .def("set_threads", &deglib::builder::EvenRegularGraphBuilder::setThreadCount, py::arg("threads"))
        .def("set_batch_size", &deglib::builder::EvenRegularGraphBuilder::setBatchSize)
        .def("get_batch_size", &deglib::builder::EvenRegularGraphBuilder::getBatchSize)

        .def(
            "build", [](deglib::builder::EvenRegularGraphBuilder& builder, std::function<void(deglib::builder::BuilderStatus&)> callback,
                        const bool infinite) { return builder.build(callback, infinite); }
        )
        .def(
            "build_silent",
            [](deglib::builder::EvenRegularGraphBuilder& builder, const bool infinite) {
                py::gil_scoped_release release;
                return builder.build([](deglib::builder::BuilderStatus&) {}, infinite);
            }
        )
        .def("stop", &deglib::builder::EvenRegularGraphBuilder::stop);
    m.def("calc_avg_edge_weight", &calc_avg_edge_weight_wrapper, py::arg("graph"), py::arg("scale") = 1);
    m.def("calc_edge_weight_histogram", &calc_edge_weight_histogram_wrapper, py::arg("graph"), py::arg("sort"), py::arg("scale") = 1);
    m.def("check_graph_weights", &check_graph_weights_wrapper, py::arg("graph"));
    m.def("check_graph_regularity", &check_graph_regularity_wrapper, py::arg("graph"), py::arg("expected_vertices"), py::arg("check_back_link") = false);
    m.def("check_graph_connectivity", &check_graph_connectivity_wrapper, py::arg("graph"));
    m.def("calc_non_mrng_edges", &calc_non_mrng_edges_wrapper, py::arg("graph"));
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
        .def_readonly("graph_quality", &deglib::analysis::GraphStats::graph_quality)
        .def_readonly("memory_bytes", &deglib::analysis::GraphStats::memory_bytes);
    m.def("analyze_graph", &analyze_graph_wrapper, py::arg("graph"), py::arg("base_top") = py::none(), py::arg("sample_size") = 0, py::arg("threads") = 0);
    m.def(
        "calc_graph_quality", &calc_graph_quality_wrapper, py::arg("graph"), py::arg("base_top"), py::arg("sample_size") = 0, py::arg("threads") = 0,
        "Computes graph quality comparing each vertex's outgoing edges against its true nearest neighbors."
    );
    m.def(
        "prune_non_mrng_edges",
        [](deglib::DynamicExplorationGraph& graph, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), threads);
        },
        py::arg("graph"), py::arg("threads") = 0
    );
    m.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, threads);
        },
        py::arg("graph"), py::arg("prune_worst"), py::arg("threads") = 0
    );

    py::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
        .def_readwrite("step", &deglib::builder::BuilderStatus::step)
        .def_readwrite("added", &deglib::builder::BuilderStatus::added)
        .def_readwrite("deleted", &deglib::builder::BuilderStatus::deleted)
        .def_readwrite("improved", &deglib::builder::BuilderStatus::improved)
        .def_readwrite("tries", &deglib::builder::BuilderStatus::tries)
        .def_readonly("step_added_ids", &deglib::builder::BuilderStatus::step_added_ids)
        .def_readonly("step_deleted_ids", &deglib::builder::BuilderStatus::step_deleted_ids)
        .def_readonly("total_added_ids", &deglib::builder::BuilderStatus::total_added_ids)
        .def_readonly("total_deleted_ids", &deglib::builder::BuilderStatus::total_deleted_ids);
}
