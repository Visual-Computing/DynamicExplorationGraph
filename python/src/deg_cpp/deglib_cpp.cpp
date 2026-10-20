//
// Created by Bruno Schilling on 29.05.24.
//

#include <deglib/deglib.h>
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
    py::buffer_info query_info = query.request();
    if constexpr (std::is_same_v<G, deglib::DynamicExplorationGraph>) {
        if (graph.hasTransformations()) {
            const size_t in_dim = graph.getInputDim();
            if (query_info.size != in_dim) {
                throw std::invalid_argument(std::format("Query vector size ({}) does not match transformations input dim ({})", query_info.size, in_dim));
            }
            const bool is_fp16 = (query_info.format == "e" || (query_info.itemsize == 2 && query_info.format != "h" && query_info.format != "H"));
            auto perform_search = [&](auto q_span) -> py::object {
                if (eps_or_ef >= 1.0f) {
                    const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
                    std::vector<deglib::graph::ObjectDistance> result = graph.search_ef(q_span, k, ef, true, filter, max_distance_computation_count);
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

                    if (return_distances)
                        return py::make_tuple(result_indices, result_distances);
                    else
                        return py::object(result_indices);
                } else {
                    deglib::graph::ResultSet result = graph.search(q_span, k, eps_or_ef, filter, max_distance_computation_count);
                    while (result.size() > k) result.pop();
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

                    if (return_distances)
                        return py::make_tuple(result_indices, result_distances);
                    else
                        return py::object(result_indices);
                }
            };

            if (is_fp16) {
                return perform_search(std::span<const uint16_t>(static_cast<const uint16_t*>(query_info.ptr), in_dim));
            } else {
                return perform_search(std::span<const float>(static_cast<const float*>(query_info.ptr), in_dim));
            }
        }
    }
    const size_t req_bytes = graph.getFeatureSpace().get_data_size();
    const size_t query_bytes = query_info.size * query_info.itemsize;
    if (query_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match required "
                "feature space data size ({})",
                query_bytes, req_bytes
            )
        );
    }
    const std::byte* query_ptr = static_cast<const std::byte*>(query_info.ptr);
    std::span<const std::byte> query_span(query_ptr, query_bytes);

    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        std::vector<deglib::graph::ObjectDistance> result = graph.search_ef(query_span, k, ef, true, filter, max_distance_computation_count);

        const size_t count = result.size();
        py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(count));
        py::buffer_info result_indices_info = result_indices.request();
        auto result_indices_ptr = static_cast<uint32_t*>(result_indices_info.ptr);

        py::array_t<float> result_distances;
        float* result_distances_ptr = nullptr;
        if (return_distances) {
            result_distances = py::array_t<float>(static_cast<py::ssize_t>(count));
            result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
        }

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

        if (return_distances) {
            return py::make_tuple(result_indices, result_distances);
        } else {
            return result_indices;
        }
    } else {
        deglib::graph::ResultSet result = graph.search(query_span, k, eps_or_ef, filter, max_distance_computation_count);

        while (result.size() > k) {
            result.pop();
        }

        const size_t count = result.size();
        py::array_t<uint32_t> result_indices(static_cast<py::ssize_t>(count));
        py::buffer_info result_indices_info = result_indices.request();
        auto result_indices_ptr = static_cast<uint32_t*>(result_indices_info.ptr);

        py::array_t<float> result_distances;
        float* result_distances_ptr = nullptr;
        if (return_distances) {
            result_distances = py::array_t<float>(static_cast<py::ssize_t>(count));
            result_distances_ptr = static_cast<float*>(result_distances.request().ptr);
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
    py::buffer_info query_info = query.request();

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
            const size_t query_stride_bytes = size_t(query_info.shape[1]) * query_info.itemsize;
            if (query_stride_bytes != req_bytes) {
                throw std::invalid_argument(
                    std::format("Query row size in bytes ({}) does not match required feature space data size ({})", query_stride_bytes, req_bytes)
                );
            }
            std::span<const std::byte> queries_span(static_cast<const std::byte*>(query_info.ptr), n_queries * req_bytes);
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
    const std::vector<uint32_t>& entry_external_labels,
    const uint32_t to_external_label,
    const float eps,
    const uint32_t k,
    const bool return_distances = true
) {
    const std::vector<deglib::graph::ObjectDistance> path = graph.hasPath(entry_external_labels, to_external_label, eps, k);

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
    if (size_t(buf1.size) * buf1.itemsize != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec1 size in bytes ({}) does not match required "
                "feature space data size ({})",
                size_t(buf1.size) * buf1.itemsize, req_bytes
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

py::array_t<float> float_space_compute_distances(const deglib::distances::FloatSpace& space, py::array query, py::array targets) {
    auto q_buf = query.request();
    auto t_buf = targets.request();
    const size_t req_bytes = space.get_data_size();

    if (size_t(q_buf.size) * q_buf.itemsize != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match required "
                "feature space data size ({})",
                size_t(q_buf.size) * q_buf.itemsize, req_bytes
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

    auto result = py::array_t<float>(num_targets);
    auto r_buf = result.request();

    deglib::distances::compute_distances(space, q_buf.ptr, 1, t_buf.ptr, num_targets, static_cast<float*>(r_buf.ptr));

    return result;
}
// ============================================================================
// Standalone Reranker Wrapper
// ============================================================================

class RerankerPy {
  private:
    py::array base_vectors_holder_;
    deglib::search::Reranker<std::byte> reranker_;

    static const std::byte* validate_and_get_ptr(const deglib::distances::FloatSpace& space, const py::array& base_vectors) {
        auto buf = base_vectors.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("base_vectors must be a 2D array");
        }
        const size_t req_bytes = space.get_data_size();
        const size_t stride_bytes = size_t(buf.shape[1]) * buf.itemsize;
        if (stride_bytes != req_bytes) {
            throw std::invalid_argument(std::format("Base vector row size ({}) does not match feature space data size ({})", stride_bytes, req_bytes));
        }
        return static_cast<const std::byte*>(buf.ptr);
    }

  public:
    RerankerPy(const deglib::distances::FloatSpace& space, py::array base_vectors)
        : base_vectors_holder_(std::move(base_vectors)),
          reranker_(space, validate_and_get_ptr(space, base_vectors_holder_), base_vectors_holder_.request().shape[0]) {}

    size_t get_num_base_vectors() const noexcept { return reranker_.getNumBaseVectors(); }

    int32_t get_po() const noexcept { return reranker_.getPo(); }
    int32_t get_pl() const noexcept { return reranker_.getPl(); }
    void set_prefetch(int32_t po, int32_t pl) { reranker_.setPrefetch(po, pl); }

    void optimize(size_t sample_count = 50, size_t num_candidates = 100, uint32_t k = 10) {
        py::gil_scoped_release release;
        reranker_.optimize(sample_count, num_candidates, k);
    }

    py::object rerank(
        py::array queries,
        py::array_t<uint32_t> candidate_indices,
        size_t k_top = 0,
        size_t num_threads = 0,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        auto q_buf = queries.request();
        if (q_buf.ndim != 2) {
            throw std::invalid_argument("queries must be a 2D array");
        }
        const size_t req_bytes = reranker_.getSpace().get_data_size();
        const size_t query_stride_bytes = size_t(q_buf.shape[1]) * q_buf.itemsize;
        if (query_stride_bytes != req_bytes) {
            throw std::invalid_argument(
                std::format("Query row size in bytes ({}) does not match required feature space data size ({})", query_stride_bytes, req_bytes)
            );
        }
        const size_t n_queries = q_buf.shape[0];

        auto cand_buf = candidate_indices.request();
        if (cand_buf.ndim != 2) {
            throw std::invalid_argument("candidate_indices must be a 2D array");
        }
        if (static_cast<size_t>(cand_buf.shape[0]) != n_queries) {
            throw std::invalid_argument("Number of candidate rows must match number of queries");
        }

        const size_t evp_k = cand_buf.shape[1];
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

        {
            py::gil_scoped_release release;
            reranker_.rerank(
                q_buf.ptr, n_queries, static_cast<const uint32_t*>(cand_buf.ptr), evp_k, static_cast<uint32_t>(k_top), ind_ptr, dist_ptr, return_distances,
                unsorted, num_threads
            );
        }

        if (return_distances) {
            return py::make_tuple(result_indices, result_distances);
        }
        return result_indices;
    }
};

// ============================================================================
// Fast Searcher Wrapper
// ============================================================================

class SearcherPy {
  private:
    std::unique_ptr<deglib::search::SearcherBase> searcher_;
    py::array base_vectors_holder_;

  public:
    SearcherPy(
        const deglib::DynamicExplorationGraph& graph,
        py::object quantizer = py::none(),
        std::optional<deglib::distances::FloatSpace> rerank_space = std::nullopt,
        std::optional<py::array> base_vectors = std::nullopt
    ) {
        const void* base_vectors_ptr = nullptr;
        size_t num_base_vectors = 0;
        bool is_fp16_base = false;

        if (base_vectors.has_value() && !base_vectors->is_none()) {
            base_vectors_holder_ = *base_vectors;
            auto buf = base_vectors_holder_.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("base_vectors must be a 2D array");
            }
            base_vectors_ptr = buf.ptr;
            num_base_vectors = buf.shape[0];
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                is_fp16_base = true;
            }
        }

        auto make_searcher_for_quant = [&]<typename QuantT>(QuantT q) {
            if (rerank_space.has_value() && base_vectors_ptr != nullptr) {
                if (is_fp16_base) {
                    using RefinerT = deglib::search::Reranker<uint16_t>;
                    searcher_ = std::make_unique<deglib::search::SearcherImpl<QuantT, RefinerT>>(
                        graph.internal(), std::move(q), RefinerT(*rerank_space, static_cast<const uint16_t*>(base_vectors_ptr), num_base_vectors)
                    );
                } else {
                    using RefinerT = deglib::search::Reranker<float>;
                    searcher_ = std::make_unique<deglib::search::SearcherImpl<QuantT, RefinerT>>(
                        graph.internal(), std::move(q), RefinerT(*rerank_space, static_cast<const float*>(base_vectors_ptr), num_base_vectors)
                    );
                }
            } else {
                using RefinerT = deglib::search::NoRefiner;
                searcher_ = std::make_unique<deglib::search::SearcherImpl<QuantT, RefinerT>>(graph.internal(), std::move(q), RefinerT{});
            }
        };

        if (quantizer.is_none()) {
            if (graph.hasTransformations() && !graph.getTransformations().empty()) {
                auto& t = graph.getTransformations().back();
                if (t->tag() == deglib::transformation::TAG_SQI8) {
                    make_searcher_for_quant(*std::dynamic_pointer_cast<deglib::transformation::ScalarQuantizerInt8>(t));
                    return;
                } else if (t->tag() == deglib::transformation::TAG_SQ8P) {
                    make_searcher_for_quant(*std::dynamic_pointer_cast<deglib::transformation::ScalarQuantizerInt8PerDim>(t));
                    return;
                } else if (t->tag() == deglib::transformation::TAG_SQU8) {
                    make_searcher_for_quant(*std::dynamic_pointer_cast<deglib::transformation::ScalarQuantizerUint8>(t));
                    return;
                } else if (t->tag() == deglib::transformation::TAG_SQUP) {
                    make_searcher_for_quant(*std::dynamic_pointer_cast<deglib::transformation::ScalarQuantizerUint8PerDim>(t));
                    return;
                } else if (t->tag() == deglib::transformation::TAG_EVPQ) {
                    make_searcher_for_quant(*std::dynamic_pointer_cast<deglib::transformation::EvpQuantizer>(t));
                    return;
                }
            }
            make_searcher_for_quant(deglib::search::NoQuantizer{});
        } else if (py::isinstance<deglib::transformation::ScalarQuantizerInt8>(quantizer)) {
            auto q = py::cast<deglib::transformation::ScalarQuantizerInt8>(quantizer);
            make_searcher_for_quant(std::move(q));
        } else if (py::isinstance<deglib::transformation::ScalarQuantizerInt8PerDim>(quantizer)) {
            auto q = py::cast<deglib::transformation::ScalarQuantizerInt8PerDim>(quantizer);
            make_searcher_for_quant(std::move(q));
        } else if (py::isinstance<deglib::transformation::ScalarQuantizerUint8>(quantizer)) {
            auto q = py::cast<deglib::transformation::ScalarQuantizerUint8>(quantizer);
            make_searcher_for_quant(std::move(q));
        } else if (py::isinstance<deglib::transformation::ScalarQuantizerUint8PerDim>(quantizer)) {
            auto q = py::cast<deglib::transformation::ScalarQuantizerUint8PerDim>(quantizer);
            make_searcher_for_quant(std::move(q));
        } else if (py::isinstance<deglib::transformation::EvpQuantizer>(quantizer)) {
            auto q = py::cast<deglib::transformation::EvpQuantizer>(quantizer);
            make_searcher_for_quant(std::move(q));
        } else {
            throw std::invalid_argument("Unsupported quantizer type provided to create_searcher.");
        }
    }

    void optimize(uint32_t n_clusters = 256, uint32_t n_iter = 20, size_t sample_size = 0, uint32_t k = 100, uint32_t seed = 7, size_t num_threads = 1) {
        py::gil_scoped_release release;
        searcher_->optimize(n_clusters, n_iter, sample_size, k, seed, num_threads);
    }

    py::object search(py::array query, uint32_t k, float eps_or_ef = 0.1f, float rerank_factor = 1.0f, bool return_distances = false, bool unsorted = false) {
        auto buf = query.request();
        if (buf.ndim != 1 && (buf.ndim != 2 || (buf.shape[0] != 1 && buf.shape[1] != 1))) {
            throw std::invalid_argument("search query must be 1D vector (or 1xDim / Dimx1 2D array)");
        }
        py::array_t<uint32_t> result(static_cast<py::ssize_t>(k));
        uint32_t* out_ptr = static_cast<uint32_t*>(result.request().ptr);

        py::array_t<float> dist_result;
        float* dist_ptr = nullptr;
        if (return_distances) {
            dist_result = py::array_t<float>(static_cast<py::ssize_t>(k));
            dist_ptr = static_cast<float*>(dist_result.request().ptr);
        }

        uint32_t count = 0;
        if (eps_or_ef >= 1.0f) {
            const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                count = searcher_->search_ef_f16(static_cast<const uint16_t*>(buf.ptr), k, ef, rerank_factor, out_ptr, dist_ptr, unsorted);
            } else {
                count = searcher_->search_ef_f32(static_cast<const float*>(buf.ptr), k, ef, rerank_factor, out_ptr, dist_ptr, unsorted);
            }
        } else {
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                count = searcher_->search_f16(static_cast<const uint16_t*>(buf.ptr), k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, unsorted);
            } else {
                count = searcher_->search_f32(static_cast<const float*>(buf.ptr), k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, unsorted);
            }
        }

        if (return_distances) {
            return py::make_tuple(result, dist_result);
        }
        return result;
    }

    py::object search_batch(
        py::array queries,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false
    ) {
        auto buf = queries.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("search_batch queries must be 2D array");
        }
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
                        static_cast<const uint16_t*>(buf.ptr), n_queries, k, ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted
                    );
                } else {
                    searcher_->search_batch_ef_f32(
                        static_cast<const float*>(buf.ptr), n_queries, k, ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted
                    );
                }
            } else {
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    searcher_->search_batch_f16(
                        static_cast<const uint16_t*>(buf.ptr), n_queries, k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted
                    );
                } else {
                    searcher_->search_batch_f32(
                        static_cast<const float*>(buf.ptr), n_queries, k, eps_or_ef, rerank_factor, out_ptr, dist_ptr, num_threads, unsorted
                    );
                }
            }
        }

        if (return_distances) {
            return py::make_tuple(result, dist_result);
        }
        return result;
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

deglib::DynamicExplorationGraph load_readonly_graph_wrapper(const char* path) { return deglib::load_readonly_graph(path); }

deglib::DynamicExplorationGraph load_sizebounded_graph_wrapper(const char* path, const uint32_t capacity = 0) {
    return deglib::load_mutable_graph(path, capacity);
}

std::tuple<py::array_t<float>, float> mips_l2_transform_wrapper(py::array_t<float, py::array::c_style> vectors) {
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
        max_norm = deglib::transformation::mips_l2_transform(static_cast<const float*>(buf.ptr), count, dim, static_cast<float*>(out_buf.ptr));
    }
    return std::make_tuple(output, max_norm);
}

py::array_t<float> mips_l2_transform_query_wrapper(py::array_t<float, py::array::c_style> queries) {
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
        deglib::transformation::mips_l2_transform_query(static_cast<const float*>(buf.ptr), count, dim, static_cast<float*>(out_buf.ptr));
    }
    return output;
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
    std::optional<py::array> custom_features = std::nullopt
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

    return custom_feature_space.has_value() ? graph.to_readonly(custom_feature_space.value(), feat_ptr) : graph.to_readonly();
}

deglib::DynamicExplorationGraph dynamic_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt,
    const uint32_t chunk_size = 1024
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

    deglib::graph::DynamicGraph dynamic_graph =
        custom_feature_space.has_value() ? deglib::graph::convert_to_dynamic_graph(graph.internal(), custom_feature_space.value(), feat_ptr, chunk_size)
                                         : deglib::graph::convert_to_dynamic_graph(graph.internal(), chunk_size);

    auto heap_graph = std::make_unique<deglib::graph::DynamicGraph>(std::move(dynamic_graph));
    return deglib::DynamicExplorationGraph(std::move(heap_graph), graph.getTransformations());
}

deglib::DynamicExplorationGraph load_dynamic_graph_wrapper(const char* path, const uint32_t chunk_size = 1024) {
    return deglib::load_dynamic_graph(path, chunk_size);
}

deglib::DynamicExplorationGraph size_bounded_graph_from_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
    std::optional<py::array> custom_features = std::nullopt,
    const uint32_t new_max_size = 0
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

    deglib::graph::SizeBoundedGraph size_bounded =
        custom_feature_space.has_value() ? deglib::graph::convert_to_sizebounded_graph(graph.internal(), custom_feature_space.value(), feat_ptr, new_max_size)
                                         : deglib::graph::convert_to_sizebounded_graph(graph.internal());

    auto heap_graph = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(size_bounded));
    return deglib::DynamicExplorationGraph(std::move(heap_graph), graph.getTransformations());
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

deglib::analysis::GraphStats analyze_graph_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::analyze_graph(graph.internal()); }

// ============================================================================

PYBIND11_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    m.def("avx_usable", &deglib::cpu::has_avx2, "Returns whether AVX2 instructions are available");
    m.def("avx512_usable", &deglib::cpu::has_avx512, "Returns whether AVX512 instructions are available");

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
    cpu_module.def("has_avx_vnni", &deglib::cpu::has_avx_vnni, "Returns whether AVX-VNNI (256-bit) instructions are available");
    cpu_module.def("has_avx512", &deglib::cpu::has_avx512, "Returns whether AVX512 instructions are available");
    cpu_module.def("has_avx512_vnni", &deglib::cpu::has_avx512_vnni, "Returns whether AVX-512 VNNI instructions are available");

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
        .value("Int8_L2", deglib::distances::MetricType::Int8_L2);

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
        .def("get_instruction", &deglib::distances::FloatSpace::get_instruction)
        .def("compute_distance", &float_space_compute_distance, py::arg("vec1"), py::arg("vec2"))
        .def("compute_distances", &float_space_compute_distances, py::arg("query"), py::arg("targets"));

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
        [](deglib::DynamicExplorationGraph& graph, const size_t num_threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), num_threads);
        },
        py::arg("graph"), py::arg("num_threads") = 0, "Prune edges that violate the MRNG condition"
    );
    optimization_module.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t num_threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, num_threads);
        },
        py::arg("graph"), py::arg("prune_worst"), py::arg("num_threads") = 0, "Prune the worst edges of each vertex"
    );

    // transformation submodule
    py::module_ transformation_module = m.def_submodule("transformation", "Feature transformation and quantization");

    auto transform_single_array = [](const deglib::transformation::FeatureTransformation& t, py::array in_arr, bool is_database) -> py::array {
        py::buffer_info buf = in_arr.request();
        bool is_1d = (buf.ndim == 1);
        size_t count = is_1d ? 1 : buf.shape[0];
        size_t dim = is_1d ? buf.shape[0] : buf.shape[1];
        if (dim != t.input_dim()) {
            throw std::invalid_argument(std::format("Input dimension ({}) does not match transformation input_dim ({})", dim, t.input_dim()));
        }
        py::array_t<float, py::array::c_style> f_arr;
        const float* f_ptr = nullptr;
        if (buf.itemsize == sizeof(float) && (buf.format == "f" || buf.format == "float")) {
            f_ptr = static_cast<const float*>(buf.ptr);
        } else {
            f_arr = py::array_t<float, py::array::c_style>(in_arr);
            f_ptr = static_cast<const float*>(f_arr.request().ptr);
        }
        const size_t out_stride = t.output_bytes();
        const uint32_t tag = t.tag();
        if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQ8P) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<int8_t> out(shape);
            if (is_database)
                t.transform_database(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                t.transform_query(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            return out;
        } else if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_SQUP || tag == deglib::transformation::TAG_EVPQ) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<uint8_t> out(shape);
            if (is_database)
                t.transform_database(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                t.transform_query(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            return out;
        } else {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(t.output_dim())}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(t.output_dim())};
            py::array_t<float> out(shape);
            if (is_database)
                t.transform_database(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                t.transform_query(f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            return out;
        }
    };

    auto apply_transformations_array = [](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr,
                                          bool is_database) -> py::array {
        if (transformations.empty()) return in_arr;
        py::buffer_info buf = in_arr.request();
        bool is_1d = (buf.ndim == 1);
        size_t count = is_1d ? 1 : buf.shape[0];
        size_t dim = is_1d ? buf.shape[0] : buf.shape[1];
        if (dim != transformations.front()->input_dim()) {
            throw std::invalid_argument(
                std::format("Input dimension ({}) does not match transformation input_dim ({})", dim, transformations.front()->input_dim())
            );
        }
        py::array_t<float, py::array::c_style> f_arr;
        const float* f_ptr = nullptr;
        if (buf.itemsize == sizeof(float) && (buf.format == "f" || buf.format == "float")) {
            f_ptr = static_cast<const float*>(buf.ptr);
        } else {
            f_arr = py::array_t<float, py::array::c_style>(in_arr);
            f_ptr = static_cast<const float*>(f_arr.request().ptr);
        }
        const size_t out_stride = transformations.back()->output_bytes();
        const uint32_t tag = transformations.back()->tag();
        if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQ8P) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<int8_t> out(shape);
            if (is_database)
                deglib::transformation::apply_database(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                deglib::transformation::apply_query(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            return out;
        } else if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_SQUP || tag == deglib::transformation::TAG_EVPQ) {
            std::vector<py::ssize_t> shape = is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(out_stride)}
                                                   : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(out_stride)};
            py::array_t<uint8_t> out(shape);
            if (is_database)
                deglib::transformation::apply_database(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                deglib::transformation::apply_query(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            return out;
        } else {
            std::vector<py::ssize_t> shape =
                is_1d ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(transformations.back()->output_dim())}
                      : std::vector<py::ssize_t>{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(transformations.back()->output_dim())};
            py::array_t<float> out(shape);
            if (is_database)
                deglib::transformation::apply_database(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
            else
                deglib::transformation::apply_query(transformations, f_ptr, reinterpret_cast<std::byte*>(out.request().ptr), count);
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
        .def_property_readonly("input_bytes", &deglib::transformation::FeatureTransformation::input_bytes)
        .def_property_readonly("output_bytes", &deglib::transformation::FeatureTransformation::output_bytes)
        .def_property_readonly(
            "output_dtype",
            [](const deglib::transformation::FeatureTransformation& t) -> py::object {
                uint32_t tag = t.tag();
                py::module_ np = py::module_::import("numpy");
                if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQ8P) return np.attr("int8");
                if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_SQUP || tag == deglib::transformation::TAG_EVPQ)
                    return np.attr("uint8");
                return np.attr("float32");
            }
        )
        .def(
            "transform_database",
            [transform_single_array](const deglib::transformation::FeatureTransformation& t, py::array in_arr) {
                return transform_single_array(t, in_arr, true);
            },
            py::arg("vectors")
        )
        .def(
            "transform_query",
            [transform_single_array](const deglib::transformation::FeatureTransformation& t, py::array in_arr) {
                return transform_single_array(t, in_arr, false);
            },
            py::arg("queries")
        )
        .def("serialize", [](const deglib::transformation::FeatureTransformation& t) -> py::bytes {
            std::ostringstream ss(std::ios::binary);
            t.serialize(ss);
            return py::bytes(ss.str());
        });

    py::class_<deglib::transformation::MipsTransform, deglib::transformation::FeatureTransformation, std::shared_ptr<deglib::transformation::MipsTransform>>(
        transformation_module, "MipsTransform"
    )
        .def(py::init<uint32_t, float>(), py::arg("dim"), py::arg("max_norm") = 0.0f)
        .def_property_readonly("max_norm", &deglib::transformation::MipsTransform::max_norm)
        .def(
            "fit",
            [](deglib::transformation::MipsTransform& t, py::array_t<float, py::array::c_style> data) {
                auto buf = data.request();
                if (buf.ndim != 2) throw std::invalid_argument("data must be a 2D float32 array");
                t.fit(static_cast<const float*>(buf.ptr), buf.shape[0]);
            },
            py::arg("data")
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::MipsTransform::deserialize(ss);
        });

    py::class_<deglib::transformation::EvpQuantizer, deglib::transformation::FeatureTransformation, std::shared_ptr<deglib::transformation::EvpQuantizer>>(
        transformation_module, "EvpQuantizer"
    )
        .def(py::init<>())
        .def(py::init<uint32_t>(), py::arg("non_zeros"))
        .def(py::init<uint32_t, uint32_t>(), py::arg("dim"), py::arg("non_zeros"))
        .def_readwrite("dim", &deglib::transformation::EvpQuantizer::dim)
        .def_readwrite("non_zeros", &deglib::transformation::EvpQuantizer::non_zeros)
        .def_property_readonly(
            "mean",
            [](const deglib::transformation::EvpQuantizer& q) -> py::object {
                if (!q.has_mean()) return py::none();
                const auto& m = q.get_mean();
                py::array_t<float> arr(static_cast<py::ssize_t>(m.size()));
                std::memcpy(arr.request().ptr, m.data(), m.size() * sizeof(float));
                return arr;
            },
            "Current 1D mean vector subtracted during quantization, or None if not set"
        )
        .def("clear_mean", &deglib::transformation::EvpQuantizer::clear_mean, "Clears the mean vector")
        .def(
            "set_mean",
            [](deglib::transformation::EvpQuantizer& q, py::array_t<float, py::array::c_style> mean) {
                py::buffer_info info = mean.request();
                if (info.ndim != 1) {
                    throw std::invalid_argument("mean must be a 1D float32 array");
                }
                q.set_mean(std::span<const float>(static_cast<const float*>(info.ptr), info.shape[0]));
            },
            py::arg("mean"), "Set the 1D dimension-wise mean vector subtracted during quantization"
        )
        .def(
            "fit_mean",
            [](deglib::transformation::EvpQuantizer& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
                const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::gil_scoped_release release;
                if (is_fp16) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                }
            },
            py::arg("vectors"), py::arg("num_threads") = 0, "Computes and sets the 1D mean vector using parallel C++ SIMD"
        )
        .def(
            "fit",
            [](deglib::transformation::EvpQuantizer& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
                const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::gil_scoped_release release;
                if (is_fp16) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                }
            },
            py::arg("vectors"), py::arg("num_threads") = 0
        )
        .def(
            "quantize",
            [](deglib::transformation::EvpQuantizer& q, py::array vectors, size_t num_threads, std::optional<py::array_t<float, py::array::c_style>> mean) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                if (dim % 8 != 0) {
                    throw std::invalid_argument(std::format("Vector dimension ({}) must be a multiple of 8 for EVP quantization", dim));
                }
                if (mean.has_value() && !mean->is_none()) {
                    py::buffer_info mean_info = mean->request();
                    if (mean_info.ndim != 1) {
                        throw std::invalid_argument("mean must be a 1D float32 array");
                    }
                    q.set_mean(std::span<const float>(static_cast<const float*>(mean_info.ptr), mean_info.shape[0]));
                }
                const bool is_fp16 = (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e"));
                const bool is_fp32 = (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float"));
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }

                const size_t evp_bytes = 2 * (dim / 8);
                py::array_t<uint8_t> output({count, evp_bytes});
                py::buffer_info out_buf = output.request();
                std::byte* dst = static_cast<std::byte*>(out_buf.ptr);

                {
                    py::gil_scoped_release release;
                    if (is_fp16) {
                        q.quantize(static_cast<const uint16_t*>(buf.ptr), dst, count, dim, num_threads);
                    } else {
                        q.quantize(static_cast<const float*>(buf.ptr), dst, count, dim, num_threads);
                    }
                }
                return output;
            },
            py::arg("vectors"), py::arg("num_threads") = 0, py::arg("mean") = py::none()
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::EvpQuantizer::deserialize(ss);
        });

    py::class_<
        deglib::transformation::ScalarQuantizerInt8, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerInt8>>(transformation_module, "ScalarQuantizerInt8")
        .def(py::init<>())
        .def(py::init<float>(), py::arg("abs_max"))
        .def(py::init<uint32_t, float>(), py::arg("dim"), py::arg("abs_max") = 1.0f)
        .def_readwrite("dim", &deglib::transformation::ScalarQuantizerInt8::dim)
        .def_readwrite("abs_max", &deglib::transformation::ScalarQuantizerInt8::abs_max)
        .def_readwrite("scale", &deglib::transformation::ScalarQuantizerInt8::scale)
        .def_readonly("is_fitted", &deglib::transformation::ScalarQuantizerInt8::is_fitted)
        .def(
            "fit",
            [](deglib::transformation::ScalarQuantizerInt8& q, py::array vectors, float drop_ratio) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
            },
            py::arg("vectors"), py::arg("drop_ratio") = 0.0f
        )
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerInt8& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                std::vector<int8_t> result;
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    result = q.quantize(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    result = q.quantize(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::array_t<int8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::memcpy(out_buf.ptr, result.data(), count * dim * sizeof(int8_t));
                return output;
            },
            py::arg("vectors"), py::arg("num_threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::ScalarQuantizerInt8::deserialize(ss);
        });

    py::class_<
        deglib::transformation::ScalarQuantizerInt8PerDim, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerInt8PerDim>>(transformation_module, "ScalarQuantizerInt8PerDim")
        .def(py::init<>())
        .def(py::init<uint32_t>(), py::arg("dim"))
        .def(py::init<std::vector<float>>(), py::arg("abs_maxs"))
        .def(py::init<uint32_t, std::vector<float>>(), py::arg("dim"), py::arg("abs_maxs") = std::vector<float>{})
        .def_readwrite("dim", &deglib::transformation::ScalarQuantizerInt8PerDim::dim)
        .def_readwrite("abs_maxs", &deglib::transformation::ScalarQuantizerInt8PerDim::abs_maxs)
        .def_readwrite("scales", &deglib::transformation::ScalarQuantizerInt8PerDim::scales)
        .def_readonly("is_fitted", &deglib::transformation::ScalarQuantizerInt8PerDim::is_fitted)
        .def(
            "fit",
            [](deglib::transformation::ScalarQuantizerInt8PerDim& q, py::array vectors, float drop_ratio) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
            },
            py::arg("vectors"), py::arg("drop_ratio") = 0.0f
        )
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerInt8PerDim& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                std::vector<int8_t> result;
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    result = q.quantize(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    result = q.quantize(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::array_t<int8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::memcpy(out_buf.ptr, result.data(), count * dim * sizeof(int8_t));
                return output;
            },
            py::arg("vectors"), py::arg("num_threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::ScalarQuantizerInt8PerDim::deserialize(ss);
        });

    py::class_<
        deglib::transformation::ScalarQuantizerUint8, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerUint8>>(transformation_module, "ScalarQuantizerUint8")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("min_val"), py::arg("max_val"))
        .def(py::init<uint32_t, float, float>(), py::arg("dim"), py::arg("min_val") = 0.0f, py::arg("max_val") = 1.0f)
        .def_readwrite("dim", &deglib::transformation::ScalarQuantizerUint8::dim)
        .def_readwrite("min_val", &deglib::transformation::ScalarQuantizerUint8::min_val)
        .def_readwrite("max_val", &deglib::transformation::ScalarQuantizerUint8::max_val)
        .def_readwrite("dif", &deglib::transformation::ScalarQuantizerUint8::dif)
        .def_readwrite("scale", &deglib::transformation::ScalarQuantizerUint8::scale)
        .def_readonly("is_fitted", &deglib::transformation::ScalarQuantizerUint8::is_fitted)
        .def(
            "fit",
            [](deglib::transformation::ScalarQuantizerUint8& q, py::array vectors, float drop_ratio) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
            },
            py::arg("vectors"), py::arg("drop_ratio") = 0.0f
        )
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerUint8& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                std::vector<uint8_t> result;
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    result = q.quantize(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    result = q.quantize(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::array_t<uint8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::memcpy(out_buf.ptr, result.data(), count * dim * sizeof(uint8_t));
                return output;
            },
            py::arg("vectors"), py::arg("num_threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::ScalarQuantizerUint8::deserialize(ss);
        });

    py::class_<
        deglib::transformation::ScalarQuantizerUint8PerDim, deglib::transformation::FeatureTransformation,
        std::shared_ptr<deglib::transformation::ScalarQuantizerUint8PerDim>>(transformation_module, "ScalarQuantizerUint8PerDim")
        .def(py::init<>())
        .def(py::init<uint32_t>(), py::arg("dim"))
        .def(py::init<std::vector<float>, std::vector<float>>(), py::arg("mins"), py::arg("maxs"))
        .def(
            py::init<uint32_t, std::vector<float>, std::vector<float>>(), py::arg("dim"), py::arg("mins") = std::vector<float>{},
            py::arg("maxs") = std::vector<float>{}
        )
        .def_readwrite("dim", &deglib::transformation::ScalarQuantizerUint8PerDim::dim)
        .def_readwrite("mins", &deglib::transformation::ScalarQuantizerUint8PerDim::mins)
        .def_readwrite("maxs", &deglib::transformation::ScalarQuantizerUint8PerDim::maxs)
        .def_readwrite("scales", &deglib::transformation::ScalarQuantizerUint8PerDim::scales)
        .def_readonly("is_fitted", &deglib::transformation::ScalarQuantizerUint8PerDim::is_fitted)
        .def(
            "fit",
            [](deglib::transformation::ScalarQuantizerUint8PerDim& q, py::array vectors, float drop_ratio) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    q.fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    q.fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
            },
            py::arg("vectors"), py::arg("drop_ratio") = 0.0f
        )
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerUint8PerDim& q, py::array vectors, size_t num_threads) {
                py::buffer_info buf = vectors.request();
                auto [count, dim] = get_2d_shape(buf);
                std::vector<uint8_t> result;
                if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                    result = q.quantize(static_cast<const uint16_t*>(buf.ptr), count, dim, num_threads);
                } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                    result = q.quantize(static_cast<const float*>(buf.ptr), count, dim, num_threads);
                } else {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                py::array_t<uint8_t> output({count, static_cast<size_t>(dim)});
                py::buffer_info out_buf = output.request();
                std::memcpy(out_buf.ptr, result.data(), count * dim * sizeof(uint8_t));
                return output;
            },
            py::arg("vectors"), py::arg("num_threads") = 0
        )
        .def_static("deserialize", [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::ScalarQuantizerUint8PerDim::deserialize(ss);
        });

    transformation_module.def(
        "make_scalar_quantizer_int8",
        [](py::array vectors, float drop_ratio) {
            py::buffer_info buf = vectors.request();
            auto [count, dim] = get_2d_shape(buf);
            auto q = std::make_shared<deglib::transformation::ScalarQuantizerInt8>();
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                q->fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                q->fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("drop_ratio") = 0.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_int8_perdim",
        [](py::array vectors, float drop_ratio) {
            py::buffer_info buf = vectors.request();
            auto [count, dim] = get_2d_shape(buf);
            auto q = std::make_shared<deglib::transformation::ScalarQuantizerInt8PerDim>();
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                q->fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                q->fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("drop_ratio") = 0.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_uint8",
        [](py::array vectors, float drop_ratio) {
            py::buffer_info buf = vectors.request();
            auto [count, dim] = get_2d_shape(buf);
            auto q = std::make_shared<deglib::transformation::ScalarQuantizerUint8>();
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                q->fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                q->fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("drop_ratio") = 0.0f
    );

    transformation_module.def(
        "make_scalar_quantizer_uint8_perdim",
        [](py::array vectors, float drop_ratio) {
            py::buffer_info buf = vectors.request();
            auto [count, dim] = get_2d_shape(buf);
            auto q = std::make_shared<deglib::transformation::ScalarQuantizerUint8PerDim>();
            if (buf.itemsize == 2 && (buf.format == "H" || buf.format == "h" || buf.format == "e")) {
                q->fit(static_cast<const uint16_t*>(buf.ptr), count, dim, drop_ratio);
            } else if (buf.itemsize == 4 && (buf.format == "f" || buf.format == "float")) {
                q->fit(static_cast<const float*>(buf.ptr), count, dim, drop_ratio);
            } else {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            return q;
        },
        py::arg("vectors"), py::arg("drop_ratio") = 0.0f
    );

    transformation_module.def(
        "make_evp_quantizer", [](uint32_t non_zeros) { return std::make_shared<deglib::transformation::EvpQuantizer>(non_zeros); }, py::arg("non_zeros")
    );

    transformation_module.def("mips_l2_transform", &mips_l2_transform_wrapper, py::arg("vectors"));
    transformation_module.def("mips_l2_transform_query", &mips_l2_transform_query_wrapper, py::arg("queries"));

    transformation_module.def(
        "serialize_transformations",
        [](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations) -> py::bytes {
            std::ostringstream ss(std::ios::binary);
            deglib::transformation::serialize_transformations(transformations, ss);
            return py::bytes(ss.str());
        },
        py::arg("transformations")
    );

    transformation_module.def(
        "deserialize_transformations",
        [](py::bytes b) {
            std::string s = b;
            std::istringstream ss(s, std::ios::binary);
            return deglib::transformation::deserialize_transformations(ss);
        },
        py::arg("bytes")
    );

    transformation_module.def(
        "apply_database",
        [apply_transformations_array](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr) {
            return apply_transformations_array(transformations, in_arr, true);
        },
        py::arg("transformations"), py::arg("vectors")
    );

    transformation_module.def(
        "apply_query",
        [apply_transformations_array](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, py::array in_arr) {
            return apply_transformations_array(transformations, in_arr, false);
        },
        py::arg("transformations"), py::arg("queries")
    );

    m.attr("FeatureTransformation") = transformation_module.attr("FeatureTransformation");
    m.attr("MipsTransform") = transformation_module.attr("MipsTransform");
    m.attr("EvpQuantizer") = transformation_module.attr("EvpQuantizer");
    m.attr("ScalarQuantizerInt8") = transformation_module.attr("ScalarQuantizerInt8");
    m.attr("ScalarQuantizerUint8") = transformation_module.attr("ScalarQuantizerUint8");
    m.attr("ScalarQuantizerInt8PerDim") = transformation_module.attr("ScalarQuantizerInt8PerDim");
    m.attr("ScalarQuantizerUint8PerDim") = transformation_module.attr("ScalarQuantizerUint8PerDim");
    m.attr("make_evp_quantizer") = transformation_module.attr("make_evp_quantizer");
    m.attr("make_scalar_quantizer_int8") = transformation_module.attr("make_scalar_quantizer_int8");
    m.attr("make_scalar_quantizer_int8_perdim") = transformation_module.attr("make_scalar_quantizer_int8_perdim");
    m.attr("make_scalar_quantizer_uint8") = transformation_module.attr("make_scalar_quantizer_uint8");
    m.attr("make_scalar_quantizer_uint8_perdim") = transformation_module.attr("make_scalar_quantizer_uint8_perdim");

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

    py::class_<RerankerPy>(search_module, "Reranker")
        .def(py::init<const deglib::distances::FloatSpace&, py::array>(), py::arg("space"), py::arg("base_vectors"), py::keep_alive<1, 3>())
        .def("get_num_base_vectors", &RerankerPy::get_num_base_vectors)
        .def("get_po", &RerankerPy::get_po, "Prefetch lookahead offset (0 disables rerank prefetching).")
        .def("get_pl", &RerankerPy::get_pl, "Cache lines prefetched per feature vector (0 prefetches the whole vector).")
        .def(
            "set_prefetch", &RerankerPy::set_prefetch, py::arg("po"), py::arg("pl"),
            "Override the rerank prefetch parameters. Values <= 0 disable the respective knob."
        )
        .def(
            "optimize", &RerankerPy::optimize, py::arg("sample_count") = 50, py::arg("num_candidates") = 100, py::arg("k") = 10,
            "Auto-tune the reranker's prefetch parameters (po, pl) by timing reranking over sampled base vectors.\n\n"
            "Args:\n"
            "    sample_count: number of base vectors used as sample queries (capped at the base count).\n"
            "    num_candidates: candidates scored per query (clamped to the base count).\n"
            "    k: candidates kept per query after reranking (clamped to num_candidates)."
        )
        .def(
            "rerank", &RerankerPy::rerank, py::arg("queries"), py::arg("candidate_indices"), py::arg("k_top") = 0, py::arg("num_threads") = 0,
            py::arg("return_distances") = false, py::arg("unsorted") = false
        );

    py::class_<SearcherPy>(search_module, "Searcher")
        .def(
            py::init<const deglib::DynamicExplorationGraph&, py::object, std::optional<deglib::distances::FloatSpace>, std::optional<py::array>>(),
            py::arg("graph"), py::arg("quantizer") = py::none(), py::arg("rerank_space") = std::nullopt, py::arg("base_vectors") = std::nullopt,
            py::keep_alive<1, 2>()
        )
        .def(
            "search", &SearcherPy::search, py::arg("query"), py::arg("k"), py::arg("eps_or_ef") = 0.1f, py::arg("rerank_factor") = 1.0f,
            py::arg("return_distances") = false, py::arg("unsorted") = false,
            "Search for nearest neighbors of a single query vector.\n\n"
            "Args:\n"
            "    query: 1D query vector (or 1xDim 2D array).\n"
            "    k: Number of nearest neighbors to return.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    return_distances: If True, returns a tuple of (indices, distances).\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance."
        )
        .def(
            "search_batch", &SearcherPy::search_batch, py::arg("queries"), py::arg("k"), py::arg("eps_or_ef") = 0.1f, py::arg("rerank_factor") = 1.0f,
            py::arg("num_threads") = 1, py::arg("return_distances") = false, py::arg("unsorted") = false,
            "Search for nearest neighbors for a batch of query vectors.\n\n"
            "Args:\n"
            "    queries: 2D query array (shape N x Dim).\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    num_threads: Number of worker threads for parallel search.\n"
            "    return_distances: If True, returns a tuple of (indices, distances).\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance."
        )
        .def(
            "optimize", &SearcherPy::optimize, py::arg("n_clusters") = 256, py::arg("n_iter") = 20, py::arg("sample_size") = 0, py::arg("k") = 100,
            py::arg("seed") = 7, py::arg("num_threads") = 1,
            "Select entry vertices via k-means medoids.\n\nArgs:\n    n_clusters: number of entry vertices to select.\n    n_iter: number of k-means "
            "iterations.\n    sample_size: vertices sampled for clustering, 0 selects 3% of the graph size.\n    k: expected result count; sets the reranker "
            "and traversal prefetch tuning operating point.\n    seed: random seed for sampling and centroid init.\n    num_threads: worker threads, 0 selects "
            "a library default."
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
               std::optional<py::array> custom_features) {
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
                if (custom_feature_space.has_value() || feat_ptr != nullptr) {
                    const auto& fs = custom_feature_space.value_or(g.getFeatureSpace());
                    return g.to_readonly(fs, feat_ptr);
                } else {
                    return g.to_readonly();
                }
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none()
        )
        .def(
            "to_mutable",
            [](const deglib::DynamicExplorationGraph& g, std::optional<deglib::distances::FloatSpace> custom_feature_space,
               std::optional<py::array> custom_features, uint32_t new_max_size) {
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
                if (custom_feature_space.has_value() || feat_ptr != nullptr) {
                    const auto& fs = custom_feature_space.value_or(g.getFeatureSpace());
                    return g.to_mutable(fs, feat_ptr, new_max_size);
                } else {
                    return g.to_mutable(new_max_size);
                }
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none(), py::arg("new_max_size") = 0
        )
        .def(
            "to_dynamic",
            [](const deglib::DynamicExplorationGraph& g, std::optional<deglib::distances::FloatSpace> custom_feature_space,
               std::optional<py::array> custom_features, uint32_t chunk_size) {
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
                if (custom_feature_space.has_value() || feat_ptr != nullptr) {
                    const auto& fs = custom_feature_space.value_or(g.getFeatureSpace());
                    return g.to_dynamic(fs, feat_ptr, chunk_size);
                } else {
                    return g.to_dynamic(chunk_size);
                }
            },
            py::arg("custom_feature_space") = py::none(), py::arg("custom_features") = py::none(), py::arg("chunk_size") = 1024
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
            "has_path", &graph_has_path_wrapper<deglib::DynamicExplorationGraph>, py::arg("entry_external_labels"), py::arg("to_external_label"),
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
            "optimize", &deglib::DynamicExplorationGraph::optimize, py::arg("sample_count") = 50, py::arg("k") = 100, py::arg("ef") = 200, py::arg("seed") = 7,
            "Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.\n\n"
            "Args:\n"
            "    sample_count: number of vertices sampled as queries (capped at the graph size).\n"
            "    k: result count per traversal (the expected query operating point).\n"
            "    ef: beam width per traversal; clamped to at least k.\n"
            "    seed: random seed for query sampling."
        );

    m.def("load_readonly_graph", &load_readonly_graph_wrapper);
    m.def("load_dynamic_graph", &load_dynamic_graph_wrapper, py::arg("path"), py::arg("chunk_size") = 1024);
    m.def("load_sizebounded_graph", &load_sizebounded_graph_wrapper, py::arg("path"), py::arg("capacity") = 0);
    m.def(
        "read_only_graph_from_graph", &read_only_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none()
    );
    m.def(
        "dynamic_graph_from_graph", &dynamic_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none(), py::arg("chunk_size") = 1024
    );
    m.def(
        "size_bounded_graph_from_graph", &size_bounded_graph_from_graph_wrapper, py::arg("graph"), py::arg("custom_feature_space") = py::none(),
        py::arg("custom_features") = py::none(), py::arg("new_max_size") = 0
    );
    m.def(
        "presort", &presort_wrapper, py::arg("vectors"), py::arg("space"), py::arg("radius_decay") = 0.9f, py::arg("threads") = 0,
        py::arg("callback") = py::none()
    );
    m.def("mips_l2_transform", &mips_l2_transform_wrapper, py::arg("vectors"));
    m.def("mips_l2_transform_query", &mips_l2_transform_query_wrapper, py::arg("queries"));

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
                uint32_t seed_val = seed.value_or(5489);
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
        .def("set_thread_count", &deglib::builder::EvenRegularGraphBuilder::setThreadCount)
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
        .def_readonly("memory_bytes", &deglib::analysis::GraphStats::memory_bytes);
    m.def("analyze_graph", &analyze_graph_wrapper, py::arg("graph"));
    m.def(
        "prune_non_mrng_edges",
        [](deglib::DynamicExplorationGraph& graph, const size_t num_threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), num_threads);
        },
        py::arg("graph"), py::arg("num_threads") = 0
    );
    m.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t num_threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, num_threads);
        },
        py::arg("graph"), py::arg("prune_worst"), py::arg("num_threads") = 0
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
