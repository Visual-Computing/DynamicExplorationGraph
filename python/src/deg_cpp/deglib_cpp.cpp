//
// Python bindings for Dynamic Exploration Graph using nanobind
//

#include <deglib/deglib.h>
#include <deglib/concurrent.h>
#include <deglib/distance/fp16.h>
#include <deglib/optimization.h>
#include <deglib/optimization/pruning.h>
#include <deglib/transformation.h>
#include <deglib/tuning.h>

#include "quantizer_pipeline.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

// ============================================================================
// NDArray Type Aliases & Creation / Inspection Helpers
// ============================================================================

using InArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
using InFloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using InUint16Array = nb::ndarray<const uint16_t, nb::c_contig, nb::device::cpu>;
using InUint32Array = nb::ndarray<const uint32_t, nb::c_contig, nb::device::cpu>;
using InIntArray = nb::ndarray<const int, nb::c_contig, nb::device::cpu>;
using InStridedUint32Array = nb::ndarray<const uint32_t, nb::device::cpu>;

inline constexpr nb::dlpack::dtype kDtypeFloat16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};

template <typename T>
inline nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig> make_numpy_1d(size_t count, T*& out_ptr) {
    T* data = new T[count > 0 ? count : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[1] = {count};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig>(data, 1, shape, owner);
}

template <typename T>
inline nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig> make_numpy_2d(size_t rows, size_t cols, T*& out_ptr) {
    const size_t total = rows * cols;
    T* data = new T[total > 0 ? total : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[2] = {rows, cols};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig>(data, 2, shape, owner);
}

inline nb::ndarray<nb::numpy, nb::c_contig> make_numpy_fp16(size_t ndim, const size_t* shape, uint16_t*& out_ptr) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= shape[i];
    }
    uint16_t* data = new uint16_t[total > 0 ? total : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint16_t*>(p); });
    return nb::ndarray<nb::numpy, nb::c_contig>(data, ndim, shape, owner, nullptr, kDtypeFloat16);
}

inline bool is_dtype_float16_only(nb::dlpack::dtype dt) noexcept { return dt == kDtypeFloat16; }

inline bool is_dtype_fp16_or_u16(nb::dlpack::dtype dt) noexcept {
    return dt.bits == 16 && dt.lanes == 1 &&
           (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) ||
            dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int));
}

inline bool is_dtype_float32(nb::dlpack::dtype dt) noexcept { return dt == nb::dtype<float>(); }

inline std::pair<size_t, uint32_t> get_2d_shape(const InArray& arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected 2D array, got {} dimensions", arr.ndim()));
    }
    return {arr.shape(0), static_cast<uint32_t>(arr.shape(1))};
}

inline std::vector<float> convert_array_to_float32(const InArray& arr) {
    const size_t n = arr.size();
    std::vector<float> out(n);
    if (n == 0) return out;
    const auto dt = arr.dtype();
    const void* src = arr.data();
    if (dt.lanes != 1) {
        throw std::invalid_argument("Unsupported array dtype with lanes != 1");
    }
    const auto code = static_cast<nb::dlpack::dtype_code>(dt.code);
    if (code == nb::dlpack::dtype_code::Float) {
        if (dt.bits == 32) {
            std::memcpy(out.data(), src, n * sizeof(float));
            return out;
        }
        if (dt.bits == 16) {
            deglib::distances::fp16::fp16_to_floats(static_cast<const uint16_t*>(src), out.data(), n);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const double*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::Int) {
        if (dt.bits == 8) {
            const auto* p = static_cast<const int8_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 16) {
            const auto* p = static_cast<const int16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 32) {
            const auto* p = static_cast<const int32_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const int64_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::UInt) {
        if (dt.bits == 8) {
            const auto* p = static_cast<const uint8_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 16) {
            const auto* p = static_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 32) {
            const auto* p = static_cast<const uint32_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const uint64_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::Bool && dt.bits == 8) {
        const auto* p = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = p[i] ? 1.0f : 0.0f;
        return out;
    }
    throw std::invalid_argument(std::format("Unsupported array dtype (code: {}, bits: {}) for float32 conversion", dt.code, dt.bits));
}

inline const void* extract_custom_features_ptr(
    const deglib::DynamicExplorationGraph& graph,
    const std::optional<deglib::distances::FloatSpace>& custom_feature_space,
    const std::optional<InArray>& custom_features
) {
    if (!custom_features.has_value() || !custom_features->is_valid()) {
        return nullptr;
    }
    const void* feat_ptr = custom_features->data();
    if (custom_feature_space.has_value()) {
        const size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
        const size_t actual_bytes = custom_features->size() * custom_features->itemsize();
        if (actual_bytes != expected_bytes) {
            throw std::invalid_argument(
                std::format("custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes)
            );
        }
    }
    return feat_ptr;
}

// ============================================================================
// Graph Search, Explore & Path Wrappers
// ============================================================================

template <typename QuerySpanT>
deglib::search::SearchResults execute_graph_search_single(
    const deglib::DynamicExplorationGraph& graph,
    QuerySpanT q_span,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const bool return_distances,
    const bool unsorted
) {
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        return graph.search_ef(q_span, k, ef, /*include_entry=*/true, filter, max_distance_computation_count, return_distances, /*pad_missing=*/true);
    }
    return graph.search(q_span, k, eps_or_ef, filter, max_distance_computation_count, return_distances, unsorted, /*pad_missing=*/true);
}

deglib::search::SearchResults graph_search_single_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InArray& query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (query.size() != in_dim) {
            throw std::invalid_argument(std::format("Query vector size ({}) does not match transformations input dim ({})", query.size(), in_dim));
        }
        const auto dt = query.dtype();
        if (is_dtype_float16_only(dt)) {
            return execute_graph_search_single(
                graph, std::span<const uint16_t>(static_cast<const uint16_t*>(query.data()), in_dim), eps_or_ef, k, filter, max_distance_computation_count,
                return_distances, unsorted
            );
        } else if (is_dtype_float32(dt)) {
            return execute_graph_search_single(
                graph, std::span<const float>(static_cast<const float*>(query.data()), in_dim), eps_or_ef, k, filter, max_distance_computation_count,
                return_distances, unsorted
            );
        } else {
            throw std::invalid_argument(std::format("Unsupported query dtype (code: {}, bits: {})", dt.code, dt.bits));
        }
    }
    const size_t req_bytes = graph.getFeatureSpace().get_data_size();
    const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
    const size_t query_bytes = query.size() * query.itemsize();
    if (query_bytes != query_req_bytes && query_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_bytes, query_req_bytes, req_bytes
            )
        );
    }
    const std::byte* query_ptr = static_cast<const std::byte*>(query.data());
    std::span<const std::byte> query_span(query_ptr, query_bytes);
    return execute_graph_search_single(graph, query_span, eps_or_ef, k, filter, max_distance_computation_count, return_distances, unsorted);
}

// Batch search wrapper for DynamicExplorationGraph
deglib::search::SearchResults dynamic_exploration_graph_search_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InArray& query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (query.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected query to have two dimensions, got {}", query.ndim()));
    }

    const uint32_t n_queries = static_cast<uint32_t>(query.shape(0));
    const size_t q_cols = query.shape(1);
    const size_t q_itemsize = query.itemsize();
    const void* q_ptr = query.data();
    const auto q_dt = query.dtype();

    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (q_cols != in_dim) {
            throw std::invalid_argument(std::format("Query row dimension ({}) does not match transformations input dim ({})", q_cols, in_dim));
        }
    } else {
        const size_t req_bytes = graph.getFeatureSpace().get_data_size();
        const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
        const size_t query_stride_bytes = q_cols * q_itemsize;
        if (query_stride_bytes != query_req_bytes && query_stride_bytes != req_bytes) {
            throw std::invalid_argument(
                std::format(
                    "Query row size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_stride_bytes, query_req_bytes,
                    req_bytes
                )
            );
        }
    }

    nb::gil_scoped_release release;
    auto run_batch = [&](auto queries_span) -> deglib::search::SearchResults {
        if (eps_or_ef >= 1.0f) {
            const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
            return graph.search_batch_ef(queries_span, n_queries, k, ef, true, filter, max_distance_computation_count, threads, return_distances, unsorted);
        }
        return graph.search_batch(queries_span, n_queries, k, eps_or_ef, filter, max_distance_computation_count, threads, return_distances, unsorted);
    };

    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (is_dtype_float16_only(q_dt)) {
            return run_batch(std::span<const uint16_t>(static_cast<const uint16_t*>(q_ptr), static_cast<size_t>(n_queries) * in_dim));
        }
        return run_batch(std::span<const float>(static_cast<const float*>(q_ptr), static_cast<size_t>(n_queries) * in_dim));
    }
    const size_t query_stride_bytes = q_cols * q_itemsize;
    return run_batch(std::span<const std::byte>(static_cast<const std::byte*>(q_ptr), static_cast<size_t>(n_queries) * query_stride_bytes));
}

deglib::search::SearchResults graph_explore_single_wrapper(
    const deglib::DynamicExplorationGraph& graph,
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
        return graph.explore_ef(entry_external_label, k, ef, max_distance_computation_count, include_entry, filter, return_distances);
    }
    return graph.explore(entry_external_label, k, max_distance_computation_count, eps_or_ef, include_entry, filter, return_distances, unsorted);
}

template <typename G>
nb::object graph_has_path_wrapper(
    const G& graph,
    const std::vector<uint32_t>& from_external_labels,
    const uint32_t to_external_label,
    const float eps,
    const uint32_t k,
    const bool return_distances = true
) {
    const std::vector<deglib::graph::ObjectDistance> path = graph.hasPath(from_external_labels, to_external_label, eps, k);

    const size_t count = path.size();
    uint32_t* path_indices_ptr = nullptr;
    auto path_indices = make_numpy_1d<uint32_t>(count, path_indices_ptr);

    nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig> path_distances;
    float* path_distances_ptr = nullptr;
    if (return_distances) {
        path_distances = make_numpy_1d<float>(count, path_distances_ptr);
    }

    for (size_t i = 0; i < count; ++i) {
        path_indices_ptr[i] = path[i].getIdentifier();
        if (path_distances_ptr) {
            path_distances_ptr[i] = path[i].getDistance();
        }
    }

    if (return_distances) {
        return nb::make_tuple(path_indices, path_distances);
    }
    return path_indices.cast();
}

deglib::search::SearchResults dynamic_exploration_graph_explore_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InUint32Array& entry_external_labels,
    const uint32_t k,
    const uint32_t max_distance_computation_count,
    const float eps_or_ef,
    const bool include_entry,
    const deglib::search::Filter* filter,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (entry_external_labels.ndim() != 1) {
        throw std::invalid_argument(std::format("Expected entry_external_labels to have 1 dimension, got {}", entry_external_labels.ndim()));
    }

    const uint32_t* entry_ptr = entry_external_labels.data();
    const uint32_t n_queries = static_cast<uint32_t>(entry_external_labels.shape(0));
    std::span<const uint32_t> entries_span(entry_ptr, n_queries);

    nb::gil_scoped_release release;
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        return graph.explore_batch_ef(entries_span, k, ef, max_distance_computation_count, include_entry, filter, threads, return_distances, unsorted);
    }
    return graph.explore_batch(entries_span, k, max_distance_computation_count, eps_or_ef, include_entry, filter, threads, return_distances, unsorted);
}

// ============================================================================
// Distance Computations
// ============================================================================

float float_space_compute_distance(const deglib::distances::FloatSpace& space, const InArray& vec1, const InArray& vec2) {
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();
    const size_t vec1_bytes = vec1.size() * vec1.itemsize();
    const size_t vec2_bytes = vec2.size() * vec2.itemsize();
    if (vec1_bytes != query_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec1 size in bytes ({}) does not match required "
                "query size ({})",
                vec1_bytes, query_bytes
            )
        );
    }
    if (vec2_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec2 size in bytes ({}) does not match required "
                "feature space data size ({})",
                vec2_bytes, req_bytes
            )
        );
    }
    return deglib::distances::compute_distance(space, vec1.data(), vec2.data());
}

nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>
float_space_compute_distances(const deglib::distances::FloatSpace& space, const InArray& query, const InArray& targets, size_t threads) {
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();

    size_t num_queries = 0;
    size_t query_stride_bytes = 0;
    if (query.ndim() == 1) {
        num_queries = 1;
        query_stride_bytes = query.size() * query.itemsize();
    } else if (query.ndim() == 2) {
        num_queries = query.shape(0);
        query_stride_bytes = query.shape(1) * query.itemsize();
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
    if (targets.ndim() == 1) {
        num_targets = 1;
        target_stride_bytes = targets.size() * targets.itemsize();
    } else if (targets.ndim() == 2) {
        num_targets = targets.shape(0);
        target_stride_bytes = targets.shape(1) * targets.itemsize();
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

    float* out = nullptr;
    auto result = make_numpy_2d<float>(num_queries, num_targets, out);
    const size_t query_stride = query_stride_bytes;
    const void* qptr = query.data();
    const void* tptr = targets.data();

    if (num_queries == 1) {
        nb::gil_scoped_release release;
        deglib::distances::compute_distances(space, qptr, 1, tptr, num_targets, out);
    } else {
        nb::gil_scoped_release release;
        deglib::concurrent::parallel_for(0, num_queries, threads, [&](size_t q, size_t) {
            deglib::distances::compute_distances(space, static_cast<const uint8_t*>(qptr) + q * query_stride, 1, tptr, num_targets, out + q * num_targets);
        });
    }

    return result;
}

// ============================================================================
// Reranker Binding Helpers (thin marshaling around C++ owning Reranker)
// ============================================================================

static std::shared_ptr<deglib::search::Reranker> reranker_from_float32(
    const deglib::distances::FloatSpace& space,
    const InFloatArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const uint32_t expected_input_dim =
        (!transformations.empty() && transformations.front()) ? transformations.front()->input_dim() : static_cast<uint32_t>(space.dim());
    if (data.shape(1) != expected_input_dim) {
        throw std::invalid_argument(
            std::format("base_vectors column count ({}) does not match expected input dimension ({})", data.shape(1), expected_input_dim)
        );
    }
    const size_t n_rows = data.shape(0);
    const uint32_t dim = static_cast<uint32_t>(data.shape(1));
    std::span<const float> data_span(data.data(), n_rows * dim);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(deglib::search::Reranker::from_data(space, data_span, std::move(transformations), threads, idx_span));
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_fp16(
    const deglib::distances::FloatSpace& space,
    const InArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    if (!is_dtype_fp16_or_u16(data.dtype())) {
        throw std::invalid_argument("reranker_from_fp16: expected float16 or uint16 array");
    }
    const uint32_t expected_input_dim =
        (!transformations.empty() && transformations.front()) ? transformations.front()->input_dim() : static_cast<uint32_t>(space.dim());
    if (data.shape(1) != expected_input_dim) {
        throw std::invalid_argument(
            std::format("base_vectors column count ({}) does not match expected input dimension ({})", data.shape(1), expected_input_dim)
        );
    }
    const size_t n_rows = data.shape(0);
    const uint32_t dim = static_cast<uint32_t>(data.shape(1));
    std::span<const uint16_t> data_span(static_cast<const uint16_t*>(data.data()), n_rows * dim);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(deglib::search::Reranker::from_data(space, data_span, std::move(transformations), threads, idx_span));
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_raw(
    const deglib::distances::FloatSpace& space,
    const InArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const size_t req_bytes = space.get_data_size();
    const size_t stride_bytes = data.shape(1) * data.itemsize();
    if (stride_bytes != req_bytes) {
        throw std::invalid_argument(std::format("Base vector row size ({}) does not match feature space data size ({})", stride_bytes, req_bytes));
    }
    const size_t n_rows = data.shape(0);
    std::span<const std::byte> data_span(static_cast<const std::byte*>(data.data()), n_rows * stride_bytes);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_raw(space, data_span, n_rows, std::move(transformations), threads, idx_span)
    );
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_graph(
    const deglib::DynamicExplorationGraph& graph,
    const deglib::distances::FloatSpace& space,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0
) {
    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(graph.to_reranker(space, std::move(transformations), threads));
}

static nb::object reranker_rerank_py(
    const deglib::search::Reranker& reranker,
    const InArray& queries,
    const InUint32Array& candidate_indices,
    size_t k_top = 0,
    size_t num_threads = 0,
    bool return_distances = false,
    bool unsorted = false
) {
    if (queries.ndim() != 2) {
        throw std::invalid_argument("queries must be a 2D array");
    }
    const size_t n_queries = queries.shape(0);

    if (candidate_indices.ndim() != 2) {
        throw std::invalid_argument("candidate_indices must be a 2D array");
    }
    if (candidate_indices.shape(0) != n_queries) {
        throw std::invalid_argument("Number of candidate rows must match number of queries");
    }

    const size_t evp_k = candidate_indices.shape(1);
    if (k_top == 0 || k_top > evp_k) {
        k_top = evp_k;
    }

    const bool is_f32_queries = is_dtype_float32(queries.dtype());
    const size_t q_dim = queries.shape(1);
    const size_t query_row_bytes = q_dim * queries.itemsize();
    const size_t req_bytes = reranker.getSpace().get_data_size();
    const bool use_f32_path =
        is_f32_queries && (!reranker.getTransformations().empty() || reranker.getSpace().metric().get_data_type() == deglib::distances::MetricDataType::FP32);

    if (!use_f32_path && query_row_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format("Query row size in bytes ({}) does not match required feature space data size ({})", query_row_bytes, req_bytes)
        );
    }

    uint32_t* ind_ptr = nullptr;
    auto result_indices = make_numpy_2d<uint32_t>(n_queries, k_top, ind_ptr);

    nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig> result_distances;
    float* dist_ptr = nullptr;
    if (return_distances) {
        result_distances = make_numpy_2d<float>(n_queries, k_top, dist_ptr);
    }

    const uint32_t* cand_ptr = candidate_indices.data();
    const void* q_raw_ptr = queries.data();

    if (use_f32_path) {
        const float* q_ptr = static_cast<const float*>(q_raw_ptr);
        nb::gil_scoped_release release;
        deglib::concurrent::parallel_batch_for(
            size_t{0}, n_queries, num_threads,
            [&](size_t begin, size_t end, size_t) {
                for (size_t q = begin; q < end; ++q) {
                    reranker.rerank_f32(
                        std::span<const float>(q_ptr + q * q_dim, q_dim), std::span<const uint32_t>(cand_ptr + q * evp_k, evp_k), static_cast<uint32_t>(k_top),
                        std::span<uint32_t>(ind_ptr + q * k_top, k_top), return_distances ? std::span<float>(dist_ptr + q * k_top, k_top) : std::span<float>{},
                        return_distances, unsorted
                    );
                }
            },
            64
        );
    } else {
        nb::gil_scoped_release release;
        reranker.rerank(
            std::span<const std::byte>(static_cast<const std::byte*>(q_raw_ptr), n_queries * req_bytes), n_queries,
            std::span<const uint32_t>(cand_ptr, n_queries * evp_k), evp_k, static_cast<uint32_t>(k_top), std::span<uint32_t>(ind_ptr, n_queries * k_top),
            return_distances ? std::span<float>(dist_ptr, n_queries * k_top) : std::span<float>{}, return_distances, unsorted, num_threads
        );
    }

    if (return_distances) {
        return nb::make_tuple(result_indices, result_distances);
    }
    return result_indices.cast();
}

// ============================================================================
// Fast Searcher Wrapper
// ============================================================================

inline deglib::search::RecallTable::Mode parse_recall_mode_py(const std::string& mode) {
    if (mode == "ef") {
        return deglib::search::RecallTable::Mode::Ef;
    }
    if (mode == "eps") {
        return deglib::search::RecallTable::Mode::Eps;
    }
    throw std::invalid_argument(std::format("RecallTable mode must be 'ef' or 'eps', got '{}'", mode));
}

class SearcherPy {
  private:
    std::unique_ptr<deglib::search::Searcher> searcher_;

  public:
    SearcherPy(
        const deglib::DynamicExplorationGraph& graph,
        std::shared_ptr<deglib::search::Reranker> shared_reranker = nullptr,
        bool normalize_query = false
    )
        : searcher_(std::make_unique<deglib::search::Searcher>(graph.to_searcher(std::move(shared_reranker), normalize_query))) {}

    bool get_normalize_query() const noexcept { return searcher_->normalize_query(); }
    void set_normalize_query(bool val) noexcept { searcher_->set_normalize_query(val); }

    void optimize(uint32_t n_clusters = 256, uint32_t n_iter = 20, size_t sample_size = 0, size_t threads = 0, bool use_routing = true) {
        nb::gil_scoped_release release;
        searcher_->optimize(n_clusters, n_iter, sample_size, threads, use_routing);
    }

    deglib::search::SearchResults search(
        const InArray& query,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) {
        const size_t ndim = query.ndim();
        const void* query_ptr = query.data();
        const auto dt = query.dtype();
        const bool is_f16 = is_dtype_fp16_or_u16(dt);
        if (!is_f16 && !is_dtype_float32(dt)) {
            throw std::invalid_argument(std::format("Unsupported query dtype (code: {}, bits: {})", dt.code, dt.bits));
        }

        // 1D query or 2D single-row/single-column query -> execute fast single-query search
        if (ndim == 1 || (ndim == 2 && (query.shape(0) == 1 || query.shape(1) == 1))) {
            const size_t q_len = query.size();
            if (is_f16) {
                return searcher_->search<uint16_t>(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(query_ptr), q_len), k, eps_or_ef, rerank_factor, return_distances, unsorted,
                    max_distance_computation_count
                );
            }
            return searcher_->search<float>(
                std::span<const float>(static_cast<const float*>(query_ptr), q_len), k, eps_or_ef, rerank_factor, return_distances, unsorted,
                max_distance_computation_count
            );
        }

        // 2D batch query
        if (ndim == 2) {
            const size_t n_queries = query.shape(0);
            const size_t q_cols = query.shape(1);
            nb::gil_scoped_release release;
            if (is_f16) {
                return searcher_->search_batch<uint16_t>(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(query_ptr), n_queries * q_cols), n_queries, k, eps_or_ef, rerank_factor, num_threads,
                    return_distances, unsorted, max_distance_computation_count
                );
            }
            return searcher_->search_batch<float>(
                std::span<const float>(static_cast<const float*>(query_ptr), n_queries * q_cols), n_queries, k, eps_or_ef, rerank_factor, num_threads,
                return_distances, unsorted, max_distance_computation_count
            );
        }

        throw std::invalid_argument("search query must be 1D vector or 2D batch array");
    }

    deglib::search::SearchResults search_batch(
        const InArray& queries,
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

    deglib::search::RecallTable to_recall_table(
        uint32_t k,
        const std::string& mode,
        float rerank_factor = 1.0f,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
        float recall_step = deglib::search::RecallTable::kDefaultRecallStep,
        std::optional<std::vector<float>> target_recalls = std::nullopt,
        size_t threads = 0
    ) const {
        const auto parsed_mode = parse_recall_mode_py(mode);
        nb::gil_scoped_release release;
        if (target_recalls.has_value()) {
            return searcher_->to_recall_table(k, parsed_mode, std::span<const float>(*target_recalls), rerank_factor, sample_count, threads);
        }
        return searcher_->to_recall_table(k, parsed_mode, rerank_factor, sample_count, recall_range, recall_step, threads);
    }

    std::vector<deglib::search::RecallTable> to_recall_tables(
        uint32_t k,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
        float recall_step = deglib::search::RecallTable::kDefaultRecallStep,
        std::optional<std::vector<float>> target_recalls = std::nullopt,
        std::optional<std::vector<float>> rerank_factors = std::nullopt,
        size_t threads = 0
    ) const {
        if (rerank_factors.has_value() && rerank_factors->empty()) {
            throw std::invalid_argument("rerank_factors must not be empty");
        }
        const std::span<const float> rf_span = rerank_factors.has_value() ? std::span<const float>(*rerank_factors) : std::span<const float>{};
        nb::gil_scoped_release release;
        if (target_recalls.has_value()) {
            return searcher_->to_recall_tables(k, std::span<const float>(*target_recalls), sample_count, rf_span, threads);
        }
        return searcher_->to_recall_tables(k, sample_count, recall_range, recall_step, rf_span, threads);
    }

    deglib::search::ParetoLine to_pareto_line(
        const std::vector<deglib::search::RecallTable>& recall_tables,
        size_t sample_count = 0,
        size_t rounds = deglib::search::ParetoLine::kDefaultRounds
    ) const {
        nb::gil_scoped_release release;
        return searcher_->to_pareto_line(std::span<const deglib::search::RecallTable>(recall_tables), sample_count, rounds);
    }
};

// ============================================================================
// GraphBuilder Wrapper (owns RNG alongside EvenRegularGraphBuilder)
// ============================================================================

class GraphBuilderPy {
  private:
    std::mt19937 rng_;
    deglib::builder::EvenRegularGraphBuilder builder_;

  public:
    GraphBuilderPy(
        deglib::DynamicExplorationGraph& graph,
        std::optional<uint32_t> seed,
        const deglib::builder::OptimizationTarget optimization_target,
        const uint8_t extend_k,
        const float extend_eps,
        const uint8_t improve_k,
        const float improve_eps,
        const uint8_t max_path_length,
        const uint32_t improve_tries
    )
        : rng_(seed.value_or(7)),
          builder_(
              [&]() -> deglib::DynamicExplorationGraph& {
                  if (!graph.isMutable()) {
                      throw std::runtime_error("Graph must be mutable to use GraphBuilder");
                  }
                  return graph;
              }(),
              rng_,
              optimization_target,
              extend_k,
              extend_eps,
              improve_k,
              improve_eps,
              max_path_length,
              improve_tries
          ) {}

    void add_entry(const InUint32Array& label, const InArray& feature) {
        if (label.ndim() != 1) {
            throw std::invalid_argument(std::format("Expected label to have 1 dimension, got {}", label.ndim()));
        }
        if (feature.ndim() != 2) {
            throw std::invalid_argument(std::format("Expected feature to have two dimensions, got {}", feature.ndim()));
        }
        const size_t num_entries = feature.shape(0);
        if (label.shape(0) != num_entries) {
            throw std::invalid_argument("Number of labels must match number of feature rows");
        }
        const uint32_t* label_ptr = label.data();
        const size_t dim = feature.shape(1);
        const size_t itemsize = feature.itemsize();
        const bool is_f32 = is_dtype_float32(feature.dtype());
        const void* raw_ptr = feature.data();

        nb::gil_scoped_release release;
        if (builder_.hasTransformations() && is_f32) {
            const float* f_ptr = static_cast<const float*>(raw_ptr);
            for (size_t i = 0; i < num_entries; ++i) {
                builder_.addEntry(label_ptr[i], std::span<const float>(f_ptr + i * dim, dim));
            }
        } else {
            const std::byte* feature_ptr = static_cast<const std::byte*>(raw_ptr);
            const size_t feature_len = itemsize * dim;
            for (size_t i = 0; i < num_entries; ++i) {
                std::vector<std::byte> feature_vec(feature_ptr + (feature_len * i), feature_ptr + (feature_len * (i + 1)));
                builder_.addEntry(label_ptr[i], std::move(feature_vec));
            }
        }
    }

    void remove_entry(const uint32_t label) { builder_.removeEntry(label); }
    size_t get_num_new_entries() { return builder_.getNumNewEntries(); }
    size_t get_num_remove_entries() { return builder_.getNumRemoveEntries(); }
    void set_threads(const uint32_t threads) { builder_.setThreadCount(threads); }
    void set_batch_size(const uint32_t tasks_per_batch, const uint32_t task_size) { builder_.setBatchSize(tasks_per_batch, task_size); }
    uint32_t get_batch_size() const { return builder_.getBatchSize(); }

    deglib::builder::BuilderStatus build(nb::object callback = nb::none(), const bool infinite = false) {
        nb::handle cb_handle = callback.is_none() ? nb::handle() : nb::handle(callback.ptr());
        std::function<void(deglib::builder::BuilderStatus&)> cb = nullptr;
        if (cb_handle.is_valid()) {
            cb = [cb_handle](deglib::builder::BuilderStatus& status) {
                nb::gil_scoped_acquire acquire;
                cb_handle(deglib::builder::BuilderStatus(status));
            };
        }
        nb::gil_scoped_release release;
        return builder_.build(cb, infinite);
    }

    deglib::builder::BuilderStatus build_silent(const bool infinite = false) {
        nb::gil_scoped_release release;
        return builder_.build([](deglib::builder::BuilderStatus&) {}, infinite);
    }

    void stop() { builder_.stop(); }
};

// ============================================================================
// FP16 Conversion Bindings
// ============================================================================

nb::ndarray<nb::numpy, uint16_t, nb::ndim<1>, nb::c_contig> floats_to_fp16_wrapper(const InFloatArray& floats) {
    if (floats.itemsize() != sizeof(float)) {
        throw std::invalid_argument("Input array must contain 32-bit floats (itemsize == 4)");
    }
    const size_t count = floats.size();
    uint16_t* out_ptr = nullptr;
    auto output = make_numpy_1d<uint16_t>(count, out_ptr);
    deglib::distances::fp16::floats_to_fp16(floats.data(), out_ptr, count);
    return output;
}

nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig> fp16_to_floats_wrapper(const InUint16Array& fp16_vals) {
    if (fp16_vals.itemsize() != sizeof(uint16_t)) {
        throw std::invalid_argument("Input array must contain 16-bit uint16/FP16 values (itemsize == 2)");
    }
    const size_t count = fp16_vals.size();
    float* out_ptr = nullptr;
    auto output = make_numpy_1d<float>(count, out_ptr);
    deglib::distances::fp16::fp16_to_floats(fp16_vals.data(), out_ptr, count);
    return output;
}

// ============================================================================
// Graph Loading / Creation / Optimization / Analysis Wrappers
// ============================================================================

deglib::DynamicExplorationGraph load_readonly_graph_wrapper(const char* path) { return deglib::DynamicExplorationGraph::load_readonly(path); }

deglib::DynamicExplorationGraph load_sizebounded_graph_wrapper(const char* path, const uint32_t capacity = 0) {
    return deglib::load_mutable_graph(path, capacity);
}

inline deglib::distances::MetricDataType parse_quant_data_type(const nb::object& quant) {
    if (nb::isinstance<deglib::distances::MetricType>(quant)) {
        return deglib::distances::Metric(nb::cast<deglib::distances::MetricType>(quant)).get_data_type();
    }
    if (nb::isinstance<nb::int_>(quant)) {
        return deglib::distances::Metric(static_cast<deglib::distances::MetricType>(nb::cast<uint8_t>(quant))).get_data_type();
    }
    if (nb::isinstance<nb::str>(quant)) {
        std::string q_lower = nb::cast<std::string>(quant);
        for (char& c : q_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (q_lower == "uint4") return deglib::distances::MetricDataType::Uint4;
        if (q_lower == "int4") return deglib::distances::MetricDataType::Int4;
        if (q_lower == "uint7") return deglib::distances::MetricDataType::Uint7;
        if (q_lower == "uint8") return deglib::distances::MetricDataType::Uint8;
        if (q_lower == "int8") return deglib::distances::MetricDataType::Int8;
        if (q_lower == "fp16") return deglib::distances::MetricDataType::FP16;
        if (q_lower == "evp") return deglib::distances::MetricDataType::EVP;
        throw std::invalid_argument("Unsupported quantizer data type: " + q_lower);
    }
    throw std::invalid_argument("quant must be a Metric or data type string (uint4, int4, uint7, uint8, int8, fp16, evp)");
}

nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig> presort_wrapper(
    const InArray& vectors,
    const deglib::distances::FloatSpace& space,
    float radius_decay = 0.9f,
    size_t threads = 0,
    nb::object callback = nb::none()
) {
    if (vectors.ndim() != 2) {
        throw std::invalid_argument("vectors must be a 2D float32/float16 array");
    }
    const auto vec_dt = vectors.dtype();
    if (!is_dtype_float32(vec_dt) && !is_dtype_float16_only(vec_dt)) {
        throw std::invalid_argument(std::format("presort expects a float32 or float16 array, got dtype code {} bits {}", vec_dt.code, vec_dt.bits));
    }
    const size_t count = vectors.shape(0);
    const size_t dim = vectors.shape(1);
    if (count > 0 && dim != space.dim()) {
        throw std::invalid_argument(std::format("Vector dimension ({}) does not match FloatSpace dimension ({})", dim, space.dim()));
    }

    PyObject* cb_ptr = callback.is_none() ? nullptr : callback.ptr();
    std::function<bool(float)> cb = nullptr;
    if (cb_ptr != nullptr) {
        cb = [cb_ptr](float progress) -> bool {
            nb::gil_scoped_acquire acquire;
            try {
                nb::object res = nb::handle(cb_ptr)(progress);
                if (!res.is_none() && nb::isinstance<nb::bool_>(res)) {
                    return nb::cast<bool>(res);
                }
            } catch (const std::exception&) {
                // Ignore callback exceptions during sorting loop
            }
            return false;
        };
    }

    const void* vec_ptr = vectors.data();
    std::vector<uint32_t> sorted_indices;
    {
        nb::gil_scoped_release release;
        sorted_indices = deglib::optimization::presort(vec_ptr, count, space, radius_decay, threads, cb);
    }

    uint32_t* out_ptr = nullptr;
    auto output = make_numpy_1d<uint32_t>(count, out_ptr);
    if (count > 0) {
        std::memcpy(out_ptr, sorted_indices.data(), count * sizeof(uint32_t));
    }
    return output;
}

deglib::DynamicExplorationGraph load_dynamic_graph_wrapper(const char* path, const uint32_t chunk_size = 1024) {
    return deglib::load_dynamic_graph(path, chunk_size);
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
    const InArray& features,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t seed,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    if (features.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", features.ndim()));
    }
    const uint32_t vertex_count = static_cast<uint32_t>(features.shape(0));
    const size_t req_bytes = feature_space.get_data_size();
    const size_t row_bytes = features.shape(1) * features.itemsize();
    if (row_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Feature row size in bytes ({}) does not match required feature space "
                "data size ({})",
                row_bytes, req_bytes
            )
        );
    }

    const std::byte* feature_data = static_cast<const std::byte*>(features.data());
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
    std::optional<InStridedUint32Array> base_top = std::nullopt,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    const uint32_t* ptr = nullptr;
    size_t rows = 0;
    size_t k = 0;
    size_t row_stride = 0;

    if (base_top.has_value() && base_top->is_valid() && base_top->size() > 0) {
        if (base_top->ndim() != 2) {
            throw std::invalid_argument("base_top must be a 2D numpy array");
        }
        rows = base_top->shape(0);
        k = base_top->shape(1);
        const int64_t col_stride = base_top->stride(1);
        if (col_stride != 1) {
            throw std::invalid_argument("base_top columns must be contiguous");
        }
        row_stride = static_cast<size_t>(base_top->stride(0));
        ptr = base_top->data();
    }

    nb::gil_scoped_release release;
    return deglib::analysis::analyze_graph(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
}

float calc_graph_quality_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InStridedUint32Array& base_top,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    if (base_top.ndim() != 2) {
        throw std::invalid_argument("base_top must be a 2D numpy array");
    }
    const size_t rows = base_top.shape(0);
    const size_t k = base_top.shape(1);
    const int64_t col_stride = base_top.stride(1);
    if (col_stride != 1) {
        throw std::invalid_argument("base_top columns must be contiguous");
    }
    const size_t row_stride = static_cast<size_t>(base_top.stride(0));
    const uint32_t* ptr = base_top.data();

    nb::gil_scoped_release release;
    return deglib::analysis::calc_graph_quality(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
}

// ============================================================================
// Transformation Binding Helpers
// ============================================================================

template <typename QuantT, typename OutElemT, bool Packed4Bit = false>
nb::ndarray<nb::numpy, OutElemT, nb::ndim<2>, nb::c_contig> quantize_scalar_2d(const QuantT& q, const InArray& vectors, size_t threads) {
    auto [count, dim] = get_2d_shape(vectors);
    const size_t out_cols = Packed4Bit ? ((static_cast<size_t>(dim) + 1) / 2) : static_cast<size_t>(dim);
    const auto dt = vectors.dtype();
    const bool is_fp16 = is_dtype_fp16_or_u16(dt);
    const bool is_fp32 = is_dtype_float32(dt);
    if (!is_fp16 && !is_fp32) {
        throw std::invalid_argument("vectors must be float32 or float16");
    }

    OutElemT* out_ptr = nullptr;
    auto output = make_numpy_2d<OutElemT>(count, out_cols, out_ptr);
    std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
    const void* src = vectors.data();

    nb::gil_scoped_release release;
    if (is_fp16) {
        q.transform_database(
            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * out_cols * sizeof(OutElemT)), threads
        );
    } else {
        q.transform_database(
            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * out_cols * sizeof(OutElemT)), threads
        );
    }
    return output;
}

template <typename T>
std::shared_ptr<T> deserialize_transform(nb::bytes b) {
    std::string s(b.c_str(), b.size());
    std::istringstream ss(s, std::ios::binary);
    return std::shared_ptr<T>(T::deserialize(ss));
}

template <typename QuantT>
std::shared_ptr<QuantT> make_scalar_quantizer_impl(const InArray& vectors, float quantile) {
    auto [count, dim] = get_2d_shape(vectors);
    if (count == 0 || dim == 0) {
        throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
    }
    const auto dt = vectors.dtype();
    const bool is_fp16 = is_dtype_fp16_or_u16(dt);
    const bool is_fp32 = is_dtype_float32(dt);
    if (!is_fp16 && !is_fp32) {
        throw std::invalid_argument("vectors must be float32 or float16");
    }
    const void* src = vectors.data();
    auto q = std::make_shared<QuantT>(dim);
    nb::gil_scoped_release release;
    if (is_fp16) {
        q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), quantile);
    } else {
        q->fit(std::span<const float>(static_cast<const float*>(src), count * dim), quantile);
    }
    return q;
}

// ============================================================================
// Module Definition
// ============================================================================

NB_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    // cpu submodule
    nb::module_ cpu_module = m.def_submodule("cpu", "CPU feature detection and InstructionSet");
    nb::enum_<deglib::cpu::InstructionSet>(cpu_module, "InstructionSet", nb::is_arithmetic())
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
    nb::module_ distances_module = m.def_submodule("distances", "Distance metrics and feature spaces");
    nb::enum_<deglib::distances::MetricType>(distances_module, "Metric", nb::is_arithmetic())
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
        .value("Uint7_InnerProduct", deglib::distances::MetricType::Uint7_InnerProduct)
        .value("Uint7_L2", deglib::distances::MetricType::Uint7_L2);

    nb::class_<deglib::distances::FloatSpace>(distances_module, "FloatSpace")
        .def(
            "__init__",
            [](deglib::distances::FloatSpace* self, const size_t dim, const deglib::distances::MetricType metric,
               const deglib::cpu::InstructionSet instruction) {
                new (self) deglib::distances::FloatSpace(dim, deglib::distances::Metric(metric), instruction);
            },
            nb::arg("dim"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("instruction") = deglib::cpu::InstructionSet::Auto
        )
        .def("dim", &deglib::distances::FloatSpace::dim)
        .def("metric", [](const deglib::distances::FloatSpace& fs) { return fs.metric().value; })
        .def("get_data_size", &deglib::distances::FloatSpace::get_data_size)
        .def("get_query_data_size", &deglib::distances::FloatSpace::get_query_data_size)
        .def("get_instruction", &deglib::distances::FloatSpace::get_instruction)
        .def("compute_distance", &float_space_compute_distance, nb::arg("vec1"), nb::arg("vec2"))
        .def("compute_distances", &float_space_compute_distances, nb::arg("query"), nb::arg("targets"), nb::arg("threads") = 0);

    // quantization functions in distances submodule
    distances_module.def("floats_to_fp16", &floats_to_fp16_wrapper, "Convert float32 array to FP16 (uint16_t)");
    distances_module.def("fp16_to_floats", &fp16_to_floats_wrapper, "Convert FP16 (uint16_t) array to float32");

    // optimization submodule
    nb::module_ optimization_module = m.def_submodule("optimization", "Graph optimization and layout utilities");
    optimization_module.def(
        "presort", &presort_wrapper, nb::arg("vectors"), nb::arg("space"), nb::arg("radius_decay") = 0.9f, nb::arg("threads") = 0,
        nb::arg("callback") = nb::none(), "Perform 1D pre-sorting of dataset feature vectors using FLAS"
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
    nb::module_ transformation_module = m.def_submodule("transformation", "Feature transformation and quantization");

    auto apply_transformations_array = [](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations,
                                          const InArray& in_arr, bool is_database, size_t threads = 0) -> nb::object {
        if (transformations.empty()) {
            return nb::cast(in_arr);
        }
        const size_t ndim = in_arr.ndim();
        if (ndim != 1 && ndim != 2) {
            throw std::invalid_argument(std::format("Expected 1D or 2D array, got {} dimensions", ndim));
        }
        const bool is_1d = (ndim == 1);
        const size_t count = is_1d ? 1 : in_arr.shape(0);
        const size_t dim = is_1d ? in_arr.shape(0) : in_arr.shape(1);
        if (dim != transformations.front()->input_dim()) {
            throw std::invalid_argument(
                std::format("Input dimension ({}) does not match transformation input_dim ({})", dim, transformations.front()->input_dim())
            );
        }

        const auto dt = in_arr.dtype();
        const bool is_fp16 = is_dtype_fp16_or_u16(dt);
        const bool is_fp32 = is_dtype_float32(dt);
        std::vector<float> f32_converted;
        const float* f_ptr = nullptr;
        const uint16_t* u16_ptr = nullptr;
        if (is_database && is_fp16) {
            u16_ptr = static_cast<const uint16_t*>(in_arr.data());
        } else if (is_fp32) {
            f_ptr = static_cast<const float*>(in_arr.data());
        } else {
            f32_converted = convert_array_to_float32(in_arr);
            f_ptr = f32_converted.data();
        }

        const size_t out_stride = transformations.back()->output_byte_size();
        auto run_transform = [&](std::byte* out_ptr) {
            nb::gil_scoped_release release;
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

        auto emit_typed = [&](auto type_tag, size_t cols) -> nb::object {
            using OutT = typename decltype(type_tag)::type;
            OutT* ptr = nullptr;
            if (is_1d) {
                auto out = make_numpy_1d<OutT>(cols, ptr);
                run_transform(reinterpret_cast<std::byte*>(ptr));
                return out.cast();
            }
            auto out = make_numpy_2d<OutT>(count, cols, ptr);
            run_transform(reinterpret_cast<std::byte*>(ptr));
            return out.cast();
        };

        const uint32_t tag = transformations.back()->tag();
        if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) {
            return emit_typed(std::type_identity<int8_t>{}, out_stride);
        } else if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
                   tag == deglib::transformation::TAG_SQI4) {
            return emit_typed(std::type_identity<uint8_t>{}, out_stride);
        } else if (tag == deglib::transformation::TAG_SQ16) {
            uint16_t* ptr = nullptr;
            const size_t out_dim = transformations.back()->output_dim();
            if (is_1d) {
                const size_t shape[1] = {out_dim};
                auto out = make_numpy_fp16(1, shape, ptr);
                run_transform(reinterpret_cast<std::byte*>(ptr));
                return out.cast();
            }
            const size_t shape[2] = {count, out_dim};
            auto out = make_numpy_fp16(2, shape, ptr);
            run_transform(reinterpret_cast<std::byte*>(ptr));
            return out.cast();
        } else {
            return emit_typed(std::type_identity<float>{}, transformations.back()->output_dim());
        }
    };

    nb::class_<deglib::transformation::FeatureTransformation>(transformation_module, "FeatureTransformation")
        .def_prop_ro("name", [](const deglib::transformation::FeatureTransformation& t) { return std::string(t.name()); })
        .def_prop_ro("tag", &deglib::transformation::FeatureTransformation::tag)
        .def_prop_ro("input_dim", &deglib::transformation::FeatureTransformation::input_dim)
        .def_prop_ro("output_dim", &deglib::transformation::FeatureTransformation::output_dim)
        .def_prop_ro("input_byte_size", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_prop_ro("output_byte_size", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_prop_ro("input_bytes", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_prop_ro("output_bytes", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_prop_ro("is_fitted", &deglib::transformation::FeatureTransformation::is_fitted)
        .def_prop_ro(
            "output_dtype",
            [](const deglib::transformation::FeatureTransformation& t) -> nb::object {
                const uint32_t tag = t.tag();
                nb::module_ np = nb::module_::import_("numpy");
                if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) return nb::object(np.attr("int8"));
                if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
                    tag == deglib::transformation::TAG_SQI4)
                    return nb::object(np.attr("uint8"));
                if (tag == deglib::transformation::TAG_SQ16) return nb::object(np.attr("float16"));
                return nb::object(np.attr("float32"));
            }
        )
        .def(
            "transform_database",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def(
            "quantize",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def(
            "transform_query",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr) {
                return apply_transformations_array({std::move(t)}, in_arr, false, 1);
            },
            nb::arg("queries")
        )
        .def(
            "dequantize",
            [](const deglib::transformation::FeatureTransformation& t, const InArray& quantized, size_t threads) -> nb::object {
                const size_t ndim = quantized.ndim();
                const bool is_1d = (ndim == 1);
                if (ndim != 1 && ndim != 2) {
                    throw std::invalid_argument(std::format("Expected 1D or 2D array, got {} dimensions", ndim));
                }
                const size_t count = is_1d ? 1 : quantized.shape(0);
                const size_t row_bytes = (is_1d ? quantized.shape(0) : quantized.shape(1)) * quantized.itemsize();
                if (row_bytes != t.output_byte_size()) {
                    throw std::invalid_argument(std::format("quantized row bytes ({}) does not match expected ({})", row_bytes, t.output_byte_size()));
                }
                const uint32_t out_dim = t.output_dim();
                const std::byte* src = static_cast<const std::byte*>(quantized.data());
                auto run_deq = [&](float* dst) {
                    if (count > 0) {
                        nb::gil_scoped_release release;
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
                };
                float* dst = nullptr;
                if (is_1d) {
                    auto output = make_numpy_1d<float>(out_dim, dst);
                    run_deq(dst);
                    return output.cast();
                }
                auto output = make_numpy_2d<float>(count, out_dim, dst);
                run_deq(dst);
                return output.cast();
            },
            nb::arg("quantized"), nb::arg("threads") = 0, "Dequantizes transformed vectors back to approximate float32 vectors."
        )
        .def("serialize", [](const deglib::transformation::FeatureTransformation& t) -> nb::bytes {
            std::ostringstream ss(std::ios::binary);
            t.serialize(ss);
            std::string s = ss.str();
            return nb::bytes(s.data(), s.size());
        });

    nb::class_<deglib::transformation::MipsTransform, deglib::transformation::FeatureTransformation>(transformation_module, "MipsTransform")
        .def_prop_ro("max_norm", &deglib::transformation::MipsTransform::max_norm)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::MipsTransform>);

    nb::class_<deglib::transformation::EvpQuantizer, deglib::transformation::FeatureTransformation>(transformation_module, "EvpQuantizer")
        .def_prop_ro("dim", &deglib::transformation::EvpQuantizer::dim, "Dimension of vectors handled by this quantizer")
        .def_prop_ro("non_zeros", &deglib::transformation::EvpQuantizer::non_zeros, "Number of non-zero bits retained per vector")
        .def_prop_ro("is_fitted", &deglib::transformation::EvpQuantizer::is_fitted, "True if non_zeros is calibrated/set and mean is computed")
        .def_prop_ro(
            "mean",
            [](const deglib::transformation::EvpQuantizer& q) -> nb::object {
                if (!q.has_mean()) return nb::none();
                auto m = q.mean();
                float* ptr = nullptr;
                auto arr = make_numpy_1d<float>(m.size(), ptr);
                if (!m.empty()) {
                    std::memcpy(ptr, m.data(), m.size() * sizeof(float));
                }
                return arr.cast();
            },
            "Current 1D mean vector subtracted during quantization, or None if not set"
        )
        .def(
            "quantize",
            [](deglib::transformation::EvpQuantizer& q, const InArray& vectors, size_t threads) {
                auto [count, dim] = get_2d_shape(vectors);
                if (q.dim() != 0 && q.dim() != dim) {
                    throw std::invalid_argument(std::format("EvpQuantizer: vector dim ({}) does not match quantizer dim ({})", dim, q.dim()));
                }
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                if (!q.is_fitted()) {
                    throw std::logic_error("EvpQuantizer: quantize called before fit()");
                }

                const size_t evp_bytes = 2 * ((dim + 7) / 8);
                uint8_t* out_ptr = nullptr;
                auto output = make_numpy_2d<uint8_t>(count, evp_bytes, out_ptr);
                std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
                const void* src = vectors.data();

                {
                    nb::gil_scoped_release release;
                    if (is_fp16) {
                        q.transform_database(
                            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * evp_bytes), threads
                        );
                    } else {
                        q.transform_database(
                            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * evp_bytes), threads
                        );
                    }
                }
                return output;
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def_static("deserialize", &deserialize_transform<deglib::transformation::EvpQuantizer>);

    nb::class_<deglib::transformation::ScalarQuantizerInt8, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerInt8")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerInt8::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerInt8::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerInt8::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerInt8::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerInt8, int8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerInt8>);

    nb::class_<deglib::transformation::ScalarQuantizerUint8, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint8")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint8::dim)
        .def_prop_ro("min_val", &deglib::transformation::ScalarQuantizerUint8::min_val)
        .def_prop_ro("max_val", &deglib::transformation::ScalarQuantizerUint8::max_val)
        .def_prop_ro("dif", &deglib::transformation::ScalarQuantizerUint8::dif)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint8::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint8::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint8, uint8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint8>);

    nb::class_<deglib::transformation::ScalarQuantizerInt4, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerInt4")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerInt4::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerInt4::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerInt4::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerInt4::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerInt4, uint8_t, true>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerInt4>);

    nb::class_<deglib::transformation::ScalarQuantizerUint7, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint7")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint7::dim)
        .def_prop_ro("min", &deglib::transformation::ScalarQuantizerUint7::min)
        .def_prop_ro("max", &deglib::transformation::ScalarQuantizerUint7::max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint7::scale)
        .def_prop_ro("inv_scale", &deglib::transformation::ScalarQuantizerUint7::inv_scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint7::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint7, int8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint7>);

    nb::class_<deglib::transformation::ScalarQuantizerUint4, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint4")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint4::dim)
        .def_prop_ro("min_val", &deglib::transformation::ScalarQuantizerUint4::min_val)
        .def_prop_ro("max_val", &deglib::transformation::ScalarQuantizerUint4::max_val)
        .def_prop_ro("dif", &deglib::transformation::ScalarQuantizerUint4::dif)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint4::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint4::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint4, uint8_t, true>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint4>);

    nb::class_<deglib::transformation::ScalarQuantizerFP16, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerFP16")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerFP16::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerFP16::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerFP16::scale)
        .def_prop_ro("inv_scale", &deglib::transformation::ScalarQuantizerFP16::inv_scale)
        .def_prop_ro("quantile", &deglib::transformation::ScalarQuantizerFP16::quantile)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerFP16::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerFP16& q, const InArray& vectors, size_t threads) {
                auto [count, dim] = get_2d_shape(vectors);
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                const size_t shape[2] = {count, static_cast<size_t>(dim)};
                uint16_t* out_ptr = nullptr;
                auto output = make_numpy_fp16(2, shape, out_ptr);
                std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
                const void* src = vectors.data();
                {
                    nb::gil_scoped_release release;
                    if (is_fp16) {
                        q.transform_database(
                            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t))
                        );
                    } else {
                        q.transform_database(
                            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t)),
                            threads
                        );
                    }
                }
                return output;
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerFP16>);

    nb::class_<deglib::transformation::L2NormalizeTransform, deglib::transformation::FeatureTransformation>(transformation_module, "L2NormalizeTransform")
        .def_prop_ro("dim", &deglib::transformation::L2NormalizeTransform::dim)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::L2NormalizeTransform>);

    nb::class_<deglib::transformation::FastHadamardRotation, deglib::transformation::FeatureTransformation>(transformation_module, "FastHadamardRotation")
        .def_prop_ro("rounds", &deglib::transformation::FastHadamardRotation::rounds)
        .def_prop_ro("dim", &deglib::transformation::FastHadamardRotation::dim)
        .def_prop_ro("seed", &deglib::transformation::FastHadamardRotation::seed)
        .def_prop_ro("is_enabled", &deglib::transformation::FastHadamardRotation::is_enabled)
        .def_prop_rw("enabled", &deglib::transformation::FastHadamardRotation::is_enabled, &deglib::transformation::FastHadamardRotation::set_enabled)
        .def_prop_ro("crest_factor", &deglib::transformation::FastHadamardRotation::crest_factor)
        .def_prop_ro("excess_kurtosis", &deglib::transformation::FastHadamardRotation::excess_kurtosis)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::FastHadamardRotation>);

    transformation_module.def(
        "make_scalar_quantizer_int8", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerInt8>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint8", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint8>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_int4", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerInt4>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint7", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint7>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint4", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint4>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f, "Factory function creating a ScalarQuantizerUint4 calibrated on vectors. If quantile < 0 (default), auto-optimizes."
    );
    transformation_module.def(
        "make_scalar_quantizer_fp16", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerFP16>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_quantizer_pipeline",
        [](const InArray& vectors, nb::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_quantizer_pipeline: vectors must be a non-empty 2D array with dim > 0");
            }
            const auto data_type = parse_quant_data_type(quant);
            const auto dt = vectors.dtype();
            const bool is_fp16 = is_dtype_fp16_or_u16(dt);
            const bool is_fp32 = is_dtype_float32(dt);
            if (!is_fp16 && !is_fp32) {
                std::vector<float> f_vec = convert_array_to_float32(vectors);
                nb::gil_scoped_release release;
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const float>(f_vec.data(), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank, threads, seed
                );
            }

            const void* src = vectors.data();
            nb::gil_scoped_release release;
            if (is_fp16) {
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank,
                    threads, seed
                );
            }
            return deglib::transformation::make_quantizer_pipeline(
                std::span<const float>(static_cast<const float*>(src), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank, threads, seed
            );
        },
        nb::arg("vectors"), nb::arg("quant"), nb::arg("quantile") = -1.0f, nb::arg("use_rerank") = false, nb::arg("threads") = 0, nb::arg("seed") = 7,
        "Builds a transformation pipeline [FastHadamardRotation, ScalarQuantizer] or [ScalarQuantizer] based on kurtosis."
    );

    transformation_module.def(
        "make_quantizer_pipeline_from_graph",
        [](const deglib::DynamicExplorationGraph& graph, nb::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed, bool normalize, bool mips) {
            const auto target_dt = parse_quant_data_type(quant);
            nb::gil_scoped_release release;
            return deglib_py::make_quantizer_pipeline_from_graph(
                graph.internal(), target_dt, quantile, use_rerank, threads, seed, normalize, mips
            );
        },
        nb::arg("graph"), nb::arg("quant"), nb::arg("quantile") = -1.0f, nb::arg("use_rerank") = false, nb::arg("threads") = 0, nb::arg("seed") = 7, nb::arg("normalize") = false, nb::arg("mips") = false,
        "Calibrate a quantizer pipeline directly from an FP32/FP16 graph's stored vectors."
    );

    transformation_module.def(
        "make_evp_quantizer",
        [](const InArray& vectors, uint32_t non_zeros, size_t threads) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
            }
            const auto dt = vectors.dtype();
            const bool is_fp16 = is_dtype_fp16_or_u16(dt);
            const bool is_fp32 = is_dtype_float32(dt);
            if (!is_fp16 && !is_fp32) {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            const void* src = vectors.data();
            auto q = std::make_shared<deglib::transformation::EvpQuantizer>(dim, non_zeros);
            nb::gil_scoped_release release;
            if (is_fp16) {
                q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), threads);
            } else {
                q->fit(std::span<const float>(static_cast<const float*>(src), count * dim), threads);
            }
            return q;
        },
        nb::arg("vectors"), nb::arg("non_zeros") = 0, nb::arg("threads") = 0, "Factory function creating an EvpQuantizer calibrated on vectors."
    );

    transformation_module.def(
        "make_mips_transform",
        [](const InArray& vectors, float max_norm) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_mips_transform: vectors must be a non-empty 2D array with dim > 0");
            }
            auto t = std::make_shared<deglib::transformation::MipsTransform>(dim, max_norm);
            if (max_norm <= 0.0f) {
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (is_fp32) {
                    const float* f_ptr = static_cast<const float*>(vectors.data());
                    nb::gil_scoped_release release;
                    t->fit(std::span<const float>(f_ptr, count * dim));
                } else if (is_fp16) {
                    const uint16_t* u_ptr = static_cast<const uint16_t*>(vectors.data());
                    std::vector<float> fp16_converted(count * dim);
                    nb::gil_scoped_release release;
                    deglib::distances::fp16::fp16_to_floats(u_ptr, fp16_converted.data(), count * dim);
                    t->fit(std::span<const float>(fp16_converted.data(), count * dim));
                } else {
                    std::vector<float> f_vec = convert_array_to_float32(vectors);
                    nb::gil_scoped_release release;
                    t->fit(std::span<const float>(f_vec.data(), count * dim));
                }
            }
            return t;
        },
        nb::arg("vectors"), nb::arg("max_norm") = 0.0f, "Factory function creating a MipsTransform (d -> d+1 spherical lift) fitted on vectors."
    );

    transformation_module.def(
        "make_l2_normalize_transform",
        [](uint32_t dim) {
            if (dim == 0) {
                throw std::invalid_argument("make_l2_normalize_transform: dim must be > 0");
            }
            return std::make_shared<deglib::transformation::L2NormalizeTransform>(dim);
        },
        nb::arg("dim"), "Factory function creating an L2NormalizeTransform."
    );

    transformation_module.def(
        "make_fast_hadamard_rotation",
        [](const InFloatArray& vectors, uint64_t seed, uint32_t rounds, bool auto_detect) {
            if (vectors.ndim() != 2) {
                throw std::invalid_argument(std::format("Expected 2D array, got {} dimensions", vectors.ndim()));
            }
            const size_t count = vectors.shape(0);
            const uint32_t dim = static_cast<uint32_t>(vectors.shape(1));
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_fast_hadamard_rotation: vectors must be a non-empty 2D array with dim > 0");
            }
            const float* f_ptr = vectors.data();
            auto t = std::make_shared<deglib::transformation::FastHadamardRotation>(dim, seed, rounds);
            {
                nb::gil_scoped_release release;
                t->fit(std::span<const float>(f_ptr, count * dim), auto_detect);
            }
            return t;
        },
        nb::arg("vectors"), nb::arg("seed") = 7, nb::arg("rounds") = 0, nb::arg("auto_detect") = true,
        "Factory function creating a FastHadamardRotation fitted on vectors."
    );

    transformation_module.def(
        "apply_on_database",
        [apply_transformations_array](
            const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, const InArray& in_arr, size_t threads
        ) { return apply_transformations_array(transformations, in_arr, true, threads); },
        nb::arg("transformations"), nb::arg("vectors"), nb::arg("threads") = 0
    );

    transformation_module.def(
        "apply_on_query",
        [apply_transformations_array](
            const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, const InArray& in_arr
        ) { return apply_transformations_array(transformations, in_arr, false); },
        nb::arg("transformations"), nb::arg("queries")
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
    nb::module_ search_module = m.def_submodule("search", "Search utilities including Filter and SearchResults");
    nb::class_<deglib::search::Filter>(search_module, "Filter");

    static uint32_t s_empty_u32 = 0;
    static float s_empty_f32 = 0.0f;
    nb::class_<deglib::search::SearchResults>(
        search_module, "SearchResults", "Unified search results holding zero-copy NumPy arrays (`indices`, optional `distances`) and traversal telemetry."
    )
        .def_prop_ro(
            "indices",
            [](nb::handle self_h) -> nb::object {
                auto& self = nb::cast<deglib::search::SearchResults&>(self_h);
                uint32_t* ptr = self.indices.empty() ? &s_empty_u32 : self.indices.data();
                if (self.is_batch) {
                    size_t shape[2] = {self.n_queries, static_cast<size_t>(self.k)};
                    return nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>, nb::c_contig>(ptr, 2, shape, self_h).cast();
                }
                size_t shape[1] = {self.indices.size()};
                return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig>(ptr, 1, shape, self_h).cast();
            },
            "Zero-copy NumPy uint32 ndarray of neighbor external labels (1D for single query, 2D for batch)."
        )
        .def_prop_ro(
            "distances",
            [](nb::handle self_h) -> nb::object {
                auto& self = nb::cast<deglib::search::SearchResults&>(self_h);
                if (!self.has_distances()) return nb::none();
                float* ptr = self.distances.empty() ? &s_empty_f32 : self.distances.data();
                if (self.is_batch) {
                    size_t shape[2] = {self.n_queries, static_cast<size_t>(self.k)};
                    return nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>(ptr, 2, shape, self_h).cast();
                }
                size_t shape[1] = {self.distances.size()};
                return nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig>(ptr, 1, shape, self_h).cast();
            },
            "Zero-copy NumPy float32 ndarray of neighbor distances, or None when return_distances=False."
        )
        .def_ro(
            "graph_distance_count", &deglib::search::SearchResults::graph_distance_count,
            "Total number of distance computations performed during graph traversal."
        )
        .def_ro(
            "rerank_distance_count", &deglib::search::SearchResults::rerank_distance_count,
            "Total number of exact distance computations performed during candidate reranking."
        )
        .def_ro(
            "graph_hop_count", &deglib::search::SearchResults::graph_hop_count, "Total number of graph vertex expansions (hops) performed during traversal."
        )
        .def("__len__", &deglib::search::SearchResults::size);

    m.attr("SearchResults") = search_module.attr("SearchResults");

    nb::class_<deglib::search::RecallTable>(search_module, "RecallTable", "Single-mode calibration table mapping target recall@k values to ef or eps.")
        .def_prop_ro("k", &deglib::search::RecallTable::k)
        .def_prop_ro("sample_count", &deglib::search::RecallTable::sample_count)
        .def_prop_ro("is_ef", [](const deglib::search::RecallTable& t) { return t.mode() == deglib::search::RecallTable::Mode::Ef; })
        .def_prop_ro("is_eps", [](const deglib::search::RecallTable& t) { return t.mode() == deglib::search::RecallTable::Mode::Eps; })
        .def_prop_ro("rerank_factor", &deglib::search::RecallTable::rerank_factor)
        .def_prop_ro("recall_bounds", &deglib::search::RecallTable::recall_bounds)
        .def_prop_ro(
            "entries",
            [](const deglib::search::RecallTable& t) {
                std::vector<std::tuple<float, float, float>> out;
                const auto& entries = t.entries();
                out.reserve(entries.size());
                for (const auto& e : entries) {
                    out.emplace_back(e.target_recall, e.eps_or_ef, e.measured_recall);
                }
                return out;
            }
        )
        .def("eps_or_ef_values", &deglib::search::RecallTable::eps_or_ef_values, "Returns deduplicated ascending calibrated eps_or_ef values.")
        .def(
            "param_for", &deglib::search::RecallTable::param_for, nb::arg("target_recall"), nb::arg("clamp") = false,
            "Looks up or linearly interpolates an eps_or_ef value for target_recall (returns None if out of bounds unless clamp=True)."
        )
        .def("__len__", &deglib::search::RecallTable::size)
        .def("__str__", &deglib::search::RecallTable::to_string)
        .def("__repr__", &deglib::search::RecallTable::to_string);

    nb::class_<deglib::search::ParetoLine>(search_module, "ParetoLine", "Unified recall-latency Pareto frontier combining one or more RecallTables.")
        .def_prop_ro("k", &deglib::search::ParetoLine::k)
        .def_prop_ro("sample_count", &deglib::search::ParetoLine::sample_count)
        .def_prop_ro("recall_bounds", &deglib::search::ParetoLine::recall_bounds)
        .def_prop_ro("recall_tables", [](const deglib::search::ParetoLine& line) { return line.recall_tables(); })
        .def_prop_ro(
            "entries",
            [](const deglib::search::ParetoLine& line) {
                std::vector<std::tuple<float, float, float, float, double, double>> out;
                const auto& entries = line.entries();
                out.reserve(entries.size());
                for (const auto& e : entries) {
                    out.emplace_back(e.target_recall, e.eps_or_ef, e.rerank_factor, e.measured_recall, e.latency_us, e.qps);
                }
                return out;
            }
        )
        .def("eps_or_ef_values", &deglib::search::ParetoLine::eps_or_ef_values, "Returns deduplicated eps_or_ef parameters along the Pareto frontier.")
        .def("params", &deglib::search::ParetoLine::params, "Returns deduplicated (eps_or_ef, rerank_factor) parameter pairs along the Pareto frontier.")
        .def(
            "param_for", &deglib::search::ParetoLine::param_for, nb::arg("target_recall"), nb::arg("clamp") = false,
            "Returns the (eps_or_ef, rerank_factor) pair of the frontier entry whose measured_recall is nearest to target_recall (returns None if out of "
            "bounds "
            "unless clamp=True)."
        )
        .def("__len__", &deglib::search::ParetoLine::size)
        .def("__str__", &deglib::search::ParetoLine::to_string)
        .def("__repr__", &deglib::search::ParetoLine::to_string);

    search_module.def("create_filter", [](const InIntArray& valid_labels, size_t max_value, size_t max_label_count) {
        if (valid_labels.ndim() != 1) {
            throw std::invalid_argument(std::format("Expected feature to have only one dimension, got {}", valid_labels.ndim()));
        }
        const int* ptr = valid_labels.data();
        const size_t size = valid_labels.shape(0);
        return std::make_unique<deglib::search::Filter>(ptr, size, max_value, max_label_count);
    });

    nb::class_<deglib::search::Reranker>(search_module, "Reranker", "Owning C++ candidate reranker.")
        .def_static(
            "from_float32", &reranker_from_float32, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_fp16", &reranker_from_fp16, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_raw", &reranker_from_raw, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_graph", &reranker_from_graph, nb::arg("graph"), nb::arg("space"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0
        )
        .def("get_num_base_vectors", &deglib::search::Reranker::getNumBaseVectors)
        .def("get_dim", [](const deglib::search::Reranker& r) { return r.getSpace().dim(); })
        .def("get_instruction", [](const deglib::search::Reranker& r) { return r.getSpace().get_instruction(); })
        .def("get_po", &deglib::search::Reranker::getPo, "Prefetch lookahead offset (0 disables rerank prefetching).")
        .def("get_pl", &deglib::search::Reranker::getPl, "Cache lines prefetched per feature vector (0 prefetches the whole vector).")
        .def(
            "optimize",
            [](deglib::search::Reranker& r, size_t sample_count, size_t num_candidates) {
                nb::gil_scoped_release release;
                r.optimize(sample_count, num_candidates);
            },
            nb::arg("sample_count") = 300, nb::arg("num_candidates") = 150,
            "Auto-tune the reranker's prefetch parameters (po, pl) by timing reranking over sampled base vectors."
        )
        .def(
            "rerank", &reranker_rerank_py, nb::arg("queries"), nb::arg("candidate_indices"), nb::arg("k_top") = 0, nb::arg("threads") = 0,
            nb::arg("return_distances") = false, nb::arg("unsorted") = false
        );

    nb::class_<SearcherPy>(search_module, "Searcher", "Search pipeline. Results are the graph's labels.")
        .def(
            nb::init<const deglib::DynamicExplorationGraph&, std::shared_ptr<deglib::search::Reranker>, bool>(), nb::arg("graph"),
            nb::arg("reranker") = nullptr, nb::arg("normalize_query") = false, nb::keep_alive<1, 2>(),
            "Construct a Searcher with optional shared owning Reranker."
        )
        .def_prop_rw("normalize_query", &SearcherPy::get_normalize_query, &SearcherPy::set_normalize_query)
        .def(
            "search", &SearcherPy::search, nb::arg("query"), nb::arg("k") = 10, nb::arg("eps_or_ef") = 0.1f, nb::arg("rerank_factor") = 1.0f,
            nb::arg("threads") = 1, nb::arg("return_distances") = false, nb::arg("unsorted") = false, nb::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors of a single query vector (1D) or batch of query vectors (2D).\n\n"
            "Args:\n"
            "    query: 1D query vector or 2D batch array.\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for batch search (default: 1).\n"
            "    return_distances: If True, populates `.distances` on the returned SearchResults.\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited).\n\n"
            "Returns:\n"
            "    SearchResults: Object with zero-copy `.indices`, optional `.distances`, and telemetry counts."
        )
        .def(
            "search_batch", &SearcherPy::search_batch, nb::arg("queries"), nb::arg("k") = 10, nb::arg("eps_or_ef") = 0.1f, nb::arg("rerank_factor") = 1.0f,
            nb::arg("threads") = 1, nb::arg("return_distances") = false, nb::arg("unsorted") = false, nb::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors for a batch of query vectors.\n\n"
            "Args:\n"
            "    queries: 2D query array (shape N x Dim).\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for parallel search.\n"
            "    return_distances: If True, populates `.distances` on the returned SearchResults.\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited).\n\n"
            "Returns:\n"
            "    SearchResults: Object with zero-copy `.indices`, optional `.distances`, and telemetry counts."
        )
        .def(
            "optimize", &SearcherPy::optimize, nb::arg("n_clusters") = 256, nb::arg("n_iter") = 20, nb::arg("sample_size") = 0, nb::arg("threads") = 0,
            nb::arg("use_routing") = true,
            "Select entry vertices via k-means medoids and tune prefetch parameters.\n\nArgs:\n    n_clusters: number of entry vertices to select (maximum "
            "512).\n    n_iter: number of "
            "k-means "
            "iterations.\n    sample_size: vertices sampled for clustering, 0 selects 3% of the graph size.\n    threads: worker threads, 0 selects "
            "a library default.\n    use_routing: build a k-means routing graph for entry selection (used by the QG/query-graph searcher)."
        )
        .def(
            "to_recall_table", &SearcherPy::to_recall_table, nb::arg("k"), nb::arg("mode"), nb::arg("rerank_factor") = 1.0f,
            nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate a single-mode RecallTable ('ef' or 'eps') for this Searcher pipeline."
        )
        .def(
            "to_recall_tables", &SearcherPy::to_recall_tables, nb::arg("k"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(),
            nb::arg("rerank_factors") = nb::none(), nb::arg("threads") = 0,
            "Calibrate ('ef', 'eps') RecallTables across rerank_factors in a shared single-pass evaluation."
        )
        .def(
            "to_pareto_line", &SearcherPy::to_pareto_line, nb::arg("recall_tables"), nb::arg("sample_count") = 0,
            nb::arg("rounds") = deglib::search::ParetoLine::kDefaultRounds,
            "Build an end-to-end ParetoLine unifying measured operating points across RecallTables by (measured_recall, latency_us) dominance."
        );

    // graphs
    nb::class_<deglib::DynamicExplorationGraph>(m, "DynamicExplorationGraph")
        .def_static(
            "create_empty",
            [](const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space) {
                return deglib::create_empty(max_vertex_count, edges_per_vertex, feature_space);
            },
            nb::arg("max_vertex_count"), nb::arg("edges_per_vertex"), nb::arg("feature_space")
        )
        .def_static(
            "create_dynamic_empty",
            [](const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t chunk_size) {
                return deglib::create_dynamic_empty(edges_per_vertex, feature_space, chunk_size);
            },
            nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("chunk_size") = 1024
        )
        .def_static(
            "create_random_graph",
            [](const InArray& features, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t seed) {
                if (features.ndim() != 2) {
                    throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", features.ndim()));
                }
                const size_t req_bytes = feature_space.get_data_size();
                const size_t row_bytes = features.shape(1) * features.itemsize();
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
                    static_cast<const std::byte*>(features.data()), static_cast<uint32_t>(features.shape(0)), edges_per_vertex, feature_space, seed
                );
            },
            nb::arg("features"), nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("seed") = 7
        )
        .def("size", &deglib::DynamicExplorationGraph::size)
        .def("get_edges_per_vertex", &deglib::DynamicExplorationGraph::getEdgesPerVertex)
        .def("get_po", &deglib::DynamicExplorationGraph::getPo)
        .def("get_pl", &deglib::DynamicExplorationGraph::getPl)
        .def("get_nl", &deglib::DynamicExplorationGraph::getNl)
        .def(
            "get_feature_space", [](const deglib::DynamicExplorationGraph& g) -> const deglib::distances::FloatSpace& { return g.getFeatureSpace(); },
            nb::rv_policy::reference_internal
        )
        .def("has_vertex", &deglib::DynamicExplorationGraph::hasVertex)
        .def(
            "get_labels",
            [](const deglib::DynamicExplorationGraph& g) {
                auto labels = g.getExternalLabels();
                uint32_t* ptr = nullptr;
                auto arr = make_numpy_1d<uint32_t>(labels.size(), ptr);
                if (!labels.empty()) {
                    std::memcpy(ptr, labels.data(), labels.size() * sizeof(uint32_t));
                }
                return arr;
            },
            "Returns a 1D numpy array of all external labels ordered by their internal graph index (0 .. size()-1)."
        )
        .def(
            "get_neighbors",
            [](const deglib::DynamicExplorationGraph& g, uint32_t external_label) { return g.getNeighbors(external_label); },
            nb::arg("external_label")
        )
        .def(
            "get_neighbors",
            [](const deglib::DynamicExplorationGraph& g, const InUint32Array& labels, size_t threads) {
                if (labels.ndim() != 1) {
                    throw std::invalid_argument(std::format("Expected labels to have 1 dimension, got {}", labels.ndim()));
                }
                const uint32_t n_labels = static_cast<uint32_t>(labels.shape(0));
                const uint8_t edges = g.getEdgesPerVertex();
                uint32_t* out_ptr = nullptr;
                auto result = make_numpy_2d<uint32_t>(n_labels, static_cast<size_t>(edges), out_ptr);
                std::span<const uint32_t> labels_span(labels.data(), n_labels);
                std::span<uint32_t> out_span(out_ptr, static_cast<size_t>(n_labels) * edges);
                {
                    nb::gil_scoped_release release;
                    g.getNeighbors(labels_span, out_span, threads);
                }
                return result;
            },
            nb::arg("external_labels"), nb::arg("threads") = 1
        )
        .def("is_mutable", &deglib::DynamicExplorationGraph::isMutable)
        .def(
            "to_readonly",
            [](const deglib::DynamicExplorationGraph& g) {
                nb::gil_scoped_release release;
                return g.to_readonly();
            }
        )
        .def(
            "to_mutable",
            [](const deglib::DynamicExplorationGraph& g, uint32_t new_max_size, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.to_mutable(new_max_size, num_threads);
            },
            nb::arg("max_size") = 0, nb::arg("threads") = 0
        )
        .def(
            "to_dynamic",
            [](const deglib::DynamicExplorationGraph& g, uint32_t chunk_size, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.to_dynamic(chunk_size, num_threads);
            },
            nb::arg("chunk_size") = 1024, nb::arg("threads") = 0
        )
        .def(
            "with_metric",
            [](const deglib::DynamicExplorationGraph& g, const deglib::distances::FloatSpace& target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.with_metric(target_space, num_threads);
            },
            nb::arg("target_space"), nb::arg("threads") = 0
        )
        .def(
            "with_transformations",
            [](const deglib::DynamicExplorationGraph& g, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
               std::optional<deglib::distances::FloatSpace> target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.with_transformations(std::move(transformations), target_space, num_threads);
            },
            nb::arg("transformations"), nb::arg("target_space") = nb::none(), nb::arg("threads") = 0
        )
        .def(
            "with_features",
            [](const deglib::DynamicExplorationGraph& g, const deglib::distances::FloatSpace& target_space, InArray features,
               std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations, size_t num_threads) {
                const void* feat_ptr = extract_custom_features_ptr(g, target_space, features);
                nb::gil_scoped_release release;
                return g.with_features(target_space, feat_ptr, std::move(transformations), num_threads);
            },
            nb::arg("target_space"), nb::arg("features"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            nb::arg("threads") = 0
        )
        .def(
            "_transform_readonly",
            [](const deglib::DynamicExplorationGraph& g, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
               std::optional<deglib::distances::FloatSpace> target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                auto combined = g.getTransformations();
                combined.insert(combined.end(), transformations.begin(), transformations.end());
                auto ro = std::make_unique<deglib::graph::ReadOnlyGraph>(
                    deglib::graph::ReadOnlyGraph::from_graph(g.internal(), target_space, transformations, num_threads)
                );
                return deglib::DynamicExplorationGraph(std::move(ro), std::move(combined));
            },
            nb::arg("transformations"), nb::arg("target_space") = nb::none(), nb::arg("threads") = 0
        )
        .def("set_transformations", &deglib::DynamicExplorationGraph::setTransformations, nb::arg("transformations"))
        .def(
            "search", &graph_search_single_wrapper, nb::arg("query"), nb::arg("eps_or_ef"), nb::arg("k"), nb::arg("filter") = nullptr,
            nb::arg("max_distance_computation_count") = 0, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "search_batch", &dynamic_exploration_graph_search_batch_wrapper, nb::arg("query"), nb::arg("eps_or_ef"), nb::arg("k"), nb::arg("filter") = nullptr,
            nb::arg("max_distance_computation_count") = 0, nb::arg("threads") = 1, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "explore", &graph_explore_single_wrapper, nb::arg("entry_external_label"), nb::arg("k"), nb::arg("max_distance_computation_count") = 0,
            nb::arg("eps_or_ef") = 0.0f, nb::arg("include_entry") = true, nb::arg("filter") = nullptr, nb::arg("return_distances") = true,
            nb::arg("unsorted") = false
        )
        .def(
            "explore_batch", &dynamic_exploration_graph_explore_batch_wrapper, nb::arg("entry_external_labels"), nb::arg("k"),
            nb::arg("max_distance_computation_count") = 0, nb::arg("eps_or_ef") = 0.0f, nb::arg("include_entry") = true, nb::arg("filter") = nullptr,
            nb::arg("threads") = 1, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "has_path", &graph_has_path_wrapper<deglib::DynamicExplorationGraph>, nb::arg("from_external_labels"), nb::arg("to_external_label"),
            nb::arg("eps") = 0.0f, nb::arg("k") = 1, nb::arg("return_distances") = true
        )
        .def(
            "save_graph", [](deglib::DynamicExplorationGraph& g, const char* path) { g.saveGraph(path); }, nb::arg("path")
        )
        .def("has_transformations", &deglib::DynamicExplorationGraph::hasTransformations)
        .def("get_transformations", [](const deglib::DynamicExplorationGraph& g) { return g.getTransformations(); })
        .def("get_entry_vertex_indices", &deglib::DynamicExplorationGraph::getEntryVertexIndices)
        .def("set_entry_vertex_indices", &deglib::DynamicExplorationGraph::setEntryVertexIndices, nb::arg("indices"))
        .def(
            "optimize", &deglib::DynamicExplorationGraph::optimize, nb::arg("sample_count") = 300, nb::arg("median_subset") = 4096,
            "Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.\n\n"
            "Args:\n"
            "    sample_count: number of vertices sampled as queries (capped at the graph size).\n"
            "    median_subset: subset size for computing the graph-median entry point (<= 1 keeps native entries)."
        )
        .def(
            "to_recall_table",
            [](const deglib::DynamicExplorationGraph& g, uint32_t k, const std::string& mode, size_t sample_count, std::pair<float, float> recall_range,
               float recall_step, std::optional<std::vector<float>> target_recalls, size_t threads) {
                const auto parsed_mode = parse_recall_mode_py(mode);
                nb::gil_scoped_release release;
                if (target_recalls.has_value()) {
                    return g.to_recall_table(k, parsed_mode, std::span<const float>(*target_recalls), sample_count, threads);
                }
                return g.to_recall_table(k, parsed_mode, sample_count, recall_range, recall_step, threads);
            },
            nb::arg("k"), nb::arg("mode"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate a single-mode RecallTable ('ef' or 'eps') mapping target recall@k values to search parameters."
        )
        .def(
            "to_recall_tables",
            [](const deglib::DynamicExplorationGraph& g, uint32_t k, size_t sample_count, std::pair<float, float> recall_range, float recall_step,
               std::optional<std::vector<float>> target_recalls, size_t threads) {
                nb::gil_scoped_release release;
                if (target_recalls.has_value()) {
                    return g.to_recall_tables(k, std::span<const float>(*target_recalls), sample_count, threads);
                }
                return g.to_recall_tables(k, sample_count, recall_range, recall_step, threads);
            },
            nb::arg("k"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate both 'ef' and 'eps' RecallTables in one pass sharing exact ground truth."
        )
        .def(
            "to_pareto_line",
            [](const deglib::DynamicExplorationGraph& g, const std::vector<deglib::search::RecallTable>& recall_tables, size_t sample_count, size_t rounds) {
                nb::gil_scoped_release release;
                return g.to_pareto_line(std::span<const deglib::search::RecallTable>(recall_tables), sample_count, rounds);
            },
            nb::arg("recall_tables"), nb::arg("sample_count") = 0, nb::arg("rounds") = deglib::search::ParetoLine::kDefaultRounds,
            "Build a ParetoLine unifying measured operating points across one or more RecallTables by (measured_recall, latency_us) dominance."
        );

    m.def("load_readonly_graph", &load_readonly_graph_wrapper);
    m.def("load_dynamic_graph", &load_dynamic_graph_wrapper, nb::arg("path"), nb::arg("chunk_size") = 1024);
    m.def("load_sizebounded_graph", &load_sizebounded_graph_wrapper, nb::arg("path"), nb::arg("capacity") = 0);
    m.def(
        "presort", &presort_wrapper, nb::arg("vectors"), nb::arg("space"), nb::arg("radius_decay") = 0.9f, nb::arg("threads") = 0,
        nb::arg("callback") = nb::none()
    );

    m.def(
        "create_size_bounded_graph", &create_size_bounded_graph, nb::arg("max_vertex_count"), nb::arg("edges_per_vertex"), nb::arg("feature_space"),
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_dynamic_graph", &create_dynamic_graph, nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("chunk_size") = 1024,
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_random_graph", &create_random_graph, nb::arg("features"), nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("seed") = 7,
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );

    // GraphBuilder (renamed from EvenRegularGraphBuilder)
    nb::enum_<deglib::builder::OptimizationTarget>(m, "OptimizationTarget", nb::is_arithmetic())
        .value("StreamingData", deglib::builder::OptimizationTarget::StreamingData)
        .value("HighLID", deglib::builder::OptimizationTarget::HighLID)
        .value("LowLID", deglib::builder::OptimizationTarget::LowLID);

    nb::class_<GraphBuilderPy>(m, "GraphBuilder")
        .def(
            nb::init<
                deglib::DynamicExplorationGraph&, std::optional<uint32_t>, deglib::builder::OptimizationTarget, uint8_t, float, uint8_t, float, uint8_t,
                uint32_t>(),
            nb::arg("graph"), nb::arg("seed") = nb::none(), nb::arg("optimization_target") = deglib::builder::OptimizationTarget::LowLID,
            nb::arg("extend_k") = 0, nb::arg("extend_eps") = 0.1f, nb::arg("improve_k") = 0, nb::arg("improve_eps") = 0.001f, nb::arg("max_path_length") = 5,
            nb::arg("improve_tries") = 0, nb::keep_alive<1, 2>()
        )
        .def("add_entry", &GraphBuilderPy::add_entry, nb::arg("label"), nb::arg("feature"))
        .def("remove_entry", &GraphBuilderPy::remove_entry)
        .def("get_num_new_entries", &GraphBuilderPy::get_num_new_entries)
        .def("get_num_remove_entries", &GraphBuilderPy::get_num_remove_entries)
        .def("set_threads", &GraphBuilderPy::set_threads, nb::arg("threads"))
        .def("set_batch_size", &GraphBuilderPy::set_batch_size)
        .def("get_batch_size", &GraphBuilderPy::get_batch_size)
        .def("build", &GraphBuilderPy::build, nb::arg("callback") = nb::none(), nb::arg("infinite") = false)
        .def("build_silent", &GraphBuilderPy::build_silent, nb::arg("infinite") = false)
        .def("stop", &GraphBuilderPy::stop);

    m.def("calc_avg_edge_weight", &calc_avg_edge_weight_wrapper, nb::arg("graph"), nb::arg("scale") = 1);
    m.def("calc_edge_weight_histogram", &calc_edge_weight_histogram_wrapper, nb::arg("graph"), nb::arg("sort"), nb::arg("scale") = 1);
    m.def("check_graph_weights", &check_graph_weights_wrapper, nb::arg("graph"));
    m.def("check_graph_regularity", &check_graph_regularity_wrapper, nb::arg("graph"), nb::arg("expected_vertices"), nb::arg("check_back_link") = false);
    m.def("check_graph_connectivity", &check_graph_connectivity_wrapper, nb::arg("graph"));
    m.def("calc_non_mrng_edges", &calc_non_mrng_edges_wrapper, nb::arg("graph"));
    m.def("calc_search_reachability", &calc_search_reachability_wrapper, nb::arg("graph"));
    m.def("calc_exploration_reach", &calc_exploration_reach_wrapper, nb::arg("graph"));

    nb::class_<deglib::analysis::GraphStats>(m, "GraphStats", nb::dynamic_attr())
        .def_ro("vertex_count", &deglib::analysis::GraphStats::vertex_count)
        .def_ro("edge_count", &deglib::analysis::GraphStats::edge_count)
        .def_ro("feature_dims", &deglib::analysis::GraphStats::feature_dims)
        .def_ro("edges_per_vertex", &deglib::analysis::GraphStats::edges_per_vertex)
        .def_ro("avg_out_degree", &deglib::analysis::GraphStats::avg_out_degree)
        .def_ro("min_out_degree", &deglib::analysis::GraphStats::min_out_degree)
        .def_ro("max_out_degree", &deglib::analysis::GraphStats::max_out_degree)
        .def_ro("avg_in_degree", &deglib::analysis::GraphStats::avg_in_degree)
        .def_ro("min_in_degree", &deglib::analysis::GraphStats::min_in_degree)
        .def_ro("max_in_degree", &deglib::analysis::GraphStats::max_in_degree)
        .def_ro("source_vertices", &deglib::analysis::GraphStats::source_vertices)
        .def_ro("search_reachability", &deglib::analysis::GraphStats::search_reachability)
        .def_ro("exploration_reachability", &deglib::analysis::GraphStats::exploration_reachability)
        .def_ro("graph_quality", &deglib::analysis::GraphStats::graph_quality)
        .def_ro("memory_bytes", &deglib::analysis::GraphStats::memory_bytes);

    m.def("analyze_graph", &analyze_graph_wrapper, nb::arg("graph"), nb::arg("base_top") = nb::none(), nb::arg("sample_size") = 0, nb::arg("threads") = 0);
    m.def(
        "calc_graph_quality", &calc_graph_quality_wrapper, nb::arg("graph"), nb::arg("base_top"), nb::arg("sample_size") = 0, nb::arg("threads") = 0,
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
        nb::arg("graph"), nb::arg("threads") = 0
    );
    m.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, threads);
        },
        nb::arg("graph"), nb::arg("prune_worst"), nb::arg("threads") = 0
    );

    nb::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
        .def_rw("step", &deglib::builder::BuilderStatus::step)
        .def_rw("added", &deglib::builder::BuilderStatus::added)
        .def_rw("deleted", &deglib::builder::BuilderStatus::deleted)
        .def_rw("improved", &deglib::builder::BuilderStatus::improved)
        .def_rw("tries", &deglib::builder::BuilderStatus::tries)
        .def_ro("step_added_ids", &deglib::builder::BuilderStatus::step_added_ids)
        .def_ro("step_deleted_ids", &deglib::builder::BuilderStatus::step_deleted_ids)
        .def_ro("total_added_ids", &deglib::builder::BuilderStatus::total_added_ids)
        .def_ro("total_deleted_ids", &deglib::builder::BuilderStatus::total_deleted_ids);

    // tuning submodule
    nb::module_ tuning_module = m.def_submodule("tuning", "Pre-build hyperparameter prediction and dataset profiling");

    nb::class_<deglib::tuning::DatasetProfile>(tuning_module, "DatasetProfile", "Base-only geometric profile of a vector dataset.")
        .def(
            "__init__",
            [](deglib::tuning::DatasetProfile* self, uint32_t sample_count, uint32_t dim, float participation_ratio, float rel_contrast_1, float pp_iqr,
               float q_mode_u7, float q_mode_u4, deglib::distances::MetricType metric) {
                new (self) deglib::tuning::DatasetProfile{sample_count, dim,       participation_ratio, rel_contrast_1,
                                                          pp_iqr,       q_mode_u7, q_mode_u4,           deglib::distances::Metric(metric)};
            },
            nb::arg("sample_count") = 0, nb::arg("dim") = 0, nb::arg("participation_ratio") = 0.0f, nb::arg("rel_contrast_1") = 0.0f, nb::arg("pp_iqr") = 0.0f,
            nb::arg("q_mode_u7") = 0.0f, nb::arg("q_mode_u4") = 0.0f, nb::arg("metric") = deglib::distances::MetricType::FP32_L2
        )
        .def_ro("sample_count", &deglib::tuning::DatasetProfile::sample_count)
        .def_ro("dim", &deglib::tuning::DatasetProfile::dim)
        .def_ro("participation_ratio", &deglib::tuning::DatasetProfile::participation_ratio)
        .def_ro("rel_contrast_1", &deglib::tuning::DatasetProfile::rel_contrast_1)
        .def_ro("pp_iqr", &deglib::tuning::DatasetProfile::pp_iqr)
        .def_ro("q_mode_u7", &deglib::tuning::DatasetProfile::q_mode_u7)
        .def_ro("q_mode_u4", &deglib::tuning::DatasetProfile::q_mode_u4)
        .def_prop_ro("metric", [](const deglib::tuning::DatasetProfile& p) -> deglib::distances::MetricType { return p.metric.value; })
        .def("__repr__", [](const deglib::tuning::DatasetProfile& p) {
            return std::format(
                "DatasetProfile(sample_count={}, dim={}, participation_ratio={:.6f}, rel_contrast_1={:.6f}, pp_iqr={:.6f}, q_mode_u7={:.6f}, "
                "q_mode_u4={:.6f}, metric={})",
                p.sample_count, p.dim, p.participation_ratio, p.rel_contrast_1, p.pp_iqr, p.q_mode_u7, p.q_mode_u4, p.metric.to_string()
            );
        });

    nb::class_<deglib::tuning::TargetPrediction>(tuning_module, "TargetPrediction", "Predicted graph construction target and expected QPS advantage.")
        .def_ro("target", &deglib::tuning::TargetPrediction::target)
        .def_ro("lowlid_advantage_pct", &deglib::tuning::TargetPrediction::lowlid_advantage_pct)
        .def_ro("profile", &deglib::tuning::TargetPrediction::profile)
        .def("__repr__", [](const deglib::tuning::TargetPrediction& r) {
            const char* name = (r.target == deglib::builder::OptimizationTarget::LowLID)
                                   ? "LowLID"
                                   : ((r.target == deglib::builder::OptimizationTarget::HighLID) ? "HighLID" : "StreamingData");
            return std::format("TargetPrediction(target={}, lowlid_advantage_pct={:+.2f}%)", name, r.lowlid_advantage_pct);
        });

    nb::class_<deglib::tuning::QuantizationPrediction>(
        tuning_module, "QuantizationPrediction", "Predicted quantizer storage metric (Uint7_L2 vs Int8_InnerProduct) and expected QPS advantage."
    )
        .def_prop_ro("metric", [](const deglib::tuning::QuantizationPrediction& r) -> deglib::distances::MetricType { return r.metric.value; })
        .def_ro("uint7_advantage_pct", &deglib::tuning::QuantizationPrediction::uint7_advantage_pct)
        .def_ro("profile", &deglib::tuning::QuantizationPrediction::profile)
        .def("__repr__", [](const deglib::tuning::QuantizationPrediction& r) {
            return std::format("QuantizationPrediction(metric={}, uint7_advantage_pct={:+.2f}%)", r.metric.to_string(), r.uint7_advantage_pct);
        });

    auto measure_profile_py = [](const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed,
                                 size_t threads) -> deglib::tuning::DatasetProfile {
        auto [count, dim] = get_2d_shape(data);
        const auto dt = data.dtype();
        const bool is_fp32 = is_dtype_float32(dt);
        const bool is_fp16 = is_dtype_float16_only(dt);
        if (!is_fp32 && !is_fp16) {
            throw std::invalid_argument("data must be a 2D float32 or float16 array");
        }
        const void* ptr = data.data();
        const deglib::distances::Metric resolved_metric(metric);
        nb::gil_scoped_release release;
        if (is_fp16) {
            return deglib::tuning::measure_dataset_profile(
                std::span<const uint16_t>(static_cast<const uint16_t*>(ptr), count * static_cast<size_t>(dim)), dim, resolved_metric, sample_count, query_count,
                seed, threads
            );
        }
        return deglib::tuning::measure_dataset_profile(
            std::span<const float>(static_cast<const float*>(ptr), count * static_cast<size_t>(dim)), dim, resolved_metric, sample_count, query_count, seed,
            threads
        );
    };

    tuning_module.def(
        "measure_dataset_profile", measure_profile_py, nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2,
        nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount, nb::arg("query_count") = deglib::tuning::kDefaultQueryCount,
        nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure the base-only geometric DatasetProfile from a 2D float32 or float16 array."
    );
    tuning_module.def(
        "predict_optimization_target", [](const deglib::tuning::DatasetProfile& profile) { return deglib::tuning::predict_optimization_target(profile); },
        nb::arg("profile"), "Predict OptimizationTarget from a precomputed DatasetProfile."
    );
    tuning_module.def(
        "predict_optimization_target",
        [measure_profile_py](
            const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed, size_t threads
        ) {
            const auto profile = measure_profile_py(data, metric, sample_count, query_count, seed, threads);
            return deglib::tuning::predict_optimization_target(profile);
        },
        nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount,
        nb::arg("query_count") = deglib::tuning::kDefaultQueryCount, nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure a DatasetProfile and predict OptimizationTarget from a 2D float32 or float16 array."
    );
    tuning_module.def(
        "predict_quantization", [](const deglib::tuning::DatasetProfile& profile) { return deglib::tuning::predict_quantization(profile); }, nb::arg("profile"),
        "Predict quantizer Metric (Uint7_L2 vs Int8_InnerProduct) from a precomputed DatasetProfile."
    );
    tuning_module.def(
        "predict_quantization",
        [measure_profile_py](
            const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed, size_t threads
        ) {
            const auto profile = measure_profile_py(data, metric, sample_count, query_count, seed, threads);
            return deglib::tuning::predict_quantization(profile);
        },
        nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount,
        nb::arg("query_count") = deglib::tuning::kDefaultQueryCount, nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure a DatasetProfile and predict quantizer Metric (Uint7_L2 vs Int8_InnerProduct) from a 2D float32 or float16 array."
    );
}
