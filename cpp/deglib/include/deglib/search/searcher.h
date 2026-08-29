#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/optimization/quantization/scalar_quantize.h"
#include "deglib/search.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace deglib::search {

// ============================================================================
// Search Result Struct for Modern C++20 API
// ============================================================================

struct SearchResult {
    std::vector<uint32_t> indices;
    std::vector<float> distances;

    size_t size() const noexcept { return indices.size(); }
    bool empty() const noexcept { return indices.empty(); }
    bool has_distances() const noexcept { return !distances.empty(); }
};

// ============================================================================
// Quantizer Concept Wrappers
// ============================================================================

struct NoQuantizer {
    template <typename InT, typename OutByteT>
    static inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) {
        std::memcpy(out_bytes, in, dim * sizeof(InT));
    }
};

struct ScalarInt8Quantizer {
    deglib::quantization::scalar::ScalarQuantizerInt8 q;
    explicit ScalarInt8Quantizer(deglib::quantization::scalar::ScalarQuantizerInt8 quant) : q(quant) {}

    template <typename InT, typename OutByteT>
    inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) const {
        int8_t* dst = reinterpret_cast<int8_t*>(out_bytes);
        if constexpr (std::is_same_v<InT, float>) {
            for (uint32_t i = 0; i < dim; ++i) dst[i] = q.transform(in[i]);
        } else if constexpr (std::is_same_v<InT, uint16_t>) {
            for (uint32_t i = 0; i < dim; ++i) {
                dst[i] = q.transform(deglib::distances::fp16::fp16_to_float(in[i]));
            }
        }
    }
};

struct ScalarInt8PerDimQuantizer {
    deglib::quantization::scalar::ScalarQuantizerInt8PerDim q;
    explicit ScalarInt8PerDimQuantizer(deglib::quantization::scalar::ScalarQuantizerInt8PerDim quant) : q(std::move(quant)) {}

    template <typename InT, typename OutByteT>
    inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) const {
        int8_t* dst = reinterpret_cast<int8_t*>(out_bytes);
        if constexpr (std::is_same_v<InT, float>) {
            for (uint32_t i = 0; i < dim; ++i) dst[i] = q.transform(in[i], i);
        } else if constexpr (std::is_same_v<InT, uint16_t>) {
            for (uint32_t i = 0; i < dim; ++i) {
                dst[i] = q.transform(deglib::distances::fp16::fp16_to_float(in[i]), i);
            }
        }
    }
};

struct ScalarUint8Quantizer {
    deglib::quantization::scalar::ScalarQuantizerUint8 q;
    explicit ScalarUint8Quantizer(deglib::quantization::scalar::ScalarQuantizerUint8 quant) : q(quant) {}

    template <typename InT, typename OutByteT>
    inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) const {
        uint8_t* dst = reinterpret_cast<uint8_t*>(out_bytes);
        if constexpr (std::is_same_v<InT, float>) {
            for (uint32_t i = 0; i < dim; ++i) dst[i] = q.transform(in[i]);
        } else if constexpr (std::is_same_v<InT, uint16_t>) {
            for (uint32_t i = 0; i < dim; ++i) {
                dst[i] = q.transform(deglib::distances::fp16::fp16_to_float(in[i]));
            }
        }
    }
};

struct ScalarUint8PerDimQuantizer {
    deglib::quantization::scalar::ScalarQuantizerUint8PerDim q;
    explicit ScalarUint8PerDimQuantizer(deglib::quantization::scalar::ScalarQuantizerUint8PerDim quant) : q(std::move(quant)) {}

    template <typename InT, typename OutByteT>
    inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) const {
        uint8_t* dst = reinterpret_cast<uint8_t*>(out_bytes);
        if constexpr (std::is_same_v<InT, float>) {
            for (uint32_t i = 0; i < dim; ++i) dst[i] = q.transform(in[i], i);
        } else if constexpr (std::is_same_v<InT, uint16_t>) {
            for (uint32_t i = 0; i < dim; ++i) {
                dst[i] = q.transform(deglib::distances::fp16::fp16_to_float(in[i]), i);
            }
        }
    }
};

struct EVPQuantizer {
    uint32_t non_zeros = 0;
    explicit EVPQuantizer(uint32_t nz) : non_zeros(nz) {}

    template <typename InT, typename OutByteT>
    inline void transform(const InT* in, uint32_t dim, OutByteT* out_bytes) const {
        if constexpr (std::is_same_v<InT, float>) {
            auto evp = deglib::quantization::evp::quantize_single(in, dim, non_zeros);
            std::memcpy(out_bytes, evp.data(), evp.size());
        } else if constexpr (std::is_same_v<InT, uint16_t>) {
            std::vector<float> f32(dim);
            deglib::distances::fp16::fp16_to_floats(in, f32.data(), dim);
            auto evp = deglib::quantization::evp::quantize_single(f32.data(), dim, non_zeros);
            std::memcpy(out_bytes, evp.data(), evp.size());
        }
    }
};

// ============================================================================
// Refiner Concept Wrappers
// ============================================================================

struct NoRefiner {
    static constexpr bool enabled = false;

    template <typename QueryT>
    static inline uint32_t rerank(
        const QueryT*, uint32_t, const uint32_t*, size_t, uint32_t,
        uint32_t*, float*, bool
    ) {
        return 0;
    }
};

template <typename RefineDataT>
struct ExactRefiner {
    static constexpr bool enabled = true;
    deglib::distances::FloatSpace space;
    const RefineDataT* base_vectors = nullptr;
    size_t num_base_vectors = 0;

    ExactRefiner(deglib::distances::FloatSpace sp, const RefineDataT* base, size_t count)
        : space(std::move(sp)), base_vectors(base), num_base_vectors(count) {}

    template <typename QueryT>
    inline uint32_t rerank(
        const QueryT* query, uint32_t dim, const uint32_t* candidate_indices, size_t num_cands,
        uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted
    ) const {
        if (!base_vectors || num_cands == 0) return 0;

        if constexpr (std::is_same_v<QueryT, RefineDataT>) {
            auto reranked = deglib::search::rerank(
                space, query, 1, base_vectors, num_base_vectors, candidate_indices, num_cands, k, 1
            );
            return populate_results(reranked, k, out_indices, out_distances, unsorted);
        } else if constexpr (std::is_same_v<QueryT, float> && std::is_same_v<RefineDataT, uint16_t>) {
            std::vector<uint16_t> q_fp16(dim);
            deglib::distances::fp16::floats_to_fp16(query, q_fp16.data(), dim);
            auto reranked = deglib::search::rerank(
                space, q_fp16.data(), 1, base_vectors, num_base_vectors, candidate_indices, num_cands, k, 1
            );
            return populate_results(reranked, k, out_indices, out_distances, unsorted);
        } else if constexpr (std::is_same_v<QueryT, uint16_t> && std::is_same_v<RefineDataT, float>) {
            std::vector<float> q_f32(dim);
            deglib::distances::fp16::fp16_to_floats(query, q_f32.data(), dim);
            auto reranked = deglib::search::rerank(
                space, q_f32.data(), 1, base_vectors, num_base_vectors, candidate_indices, num_cands, k, 1
            );
            return populate_results(reranked, k, out_indices, out_distances, unsorted);
        }
        return 0;
    }

  private:
    static inline uint32_t populate_results(
        std::vector<ResultSet>& reranked, uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted
    ) {
        if (reranked.empty()) return 0;
        auto& heap = reranked[0];
        const size_t top_n = std::min<size_t>(k, heap.size());

        if (unsorted) {
            for (size_t i = 0; i < top_n; ++i) {
                out_indices[i] = heap[i].getIdentifier();
                if (out_distances) out_distances[i] = heap[i].getDistance();
            }
        } else {
            for (size_t i = top_n; i > 0; --i) {
                const auto next = heap.top();
                out_indices[i - 1] = next.getIdentifier();
                if (out_distances) out_distances[i - 1] = next.getDistance();
                heap.pop();
            }
        }
        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
            if (out_distances) out_distances[i] = std::numeric_limits<float>::max();
        }
        return static_cast<uint32_t>(top_n);
    }
};

// ============================================================================
// Abstract Searcher Interface
// ============================================================================

class SearcherBase {
  public:
    virtual ~SearcherBase() = default;

    virtual void set_query_arguments(float search_eps, float rerank_factor = 1.0f) = 0;
    virtual void set_search_eps(float search_eps) = 0;
    virtual float get_search_eps() const = 0;
    virtual void set_rerank_factor(float rerank_factor) = 0;
    virtual float get_rerank_factor() const = 0;

    // --- Raw buffer API (Zero Overhead) ---
    virtual uint32_t search_f32(const float* query, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;
    virtual uint32_t search_f16(const uint16_t* query, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;

    virtual void search_batch_f32(const float* queries, size_t n_queries, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;
    virtual void search_batch_f16(const uint16_t* queries, size_t n_queries, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;

    // --- Modern C++20 std::span and std::vector Convenience API ---
    template <typename T>
    uint32_t search(
        std::span<const T> query,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool unsorted = false
    ) const {
        if (out_indices.size() < k) {
            throw std::invalid_argument("Searcher::search: out_indices span is smaller than k");
        }
        if (!out_distances.empty() && out_distances.size() < k) {
            throw std::invalid_argument("Searcher::search: out_distances span is smaller than k");
        }
        float* d_ptr = out_distances.empty() ? nullptr : out_distances.data();
        if constexpr (std::is_same_v<T, float>) {
            return search_f32(query.data(), k, out_indices.data(), d_ptr, unsorted);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return search_f16(query.data(), k, out_indices.data(), d_ptr, unsorted);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported query type for search: must be float or uint16_t (fp16)");
        }
    }

    template <typename T>
    SearchResult search(std::span<const T> query, uint32_t k, bool return_distances = false, bool unsorted = false) const {
        SearchResult res;
        res.indices.resize(k);
        if (return_distances) res.distances.resize(k);
        uint32_t count = search<T>(
            query, k, std::span<uint32_t>(res.indices),
            return_distances ? std::span<float>(res.distances) : std::span<float>{},
            unsorted
        );
        res.indices.resize(count);
        if (return_distances) res.distances.resize(count);
        return res;
    }

    template <typename T>
    std::vector<SearchResult> search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        const size_t dim = queries.size() / n_queries;
        std::vector<SearchResult> results(n_queries);

        std::vector<uint32_t> all_indices(n_queries * k);
        std::vector<float> all_distances(return_distances ? (n_queries * k) : 0);

        if constexpr (std::is_same_v<T, float>) {
            search_batch_f32(queries.data(), n_queries, k, all_indices.data(), return_distances ? all_distances.data() : nullptr, threads, unsorted);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            search_batch_f16(queries.data(), n_queries, k, all_indices.data(), return_distances ? all_distances.data() : nullptr, threads, unsorted);
        }

        for (size_t q = 0; q < n_queries; ++q) {
            results[q].indices.assign(all_indices.begin() + q * k, all_indices.begin() + (q + 1) * k);
            if (return_distances) {
                results[q].distances.assign(all_distances.begin() + q * k, all_distances.begin() + (q + 1) * k);
            }
        }
        return results;
    }
};

// ============================================================================
// Pure Templated Searcher Implementation (Zero-Dispatch / Zero-Branch Loop)
// ============================================================================

template <typename QuantT, typename RefinerT>
class SearcherImpl : public SearcherBase {
  private:
    const deglib::graph::InternalGraph* graph_ = nullptr;
    QuantT quantizer_;
    RefinerT refiner_;

    float search_eps_ = 0.1f;
    float rerank_factor_ = 1.0f;

  public:
    SearcherImpl(
        const deglib::graph::InternalGraph& graph,
        QuantT quantizer,
        RefinerT refiner,
        float search_eps = 0.1f,
        float rerank_factor = 1.0f
    )
        : graph_(&graph),
          quantizer_(std::move(quantizer)),
          refiner_(std::move(refiner)),
          search_eps_(search_eps),
          rerank_factor_(rerank_factor) {}

    void set_query_arguments(float search_eps, float rerank_factor = 1.0f) override {
        search_eps_ = search_eps;
        rerank_factor_ = rerank_factor;
    }

    void set_search_eps(float search_eps) override { search_eps_ = search_eps; }
    float get_search_eps() const override { return search_eps_; }

    void set_rerank_factor(float rerank_factor) override { rerank_factor_ = rerank_factor; }
    float get_rerank_factor() const override { return rerank_factor_; }

    template <typename QueryT>
    inline uint32_t search_single_typed(
        const QueryT* query, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false
    ) const {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        const uint32_t fetch_k = std::max(k, static_cast<uint32_t>(std::round(k * rerank_factor_)));
        const size_t graph_feature_bytes = graph_->getFeatureSpace().get_data_size();

        // 1. Static Query Transformation
        std::vector<std::byte> query_bytes(graph_feature_bytes);
        quantizer_.transform(query, dim, query_bytes.data());

        // 2. Direct Graph Search
        auto result = graph_->search(
            std::span<const std::byte>(query_bytes.data(), graph_feature_bytes),
            fetch_k, search_eps_, nullptr, 0
        );
        const size_t found_count = result.size();

        // 3. Static Compile-Time Reranker Check
        if constexpr (RefinerT::enabled) {
            if (fetch_k > k) {
                std::vector<uint32_t> candidate_indices(found_count);
                for (size_t i = 0; i < found_count; ++i) {
                    candidate_indices[i] = graph_->getExternalLabel(result[i].getIdentifier());
                }

                uint32_t ref_count = refiner_.rerank(query, dim, candidate_indices.data(), found_count, k, out_indices, out_distances, unsorted);
                if (ref_count > 0) {
                    return ref_count;
                }
            }
        }

        // Direct return without rerank
        while (result.size() > k) {
            result.pop();
        }
        const size_t top_n = result.size();
        if (unsorted) {
            for (size_t i = 0; i < top_n; ++i) {
                out_indices[i] = graph_->getExternalLabel(result[i].getIdentifier());
                if (out_distances) out_distances[i] = result[i].getDistance();
            }
        } else {
            for (size_t i = top_n; i > 0; --i) {
                const auto next = result.top();
                out_indices[i - 1] = graph_->getExternalLabel(next.getIdentifier());
                if (out_distances) out_distances[i - 1] = next.getDistance();
                result.pop();
            }
        }
        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
            if (out_distances) out_distances[i] = std::numeric_limits<float>::max();
        }
        return static_cast<uint32_t>(top_n);
    }

    uint32_t search_f32(const float* query, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_typed<float>(query, k, out_indices, out_distances, unsorted);
    }

    uint32_t search_f16(const uint16_t* query, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_typed<uint16_t>(query, k, out_indices, out_distances, unsorted);
    }

    void search_batch_f32(const float* queries, size_t n_queries, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f32(queries + q * dim, k, out_indices + q * k, d_ptr, unsorted);
        });
    }

    void search_batch_f16(const uint16_t* queries, size_t n_queries, uint32_t k, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f16(queries + q * dim, k, out_indices + q * k, d_ptr, unsorted);
        });
    }
};

// ============================================================================
// Modern C++20 Factory Functions
// ============================================================================

template <typename QuantT = NoQuantizer, typename RefinerT = NoRefiner>
inline std::unique_ptr<SearcherBase> make_searcher(
    const deglib::graph::InternalGraph& graph,
    QuantT quantizer = NoQuantizer{},
    RefinerT refiner = NoRefiner{},
    float search_eps = 0.1f,
    float rerank_factor = 1.0f
) {
    return std::make_unique<SearcherImpl<QuantT, RefinerT>>(
        graph, std::move(quantizer), std::move(refiner), search_eps, rerank_factor
    );
}

}  // namespace deglib::search
