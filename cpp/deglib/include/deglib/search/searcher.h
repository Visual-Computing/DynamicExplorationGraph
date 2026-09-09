#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/optimization/quantization/quantizer_concept.h"
#include "deglib/optimization/quantization/scalar_quantize.h"
#include "deglib/search.h"
#include "deglib/search/kmeans.h"

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
// Search Result Structs for Modern C++20 API
// ============================================================================

struct SearchResult {
    std::vector<uint32_t> indices;
    std::vector<float> distances;

    size_t size() const noexcept { return indices.size(); }
    bool empty() const noexcept { return indices.empty(); }
    bool has_distances() const noexcept { return !distances.empty(); }
};

/**
 * Contiguous, zero-overhead batch search result structure.
 * Avoids N separate vector heap allocations when searching batches.
 */
struct SearchResultBatch {
    size_t n_queries = 0;
    uint32_t k = 0;
    std::vector<uint32_t> indices;
    std::vector<float> distances;

    size_t size() const noexcept { return n_queries; }
    bool empty() const noexcept { return n_queries == 0; }
    bool has_distances() const noexcept { return !distances.empty(); }

    std::span<const uint32_t> get_indices(size_t q) const noexcept { return {indices.data() + q * k, k}; }
    std::span<const float> get_distances(size_t q) const noexcept {
        return distances.empty() ? std::span<const float>{} : std::span<const float>{distances.data() + q * k, k};
    }
};

// ============================================================================
// Quantizer Concept
// A Quantizer Q satisfies deglib::quantization::Quantizer: it exposes its
// packed element type via output_type and pointer- plus span-based quantize
// methods for float and uint16_t (fp16) inputs, in-place and returning.
// NoQuantizer bypasses quantization via zero-copy and is exempt.
// ============================================================================

struct NoQuantizer {};

// ============================================================================
// Refiner Concept Wrappers
// ============================================================================

struct NoRefiner {
    static constexpr bool enabled = false;

    template <typename QueryT>
    static inline uint32_t rerank(const QueryT*, uint32_t, const uint32_t*, size_t, uint32_t, uint32_t*, float*, bool, bool) {
        return 0;
    }
};

template <typename RefineDataT>
struct ExactRefiner {
    static constexpr bool enabled = true;
    deglib::distances::FloatSpace space;
    const RefineDataT* base_vectors = nullptr;
    size_t num_base_vectors = 0;

    ExactRefiner(deglib::distances::FloatSpace sp, const RefineDataT* base, size_t count) : space(std::move(sp)), base_vectors(base), num_base_vectors(count) {}

    template <typename QueryT>
    inline uint32_t rerank(
        const QueryT* query,
        uint32_t dim,
        const uint32_t* candidate_indices,
        size_t num_cands,
        uint32_t k,
        uint32_t* out_indices,
        float* out_distances,
        bool return_distances,
        bool unsorted
    ) const {
        if (!base_vectors || num_cands == 0 || k == 0) return 0;

        auto cands_span = std::span<const uint32_t>(candidate_indices, num_cands);
        auto out_idx_span = std::span<uint32_t>(out_indices, k);
        auto out_dist_span = (return_distances && out_distances) ? std::span<float>(out_distances, k) : std::span<float>{};

        if constexpr (std::is_same_v<QueryT, RefineDataT>) {
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(query), dim * sizeof(QueryT));
            return deglib::search::rerank(
                space, q_span, base_vectors, num_base_vectors, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted
            );
        } else if constexpr (std::is_same_v<QueryT, float> && std::is_same_v<RefineDataT, uint16_t>) {
            std::vector<uint16_t> q_fp16(dim);
            deglib::distances::fp16::floats_to_fp16(query, q_fp16.data(), dim);
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(q_fp16.data()), dim * sizeof(uint16_t));
            return deglib::search::rerank(
                space, q_span, base_vectors, num_base_vectors, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted
            );
        } else if constexpr (std::is_same_v<QueryT, uint16_t> && std::is_same_v<RefineDataT, float>) {
            std::vector<float> q_f32(dim);
            deglib::distances::fp16::fp16_to_floats(query, q_f32.data(), dim);
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(q_f32.data()), dim * sizeof(float));
            return deglib::search::rerank(
                space, q_span, base_vectors, num_base_vectors, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted
            );
        }
        return 0;
    }
};

// ============================================================================
// Abstract Searcher Interface
// ============================================================================

class SearcherBase {
  public:
    virtual ~SearcherBase() = default;

    // --- Raw buffer API (Zero Overhead). ---
    virtual uint32_t search_f32(
        const float* query,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const = 0;
    virtual uint32_t search_f16(
        const uint16_t* query,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const = 0;

    virtual uint32_t search_ef_f32(
        const float* query,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const = 0;
    virtual uint32_t search_ef_f16(
        const uint16_t* query,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const = 0;

    virtual void search_batch_ef_f32(
        const float* queries,
        size_t n_queries,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const = 0;
    virtual void search_batch_ef_f16(
        const uint16_t* queries,
        size_t n_queries,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const = 0;

    virtual void search_batch_f32(
        const float* queries,
        size_t n_queries,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const = 0;
    virtual void search_batch_f16(
        const uint16_t* queries,
        size_t n_queries,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const = 0;

    /**
     * Select entry vertices via k-means medoids.
     *
     * @param n_clusters  Number of entry vertices to select.
     * @param n_iter      Number of k-means iterations.
     * @param sample_size Number of vertices sampled for clustering, 0 selects 3% of the graph size.
     * @param seed        Random seed for sampling and centroid init.
     * @param threads     Number of worker threads, 0 selects a library default.
     */
    virtual void optimize(uint32_t n_clusters = 128, uint32_t n_iter = 15, size_t sample_size = 0, uint32_t seed = 42, size_t threads = 0) = 0;

    // --- Modern C++20 std::span and std::vector Convenience API ---

    /**
     * Single query search directly writing into pre-allocated destination spans.
     *
     * @param query             Query vector span (float or uint16_t fp16)
     * @param k                 Number of nearest neighbors to return
     * @param out_indices       Destination span for nearest neighbor IDs (size must be >= k)
     * @param out_distances     Destination span for neighbor distances (size must be >= k if return_distances is true)
     * @param eps               Search expansion factor (trade-off speed vs recall)
     * @param rerank_factor     Candidate expansion factor for reranking
     * @param return_distances  Explicit boolean flag indicating whether distances should be written
     * @param unsorted          If true, returns results in fast unsorted heap order; if false, sorted nearest-first
     * @return                  Actual number of valid results written (<= k)
     */
    template <typename T>
    uint32_t search(
        std::span<const T> query,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        if (out_indices.size() < k) {
            throw std::invalid_argument("Searcher::search: out_indices span is smaller than k");
        }
        if (return_distances && out_distances.size() < k) {
            throw std::invalid_argument("Searcher::search: return_distances is true but out_distances span is smaller than k");
        }
        float* d_ptr = (return_distances && !out_distances.empty()) ? out_distances.data() : nullptr;
        if constexpr (std::is_same_v<T, float>) {
            return search_f32(query.data(), k, eps, rerank_factor, out_indices.data(), d_ptr, unsorted);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return search_f16(query.data(), k, eps, rerank_factor, out_indices.data(), d_ptr, unsorted);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported query type for search: must be float or uint16_t (fp16)");
        }
    }

    /**
     * Single query search returning a SearchResult structure with std::vector containers.
     *
     * @param query             Query vector span (float or uint16_t fp16)
     * @param k                 Number of nearest neighbors to return
     * @param eps               Search expansion factor (trade-off speed vs recall)
     * @param rerank_factor     Candidate expansion factor for reranking
     * @param return_distances  Explicit boolean flag indicating whether distances should be returned
     * @param unsorted          If true, returns results in fast unsorted heap order; if false, sorted nearest-first
     * @return                  SearchResult containing vector of indices and optional distances
     */
    template <typename T>
    SearchResult
    search(std::span<const T> query, uint32_t k, float eps = 0.1f, float rerank_factor = 1.0f, bool return_distances = false, bool unsorted = false) const {
        SearchResult res;
        res.indices.resize(k);
        if (return_distances) res.distances.resize(k);
        uint32_t count = search<T>(
            query, k, std::span<uint32_t>(res.indices), return_distances ? std::span<float>(res.distances) : std::span<float>{}, eps, rerank_factor,
            return_distances, unsorted
        );
        res.indices.resize(count);
        if (return_distances) res.distances.resize(count);
        return res;
    }

    /**
     * Batch query search directly writing into pre-allocated destination spans.
     *
     * @param queries           Contiguous query vectors span of size n_queries * dim
     * @param n_queries         Number of queries in the batch
     * @param k                 Number of nearest neighbors to return per query
     * @param out_indices       Destination span for result indices of size >= n_queries * k
     * @param out_distances     Destination span for result distances of size >= n_queries * k (if return_distances is true)
     * @param eps               Search expansion factor (trade-off speed vs recall)
     * @param rerank_factor     Candidate expansion factor for reranking
     * @param threads           Number of worker threads for parallel search
     * @param return_distances  Explicit boolean flag indicating whether distances should be written
     * @param unsorted          If true, returns results in fast unsorted heap order; if false, sorted nearest-first
     */
    template <typename T>
    void search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        if (out_indices.size() < n_queries * k) {
            throw std::invalid_argument("Searcher::search_batch: out_indices span smaller than n_queries * k");
        }
        if (return_distances && out_distances.size() < n_queries * k) {
            throw std::invalid_argument("Searcher::search_batch: return_distances is true but out_distances span smaller than n_queries * k");
        }
        float* d_ptr = (return_distances && !out_distances.empty()) ? out_distances.data() : nullptr;
        if constexpr (std::is_same_v<T, float>) {
            search_batch_f32(queries.data(), n_queries, k, eps, rerank_factor, out_indices.data(), d_ptr, threads, unsorted);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            search_batch_f16(queries.data(), n_queries, k, eps, rerank_factor, out_indices.data(), d_ptr, threads, unsorted);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported query type for search_batch: must be float or uint16_t (fp16)");
        }
    }

    /**
     * Batch query search returning a flat contiguous SearchResultBatch structure.
     *
     * @param queries           Contiguous query vectors span of size n_queries * dim
     * @param n_queries         Number of queries in the batch
     * @param k                 Number of nearest neighbors to return per query
     * @param eps               Search expansion factor (trade-off speed vs recall)
     * @param rerank_factor     Candidate expansion factor for reranking
     * @param threads           Number of worker threads for parallel search
     * @param return_distances  Explicit boolean flag indicating whether distances should be returned
     * @param unsorted          If true, returns results in fast unsorted heap order; if false, sorted nearest-first
     * @return                  SearchResultBatch containing flat vectors of indices and optional distances
     */
    template <typename T>
    SearchResultBatch search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        float eps = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        SearchResultBatch result;
        result.n_queries = n_queries;
        result.k = k;
        result.indices.resize(n_queries * k);
        if (return_distances) {
            result.distances.resize(n_queries * k);
        }
        search_batch<T>(
            queries, n_queries, k, std::span<uint32_t>(result.indices), return_distances ? std::span<float>(result.distances) : std::span<float>{}, eps,
            rerank_factor, threads, return_distances, unsorted
        );
        return result;
    }
};

// ============================================================================
// Pure Templated Searcher Implementation (Zero-Dispatch / Zero-Branch Loop)
// ============================================================================

template <typename QuantT, typename RefinerT>
    requires deglib::quantization::Quantizer<QuantT> || std::is_same_v<QuantT, NoQuantizer>
class SearcherImpl : public SearcherBase {
  private:
    const deglib::graph::InternalGraph* graph_ = nullptr;
    QuantT quantizer_;
    RefinerT refiner_;
    KMeansEntrySelector entry_selector_;

  public:
    SearcherImpl(const deglib::graph::InternalGraph& graph, QuantT quantizer, RefinerT refiner)
        : graph_(&graph), quantizer_(std::move(quantizer)), refiner_(std::move(refiner)), entry_selector_(graph) {}

    /**
     * Select entry vertices via k-means medoids.
     *
     * @param n_clusters  Number of entry vertices to select.
     * @param n_iter      Number of k-means iterations.
     * @param sample_size Number of vertices sampled for clustering, 0 selects 3% of the graph size.
     * @param seed        Random seed for sampling and centroid init.
     * @param threads     Number of worker threads.
     */
    void optimize(uint32_t n_clusters = 128, uint32_t n_iter = 15, size_t sample_size = 0, uint32_t seed = 7, size_t threads = 1) override {
        // 1. K-Means entry points selection
        entry_selector_.optimize(n_clusters, n_iter, sample_size, seed, threads);

        // 2. Pure C++ prefetch auto-tuning directly within optimize()
        if (graph_->size() < 10) return;

        const size_t test_q_count = std::min(size_t(50), size_t(graph_->size()));
        const uint32_t dim = graph_->getFeatureSpace().dim();
        std::vector<float> sample_queries(test_q_count * size_t(dim));

        // Sample queries from graph vertices (zero Python dependency)
        for (size_t i = 0; i < test_q_count; ++i) {
            const auto* feat = reinterpret_cast<const float*>(graph_->getFeatureVector(static_cast<uint32_t>(i)));
            std::memcpy(sample_queries.data() + i * size_t(dim), feat, size_t(dim) * sizeof(float));
        }

        const std::vector<int32_t> try_pos = {4, 8, 12, 16};
        const std::vector<int32_t> try_pls = {2, 3, 4};
        const std::vector<int32_t> try_nls = {2, 3, 4};

        int32_t best_po = graph_->getPo();
        int32_t best_pl = graph_->getPl();
        int32_t best_nl = graph_->getNl();
        double best_time = std::numeric_limits<double>::max();

        std::vector<uint32_t> dummy_out(100);

        for (int32_t po : try_pos) {
            for (int32_t pl : try_pls) {
                for (int32_t nl : try_nls) {
                    const_cast<deglib::graph::InternalGraph*>(graph_)->setPrefetch(po, pl, nl);

                    // Warmup
                    for (size_t i = 0; i < std::min(size_t(3), test_q_count); ++i) {
                        search_single_ef_typed(sample_queries.data() + i * size_t(dim), 100, 200, 1.0f, dummy_out.data(), nullptr, true);
                    }

                    auto t_start = std::chrono::high_resolution_clock::now();
                    for (size_t i = 0; i < test_q_count; ++i) {
                        search_single_ef_typed(sample_queries.data() + i * size_t(dim), 100, 200, 1.0f, dummy_out.data(), nullptr, true);
                    }
                    auto t_end = std::chrono::high_resolution_clock::now();
                    double dur = std::chrono::duration<double, std::micro>(t_end - t_start).count();

                    if (dur < best_time) {
                        best_time = dur;
                        best_po = po;
                        best_pl = pl;
                        best_nl = nl;
                    }
                }
            }
        }

        const_cast<deglib::graph::InternalGraph*>(graph_)->setPrefetch(best_po, best_pl, best_nl);
    }

    template <typename QueryT>
    inline uint32_t search_single_typed(
        const QueryT* query,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        const uint32_t fetch_k = std::max(k, static_cast<uint32_t>(std::round(k * rerank_factor)));
        const size_t graph_feature_bytes = graph_->getFeatureSpace().get_data_size();

        if (graph_->size() == 0) throw std::invalid_argument("Searcher: graph is empty");

        // 1. Static Query Transformation (Zero-copy bypass for NoQuantizer)
        const std::byte* query_bytes = nullptr;
        alignas(64) std::byte stack_query_bytes[512];
        std::unique_ptr<std::byte[]> heap_query_bytes;

        if constexpr (std::is_same_v<QuantT, NoQuantizer>) {
            // Direct zero-copy: query is already in unquantized format
            query_bytes = reinterpret_cast<const std::byte*>(query);
        } else {
            std::byte* q_buf = stack_query_bytes;
            if (graph_feature_bytes > sizeof(stack_query_bytes)) {
                heap_query_bytes = std::make_unique<std::byte[]>(graph_feature_bytes);
                q_buf = heap_query_bytes.get();
            }

            quantizer_.quantize(query, reinterpret_cast<typename QuantT::output_type*>(q_buf), 1, dim);
            query_bytes = q_buf;
        }

        // 2. Entry selection via selector (falls back to vertex 0) + Direct Graph Search
        const std::span<const std::byte> query_span(query_bytes, graph_feature_bytes);
        const std::vector<uint32_t> entries = entry_selector_.top_entries(query_span, 2);
        deglib::graph::ResultSet result =
            entries.empty() ? graph_->search(query_span, fetch_k, eps, nullptr, 0) : graph_->search(query_span, entries, fetch_k, eps, nullptr, 0);

        // 3. Static Compile-Time Reranker Check
        if constexpr (RefinerT::enabled) {
            if (fetch_k > k) {
                const size_t found_count = result.size();

                // Stack fast path for up to 256 candidates, heap fallback beyond that without result limit.
                uint32_t stack_cand_indices[256];
                std::unique_ptr<uint32_t[]> heap_cand_indices;
                uint32_t* cands = stack_cand_indices;
                if (found_count > (sizeof(stack_cand_indices) / sizeof(uint32_t))) {
                    heap_cand_indices = std::make_unique<uint32_t[]>(found_count);
                    cands = heap_cand_indices.get();
                }

                for (size_t i = 0; i < found_count; ++i) {
                    cands[i] = graph_->getExternalLabel(result[i].getIdentifier());
                }

                const bool return_distances = (out_distances != nullptr);
                uint32_t ref_count = refiner_.rerank(query, dim, cands, found_count, k, out_indices, out_distances, return_distances, unsorted);
                if (ref_count > 0) {
                    return ref_count;
                }
            }
        }

        // Direct return without rerank
        while (result.size() > k) {
            result.pop();
        }

        if (out_distances) {
            return populate_graph_results<true>(result, k, out_indices, out_distances, unsorted);
        } else {
            return populate_graph_results<false>(result, k, out_indices, nullptr, unsorted);
        }
    }

    template <bool HAS_DISTANCES, typename ResultSetT>
    inline uint32_t populate_graph_results(ResultSetT& result, uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted) const {
        const size_t top_n = result.size();
        if (unsorted) {
            for (size_t i = 0; i < top_n; ++i) {
                out_indices[i] = graph_->getExternalLabel(result[i].getIdentifier());
            }
            if constexpr (HAS_DISTANCES) {
                for (size_t i = 0; i < top_n; ++i) {
                    out_distances[i] = result[i].getDistance();
                }
            }
        } else {
            for (size_t i = top_n; i > 0; --i) {
                const auto next = result.top();
                out_indices[i - 1] = graph_->getExternalLabel(next.getIdentifier());
                if constexpr (HAS_DISTANCES) {
                    out_distances[i - 1] = next.getDistance();
                }
                result.pop();
            }
        }
        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
        }
        if constexpr (HAS_DISTANCES) {
            for (size_t i = top_n; i < k; ++i) {
                out_distances[i] = std::numeric_limits<float>::max();
            }
        }
        return static_cast<uint32_t>(top_n);
    }

    uint32_t search_f32(
        const float* query,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const override {
        return search_single_typed<float>(query, k, eps, rerank_factor, out_indices, out_distances, unsorted);
    }

    uint32_t search_f16(
        const uint16_t* query,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const override {
        return search_single_typed<uint16_t>(query, k, eps, rerank_factor, out_indices, out_distances, unsorted);
    }

    void search_batch_f32(
        const float* queries,
        size_t n_queries,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f32(queries + q * dim, k, eps, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    void search_batch_f16(
        const uint16_t* queries,
        size_t n_queries,
        uint32_t k,
        float eps,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f16(queries + q * dim, k, eps, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    template <typename QueryT>
    inline uint32_t search_single_ef_typed(
        const QueryT* query,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        const uint32_t fetch_k = std::max(k, static_cast<uint32_t>(std::round(k * rerank_factor)));
        const size_t graph_feature_bytes = graph_->getFeatureSpace().get_data_size();

        if (graph_->size() == 0) throw std::invalid_argument("Searcher: graph is empty");

        // 1. Static Query Transformation
        const std::byte* query_bytes = nullptr;
        alignas(64) std::byte stack_query_bytes[512];
        std::unique_ptr<std::byte[]> heap_query_bytes;

        if constexpr (std::is_same_v<QuantT, NoQuantizer>) {
            query_bytes = reinterpret_cast<const std::byte*>(query);
        } else {
            std::byte* q_buf = stack_query_bytes;
            if (graph_feature_bytes > sizeof(stack_query_bytes)) {
                heap_query_bytes = std::make_unique<std::byte[]>(graph_feature_bytes);
                q_buf = heap_query_bytes.get();
            }

            quantizer_.quantize(query, reinterpret_cast<typename QuantT::output_type*>(q_buf), 1, dim);
            query_bytes = q_buf;
        }

        // 2. Entry selection via selector
        const std::span<const std::byte> query_span(query_bytes, graph_feature_bytes);
        const std::vector<uint32_t> entries = entry_selector_.top_entries(query_span, 2);

        // 3. Search using search_ef
        const auto pool = entries.empty() ? graph_->search_ef(query_span, fetch_k, ef) : graph_->search_ef(query_span, entries, fetch_k, ef);

        // 4. Rerank or return
        const size_t found_count = static_cast<size_t>(pool.size());
        if (found_count == 0) return 0;

        if constexpr (RefinerT::enabled) {
            if (fetch_k > k) {
                uint32_t stack_cand_indices[256];
                std::unique_ptr<uint32_t[]> heap_cand_indices;
                uint32_t* cands = stack_cand_indices;
                if (found_count > 256) {
                    heap_cand_indices = std::make_unique<uint32_t[]>(found_count);
                    cands = heap_cand_indices.get();
                }

                for (size_t i = 0; i < found_count; ++i) {
                    cands[i] = graph_->getExternalLabel(pool[i].getIdentifier());
                }

                const bool return_distances = (out_distances != nullptr);
                return refiner_.rerank(query, dim, cands, found_count, k, out_indices, out_distances, return_distances, unsorted);
            }
        }

        const uint32_t result_count = std::min(k, static_cast<uint32_t>(found_count));
        for (uint32_t i = 0; i < result_count; ++i) {
            out_indices[i] = graph_->getExternalLabel(pool[i].getIdentifier());
        }
        if (out_distances != nullptr) {
            for (uint32_t i = 0; i < result_count; ++i) {
                out_distances[i] = pool[i].getDistance();
            }
        }
        return result_count;
    }

    uint32_t search_ef_f32(
        const float* query,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const override {
        return search_single_ef_typed(query, k, ef, rerank_factor, out_indices, out_distances, unsorted);
    }

    uint32_t search_ef_f16(
        const uint16_t* query,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        bool unsorted = false
    ) const override {
        return search_single_ef_typed(query, k, ef, rerank_factor, out_indices, out_distances, unsorted);
    }

    void search_batch_ef_f32(
        const float* queries,
        size_t n_queries,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        for (size_t q = 0; q < n_queries; ++q) {
            search_ef_f32(queries + q * dim, k, ef, rerank_factor, out_indices + q * k, out_distances ? out_distances + q * k : nullptr, unsorted);
        }
    }

    void search_batch_ef_f16(
        const uint16_t* queries,
        size_t n_queries,
        uint32_t k,
        uint32_t ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        for (size_t q = 0; q < n_queries; ++q) {
            search_ef_f16(queries + q * dim, k, ef, rerank_factor, out_indices + q * k, out_distances ? out_distances + q * k : nullptr, unsorted);
        }
    }
};

}  // namespace deglib::search
