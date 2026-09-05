#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/optimization/quantization/scalar_quantize.h"
#include "deglib/search.h"
#include "deglib/utils/cpu.h"

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

    std::span<const uint32_t> get_indices(size_t q) const noexcept {
        return {indices.data() + q * k, k};
    }
    std::span<const float> get_distances(size_t q) const noexcept {
        return distances.empty() ? std::span<const float>{} : std::span<const float>{distances.data() + q * k, k};
    }
};

// ============================================================================
// Quantizer Concept
// A Quantizer Q provides:
//   q.quantize(const InT* in, OutByteT* out_bytes, size_t count = 1, uint32_t dim)
// Both ScalarQuantizers and EVPQuantizer provide this method.
// ============================================================================

struct NoQuantizer {};

struct EVPQuantizer {
    uint32_t non_zeros = 0;
    explicit EVPQuantizer(uint32_t nz) : non_zeros(nz) {}

    template <typename InT, typename OutByteT>
    inline void quantize(const InT* in, OutByteT* out_bytes, size_t count, uint32_t dim) const {
        for (size_t i = 0; i < count; ++i) {
            const size_t mask_bytes = dim / 8;
            deglib::quantization::evp::quantize_single_into(
                in + i * dim, dim, non_zeros, reinterpret_cast<std::byte*>(out_bytes) + i * 2 * mask_bytes
            );
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
        uint32_t*, float*, bool, bool
    ) {
        return 0;
    }
};

#if defined(DEGLIB_X86)
// AVX-512 FP16 inner-product reranker: D=200 4-way batch fast path plus a generic
// 16-wide chunked path. Kept in its own target-attributed function so portable
// (non-AVX-512) builds keep generic code free of AVX-512 instructions; only
// invoked on AVX-512 CPUs.
static DEGLIB_TARGET_AVX512 uint32_t fp16_ip_rerank_avx512(
    const float* query,
    uint32_t dim,
    const uint16_t* base_vectors,
    size_t num_cands,
    const uint32_t* candidate_indices,
    uint32_t k,
    uint32_t* out_indices,
    float* out_distances,
    bool return_distances,
    bool unsorted
) {
        struct CandDist {
            uint32_t id;
            float dist;
        };
        alignas(64) CandDist stack_cand_dists[512];
        std::unique_ptr<CandDist[]> heap_cand_dists;
        CandDist* items = stack_cand_dists;
        if (num_cands > 512) {
            heap_cand_dists = std::make_unique<CandDist[]>(num_cands);
            items = heap_cand_dists.get();
        }

        if (dim == 200) {
            const __m512 q0  = _mm512_loadu_ps(query + 0);
            const __m512 q1  = _mm512_loadu_ps(query + 16);
            const __m512 q2  = _mm512_loadu_ps(query + 32);
            const __m512 q3  = _mm512_loadu_ps(query + 48);
            const __m512 q4  = _mm512_loadu_ps(query + 64);
            const __m512 q5  = _mm512_loadu_ps(query + 80);
            const __m512 q6  = _mm512_loadu_ps(query + 96);
            const __m512 q7  = _mm512_loadu_ps(query + 112);
            const __m512 q8  = _mm512_loadu_ps(query + 128);
            const __m512 q9  = _mm512_loadu_ps(query + 144);
            const __m512 q10 = _mm512_loadu_ps(query + 160);
            const __m512 q11 = _mm512_loadu_ps(query + 176);
            const __m512 q12 = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_loadu_ps(query + 192), 0);

            size_t j = 0;
            for (; j + 4 <= num_cands; j += 4) {
                uint32_t idx0 = candidate_indices[j];
                uint32_t idx1 = candidate_indices[j + 1];
                uint32_t idx2 = candidate_indices[j + 2];
                uint32_t idx3 = candidate_indices[j + 3];

                if (j + 8 <= num_cands) {
                    const char* pf0 = reinterpret_cast<const char*>(base_vectors + static_cast<size_t>(candidate_indices[j + 4]) * 200);
                    const char* pf1 = reinterpret_cast<const char*>(base_vectors + static_cast<size_t>(candidate_indices[j + 5]) * 200);
                    _mm_prefetch(pf0, _MM_HINT_T0);
                    _mm_prefetch(pf0 + 64, _MM_HINT_T0);
                    _mm_prefetch(pf0 + 128, _MM_HINT_T0);
                    _mm_prefetch(pf0 + 192, _MM_HINT_T0);
                    _mm_prefetch(pf1, _MM_HINT_T0);
                    _mm_prefetch(pf1 + 64, _MM_HINT_T0);
                    _mm_prefetch(pf1 + 128, _MM_HINT_T0);
                    _mm_prefetch(pf1 + 192, _MM_HINT_T0);
                }

                const uint16_t* ptr0 = base_vectors + static_cast<size_t>(idx0) * 200;
                const uint16_t* ptr1 = base_vectors + static_cast<size_t>(idx1) * 200;
                const uint16_t* ptr2 = base_vectors + static_cast<size_t>(idx2) * 200;
                const uint16_t* ptr3 = base_vectors + static_cast<size_t>(idx3) * 200;

                #define FMA_CHUNK(offset, q_reg) \
                    sum0 = _mm512_fmadd_ps(q_reg, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr0 + offset))), sum0); \
                    sum1 = _mm512_fmadd_ps(q_reg, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr1 + offset))), sum1); \
                    sum2 = _mm512_fmadd_ps(q_reg, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr2 + offset))), sum2); \
                    sum3 = _mm512_fmadd_ps(q_reg, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr3 + offset))), sum3);

                __m512 sum0 = _mm512_setzero_ps();
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();

                FMA_CHUNK(0, q0);
                FMA_CHUNK(16, q1);
                FMA_CHUNK(32, q2);
                FMA_CHUNK(48, q3);
                FMA_CHUNK(64, q4);
                FMA_CHUNK(80, q5);
                FMA_CHUNK(96, q6);
                FMA_CHUNK(112, q7);
                FMA_CHUNK(128, q8);
                FMA_CHUNK(144, q9);
                FMA_CHUNK(160, q10);
                FMA_CHUNK(176, q11);

                #undef FMA_CHUNK

                __m512 t0 = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr0 + 192))), 0);
                __m512 t1 = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr1 + 192))), 0);
                __m512 t2 = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr2 + 192))), 0);
                __m512 t3 = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr3 + 192))), 0);

                sum0 = _mm512_fmadd_ps(q12, t0, sum0);
                sum1 = _mm512_fmadd_ps(q12, t1, sum1);
                sum2 = _mm512_fmadd_ps(q12, t2, sum2);
                sum3 = _mm512_fmadd_ps(q12, t3, sum3);

                float d0 = 1.0f - _mm512_reduce_add_ps(sum0);
                float d1 = 1.0f - _mm512_reduce_add_ps(sum1);
                float d2 = 1.0f - _mm512_reduce_add_ps(sum2);
                float d3 = 1.0f - _mm512_reduce_add_ps(sum3);

                items[j]     = {idx0, d0};
                items[j + 1] = {idx1, d1};
                items[j + 2] = {idx2, d2};
                items[j + 3] = {idx3, d3};
            }

            for (; j < num_cands; ++j) {
                uint32_t idx = candidate_indices[j];
                const uint16_t* ptr = base_vectors + static_cast<size_t>(idx) * 200;

                __m512 sum = _mm512_setzero_ps();
                sum = _mm512_fmadd_ps(q0,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 0))),   sum);
                sum = _mm512_fmadd_ps(q1,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 16))),  sum);
                sum = _mm512_fmadd_ps(q2,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 32))),  sum);
                sum = _mm512_fmadd_ps(q3,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 48))),  sum);
                sum = _mm512_fmadd_ps(q4,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 64))),  sum);
                sum = _mm512_fmadd_ps(q5,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 80))),  sum);
                sum = _mm512_fmadd_ps(q6,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 96))),  sum);
                sum = _mm512_fmadd_ps(q7,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 112))), sum);
                sum = _mm512_fmadd_ps(q8,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 128))), sum);
                sum = _mm512_fmadd_ps(q9,  _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 144))), sum);
                sum = _mm512_fmadd_ps(q10, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 160))), sum);
                sum = _mm512_fmadd_ps(q11, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 176))), sum);

                __m512 t = _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 192))), 0);
                sum = _mm512_fmadd_ps(q12, t, sum);

                float d = 1.0f - _mm512_reduce_add_ps(sum);
                items[j] = {idx, d};
            }
        } else {
            const size_t chunks16 = dim / 16;
            for (size_t j = 0; j < num_cands; ++j) {
                uint32_t idx = candidate_indices[j];
                const uint16_t* ptr = base_vectors + static_cast<size_t>(idx) * dim;
                __m512 sum = _mm512_setzero_ps();
                for (size_t c = 0; c < chunks16; ++c) {
                    __m512 q = _mm512_loadu_ps(query + c * 16);
                    __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + c * 16)));
                    sum = _mm512_fmadd_ps(q, v, sum);
                }
                float dot = _mm512_reduce_add_ps(sum);
                for (size_t c = chunks16 * 16; c < dim; ++c) {
                    dot += query[c] * deglib::distances::fp16::fp16_to_float(ptr[c]);
                }
                items[j] = {idx, 1.0f - dot};
            }
        }

        const size_t top_n = std::min<size_t>(k, num_cands);
        if (unsorted) {
            if (top_n < num_cands) {
                std::nth_element(items, items + top_n, items + num_cands, [](const CandDist& a, const CandDist& b) noexcept {
                    return a.dist < b.dist;
                });
            }
        } else {
            if (top_n < num_cands) {
                std::partial_sort(items, items + top_n, items + num_cands, [](const CandDist& a, const CandDist& b) noexcept {
                    return a.dist < b.dist;
                });
            } else {
                std::sort(items, items + top_n, [](const CandDist& a, const CandDist& b) noexcept {
                    return a.dist < b.dist;
                });
            }
        }

        for (size_t i = 0; i < top_n; ++i) {
            out_indices[i] = items[i].id;
        }
        if (return_distances && out_distances) {
            for (size_t i = 0; i < top_n; ++i) {
                out_distances[i] = items[i].dist;
            }
        }
        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
            if (return_distances && out_distances) {
                out_distances[i] = std::numeric_limits<float>::max();
            }
        }
        return static_cast<uint32_t>(top_n);
}
#endif

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
        uint32_t k, uint32_t* out_indices, float* out_distances, bool return_distances, bool unsorted
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
#if defined(DEGLIB_X86)
            if (space.metric() == deglib::distances::Metric::FP16_InnerProduct && deglib::cpu::has_avx512()) {
                return fp16_ip_rerank_avx512(query, dim, base_vectors, num_cands, candidate_indices, k, out_indices, out_distances, return_distances, unsorted);
            }
#endif
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

    // --- Raw buffer API (Zero Overhead) ---
    virtual uint32_t search_f32(const float* query, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;
    virtual uint32_t search_f16(const uint16_t* query, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;

    virtual void search_batch_f32(const float* queries, size_t n_queries, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;
    virtual void search_batch_f16(const uint16_t* queries, size_t n_queries, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;

    virtual uint32_t search_ef_f32(const float* query, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;
    virtual uint32_t search_ef_f16(const uint16_t* query, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const = 0;

    virtual void search_batch_ef_f32(const float* queries, size_t n_queries, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;
    virtual void search_batch_ef_f16(const uint16_t* queries, size_t n_queries, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const = 0;

    virtual void set_prefetch(int32_t po, int32_t pl) = 0;
    virtual std::pair<int32_t, int32_t> get_prefetch() const = 0;

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
    SearchResult search(
        std::span<const T> query,
        uint32_t k,
        float eps = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false
    ) const {
        SearchResult res;
        res.indices.resize(k);
        if (return_distances) res.distances.resize(k);
        uint32_t count = search<T>(
            query, k, std::span<uint32_t>(res.indices),
            return_distances ? std::span<float>(res.distances) : std::span<float>{},
            eps, rerank_factor, return_distances, unsorted
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
            queries, n_queries, k,
            std::span<uint32_t>(result.indices),
            return_distances ? std::span<float>(result.distances) : std::span<float>{},
            eps, rerank_factor, threads, return_distances, unsorted
        );
        return result;
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

  public:
    SearcherImpl(
        const deglib::graph::InternalGraph& graph,
        QuantT quantizer,
        RefinerT refiner
    )
        : graph_(&graph),
          quantizer_(std::move(quantizer)),
          refiner_(std::move(refiner)) {}

    template <typename QueryT>
    inline uint32_t search_single_typed(
        const QueryT* query, uint32_t k, float eps, float rerank_factor,
        uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false
    ) const {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        const uint32_t fetch_k = std::max(k, static_cast<uint32_t>(std::round(k * rerank_factor)));
        const size_t graph_feature_bytes = graph_->getFeatureSpace().get_data_size();

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

            if constexpr (requires { quantizer_.quantize(query, q_buf, 1, dim); }) {
                quantizer_.quantize(query, q_buf, 1, dim);
            } else {
                quantizer_.quantize(query, reinterpret_cast<typename QuantT::output_type*>(q_buf), 1, dim);
            }
            query_bytes = q_buf;
        }

        // 2. Direct Graph Search
        auto result = graph_->search(
            std::span<const std::byte>(query_bytes, graph_feature_bytes),
            fetch_k, eps, nullptr, 0
        );
        const size_t found_count = result.size();

        // 3. Static Compile-Time Reranker Check
        if constexpr (RefinerT::enabled) {
            if (fetch_k > k) {
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
                uint32_t ref_count = refiner_.rerank(
                    query, dim, cands, found_count, k, out_indices, out_distances, return_distances, unsorted
                );
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
    inline uint32_t populate_graph_results(
        ResultSetT& result, uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted
    ) const {
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

    template <typename QueryT>
    inline uint32_t search_single_ef_typed(
        const QueryT* query, uint32_t k, uint32_t ef, float rerank_factor,
        uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false
    ) const {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        const uint32_t fetch_k = std::max(k, static_cast<uint32_t>(std::round(k * rerank_factor)));
        const size_t graph_feature_bytes = graph_->getFeatureSpace().get_data_size();

        // 1. Static Query Transformation (Zero-copy bypass for NoQuantizer)
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

            if constexpr (requires { quantizer_.quantize(query, q_buf, 1, dim); }) {
                quantizer_.quantize(query, q_buf, 1, dim);
            } else {
                quantizer_.quantize(query, reinterpret_cast<typename QuantT::output_type*>(q_buf), 1, dim);
            }
            query_bytes = q_buf;
        }

        // 2. Direct Graph Search with LinearPool
        auto pool = graph_->search_ef(
            std::span<const std::byte>(query_bytes, graph_feature_bytes),
            fetch_k, ef
        );
        const size_t found_count = pool.size();

        // 3. Static Compile-Time Reranker Check
        if constexpr (RefinerT::enabled) {
            if (fetch_k > k && found_count > 0) {
                uint32_t stack_cand_indices[256];
                std::unique_ptr<uint32_t[]> heap_cand_indices;
                uint32_t* cands = stack_cand_indices;
                const size_t cands_to_refine = std::min<size_t>(found_count, fetch_k);
                if (cands_to_refine > (sizeof(stack_cand_indices) / sizeof(uint32_t))) {
                    heap_cand_indices = std::make_unique<uint32_t[]>(cands_to_refine);
                    cands = heap_cand_indices.get();
                }

                for (size_t i = 0; i < cands_to_refine; ++i) {
                    cands[i] = graph_->getExternalLabel(pool.id(static_cast<int32_t>(i)));
                }

                uint32_t ref_count = refiner_.rerank(
                    query, dim, cands, cands_to_refine, k,
                    out_indices, out_distances, out_distances != nullptr, unsorted
                );
                if (ref_count > 0) {
                    return ref_count;
                }
            }
        }

        // Direct return without rerank
        const size_t top_n = std::min<size_t>(k, found_count);
        for (size_t i = 0; i < top_n; ++i) {
            out_indices[i] = graph_->getExternalLabel(pool.id(static_cast<int32_t>(i)));
            if (out_distances) {
                out_distances[i] = pool.dist(static_cast<int32_t>(i));
            }
        }
        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
            if (out_distances) {
                out_distances[i] = std::numeric_limits<float>::max();
            }
        }
        return static_cast<uint32_t>(top_n);
    }

    uint32_t search_ef_f32(const float* query, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_ef_typed<float>(query, k, ef, rerank_factor, out_indices, out_distances, unsorted);
    }

    uint32_t search_ef_f16(const uint16_t* query, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_ef_typed<uint16_t>(query, k, ef, rerank_factor, out_indices, out_distances, unsorted);
    }

    void search_batch_ef_f32(const float* queries, size_t n_queries, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_ef_f32(queries + q * dim, k, ef, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    void search_batch_ef_f16(const uint16_t* queries, size_t n_queries, uint32_t k, uint32_t ef, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_ef_f16(queries + q * dim, k, ef, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    uint32_t search_f32(const float* query, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_typed<float>(query, k, eps, rerank_factor, out_indices, out_distances, unsorted);
    }

    uint32_t search_f16(const uint16_t* query, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, bool unsorted = false) const override {
        return search_single_typed<uint16_t>(query, k, eps, rerank_factor, out_indices, out_distances, unsorted);
    }

    void search_batch_f32(const float* queries, size_t n_queries, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f32(queries + q * dim, k, eps, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    void search_batch_f16(const uint16_t* queries, size_t n_queries, uint32_t k, float eps, float rerank_factor, uint32_t* out_indices, float* out_distances = nullptr, size_t threads = 1, bool unsorted = false) const override {
        const uint32_t dim = graph_->getFeatureSpace().dim();
        deglib::concurrent::parallel_for(0, n_queries, threads, [&](size_t q, size_t) {
            float* d_ptr = out_distances ? (out_distances + q * k) : nullptr;
            search_f16(queries + q * dim, k, eps, rerank_factor, out_indices + q * k, d_ptr, unsorted);
        });
    }

    void set_prefetch(int32_t po, int32_t pl) override {
        graph_->setPrefetch(po, pl);
    }

    std::pair<int32_t, int32_t> get_prefetch() const override {
        return {graph_->getPo(), graph_->getPl()};
    }
};

// ============================================================================
// Modern C++20 Factory Functions
// ============================================================================

template <typename QuantT = NoQuantizer, typename RefinerT = NoRefiner>
inline std::unique_ptr<SearcherBase> make_searcher(
    const deglib::graph::InternalGraph& graph,
    QuantT quantizer = NoQuantizer{},
    RefinerT refiner = NoRefiner{}
) {
    return std::make_unique<SearcherImpl<QuantT, RefinerT>>(
        graph, std::move(quantizer), std::move(refiner)
    );
}

}  // namespace deglib::search
