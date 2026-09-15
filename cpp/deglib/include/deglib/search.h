#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <chrono>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace deglib::search {

/**
 * High-performance candidate reranker holding base vectors and the distance metric used to
 * re-evaluate graph candidates against full-precision features.
 *
 * @tparam DataT Element type of the base feature vectors (e.g. float, uint16_t for fp16).
 */
template <typename DataT>
class Reranker {
  private:
    deglib::distances::FloatSpace space_;
    const DataT* base_vectors_ = nullptr;
    size_t num_base_vectors_ = 0;
    int32_t po_ = 8;  // Prefetch lookahead offset for candidate feature vectors (0 disables prefetch).
    int32_t pl_ = 0;  // Cache lines to prefetch per feature vector (0 = prefetch the full vector).

    /**
     * Internal core: Evaluates candidate distances against base vectors using a distance function object
     * and returns the top-k nearest candidates in a max-heap.
     *
     * @param po Lookahead offset for prefetching candidate feature vectors (0 to disable)
     * @param pl Number of cache lines to prefetch per feature vector (0 = prefetch full vector)
     */
    template <typename DistFuncObj, typename DistFuncParam>
    static inline deglib::graph::ResultSet rerank_single_heap(
        const DistFuncObj& dist_func_obj,
        DistFuncParam param,
        const uint8_t* query_ptr,
        const uint8_t* target_vectors,
        size_t target_count,
        size_t byte_stride_target,
        const uint32_t* candidate_indices,
        size_t num_candidates,
        size_t k,
        int32_t po = 0,
        int32_t pl = 0
    ) {
        deglib::graph::ResultSet heap;
        heap.reserve(k);
        float max_dist = std::numeric_limits<float>::max();

        const auto accumulate = [&](uint32_t cand_idx, const uint8_t* cand_ptr) {
            const float dist = dist_func_obj.compare(query_ptr, cand_ptr, param);
            if (heap.size() < k) {
                heap.emplace(cand_idx, dist);
                if (heap.size() == k) {
                    max_dist = heap.top().getDistance();
                }
            } else if (dist < max_dist) {
                max_dist = heap.replace_top(cand_idx, dist).getDistance();
            }
        };

        if (po > 0) {
            const size_t prefetch_bytes = (pl > 0) ? (static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE) : byte_stride_target;
            const auto prefetch_feature = [prefetch_bytes](const char* ptr) { deglib::memory::prefetch(ptr, prefetch_bytes); };

            const size_t init_prefetch = std::min<size_t>(static_cast<size_t>(po), num_candidates);
            for (size_t i = 0; i < init_prefetch; ++i) {
                const uint32_t c_idx = candidate_indices[i];
                if (c_idx < target_count) {
                    prefetch_feature(reinterpret_cast<const char*>(target_vectors + c_idx * byte_stride_target));
                }
            }

            for (size_t j = 0; j < num_candidates; ++j) {
                if (j + static_cast<size_t>(po) < num_candidates) {
                    const uint32_t next_idx = candidate_indices[j + static_cast<size_t>(po)];
                    if (next_idx < target_count) {
                        prefetch_feature(reinterpret_cast<const char*>(target_vectors + next_idx * byte_stride_target));
                    }
                }
                const uint32_t cand_idx = candidate_indices[j];
                if (cand_idx >= target_count) {
                    continue;
                }
                accumulate(cand_idx, target_vectors + cand_idx * byte_stride_target);
            }
        } else {
            for (size_t j = 0; j < num_candidates; ++j) {
                const uint32_t cand_idx = candidate_indices[j];
                if (cand_idx >= target_count) {
                    continue;
                }
                accumulate(cand_idx, target_vectors + cand_idx * byte_stride_target);
            }
        }
        return heap;
    }

    /**
     * Internal helper: Drains a ResultSet into pre-allocated destination spans without inner branching.
     */
    static inline uint32_t drain_heap_into(
        deglib::graph::ResultSet& heap,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances,
        bool return_distances,
        bool unsorted
    ) {
        const size_t top_n = std::min<size_t>(k, heap.size());

        if (unsorted) {
            for (size_t i = 0; i < top_n; ++i) {
                out_indices[i] = heap[i].getIdentifier();
            }
            if (return_distances) {
                for (size_t i = 0; i < top_n; ++i) {
                    out_distances[i] = heap[i].getDistance();
                }
            }
        } else {
            if (return_distances) {
                for (size_t i = top_n; i > 0; --i) {
                    const auto next = heap.top();
                    out_indices[i - 1] = next.getIdentifier();
                    out_distances[i - 1] = next.getDistance();
                    heap.pop();
                }
            } else {
                for (size_t i = top_n; i > 0; --i) {
                    out_indices[i - 1] = heap.top().getIdentifier();
                    heap.pop();
                }
            }
        }

        for (size_t i = top_n; i < k; ++i) {
            out_indices[i] = std::numeric_limits<uint32_t>::max();
        }
        if (return_distances) {
            for (size_t i = top_n; i < k; ++i) {
                out_distances[i] = std::numeric_limits<float>::max();
            }
        }
        return static_cast<uint32_t>(top_n);
    }

  public:
    Reranker(deglib::distances::FloatSpace space, const DataT* base_vectors, size_t num_base_vectors)
        : space_(std::move(space)), base_vectors_(base_vectors), num_base_vectors_(num_base_vectors) {}

    const deglib::distances::FloatSpace& getSpace() const noexcept { return space_; }
    const DataT* getBaseVectors() const noexcept { return base_vectors_; }
    size_t getNumBaseVectors() const noexcept { return num_base_vectors_; }
    int32_t getPo() const noexcept { return po_; }
    int32_t getPl() const noexcept { return pl_; }
    void setPo(int32_t po) noexcept { po_ = (po > 0) ? po : 0; }
    void setPl(int32_t pl) noexcept { pl_ = (pl > 0) ? pl : 0; }
    void setPrefetch(int32_t po, int32_t pl) noexcept {
        po_ = (po > 0) ? po : 0;
        pl_ = (pl > 0) ? pl : 0;
    }

    /**
     * Auto-tunes the rerank prefetch parameters (po, pl) using empirical timing on sample queries.
     * The tuning reflects the real operating point: each query scores `num_candidates` candidates and
     * reduces them to the top `k`. Prefetch is on by default (po = 8, pl = 0); tuning only replaces one good setting with another.
     *
     * @param sample_queries  Pointer to contiguous sample queries of dimension dim.
     * @param n_queries       Number of sample queries.
     * @param num_candidates  Candidates scored per query (clamped to the base count).
     * @param k               Candidates kept per query after reranking (clamped to num_candidates).
     */
    void optimize(const DataT* sample_queries, size_t n_queries, size_t num_candidates, uint32_t k) {
        const uint32_t dim = space_.dim();
        if (!base_vectors_ || num_base_vectors_ == 0 || n_queries == 0 || dim == 0 || num_candidates == 0 || k == 0) return;

        num_candidates = std::min(num_candidates, num_base_vectors_);
        k = std::min<uint32_t>(k, static_cast<uint32_t>(num_candidates));

        std::vector<uint32_t> sample_cands(n_queries * num_candidates);
        uint32_t rng = 1337;
        for (size_t i = 0; i < sample_cands.size(); ++i) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            sample_cands[i] = rng % static_cast<uint32_t>(num_base_vectors_);
        }

        const std::vector<int32_t> try_pos = {0, 4, 8, 12, 16};
        const std::vector<int32_t> try_pls = {0, 2, 4};

        int32_t best_po = po_;
        int32_t best_pl = pl_;
        double best_time = std::numeric_limits<double>::max();

        std::vector<uint32_t> dummy_out_idx(k);
        std::vector<float> dummy_out_dist(k);

        for (int32_t test_po : try_pos) {
            for (int32_t test_pl : try_pls) {
                po_ = test_po;
                pl_ = test_pl;

                for (size_t q = 0; q < std::min<size_t>(3, n_queries); ++q) {
                    rerank(
                        sample_queries + q * dim, dim, sample_cands.data() + q * num_candidates, num_candidates, k, dummy_out_idx.data(), dummy_out_dist.data(),
                        false, true
                    );
                }

                auto t_start = std::chrono::high_resolution_clock::now();
                for (size_t q = 0; q < n_queries; ++q) {
                    rerank(
                        sample_queries + q * dim, dim, sample_cands.data() + q * num_candidates, num_candidates, k, dummy_out_idx.data(), dummy_out_dist.data(),
                        false, true
                    );
                }
                auto t_end = std::chrono::high_resolution_clock::now();
                double dur = std::chrono::duration<double, std::micro>(t_end - t_start).count();

                if (dur < best_time) {
                    best_time = dur;
                    best_po = test_po;
                    best_pl = test_pl;
                }
            }
        }

        po_ = best_po;
        pl_ = best_pl;
    }

    /**
     * Auto-tunes prefetch parameters (po, pl) by sampling the reranker's own base vectors as queries.
     * Type-correct for any base feature type (fp32, fp16, ...): no external query buffer or conversion.
     *
     * @param sample_count    Number of base vectors used as sample queries (capped at the base count).
     * @param num_candidates  Candidates scored per query (clamped to the base count).
     * @param k               Candidates kept per query after reranking (clamped to num_candidates).
     */
    void optimize(size_t sample_count = 50, size_t num_candidates = 100, uint32_t k = 10) {
        const size_t n = std::min<size_t>(sample_count, num_base_vectors_);
        optimize(base_vectors_, n, num_candidates, k);
    }

    /**
     * Single-query candidate reranking returning a ResultSet.
     *
     * @param query             Byte span representing the query vector
     * @param candidate_indices Span containing candidate vertex IDs to rerank
     * @param k                 Number of top results to return
     * @return                  ResultSet containing top-k nearest candidates in max-heap order
     */
    inline deglib::graph::ResultSet rerank(std::span<const std::byte> query, std::span<const uint32_t> candidate_indices, uint32_t k) const {
        if (query.empty() || candidate_indices.empty() || base_vectors_ == nullptr || k == 0) {
            return deglib::graph::ResultSet{};
        }

        const size_t byte_stride_target = space_.get_data_size();
        const uint8_t* query_ptr = reinterpret_cast<const uint8_t*>(query.data());
        const uint8_t* t_ptr = reinterpret_cast<const uint8_t*>(base_vectors_);
        const auto param = space_.get_dist_func_param();

        deglib::graph::ResultSet heap;
        space_.compute([&](const auto& dist_func_obj) {
            heap = rerank_single_heap(
                dist_func_obj, param, query_ptr, t_ptr, num_base_vectors_, byte_stride_target, candidate_indices.data(), candidate_indices.size(), k, po_, pl_
            );
        });
        return heap;
    }

    /**
     * Zero-allocation C++20 single-query candidate reranking writing directly into pre-allocated destination spans.
     *
     * @param query             Byte span representing the query vector
     * @param candidate_indices Span containing candidate vertex IDs to rerank
     * @param k                 Number of top results to return
     * @param out_indices       Destination span for the top-k result indices (must have size >= k)
     * @param out_distances     Optional destination span for distances (must have size >= k if return_distances is true)
     * @param return_distances  Explicit boolean flag indicating whether to compute and write output distances
     * @param unsorted          If true, returns results in fast unsorted heap order; if false, sorts nearest-first
     * @return                  Actual number of valid results written (<= k)
     */
    inline uint32_t rerank(
        std::span<const std::byte> query,
        std::span<const uint32_t> candidate_indices,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool return_distances = false,
        bool unsorted = false
    ) const {
        if (out_indices.size() < k) {
            throw std::invalid_argument("rerank: out_indices span is smaller than k");
        }
        if (return_distances && out_distances.size() < k) {
            throw std::invalid_argument("rerank: return_distances is true but out_distances span is smaller than k");
        }

        deglib::graph::ResultSet heap = rerank(query, candidate_indices, k);
        return drain_heap_into(heap, k, out_indices, out_distances, return_distances, unsorted);
    }

    /**
     * Typed single-query rerank used by the Searcher refiner integration. Handles float <-> fp16 query
     * conversion when the query type differs from the base feature type.
     */
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
        if (!base_vectors_ || num_cands == 0 || k == 0) return 0;

        auto cands_span = std::span<const uint32_t>(candidate_indices, num_cands);
        auto out_idx_span = std::span<uint32_t>(out_indices, k);
        auto out_dist_span = (return_distances && out_distances) ? std::span<float>(out_distances, k) : std::span<float>{};

        if constexpr (std::is_same_v<QueryT, DataT>) {
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(query), dim * sizeof(QueryT));
            return rerank(q_span, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted);
        } else if constexpr (std::is_same_v<QueryT, float> && std::is_same_v<DataT, uint16_t>) {
            std::vector<uint16_t> q_fp16(dim);
            deglib::distances::fp16::floats_to_fp16(query, q_fp16.data(), dim);
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(q_fp16.data()), dim * sizeof(uint16_t));
            return rerank(q_span, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted);
        } else if constexpr (std::is_same_v<QueryT, uint16_t> && std::is_same_v<DataT, float>) {
            std::vector<float> q_f32(dim);
            deglib::distances::fp16::fp16_to_floats(query, q_f32.data(), dim);
            auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(q_f32.data()), dim * sizeof(float));
            return rerank(q_span, cands_span, k, out_idx_span, out_dist_span, return_distances, unsorted);
        }
        return 0;
    }

    /**
     * Batch candidate reranking returning a vector of ResultSets.
     *
     * @param queries              Pointer to [num_queries x dim] query vectors
     * @param num_queries          Number of query vectors
     * @param base_candidates      Pointer to [num_queries x candidates_per_query] candidate indices
     * @param candidates_per_query Number of candidate indices provided per query
     * @param k_top                Number of top candidates to keep per query (0 = all)
     * @param num_threads          Number of worker threads (0 = auto-detect)
     * @return                     std::vector<ResultSet> containing the top-k result sets per query (unsorted heap order)
     */
    inline std::vector<deglib::graph::ResultSet> rerank(
        const void* queries,
        size_t num_queries,
        const uint32_t* base_candidates,
        size_t candidates_per_query,
        size_t k_top = 0,
        size_t num_threads = 0
    ) const {
        if (queries == nullptr || base_candidates == nullptr) {
            throw std::invalid_argument("rerank: queries and base_candidates must not be null");
        }

        if (k_top == 0 || k_top > candidates_per_query) {
            k_top = candidates_per_query;
        }

        const size_t byte_stride_query = space_.get_data_size();
        const size_t byte_stride_target = space_.get_data_size();

        const uint8_t* q_ptr = static_cast<const uint8_t*>(queries);
        const uint8_t* t_ptr = reinterpret_cast<const uint8_t*>(base_vectors_);

        const auto param = space_.get_dist_func_param();

        std::vector<deglib::graph::ResultSet> results(num_queries);

        space_.compute([&](const auto& dist_func_obj) {
            deglib::concurrent::parallel_for(0, num_queries, num_threads, [&](size_t i, size_t) {
                const uint8_t* query_ptr = q_ptr + i * byte_stride_query;
                const uint32_t* cand_row = base_candidates + i * candidates_per_query;

                results[i] = rerank_single_heap(
                    dist_func_obj, param, query_ptr, t_ptr, num_base_vectors_, byte_stride_target, cand_row, candidates_per_query, k_top, po_, pl_
                );
            });
        });

        return results;
    }
};

}  // namespace deglib::search
