#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/transformation.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
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
 * Indexing Contract:
 * The Reranker is index-agnostic: candidate index c is evaluated directly against `base_vectors[c]`.
 *
 * - With Searcher (Most Common):
 *   Searcher passes the graph's internal indices (0 .. size()-1) into the Reranker, and subsequently
 *   remaps the winning results to external labels.
 *   * Standard / Static Graphs (Default): Vertex i is simply the i-th vector given to the builder.
 *     Pass the same base feature matrix directly!
 *   * Custom Labels or Dynamic Graphs with Deletions:
 *     Base vectors must be aligned so row i contains the vector for label `graph.getExternalLabels()[i]`.
 *
 * - Standalone Usage:
 *   The row ordering of `base_vectors` must match whatever ID space you pass as candidates.
 *   For example, if candidates are external labels in 0..N-1, row l must contain feature vector l.
 */
class Reranker {
  private:
    deglib::distances::FloatSpace space_;
    const void* base_vectors_ = nullptr;
    size_t num_base_vectors_ = 0;
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations_;
    int32_t po_ = 8;  // Prefetch lookahead offset for candidate feature vectors (0 disables prefetch).
    int32_t pl_ = 0;  // Cache lines to prefetch per feature vector (0 = prefetch the full vector).

    // Per-thread scratch pools. The Reranker is used const from multiple threads (one per Searcher
    // worker), so all per-query scratch is thread_local: reused across queries (zero allocation),
    // private per thread (no data race).
    static inline thread_local std::vector<deglib::graph::ObjectDistance> flat_scratch_;
    static inline thread_local std::vector<std::byte> transform_buf1_;
    static inline thread_local std::vector<std::byte> transform_buf2_;
    static inline thread_local std::vector<float> fp16_query_buf_;

    const std::byte* base() const noexcept { return static_cast<const std::byte*>(base_vectors_); }
    size_t row_bytes() const noexcept { return space_.get_data_size(); }

    /**
     * Scores valid candidates against the query into `scratch` (dtype-agnostic).
     * Prefetches the `po`-th next row, `pl` cache lines per row, with `po > 0` hoisted out of the loop.
     * Out-of-range base indices are skipped directly (no sentinel + isfinite scan needed).
     * Returns the number of valid entries written to `scratch`.
     */
    template <typename DistT, typename ParamT>
    static inline size_t score_all(
        std::vector<deglib::graph::ObjectDistance>& scratch,
        const DistT& dist,
        ParamT param,
        const std::byte* base,
        size_t row_bytes,
        size_t num_base,
        std::span<const std::byte> query,
        std::span<const uint32_t> candidates,
        int32_t po,
        int32_t pl
    ) {
        const auto* q = reinterpret_cast<const uint8_t*>(query.data());
        const auto* base_bytes = reinterpret_cast<const uint8_t*>(base);
        const size_t n = candidates.size();
        if (scratch.size() < n) scratch.resize(n);
        auto* dst = scratch.data();
        size_t valid = 0;

        if (po > 0) {
            const size_t prefetch_bytes = (pl > 0) ? static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE : row_bytes;
            const auto prefetch_row = [prefetch_bytes, base_bytes, row_bytes, num_base](uint32_t row) {
                if (row < num_base) {
                    deglib::memory::prefetch(reinterpret_cast<const char*>(base_bytes + static_cast<size_t>(row) * row_bytes), prefetch_bytes);
                }
            };

            const size_t lookahead = static_cast<size_t>(po);
            const size_t warmup = std::min(lookahead, n);
            for (size_t i = 0; i < warmup; ++i) {
                prefetch_row(candidates[i]);
            }

            for (size_t j = 0; j < n; ++j) {
                if (j + lookahead < n) {
                    prefetch_row(candidates[j + lookahead]);
                }
                const uint32_t cand = candidates[j];
                if (cand >= num_base) continue;
                const float d = dist.compare(q, base_bytes + static_cast<size_t>(cand) * row_bytes, param);
                dst[valid++] = deglib::graph::ObjectDistance(cand, d);
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                const uint32_t cand = candidates[j];
                if (cand >= num_base) continue;
                const float d = dist.compare(q, base_bytes + static_cast<size_t>(cand) * row_bytes, param);
                dst[valid++] = deglib::graph::ObjectDistance(cand, d);
            }
        }
        return valid;
    }

    /**
     * Selects the top-k smallest distances from `scratch[0..valid)` into the output spans
     * (sorted via partial_sort/sort, unsorted via nth_element). Pads the tail with max sentinels.
     */
    static inline uint32_t select_top_k(
        std::vector<deglib::graph::ObjectDistance>& scratch,
        size_t valid,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances,
        bool return_distances,
        bool unsorted
    ) {
        const size_t top = std::min<size_t>(k, valid);
        if (top > 0) {
            auto begin = scratch.begin();
            auto end = begin + static_cast<std::ptrdiff_t>(valid);
            if (unsorted) {
                if (top < valid) {
                    std::nth_element(begin, begin + static_cast<std::ptrdiff_t>(top), end);
                }
            } else {
                if (top < valid) {
                    std::partial_sort(begin, begin + static_cast<std::ptrdiff_t>(top), end);
                } else {
                    std::sort(begin, end);
                }
            }
            for (size_t i = 0; i < top; ++i) out_indices[i] = scratch[i].getIdentifier();
            if (return_distances) {
                for (size_t i = 0; i < top; ++i) out_distances[i] = scratch[i].getDistance();
            }
        }
        for (size_t i = top; i < k; ++i) out_indices[i] = std::numeric_limits<uint32_t>::max();
        if (return_distances) {
            for (size_t i = top; i < k; ++i) out_distances[i] = std::numeric_limits<float>::max();
        }
        return static_cast<uint32_t>(top);
    }

  public:
    /**
     * Constructs a candidate reranker.
     *
     * @param space            Distance metric and dimensionality for scoring.
     * @param base_vectors     Contiguous byte span of the row-addressed feature matrix; row c is scored for candidate index c.
     * @param num_base_vectors Number of rows in ``base_vectors``.
     * @param transformations  Optional transformations executed on queries before scoring against base vectors.
     */
    Reranker(
        deglib::distances::FloatSpace space,
        std::span<const std::byte> base_vectors,
        size_t num_base_vectors,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    )
        : space_(std::move(space)), base_vectors_(base_vectors.data()), num_base_vectors_(num_base_vectors), transformations_(std::move(transformations)) {
        if (num_base_vectors_ > 0) {
            if (base_vectors.data() == nullptr) {
                throw std::invalid_argument("Reranker: base_vectors buffer is null");
            }
            const size_t required_bytes = num_base_vectors_ * space_.get_data_size();
            if (base_vectors.size() < required_bytes) {
                throw std::invalid_argument(
                    std::format("Reranker: base_vectors span ({} bytes) is smaller than required ({} bytes)", base_vectors.size(), required_bytes)
                );
            }
        }
    }

    const deglib::distances::FloatSpace& getSpace() const noexcept { return space_; }
    const void* getBaseVectors() const noexcept { return base_vectors_; }
    size_t getNumBaseVectors() const noexcept { return num_base_vectors_; }
    const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& getTransformations() const noexcept { return transformations_; }
    int32_t getPo() const noexcept { return po_; }
    int32_t getPl() const noexcept { return pl_; }

  private:
    void setPrefetch(int32_t po, int32_t pl) noexcept {
        po_ = (po > 0) ? po : 0;
        pl_ = (pl > 0) ? pl : 0;
    }

  public:
    /**
     * @brief Auto-tunes the prefetch parameters (po, pl) by timing the single-query rerank hot path
     * on random candidate sets drawn from `base_vectors`.
     *
     * Sweeps `po` in `{0, 1, 2, 4, 8}` and `pl` up to `code_size / 64`; keeps the minimum of 3 timed
     * passes per combination (single-threaded).
     *
     * @param num_sample_queries Sample queries drawn from `base_vectors` (default 300).
     * @param num_candidates     Candidates scored per query (default 150).
     */
    void optimize(size_t num_sample_queries = 300, size_t num_candidates = 150) {
        if (base_vectors_ == nullptr || num_base_vectors_ == 0 || num_candidates == 0) return;

        const size_t stride = space_.get_data_size();
        if (stride == 0) return;

        num_candidates = std::min(num_candidates, num_base_vectors_);
        const uint32_t k = std::min<uint32_t>(100, static_cast<uint32_t>(num_candidates));
        num_sample_queries = std::min(num_sample_queries, num_base_vectors_);

        // Deterministic random candidate sets (fixed seed -> reproducible tuning).
        std::vector<uint32_t> sample_cands(num_sample_queries * num_candidates);
        uint32_t rng = 1337;
        for (auto& c : sample_cands) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            c = rng % static_cast<uint32_t>(num_base_vectors_);
        }

        // pl grid (cache lines per candidate row); 0 = full row. Capped at code_size/64.
        const int32_t pl_max = static_cast<int32_t>((stride + deglib::memory::L1_CACHE_LINE_SIZE - 1) / deglib::memory::L1_CACHE_LINE_SIZE);
        std::vector<int32_t> try_pls{0};
        for (const int32_t lines : {2, 4, 8, 16, 32, 64}) {
            if (lines <= pl_max) try_pls.push_back(lines);
        }
        // po lookahead beyond 8 gives no measurable gain at the rerank candidate count (~150).
        const std::vector<int32_t> try_pos = {0, 1, 2, 4, 8};

        std::vector<uint32_t> out(k);
        double best_time = std::numeric_limits<double>::max();
        int32_t best_po = po_;
        int32_t best_pl = pl_;

        // Time the real single-query hot path; keep the minimum of 3 passes so OS jitter cannot inflate the winner.
        auto timeit = [&]() -> double {
            double best = std::numeric_limits<double>::max();
            for (int pass = 0; pass < 3; ++pass) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < num_sample_queries; ++i) {
                    rerank(
                        std::span<const std::byte>(base() + i * stride, stride),
                        std::span<const uint32_t>(sample_cands.data() + i * num_candidates, num_candidates), k, std::span<uint32_t>(out.data(), k),
                        std::span<float>{}, false, true
                    );
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            return best;
        };

        for (const int32_t po : try_pos) {
            for (const int32_t pl : try_pls) {
                setPrefetch(po, pl);
                timeit();  // warm-up
                if (const double dur = timeit(); dur < best_time) {
                    best_time = dur;
                    best_po = po;
                    best_pl = pl;
                }
            }
        }

        setPrefetch(best_po, best_pl);
    }

    /**
     * Single-query rerank returning a ResultSet (top-k, ordered by distance).
     *
     * @param query     Raw query bytes (base encoding).
     * @param candidates Row indices into base_vectors; each is scored as base_vectors[c].
     */
    inline deglib::graph::ResultSet rerank(std::span<const std::byte> query, std::span<const uint32_t> candidates, uint32_t k) const {
        deglib::graph::ResultSet result;
        if (base_vectors_ == nullptr || query.empty() || candidates.empty() || k == 0) {
            return result;
        }
        auto& scratch = flat_scratch_;
        const auto param = space_.get_dist_func_param();
        space_.compute([&](const auto& dist) {
            const size_t valid = score_all(scratch, dist, param, base(), row_bytes(), num_base_vectors_, query, candidates, po_, pl_);
            const size_t top = std::min<size_t>(k, valid);
            if (top > 0) {
                auto begin = scratch.begin();
                auto end = begin + static_cast<std::ptrdiff_t>(valid);
                if (top < valid) {
                    std::partial_sort(begin, begin + static_cast<std::ptrdiff_t>(top), end);
                } else {
                    std::sort(begin, end);
                }
                result.reserve(top);
                for (size_t i = 0; i < top; ++i) result.emplace(scratch[i].getIdentifier(), scratch[i].getDistance());
            }
        });
        return result;
    }

    /**
     * Zero-allocation single-query rerank writing top-k into pre-allocated destination spans.
     */
    inline uint32_t rerank(
        std::span<const std::byte> query,
        std::span<const uint32_t> candidates,
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
        if (base_vectors_ == nullptr || query.empty() || candidates.empty() || k == 0) {
            std::fill_n(out_indices.begin(), k, std::numeric_limits<uint32_t>::max());
            if (return_distances) std::fill_n(out_distances.begin(), k, std::numeric_limits<float>::max());
            return 0;
        }
        auto& scratch = flat_scratch_;
        const auto param = space_.get_dist_func_param();
        size_t valid = 0;
        space_.compute([&](const auto& dist) { valid = score_all(scratch, dist, param, base(), row_bytes(), num_base_vectors_, query, candidates, po_, pl_); });
        return select_top_k(scratch, valid, k, out_indices, out_distances, return_distances, unsorted);
    }

    /**
     * Typed single-query rerank for float32 queries. Applies attached transformations if present,
     * otherwise scores directly when the base space is FP32. Query dimension is taken from the span.
     */
    inline uint32_t rerank_f32(
        std::span<const float> query,
        std::span<const uint32_t> candidates,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool return_distances = false,
        bool unsorted = false
    ) const {
        if (!transformations_.empty()) {
            const float* current = query.data();
            for (size_t t = 0; t < transformations_.size(); ++t) {
                const auto& trans = transformations_[t];
                const bool is_last = (t + 1 == transformations_.size());
                const size_t out_bytes = (trans->output_byte_size() > 0) ? trans->output_byte_size() : row_bytes();
                const std::span<const float> in(current, trans->input_dim());
                if (is_last) {
                    if (transform_buf1_.size() < out_bytes) transform_buf1_.resize(out_bytes);
                    trans->transform_query(in, std::span<std::byte>(transform_buf1_.data(), out_bytes));
                    return rerank(
                        std::span<const std::byte>(transform_buf1_.data(), out_bytes), candidates, k, out_indices, out_distances, return_distances, unsorted
                    );
                }
                if (transform_buf2_.size() < out_bytes) transform_buf2_.resize(out_bytes);
                trans->transform_query(in, std::span<std::byte>(transform_buf2_.data(), out_bytes));
                current = reinterpret_cast<const float*>(transform_buf2_.data());
            }
        }

        if (space_.metric().get_data_type() == deglib::distances::MetricDataType::FP32) {
            return rerank(std::as_bytes(query), candidates, k, out_indices, out_distances, return_distances, unsorted);
        }
        throw std::invalid_argument(
            std::format(
                "Reranker: float32 query passed to Reranker with base data type {:#x}, but no transformations were configured. "
                "A transformation (e.g. ScalarQuantizerFP16) is required to convert the query.",
                static_cast<uint8_t>(space_.metric().get_data_type())
            )
        );
    }

    /**
     * Typed single-query rerank for float16 queries (supported for FP16 and FP32 base spaces).
     */
    inline uint32_t rerank_f16(
        std::span<const uint16_t> query,
        std::span<const uint32_t> candidates,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool return_distances = false,
        bool unsorted = false
    ) const {
        const auto dt = space_.metric().get_data_type();
        if (dt == deglib::distances::MetricDataType::FP16) {
            return rerank(std::as_bytes(query), candidates, k, out_indices, out_distances, return_distances, unsorted);
        }
        if (dt == deglib::distances::MetricDataType::FP32) {
            fp16_query_buf_.resize(query.size());
            deglib::distances::fp16::fp16_to_floats(query.data(), fp16_query_buf_.data(), query.size());
            return rerank_f32(std::span<const float>(fp16_query_buf_), candidates, k, out_indices, out_distances, return_distances, unsorted);
        }
        throw std::invalid_argument(
            std::format(
                "Reranker: float16 query passed to Reranker with base data type {:#x}. Direct FP16 queries are only supported for FP16 or FP32 base spaces.",
                static_cast<uint8_t>(dt)
            )
        );
    }

    /**
     * Typed single-query rerank dispatch (float -> rerank_f32, uint16_t -> rerank_f16).
     */
    template <typename QueryT>
    inline uint32_t rerank(
        std::span<const QueryT> query,
        std::span<const uint32_t> candidates,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool return_distances = false,
        bool unsorted = false
    ) const {
        if constexpr (std::is_same_v<QueryT, float>) {
            return rerank_f32(query, candidates, k, out_indices, out_distances, return_distances, unsorted);
        } else if constexpr (std::is_same_v<QueryT, uint16_t>) {
            return rerank_f16(query, candidates, k, out_indices, out_distances, return_distances, unsorted);
        } else {
            static_assert(sizeof(QueryT) == 0, "Unsupported QueryT: must be float or uint16_t");
        }
    }

    /**
     * Batch rerank returning one ResultSet per query (top-k, ordered by distance).
     *
     * @param queries             Flat query bytes, num_queries rows of get_data_size() each.
     * @param candidates          Flat candidate indices, num_queries rows of candidates_per_query each.
     */
    inline std::vector<deglib::graph::ResultSet> rerank(
        std::span<const std::byte> queries,
        size_t num_queries,
        std::span<const uint32_t> candidates,
        size_t candidates_per_query,
        uint32_t k_top = 0,
        size_t num_threads = 0
    ) const {
        if (queries.data() == nullptr || candidates.data() == nullptr) {
            throw std::invalid_argument("rerank: queries and candidates must not be null");
        }
        if (k_top == 0 || k_top > candidates_per_query) {
            k_top = static_cast<uint32_t>(candidates_per_query);
        }

        const size_t stride = row_bytes();
        std::vector<deglib::graph::ResultSet> results(num_queries);
        const auto param = space_.get_dist_func_param();

        space_.compute([&](const auto& dist) {
            deglib::concurrent::parallel_batch_for(0, num_queries, num_threads, [&](size_t begin, size_t end, size_t) {
                auto& scratch = flat_scratch_;
                for (size_t i = begin; i < end; ++i) {
                    const auto q = queries.subspan(i * stride, stride);
                    const auto c = candidates.subspan(i * candidates_per_query, candidates_per_query);
                    auto& result = results[i];
                    if (c.empty() || k_top == 0) continue;

                    const size_t valid = score_all(scratch, dist, param, base(), stride, num_base_vectors_, q, c, po_, pl_);
                    const size_t top = std::min<size_t>(k_top, valid);
                    if (top > 0) {
                        auto it_begin = scratch.begin();
                        auto it_end = it_begin + static_cast<std::ptrdiff_t>(valid);
                        if (top < valid) {
                            std::partial_sort(it_begin, it_begin + static_cast<std::ptrdiff_t>(top), it_end);
                        } else {
                            std::sort(it_begin, it_end);
                        }
                        result.reserve(top);
                        for (size_t j = 0; j < top; ++j) result.emplace(scratch[j].getIdentifier(), scratch[j].getDistance());
                    }
                }
            });
        });

        return results;
    }

    /**
     * Batch rerank writing top-k per query into pre-allocated row-major destination spans.
     */
    inline void rerank(
        std::span<const std::byte> queries,
        size_t num_queries,
        std::span<const uint32_t> candidates,
        size_t candidates_per_query,
        uint32_t k_top,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances,
        bool return_distances,
        bool unsorted,
        size_t num_threads = 0
    ) const {
        if (queries.data() == nullptr || candidates.data() == nullptr || out_indices.data() == nullptr) {
            throw std::invalid_argument("rerank: queries, candidates, and out_indices must not be null");
        }
        if (return_distances && out_distances.data() == nullptr) {
            throw std::invalid_argument("rerank: return_distances is true but out_distances is null");
        }
        if (k_top == 0 || k_top > candidates_per_query) {
            k_top = static_cast<uint32_t>(candidates_per_query);
        }

        const size_t stride = row_bytes();
        const auto param = space_.get_dist_func_param();

        space_.compute([&](const auto& dist) {
            deglib::concurrent::parallel_batch_for(0, num_queries, num_threads, [&](size_t begin, size_t end, size_t) {
                std::vector<deglib::graph::ObjectDistance> scratch;
                scratch.reserve(candidates_per_query);

                for (size_t i = begin; i < end; ++i) {
                    const auto q = queries.subspan(i * stride, stride);
                    const auto c = candidates.subspan(i * candidates_per_query, candidates_per_query);

                    const size_t valid = score_all(scratch, dist, param, base(), stride, num_base_vectors_, q, c, po_, pl_);
                    select_top_k(
                        scratch, valid, k_top, out_indices.subspan(i * k_top, k_top),
                        return_distances ? out_distances.subspan(i * k_top, k_top) : std::span<float>{}, return_distances, unsorted
                    );
                }
            });
        });
    }
};

}  // namespace deglib::search
