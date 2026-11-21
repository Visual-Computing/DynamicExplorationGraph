#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/transformation.h"
#include "deglib/transformation/scalar/fp16.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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
    std::unique_ptr<std::byte[]> storage_;
    std::byte* base_vectors_ = nullptr;
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

    const std::byte* base() const noexcept { return base_vectors_; }
    size_t row_bytes() const noexcept { return space_.get_data_size(); }

    Reranker(
        deglib::distances::FloatSpace space,
        std::unique_ptr<std::byte[]> storage,
        std::byte* base_vectors,
        size_t num_base_vectors,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations
    ) noexcept
        : space_(std::move(space)),
          storage_(std::move(storage)),
          base_vectors_(base_vectors),
          num_base_vectors_(num_base_vectors),
          transformations_(std::move(transformations)) {}

    static inline void transform_single_row(
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        std::span<const float> src_row,
        std::span<std::byte> dst_row,
        std::vector<std::byte>& tmp1,
        std::vector<std::byte>& tmp2
    ) {
        const float* current = src_row.data();
        for (size_t t = 0; t < transformations.size(); ++t) {
            const auto& trans = transformations[t];
            const bool is_last = (t + 1 == transformations.size());
            const std::span<const float> in(current, trans->input_dim());
            if (is_last) {
                trans->transform_query(in, dst_row);
                return;
            }
            const size_t out_bytes = trans->output_byte_size();
            auto& buf = (t % 2 == 0) ? tmp1 : tmp2;
            if (buf.size() < out_bytes) buf.resize(out_bytes);
            trans->transform_query(in, std::span<std::byte>(buf.data(), out_bytes));
            current = reinterpret_cast<const float*>(buf.data());
        }
    }

    /**
     * Scores valid candidates against the query into `scratch` (dtype-agnostic).
     * Batches valid candidates into fixed-size stack chunks and dispatches through
     * FloatSpace::get_batch_dist_func() so the distance kernel stays fully inlined
     * with lookahead prefetching without instantiating Reranker per distance type.
     * Out-of-range base indices are skipped directly (no sentinel + isfinite scan needed).
     * Returns the number of valid entries written to `scratch`.
     */
    static inline size_t score_all(
        std::vector<deglib::graph::ObjectDistance>& scratch,
        const deglib::distances::FloatSpace& space,
        const std::byte* base,
        size_t row_bytes,
        size_t num_base,
        std::span<const std::byte> query,
        std::span<const uint32_t> candidates,
        int32_t po,
        int32_t pl
    ) {
        const auto* q = query.data();
        const auto* base_bytes = reinterpret_cast<const uint8_t*>(base);
        const size_t n = candidates.size();
        if (scratch.size() < n) scratch.resize(n);
        auto* dst = scratch.data();
        const auto batch_dist_func = space.get_batch_dist_func();
        const auto param = space.get_dist_func_param();
        const size_t prefetch_bytes = (pl > 0) ? static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE : row_bytes;

        constexpr size_t kBatchSize = 64;
        const void* ptrs[kBatchSize];
        uint32_t ids[kBatchSize];
        float dists[kBatchSize];

        size_t valid = 0;
        size_t batch = 0;
        for (size_t j = 0; j < n; ++j) {
            const uint32_t cand = candidates[j];
            if (cand >= num_base) continue;
            ids[batch] = cand;
            ptrs[batch] = base_bytes + static_cast<size_t>(cand) * row_bytes;
            if (++batch == kBatchSize) {
                batch_dist_func(q, ptrs, kBatchSize, param, dists, po, prefetch_bytes);
                for (size_t i = 0; i < kBatchSize; ++i) {
                    dst[valid++] = deglib::graph::ObjectDistance(ids[i], dists[i]);
                }
                batch = 0;
            }
        }
        if (batch > 0) {
            batch_dist_func(q, ptrs, batch, param, dists, po, prefetch_bytes);
            for (size_t i = 0; i < batch; ++i) {
                dst[valid++] = deglib::graph::ObjectDistance(ids[i], dists[i]);
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

    static void validate_transformations_for_space(
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        const deglib::distances::FloatSpace& target_space,
        const char* caller
    ) {
        for (const auto& t : transformations) {
            if (!t) {
                throw std::invalid_argument(std::format("{}: transformation pipeline contains null entry", caller));
            }
            if (!t->is_fitted()) {
                throw std::invalid_argument(std::format("{}: transformation '{}' is not fitted", caller, t->name()));
            }
        }
        for (size_t i = 0; i + 1 < transformations.size(); ++i) {
            if (transformations[i]->output_dim() != transformations[i + 1]->input_dim()) {
                throw std::invalid_argument(std::format("{}: transformation chain output_dim does not match next input_dim", caller));
            }
        }
        if (!transformations.empty()) {
            const auto& last = transformations.back();
            if (last->output_dim() != target_space.dim() || last->output_byte_size() != target_space.get_data_size()) {
                throw std::invalid_argument(std::format("{}: last transformation output does not match target FloatSpace", caller));
            }
        }
    }

  public:
    Reranker(const Reranker&) = delete;
    Reranker& operator=(const Reranker&) = delete;
    Reranker(Reranker&&) noexcept = default;
    Reranker& operator=(Reranker&&) noexcept = default;

    /**
     * Constructs an owning candidate reranker by copying `base_vectors` into internal C++ storage.
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
        : space_(std::move(space)),
          num_base_vectors_(num_base_vectors),
          transformations_(std::move(transformations)) {
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
            deglib::memory::allocate_aligned(storage_, base_vectors_, required_bytes);
            std::memcpy(base_vectors_, base_vectors.data(), required_bytes);
        }
    }

    /**
     * Builds an owning Reranker directly from float32 vectors in C++.
     * `space` defines the target distance metric, output dimensionality, and SIMD instruction set.
     * If `space.metric()` is non-FP32 (e.g. FP16) and `transformations` is empty, fits the quantizer
     * on `data` (`dim = space.dim()`) and converts directly into internal storage (zero intermediate full-matrix copies).
     * When `row_indices` is non-empty, gathers rows in `row_indices` order directly into storage.
     */
    static Reranker from_data(
        deglib::distances::FloatSpace space,
        std::span<const float> data,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t threads = 0,
        std::span<const uint32_t> row_indices = {}
    ) {
        validate_transformations_for_space(transformations, space, "Reranker::from_data");
        const uint32_t dim = transformations.empty() ? static_cast<uint32_t>(space.dim()) : transformations.front()->input_dim();
        if (dim == 0) throw std::invalid_argument("Reranker::from_data: dim must be > 0");
        if (data.size() % dim != 0) throw std::invalid_argument("Reranker::from_data: data size must be a multiple of dim");
        const size_t src_rows = data.size() / dim;
        const size_t num_rows = row_indices.empty() ? src_rows : row_indices.size();
        for (uint32_t idx : row_indices) {
            if (idx >= src_rows) {
                throw std::out_of_range(std::format("Reranker::from_data: row index {} out of bounds (src_rows={})", idx, src_rows));
            }
        }

        const auto target_metric = space.metric();
        const auto target_dt = target_metric.get_data_type();
        if (transformations.empty() && target_dt == deglib::distances::MetricDataType::FP32) {
            const size_t row_bytes = space.get_data_size();
            std::unique_ptr<std::byte[]> storage;
            std::byte* dst_base = nullptr;
            deglib::memory::allocate_aligned(storage, dst_base, num_rows * row_bytes);
            if (num_rows > 0) {
                if (row_indices.empty()) {
                    std::memcpy(dst_base, data.data(), num_rows * row_bytes);
                } else {
                    deglib::concurrent::parallel_batch_for(
                        size_t{0}, num_rows, threads,
                        [&](size_t begin, size_t end, size_t) {
                            for (size_t i = begin; i < end; ++i) {
                                std::memcpy(dst_base + i * row_bytes, data.data() + static_cast<size_t>(row_indices[i]) * dim, row_bytes);
                            }
                        },
                        64
                    );
                }
            }
            return Reranker(std::move(space), std::move(storage), dst_base, num_rows, {});
        }

        if (transformations.empty()) {
            if (target_dt == deglib::distances::MetricDataType::FP16) {
                float global_abs_max = 0.0f;
                if (num_rows > 0) {
                    const size_t actual_threads = (threads == 0)
                        ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64})
                        : threads;
                    std::vector<float> thread_max(actual_threads, 0.0f);
                    deglib::concurrent::parallel_batch_for(
                        size_t{0}, num_rows, actual_threads,
                        [&](size_t begin, size_t end, size_t tid) {
                            float local_max = thread_max[tid];
                            for (size_t i = begin; i < end; ++i) {
                                const size_t src_row = row_indices.empty() ? i : row_indices[i];
                                const float* vec = data.data() + src_row * dim;
                                for (uint32_t d = 0; d < dim; ++d) {
                                    const float v = std::abs(vec[d]);
                                    if (v > local_max) local_max = v;
                                }
                            }
                            thread_max[tid] = local_max;
                        },
                        64
                    );
                    for (float m : thread_max) {
                        if (m > global_abs_max) global_abs_max = m;
                    }
                }
                auto q = std::make_shared<deglib::transformation::ScalarQuantizerFP16>(dim, global_abs_max);
                transformations.push_back(std::move(q));
            } else {
                if (row_indices.empty()) {
                    transformations = deglib::transformation::make_quantizer_pipeline(data, dim, target_metric, -1.0f, true, threads);
                } else {
                    const size_t sample_rows = std::min<size_t>(num_rows, 10000);
                    std::vector<float> sample(sample_rows * dim);
                    for (size_t s = 0; s < sample_rows; ++s) {
                        const size_t r = row_indices[(s * num_rows) / sample_rows];
                        std::memcpy(sample.data() + s * dim, data.data() + r * dim, dim * sizeof(float));
                    }
                    transformations = deglib::transformation::make_quantizer_pipeline(std::span<const float>(sample), dim, target_metric, -1.0f, true, threads);
                }
            }
        }

        const size_t row_bytes = space.get_data_size();
        std::unique_ptr<std::byte[]> storage;
        std::byte* dst_base = nullptr;
        deglib::memory::allocate_aligned(storage, dst_base, num_rows * row_bytes);
        if (num_rows > 0) {
            deglib::concurrent::parallel_batch_for(
                size_t{0}, num_rows, threads,
                [&](size_t begin, size_t end, size_t) {
                    std::vector<std::byte> tmp1, tmp2;
                    for (size_t i = begin; i < end; ++i) {
                        const size_t src_row = row_indices.empty() ? i : row_indices[i];
                        transform_single_row(
                            transformations,
                            std::span<const float>(data.data() + src_row * dim, dim),
                            std::span<std::byte>(dst_base + i * row_bytes, row_bytes),
                            tmp1, tmp2
                        );
                    }
                },
                64
            );
        }
        return Reranker(std::move(space), std::move(storage), dst_base, num_rows, std::move(transformations));
    }

    /**
     * Builds an owning Reranker from already-encoded/quantized byte rows (e.g. FP16 or INT8).
     * When `row_indices` is non-empty, gathers rows in `row_indices` order directly into storage.
     */
    static Reranker from_raw(
        deglib::distances::FloatSpace space,
        std::span<const std::byte> data,
        size_t num_source_rows,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t threads = 0,
        std::span<const uint32_t> row_indices = {}
    ) {
        const size_t row_bytes = space.get_data_size();
        if (data.size() < num_source_rows * row_bytes) {
            throw std::invalid_argument("Reranker::from_raw: data span is smaller than num_source_rows * row_bytes");
        }
        const size_t num_rows = row_indices.empty() ? num_source_rows : row_indices.size();
        for (uint32_t idx : row_indices) {
            if (idx >= num_source_rows) {
                throw std::out_of_range(std::format("Reranker::from_raw: row index {} out of bounds (num_source_rows={})", idx, num_source_rows));
            }
        }
        std::unique_ptr<std::byte[]> storage;
        std::byte* dst_base = nullptr;
        deglib::memory::allocate_aligned(storage, dst_base, num_rows * row_bytes);
        if (num_rows > 0) {
            if (row_indices.empty()) {
                std::memcpy(dst_base, data.data(), num_rows * row_bytes);
            } else {
                deglib::concurrent::parallel_batch_for(
                    size_t{0}, num_rows, threads,
                    [&](size_t begin, size_t end, size_t) {
                        for (size_t i = begin; i < end; ++i) {
                            std::memcpy(dst_base + i * row_bytes, data.data() + static_cast<size_t>(row_indices[i]) * row_bytes, row_bytes);
                        }
                    },
                    64
                );
            }
        }
        return Reranker(std::move(space), std::move(storage), dst_base, num_rows, std::move(transformations));
    }

    /**
     * Builds an owning Reranker directly from an InternalGraph (FP32 or FP16) in internal vertex order,
     * quantizing on the fly into internal storage without intermediate copies.
     */
    static Reranker from_graph(
        const deglib::graph::InternalGraph& graph,
        deglib::distances::FloatSpace target_space,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t threads = 0
    ) {
        const auto& src_space = graph.getFeatureSpace();
        const auto src_dt = src_space.metric().get_data_type();
        if (src_dt != deglib::distances::MetricDataType::FP32 && src_dt != deglib::distances::MetricDataType::FP16) {
            throw std::invalid_argument("Cannot build Reranker from a non-float graph without external data");
        }

        const size_t count = graph.size();
        const uint32_t dim = src_space.dim();
        validate_transformations_for_space(transformations, target_space, "Reranker::from_graph");
        if (transformations.empty()) {
            if (target_space.dim() != dim) {
                throw std::invalid_argument(
                    std::format("Reranker::from_graph: target FloatSpace dim ({}) does not match graph dim ({})", target_space.dim(), dim)
                );
            }
        } else if (transformations.front()->input_dim() != dim) {
            throw std::invalid_argument(
                std::format(
                    "Reranker::from_graph: first transformation input_dim ({}) does not match graph dim ({})",
                    transformations.front()->input_dim(), dim
                )
            );
        }

        const bool src_is_fp16 = (src_dt == deglib::distances::MetricDataType::FP16);
        const auto target_metric = target_space.metric();
        const auto target_dt = target_metric.get_data_type();

        if (count == 0) {
            return Reranker(std::move(target_space), std::unique_ptr<std::byte[]>{}, nullptr, 0, std::move(transformations));
        }

        // Direct copy when source and target are both FP16 (or both FP32 with no transformations)
        if ((src_is_fp16 && target_dt == deglib::distances::MetricDataType::FP16 && transformations.empty()) ||
            (!src_is_fp16 && target_dt == deglib::distances::MetricDataType::FP32 && transformations.empty())) {
            const size_t row_bytes = target_space.get_data_size();
            std::unique_ptr<std::byte[]> storage;
            std::byte* dst_base = nullptr;
            deglib::memory::allocate_aligned(storage, dst_base, count * row_bytes);
            deglib::concurrent::parallel_batch_for(
                size_t{0}, count, threads,
                [&](size_t begin, size_t end, size_t) {
                    for (size_t i = begin; i < end; ++i) {
                        std::memcpy(dst_base + i * row_bytes, graph.getFeatureVector(static_cast<uint32_t>(i)), row_bytes);
                    }
                },
                64
            );
            return Reranker(std::move(target_space), std::move(storage), dst_base, count, {});
        }

        // FP16 graph -> FP32 reranker
        if (src_is_fp16 && target_dt == deglib::distances::MetricDataType::FP32 && transformations.empty()) {
            const size_t row_bytes = target_space.get_data_size();
            std::unique_ptr<std::byte[]> storage;
            std::byte* dst_base = nullptr;
            deglib::memory::allocate_aligned(storage, dst_base, count * row_bytes);
            deglib::concurrent::parallel_batch_for(
                size_t{0}, count, threads,
                [&](size_t begin, size_t end, size_t) {
                    for (size_t i = begin; i < end; ++i) {
                        const auto* src = reinterpret_cast<const uint16_t*>(graph.getFeatureVector(static_cast<uint32_t>(i)));
                        auto* dst = reinterpret_cast<float*>(dst_base + i * row_bytes);
                        deglib::distances::fp16::fp16_to_floats(src, dst, dim);
                    }
                },
                64
            );
            return Reranker(std::move(target_space), std::move(storage), dst_base, count, {});
        }

        // FP32 graph -> FP16 (auto-fit ScalarQuantizerFP16 if no transformations supplied)
        if (!src_is_fp16 && target_dt == deglib::distances::MetricDataType::FP16 && transformations.empty()) {
            const size_t actual_threads = (threads == 0)
                ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64})
                : threads;
            std::vector<float> thread_max(actual_threads, 0.0f);
            deglib::concurrent::parallel_batch_for(
                size_t{0}, count, actual_threads,
                [&](size_t begin, size_t end, size_t tid) {
                    float local_max = thread_max[tid];
                    for (size_t i = begin; i < end; ++i) {
                        const float* vec = reinterpret_cast<const float*>(graph.getFeatureVector(static_cast<uint32_t>(i)));
                        for (uint32_t d = 0; d < dim; ++d) {
                            const float v = std::abs(vec[d]);
                            if (v > local_max) local_max = v;
                        }
                    }
                    thread_max[tid] = local_max;
                },
                64
            );
            float global_abs_max = 0.0f;
            for (float m : thread_max) {
                if (m > global_abs_max) global_abs_max = m;
            }
            transformations.push_back(std::make_shared<deglib::transformation::ScalarQuantizerFP16>(dim, global_abs_max));
        } else if (transformations.empty()) {
            // Other quantized target metric from float graph: fit pipeline on strided sample
            const size_t sample_rows = std::min<size_t>(count, 10000);
            std::vector<float> sample(sample_rows * dim);
            for (size_t s = 0; s < sample_rows; ++s) {
                const uint32_t v = static_cast<uint32_t>((s * count) / sample_rows);
                if (src_is_fp16) {
                    deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(graph.getFeatureVector(v)), sample.data() + s * dim, dim);
                } else {
                    std::memcpy(sample.data() + s * dim, graph.getFeatureVector(v), dim * sizeof(float));
                }
            }
            transformations = deglib::transformation::make_quantizer_pipeline(std::span<const float>(sample), dim, target_metric, -1.0f, true, threads);
        }

        const size_t row_bytes = target_space.get_data_size();
        std::unique_ptr<std::byte[]> storage;
        std::byte* dst_base = nullptr;
        deglib::memory::allocate_aligned(storage, dst_base, count * row_bytes);
        deglib::concurrent::parallel_batch_for(
            size_t{0}, count, threads,
            [&](size_t begin, size_t end, size_t) {
                std::vector<std::byte> tmp1, tmp2;
                std::vector<float> f32_row(src_is_fp16 ? dim : 0);
                for (size_t i = begin; i < end; ++i) {
                    const std::byte* raw = graph.getFeatureVector(static_cast<uint32_t>(i));
                    const float* vec = nullptr;
                    if (src_is_fp16) {
                        deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(raw), f32_row.data(), dim);
                        vec = f32_row.data();
                    } else {
                        vec = reinterpret_cast<const float*>(raw);
                    }
                    transform_single_row(
                        transformations,
                        std::span<const float>(vec, dim),
                        std::span<std::byte>(dst_base + i * row_bytes, row_bytes),
                        tmp1, tmp2
                    );
                }
            },
            64
        );
        return Reranker(std::move(target_space), std::move(storage), dst_base, count, std::move(transformations));
    }

    const deglib::distances::FloatSpace& getSpace() const noexcept { return space_; }
    const void* getBaseVectors() const noexcept { return base(); }
    const std::byte* getBaseVector(size_t index) const noexcept { return base_vectors_ + index * row_bytes(); }
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

        const size_t data_bytes = space_.get_data_size();
        const size_t stride = row_bytes();
        if (data_bytes == 0 || stride == 0) return;

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
        const int32_t pl_max = static_cast<int32_t>((data_bytes + deglib::memory::L1_CACHE_LINE_SIZE - 1) / deglib::memory::L1_CACHE_LINE_SIZE);
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
                        std::span<const std::byte>(base() + i * stride, data_bytes),
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
        const size_t valid = score_all(scratch, space_, base(), row_bytes(), num_base_vectors_, query, candidates, po_, pl_);
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
        const size_t valid = score_all(scratch, space_, base(), row_bytes(), num_base_vectors_, query, candidates, po_, pl_);
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

        deglib::concurrent::parallel_batch_for(0, num_queries, num_threads, [&](size_t begin, size_t end, size_t) {
            auto& scratch = flat_scratch_;
            for (size_t i = begin; i < end; ++i) {
                const auto q = queries.subspan(i * stride, stride);
                const auto c = candidates.subspan(i * candidates_per_query, candidates_per_query);
                auto& result = results[i];
                if (c.empty() || k_top == 0) continue;

                const size_t valid = score_all(scratch, space_, base(), stride, num_base_vectors_, q, c, po_, pl_);
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

        deglib::concurrent::parallel_batch_for(0, num_queries, num_threads, [&](size_t begin, size_t end, size_t) {
            std::vector<deglib::graph::ObjectDistance> scratch;
            scratch.reserve(candidates_per_query);

            for (size_t i = begin; i < end; ++i) {
                const auto q = queries.subspan(i * stride, stride);
                const auto c = candidates.subspan(i * candidates_per_query, candidates_per_query);

                const size_t valid = score_all(scratch, space_, base(), stride, num_base_vectors_, q, c, po_, pl_);
                select_top_k(
                    scratch, valid, k_top, out_indices.subspan(i * k_top, k_top),
                    return_distances ? out_distances.subspan(i * k_top, k_top) : std::span<float>{}, return_distances, unsorted
                );
            }
        });
    }
};

}  // namespace deglib::search
