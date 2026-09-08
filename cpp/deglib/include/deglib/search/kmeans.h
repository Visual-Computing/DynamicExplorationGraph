#pragma once

// K-Means medoid selection for search entry vertices.
//
// Design:
// - Rows are addressed through pointer-of-pointer (`span<const float* const>`), so
//   float features stored in the graph are referenced directly without copying.
//   Only quantized feature types (FP16/UInt8/Int8/EVP) are decoded once into a
//   scratch buffer; distances always run on the library SIMD kernels.
// - The distance is a C++20 concept-constrained callable, letting the compiler
//   inline the selected SIMD kernel into the Lloyd loop (no indirect calls).
// - Medoids are returned as positions into the row span; mapping those to graph
//   internal indices is the caller's job (see `graph_kmeans_medoids`).

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace deglib::search {
namespace cluster {

// Callable invoked as `distance(a, b)` returning a float distance between two
// dim-sized float vectors. Lower means more similar.
template <typename F>
concept DistanceCallable = requires(F f, const float* a, const float* b) {
    { f(a, b) } -> std::convertible_to<float>;
};

/**
 * Decodes one native graph feature vector into `dim` floats.
 *
 * Handles FP32 (direct copy), FP16, UInt8, Int8, and EVP binary packed quantized representations.
 *
 * @param space   FloatSpace containing metric metadata and dimensionality.
 * @param native  Raw byte pointer to the feature vector in native storage format.
 * @param out     Destination float buffer of size `space.dim()`.
 */
inline void decode_feature_vector(const deglib::distances::FloatSpace& space, const std::byte* native, float* out) {
    const size_t dim = space.dim();
    using deglib::distances::MetricDataType;
    switch (space.metric().get_data_type()) {
        case MetricDataType::FP32:
            std::copy(reinterpret_cast<const float*>(native), reinterpret_cast<const float*>(native) + dim, out);
            break;
        case MetricDataType::FP16:
            deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(native), out, dim);
            break;
        case MetricDataType::Uint8: {
            const auto* v = reinterpret_cast<const uint8_t*>(native);
            for (size_t d = 0; d < dim; ++d) {
                out[d] = static_cast<float>(v[d]);
            }
            break;
        }
        case MetricDataType::Int8: {
            const auto* v = reinterpret_cast<const int8_t*>(native);
            for (size_t d = 0; d < dim; ++d) {
                out[d] = static_cast<float>(v[d]);
            }
            break;
        }
        case MetricDataType::EVP: {
            // Layout [ones (dim/8 bytes)][negs (dim/8 bytes)], dimension d lives
            // in bit (d % 8) of byte (d / 8), matching the EVP quantizer.
            const size_t mask_bytes = dim / 8;
            const auto* bytes = reinterpret_cast<const uint8_t*>(native);
            const uint8_t* ones = bytes;
            const uint8_t* negs = bytes + mask_bytes;
            for (size_t d = 0; d < dim; ++d) {
                const uint8_t bit = static_cast<uint8_t>(1u << (d % 8));
                out[d] = (ones[d / 8] & bit) ? 1.0f : ((negs[d / 8] & bit) ? -1.0f : 0.0f);
            }
            break;
        }
        default:
            throw std::invalid_argument("decode_feature_vector: unsupported metric data type");
    }
}

/**
 * Standard Lloyd k-means clustering over caller-provided row pointers.
 * Centroids are normalized means; empty or degenerate clusters preserve previous centroids.
 *
 * @tparam DistFn    Inlined SIMD distance functor satisfying DistanceCallable.
 * @param rows       Span of pointers to float vectors (dim-sized).
 * @param dim        Vector dimensionality.
 * @param distance   Distance callable invoked as `distance(a, b)`.
 * @param n_clusters Target number of clusters (medoids).
 * @param n_iter     Maximum iterations of Lloyd update loop.
 * @param seed       Random seed for centroid initialization.
 * @param threads    Number of parallel worker threads.
 * @return           Indices (medoid positions) relative to `rows`.
 */
template <DistanceCallable DistFn>
[[nodiscard]] std::vector<uint32_t>
kmeans_medoids(std::span<const float* const> rows, uint32_t dim, DistFn&& distance, uint32_t n_clusters, uint32_t n_iter, uint32_t seed, size_t threads) {
    if (rows.empty() || dim == 0) {
        throw std::invalid_argument("kmeans_medoids: empty input");
    }
    if (n_clusters == 0) {
        throw std::invalid_argument("kmeans_medoids: n_clusters must be > 0");
    }
    n_clusters = std::min(n_clusters, static_cast<uint32_t>(rows.size()));
    std::mt19937 rng(seed);

    // K-means++ style init omitted on purpose: partial Fisher-Yates over the
    // rows is cheaper and matches the established entry-point quality.
    std::vector<uint32_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> centroids(static_cast<size_t>(n_clusters) * dim);
    for (uint32_t c = 0; c < n_clusters; ++c) {
        const size_t remaining = rows.size() - c;
        const size_t j = c + (remaining > 1 ? rng() % remaining : 0);
        std::swap(order[c], order[j]);
        const float* vec = rows[order[c]];
        std::copy(vec, vec + dim, centroids.data() + static_cast<size_t>(c) * dim);
    }

    std::vector<uint32_t> labels(rows.size());
    std::vector<float> sums(static_cast<size_t>(n_clusters) * dim);
    std::vector<uint32_t> counts(n_clusters);
    for (uint32_t iter = 0; iter < n_iter; ++iter) {
        // Assignment: parallel over rows.
        deglib::concurrent::parallel_for(0, rows.size(), threads, [&](size_t i, size_t) {
            const float* vec = rows[i];
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < n_clusters; ++c) {
                const float dist = distance(vec, centroids.data() + static_cast<size_t>(c) * dim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            labels[i] = best_c;
        });

        // Update: single bandwidth-bound pass, kept serial.
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < rows.size(); ++i) {
            const uint32_t c = labels[i];
            counts[c]++;
            const float* vec = rows[i];
            float* sum = sums.data() + static_cast<size_t>(c) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                sum[d] += vec[d];
            }
        }
        for (uint32_t c = 0; c < n_clusters; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            const float* sum = sums.data() + static_cast<size_t>(c) * dim;
            const float inv = 1.0f / static_cast<float>(counts[c]);
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                norm_sq += (sum[d] * inv) * (sum[d] * inv);
            }
            if (norm_sq <= 1e-12f) {
                continue;
            }
            const float norm = std::sqrt(norm_sq);
            float* cent = centroids.data() + static_cast<size_t>(c) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                cent[d] = (sum[d] * inv) / norm;
            }
        }
    }

    // Medoid per cluster: member closest to the final centroid.
    std::vector<uint32_t> medoids(n_clusters);
    deglib::concurrent::parallel_for(0, n_clusters, threads, [&](size_t c, size_t) {
        const float* cent = centroids.data() + c * dim;
        float best_dist = std::numeric_limits<float>::max();
        size_t best_i = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            const float dist = distance(rows[i], cent);
            if (dist < best_dist) {
                best_dist = dist;
                best_i = i;
            }
        }
        medoids[c] = static_cast<uint32_t>(best_i);
    });
    return medoids;
}

/**
 * Entry-vertex optimization over graph-native features.
 * Samples `sample_size` vertices (0 = all), references FP32 features in place and decodes quantized
 * types once, clusters with the FP32 SIMD kernel matching the graph metric kind (L2 vs Inner Product),
 * and returns medoids as graph internal vertex indices.
 *
 * @param graph       Internal graph containing feature data and topology.
 * @param n_clusters  Number of clusters / entry vertices to identify.
 * @param n_iter      Maximum number of k-means iterations.
 * @param sample_size Number of graph vertices to sample for clustering (0 = all).
 * @param seed        Random seed for sampling and initial centroids.
 * @param threads     Number of worker threads for parallel assignment and medoid search.
 * @return            List of graph vertex indices representing cluster medoids.
 */
[[nodiscard]] inline std::vector<uint32_t> graph_kmeans_medoids(
    const deglib::graph::InternalGraph& graph,
    uint32_t n_clusters,
    uint32_t n_iter,
    size_t sample_size,
    uint32_t seed = 7,
    size_t threads = 1
) {
    const size_t n_vectors = graph.size();
    if (n_vectors == 0) {
        throw std::invalid_argument("graph_kmeans_medoids: graph is empty");
    }
    const deglib::distances::FloatSpace& space = graph.getFeatureSpace();
    const size_t dim = space.dim();
    const size_t actual_sample = sample_size == 0 ? n_vectors : std::min(sample_size, n_vectors);

    std::vector<uint32_t> sample(n_vectors);
    std::iota(sample.begin(), sample.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(actual_sample);

    // Row pointers into graph memory (FP32) or into the decode scratch buffer.
    std::vector<const float*> rows(actual_sample);
    std::vector<float> decoded;
    if (space.metric().get_data_type() == deglib::distances::MetricDataType::FP32) {
        for (size_t i = 0; i < actual_sample; ++i) {
            rows[i] = reinterpret_cast<const float*>(graph.getFeatureVector(sample[i]));
        }
    } else {
        decoded.resize(actual_sample * dim);
        deglib::concurrent::parallel_for(0, actual_sample, threads, [&](size_t i, size_t) {
            decode_feature_vector(space, graph.getFeatureVector(sample[i]), decoded.data() + i * dim);
        });
        for (size_t i = 0; i < actual_sample; ++i) {
            rows[i] = decoded.data() + i * dim;
        }
    }

    const bool is_l2 = space.metric().get_distance_kind() == deglib::distances::MetricDistanceKind::L2;
    const deglib::distances::FloatSpace float_space(dim, is_l2 ? deglib::distances::Metric::FP32_L2 : deglib::distances::Metric::FP32_InnerProduct);
    const std::span<const float* const> row_view(rows.data(), rows.size());
    const uint32_t dim32 = static_cast<uint32_t>(dim);
    const std::vector<uint32_t> positions = float_space.compute([&](const auto& kernel) {
        // Static dispatch: the concrete SIMD kernel inlines into the loop.
        return kmeans_medoids(
            row_view, dim32,
            [&](const float* a, const float* b) -> float { return std::decay_t<decltype(kernel)>::compare(a, b, float_space.get_dist_func_param()); },
            n_clusters, n_iter, seed, threads
        );
    });

    std::vector<uint32_t> medoids;
    medoids.reserve(positions.size());
    for (const uint32_t p : positions) {
        medoids.push_back(sample[p]);
    }
    return medoids;
}

}  // namespace cluster

/**
 * Manages k-means entry vertices (medoids) for fast query initialization on a graph.
 *
 * During optimization (`optimize`), k-means clustering identifies representative medoids
 * distributed across the graph. At search time (`top_entries`), SIMD vectorized batch distance
 * functions evaluate distances to all candidate medoids simultaneously, selecting the nearest
 * start vertices with minimal latency and zero heap allocations for standard query configurations.
 *
 * @param graph Reference to the underlying InternalGraph (must outlive this selector).
 */
class KMeansEntrySelector {
  public:
    explicit KMeansEntrySelector(const deglib::graph::InternalGraph& graph) : graph_(&graph) {}

    /**
     * Compute and store entry vertices using k-means medoid clustering on a sample of graph vertices.
     *
     * @param n_clusters  Number of cluster medoids (entry vertices) to generate (e.g. 128..512).
     * @param n_iter      Maximum number of Lloyd k-means iterations.
     * @param sample_size Number of vertices sampled for clustering (0 automatically selects 3% of graph size).
     * @param seed        Random seed for deterministic initialization and sampling.
     * @param threads     Number of worker threads used for parallel clustering.
     */
    void optimize(uint32_t n_clusters = 128, uint32_t n_iter = 15, size_t sample_size = 0, uint32_t seed = 7, size_t threads = 1) {
        if (sample_size == 0) {
            const size_t n = graph_->size();
            sample_size = n * 3 / 100;
            if (sample_size == 0 && n > 0) sample_size = n;
        }
        entries_ = cluster::graph_kmeans_medoids(*graph_, n_clusters, n_iter, sample_size, seed, threads);
    }

    /**
     * Finds the nearest entry vertices for a given query vector, ordered nearest-first.
     *
     * Evaluates distances to all medoids in a single SIMD-vectorized batch comparison call (AVX-512 / AVX2).
     * For `count == 1` and `count == 2` (common DEG search entry sizes), fast single-pass selection
     * runs completely on stack buffers without heap allocations.
     * Falls back to vertex 0 when no entries exist.
     *
     * @param query Native query vector bytes sized to the graph feature space.
     * @param count Maximum number of nearest entry vertex indices to return (typically 1 or 2).
     * @return std::vector<uint32_t> List of up to `count` internal graph vertex indices.
     */
    [[nodiscard]] std::vector<uint32_t> top_entries(std::span<const std::byte> query, size_t count = 2) const {
        const uint32_t n = graph_->size();
        if (n == 0 || count == 0) return {};
        if (entries_.empty()) return {0};
        if (entries_.size() <= count) return entries_;

        const size_t num_entries = entries_.size();
        const auto& feature_space = graph_->getFeatureSpace();
        const auto batch_dist_func = feature_space.get_batch_dist_func();
        const auto dist_param = feature_space.get_dist_func_param();

        // Stack-allocated scratch arrays for up to 512 medoids to eliminate heap allocation overhead.
        // Falls back to dynamic heap buffers only for exceptionally large cluster counts (> 512).
        const void* stack_ptrs[512];
        float stack_dists[512];
        std::unique_ptr<const void*[]> heap_ptrs;
        std::unique_ptr<float[]> heap_dists;

        const void** ptrs = stack_ptrs;
        float* dists = stack_dists;
        if (num_entries > 512) {
            heap_ptrs = std::make_unique<const void*[]>(num_entries);
            heap_dists = std::make_unique<float[]>(num_entries);
            ptrs = heap_ptrs.get();
            dists = heap_dists.get();
        }

        // Collect feature vector pointers for all candidate medoids
        for (size_t i = 0; i < num_entries; ++i) {
            ptrs[i] = graph_->getFeatureVector(entries_[i]);
        }

        // Compute distances from the query to all medoids simultaneously via SIMD kernel
        batch_dist_func(query.data(), ptrs, num_entries, dist_param, dists);

        // Fast-path: Single nearest entry point (linear scan over computed distances, 0 allocations)
        if (count == 1) {
            size_t min_idx = 0;
            float min_dist = dists[0];
            for (size_t i = 1; i < num_entries; ++i) {
                if (dists[i] < min_dist) {
                    min_dist = dists[i];
                    min_idx = i;
                }
            }
            return {entries_[min_idx]};
        }

        // Fast-path: Top 2 entry points (single-pass min1/min2 tracking, 0 allocations)
        if (count == 2) {
            uint32_t ep1 = entries_[0], ep2 = entries_[1];
            float dist1 = dists[0], dist2 = dists[1];
            if (dist2 < dist1) {
                std::swap(dist1, dist2);
                std::swap(ep1, ep2);
            }
            for (size_t i = 2; i < num_entries; ++i) {
                const float d = dists[i];
                const uint32_t ep = entries_[i];
                if (d < dist1) {
                    dist2 = dist1;
                    ep2 = ep1;
                    dist1 = d;
                    ep1 = ep;
                } else if (d < dist2) {
                    dist2 = d;
                    ep2 = ep;
                }
            }
            if (ep1 == ep2) return {ep1};
            return {ep1, ep2};
        }

        // General fallback for count > 2: Partial sort via std::nth_element in O(N)
        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(num_entries);
        for (size_t i = 0; i < num_entries; ++i) {
            scored.emplace_back(dists[i], entries_[i]);
        }
        std::nth_element(scored.begin(), scored.begin() + count, scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        scored.resize(count);
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<uint32_t> top;
        top.reserve(count);
        for (const auto& s : scored) top.push_back(s.second);
        return top;
    }

    /** Stored entry vertices as graph internal indices. */
    [[nodiscard]] const std::vector<uint32_t>& entries() const noexcept { return entries_; }

    /** True when optimize() has not produced entries yet. */
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    /** Number of stored entry vertices. */
    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

  private:
    const deglib::graph::InternalGraph* graph_ = nullptr;
    std::vector<uint32_t> entries_;
};

}  // namespace deglib::search
