#pragma once

// K-Means medoid selection for query routing entry vertices.
//
// Design:
// - Rows are addressed through pointer-of-pointer (`span<const float* const>`), so
//   float features stored in the graph or base vectors are referenced directly without copying.
// - Medoids are dynamic query-routing centers owned exclusively by KMeansEntrySelector
//   in Searcher. They do not mutate or pollute InternalGraph's topological entry vertices.
// - Distance evaluation during query routing evaluates SIMD batch distances in the native
//   feature space of the graph.

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

        // Update: single bandwidth-bound pass.
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

}  // namespace cluster

/**
 * Manages k-means entry vertices (medoids) for fast query initialization on a graph.
 *
 * `fit()` clusters caller-provided candidate features, maps medoid rows to final graph vertex IDs,
 * and copies medoid features into a compact contiguous cache. At search time, distances to all
 * candidate medoids are computed with the graph's native batch-distance kernel. The number of medoids
 * is bounded to 512 so the routing scan can use fixed-size scratch arrays.
 *
 * Crucially, KMeansEntrySelector is a query router owned by Searcher; it never mutates the graph.
 */
class KMeansEntrySelector {
  public:
    /** Maximum number of routing medoids supported by the fixed-size top-entry scratch buffers. */
    static constexpr size_t kMaxMedoids = 512;

    explicit KMeansEntrySelector(const deglib::graph::InternalGraph& graph) : graph_(&graph) {
        // If graph already has non-default entry vertices (e.g. loaded from .deg TAG_ENTR), initialize with them
        if (graph.getEntryVertexIndices().size() > 1 || (!graph.getEntryVertexIndices().empty() && graph.getEntryVertexIndices()[0] != 0)) {
            set_graph_entries(graph.getEntryVertexIndices());
        }
    }

    /**
     * Fit cluster medoids on provided candidate features and store final graph entry IDs.
     *
     * @param candidate_features Pointers to dim-sized float vectors used by K-Means.
     * @param candidate_ids      Final graph vertex IDs matching `candidate_features` row-by-row.
     * @param dim                Vector dimensionality.
     * @param is_l2              True for L2 metric, false for Inner Product.
     * @param n_clusters         Number of clusters / medoids to generate. Must be <= 512.
     * @param n_iter             Maximum number of Lloyd iterations.
     * @param seed               Random seed.
     * @param threads            Worker thread count.
     */
    void fit(
        std::span<const float* const> candidate_features,
        std::span<const uint32_t> candidate_ids,
        uint32_t dim,
        bool is_l2,
        uint32_t n_clusters = 128,
        uint32_t n_iter = 15,
        uint32_t seed = 7,
        size_t threads = 1
    ) {
        if (candidate_features.empty() || dim == 0) return;
        if (n_clusters == 0) {
            throw std::invalid_argument("KMeansEntrySelector::fit: n_clusters must be > 0");
        }
        if (n_clusters > kMaxMedoids) {
            throw std::invalid_argument("KMeansEntrySelector::fit: n_clusters must be <= 512");
        }
        if (candidate_ids.size() != candidate_features.size()) {
            throw std::invalid_argument("KMeansEntrySelector::fit: candidate_ids must match candidate_features size");
        }

        const deglib::distances::FloatSpace float_space(dim, is_l2 ? deglib::distances::Metric::FP32_L2 : deglib::distances::Metric::FP32_InnerProduct);
        const auto positions = float_space.compute([&](const auto& kernel) {
            return cluster::kmeans_medoids(
                candidate_features, dim,
                [&](const float* a, const float* b) -> float { return std::decay_t<decltype(kernel)>::compare(a, b, float_space.get_dist_func_param()); },
                n_clusters, n_iter, seed, threads
            );
        });

        entries_.clear();
        entries_.reserve(positions.size());
        for (const uint32_t pos : positions) {
            if (pos >= candidate_ids.size()) {
                throw std::runtime_error("KMeansEntrySelector::fit: medoid index is outside candidate_ids");
            }
            entries_.push_back(candidate_ids[pos]);
        }
        if (entries_.size() > kMaxMedoids) {
            throw std::runtime_error("KMeansEntrySelector::fit: produced more than 512 medoids");
        }

        update_medoid_features();
    }

    /**
     * Finds the nearest entry vertices for a given native query vector, ordered nearest-first.
     *
     * Computes medoid distances with the graph's native batch-distance kernel from a contiguous medoid
     * cache. Falls back to vertex 0 when no entries exist.
     *
     * @param query Native query vector bytes sized to the graph feature space.
     * @param count Maximum number of nearest entry vertex indices to return (typically 1 or 2).
     * @return List of up to `count` internal graph vertex indices.
     */
    [[nodiscard]] std::vector<uint32_t> top_entries(std::span<const std::byte> query, size_t count = 2) const {
        if (!graph_ || graph_->size() == 0 || count == 0) return {};
        if (entries_.empty()) return {0};
        if (entries_.size() <= count) return entries_;

        const size_t num_entries = entries_.size();
        const auto& feature_space = graph_->getFeatureSpace();
        const auto dist_param = feature_space.get_dist_func_param();

        // Fixed-size scratch array for the enforced <= 512 medoid limit.
        float dists[kMaxMedoids];
        if (num_entries > kMaxMedoids) return entries_;

        const auto batch_dist_func = feature_space.get_batch_dist_func();
        if (medoid_ptrs_[0] == nullptr) return entries_;

        batch_dist_func(query.data(), medoid_ptrs_.data(), num_entries, dist_param, dists);

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

    /** True when no medoids are stored. */
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    /** True when medoids are stored. */
    [[nodiscard]] bool has_entries() const noexcept { return !entries_.empty(); }

    /** Number of stored entry vertices. */
    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

  private:
    const deglib::graph::InternalGraph* graph_ = nullptr;
    std::vector<uint32_t> entries_;
    std::vector<std::byte> medoid_features_;
    std::array<const void*, kMaxMedoids> medoid_ptrs_{};

    void set_graph_entries(std::vector<uint32_t> entries) {
        if (entries.size() > kMaxMedoids) {
            throw std::invalid_argument("KMeansEntrySelector: maximum number of medoids is 512");
        }
        entries_ = std::move(entries);
        update_medoid_features();
    }

    void update_medoid_features() {
        medoid_ptrs_.fill(nullptr);
        medoid_features_.clear();
        if (!graph_ || entries_.empty()) return;

        const size_t bytes = graph_->getFeatureSpace().get_data_size();
        if (bytes == 0 || entries_.size() > medoid_ptrs_.size()) return;

        medoid_features_.resize(entries_.size() * bytes);
        for (size_t i = 0; i < entries_.size(); ++i) {
            std::byte* dst = medoid_features_.data() + i * bytes;
            std::memcpy(dst, graph_->getFeatureVector(entries_[i]), bytes);
            medoid_ptrs_[i] = static_cast<const void*>(dst);
        }
    }
};

}  // namespace deglib::search
