#pragma once

#include "deglib/analysis.h"
#include "deglib/concurrent.h"
#include "deglib/graph.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace deglib::optimization::pruning {

/**
 * @brief Prune the worst (highest-weight) neighbors of each vertex.
 *
 * For each vertex, sorts neighbors by descending weight and replaces the top
 * `prune_worst` highest-weight neighbors with self-loops (index=u, weight=0.0).
 * The resulting neighbor list is re-sorted by index to maintain the graph's
 * sorted-by-index invariant. Multi-threaded via parallel_for.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param prune_worst Number of worst neighbors to replace with self-loops per vertex.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 */
inline void prune_worst_edges(deglib::graph::MutableGraph& graph, const uint8_t prune_worst, const size_t numThreads = 0) {
    if (prune_worst <= 0) return;

    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();
    const auto thread_count = numThreads == 0 ? std::thread::hardware_concurrency() : numThreads;

    deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&](size_t vertex_index, size_t thread_id) {
        const auto u = static_cast<uint32_t>(vertex_index);

        // Copy current neighbors
        const auto neighbor_indices = graph.getNeighborIndices(u);
        const auto neighbor_weights = graph.getNeighborWeights(u);

        std::vector<std::pair<uint32_t, float>> neighbors;
        neighbors.reserve(edge_per_vertex);
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
            neighbors.emplace_back(neighbor_indices[n], neighbor_weights[n]);
        }

        // Sort by descending weight (worst/highest first)
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y) { return x.second > y.second; });

        // Replace top prune_worst with self-loops
        for (uint32_t n = 0; n < prune_worst && n < edge_per_vertex; n++) {
            neighbors[n].first = u;
            neighbors[n].second = 0.0f;
        }

        // Re-sort by index to maintain sorted-by-index invariant
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y) { return x.first < y.first; });

        std::vector<uint32_t> sorted_indices;
        std::vector<float> sorted_weights;
        sorted_indices.reserve(edge_per_vertex);
        sorted_weights.reserve(edge_per_vertex);
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
            sorted_indices.push_back(neighbors[n].first);
            sorted_weights.push_back(neighbors[n].second);
        }

        graph.changeEdges(u, sorted_indices.data(), sorted_weights.data());
    });
}

/**
 * @brief Remove all edges that do not satisfy the MRNG condition.
 *
 * Iterates over all vertices and their neighbors, removing any edge that does
 * not satisfy the RNG condition. Each vertex is processed in parallel via
 * parallel_for; non-conforming edges are replaced with self-loops.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_mrng_edges(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();
    const auto thread_count = numThreads == 0 ? std::thread::hardware_concurrency() : numThreads;

    auto removed_rng_edges_per_thread = std::vector<uint32_t>(thread_count);
    deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&](size_t vertex_index, size_t thread_id) {
        uint32_t removed_rng_edges = 0;
        const auto u = static_cast<uint32_t>(vertex_index);

        const auto neighbor_indices = graph.getNeighborIndices(u);
        const auto neighbor_weights = graph.getNeighborWeights(u);

        // find all none rng conform neighbors
        std::vector<uint32_t> remove_neighbor_ids;
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
            const auto neighbor_index = neighbor_indices[n];
            const auto neighbor_weight = neighbor_weights[n];

            if (deglib::analysis::checkRNG(graph, edge_per_vertex, u, neighbor_index, neighbor_weight) == false) {
                remove_neighbor_ids.emplace_back(neighbor_index);
            }
        }

        for (uint32_t n = 0; n < remove_neighbor_ids.size(); n++) {
            graph.changeEdge(u, remove_neighbor_ids[n], u, 0);
            removed_rng_edges++;
        }
        removed_rng_edges_per_thread[thread_id] += removed_rng_edges;
    });

    // aggregate
    uint32_t removed_rng_edges = 0;
    for (uint32_t i = 0; i < thread_count; i++) removed_rng_edges += removed_rng_edges_per_thread[i];

    return removed_rng_edges;
}

/**
 * @brief Remove non-MRNG edges using a weight-sorted global strategy.
 *
 * Collects all non-RNG edges across the graph, sorts them by weight (ascending),
 * then removes them in that order. This allows lower-weight (more important) edges
 * to be considered first, potentially preserving better graph connectivity.
 * Collection is multi-threaded; removal is single-threaded.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use for collection (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_mrng_edges_weight_sorted(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    struct WeightedEdge {
        uint32_t from_vertex;
        uint32_t to_vertex;
        float weight;
    };

    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();
    const auto thread_count = numThreads == 0 ? std::thread::hardware_concurrency() : numThreads;

    // Collect all non-RNG edges (multi-threaded, per-thread buffers to avoid data races)
    std::vector<std::vector<WeightedEdge>> nonMRNG_edges_per_thread(thread_count);
    deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&](size_t vertex_index, size_t thread_id) {
        const auto u = static_cast<uint32_t>(vertex_index);
        const auto neighbor_indices = graph.getNeighborIndices(u);
        const auto neighbor_weights = graph.getNeighborWeights(u);

        for (uint32_t n = 0; n < edge_per_vertex; n++) {
            const auto neighbor_index = neighbor_indices[n];
            const auto neighbor_weight = neighbor_weights[n];
            if (deglib::analysis::checkRNG(graph, edge_per_vertex, u, neighbor_index, neighbor_weight) == false) {
                nonMRNG_edges_per_thread[thread_id].push_back({u, neighbor_index, neighbor_weight});
            }
        }
    });

    // Merge per-thread results
    std::vector<WeightedEdge> nonMRNG_edges;
    for (size_t i = 0; i < thread_count; i++) {
        nonMRNG_edges.insert(nonMRNG_edges.end(), nonMRNG_edges_per_thread[i].begin(), nonMRNG_edges_per_thread[i].end());
    }

    // Sort by weight ascending
    std::sort(nonMRNG_edges.begin(), nonMRNG_edges.end(), [](const auto& x, const auto& y) { return x.weight < y.weight; });

    // Remove edges that are still non-RNG (single-threaded)
    size_t removed_rng_edges = 0;
    for (size_t i = 0; i < nonMRNG_edges.size(); i++) {
        const auto& edge = nonMRNG_edges[i];
        if (deglib::analysis::checkRNG(graph, edge_per_vertex, edge.from_vertex, edge.to_vertex, edge.weight) == false) {
            graph.changeEdge(edge.from_vertex, edge.to_vertex, edge.from_vertex, 0);
            removed_rng_edges++;
        }
    }

    return static_cast<uint32_t>(removed_rng_edges);
}

/**
 * @brief Remove non-MRNG edges using an iterative per-vertex strategy.
 *
 * For each vertex, iteratively removes non-RNG edges in a do-while loop until
 * no more edges can be removed. This accounts for cascading effects where
 * removing one edge may make another edge RNG-conform (or vice versa).
 * Multi-threaded per vertex.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_mrng_edges_iterative(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();
    const auto thread_count = numThreads == 0 ? std::thread::hardware_concurrency() : numThreads;

    auto removed_rng_edges_per_thread = std::vector<uint32_t>(thread_count);
    deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&](size_t vertex_index, size_t thread_id) {
        uint32_t removed_rng_edges = 0;
        const auto vertex_index_u32 = static_cast<uint32_t>(vertex_index);

        // Sort neighbors by their weight (highest to lowest)
        std::vector<std::pair<uint32_t, float>> neighbors;
        {
            const auto neighbor_indices = graph.getNeighborIndices(vertex_index_u32);
            const auto neighbor_weights = graph.getNeighborWeights(vertex_index_u32);
            for (uint32_t n = 0; n < edge_per_vertex; n++) {
                neighbors.emplace_back(neighbor_indices[n], neighbor_weights[n]);
            }
            std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y) { return x.second < y.second; });
        }

        // Find all non-RNG conform neighbors (indices into the sorted neighbors vector)
        std::vector<uint32_t> nonMRNG_edges;
        for (uint32_t n = 0; n < neighbors.size(); n++) {
            const auto neighbor_index = neighbors[n].first;
            const auto neighbor_weight = neighbors[n].second;
            if (deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index_u32, neighbor_index, neighbor_weight) == false) nonMRNG_edges.emplace_back(n);
        }

        // Iteratively remove edges until stable
        bool removed_edge = false;
        do {
            removed_edge = false;
            for (uint32_t n = 0; n < nonMRNG_edges.size(); n++) {
                const auto neighbor_index = neighbors[nonMRNG_edges[n]].first;
                const auto neighbor_weight = neighbors[nonMRNG_edges[n]].second;

                if (deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index_u32, neighbor_index, neighbor_weight) == false) {
                    nonMRNG_edges.erase(nonMRNG_edges.begin() + n);
                    graph.changeEdge(vertex_index_u32, neighbor_index, vertex_index_u32, 0);
                    removed_rng_edges++;
                    removed_edge = true;
                    break;
                }
            }
        } while (removed_edge);

        removed_rng_edges_per_thread[thread_id] += removed_rng_edges;
    });

    // Aggregate
    uint32_t removed_rng_edges = 0;
    for (uint32_t i = 0; i < thread_count; i++) removed_rng_edges += removed_rng_edges_per_thread[i];

    return removed_rng_edges;
}

}  // namespace deglib::optimization::pruning
