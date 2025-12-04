#pragma once

/**
 * @file stats.h
 * @brief Graph statistics computation utilities for deglib benchmark.
 * 
 * Provides functions to compute various graph quality metrics including:
 * - Seed reachability: How many vertices can be reached from entry points
 * - Average reach: Average number of vertices reachable from each vertex
 * - Graph quality: Ratio of neighbors that are "perfect" (in ground truth top-k)
 * - In-degree statistics: Distribution of incoming edges
 */

#include <vector>
#include <cstdint>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <unordered_set>

#include <fmt/core.h>

#include "deglib.h"
#include "logging.h"
#include "file_io.h"
#include "stopwatch.h"

namespace deglib::benchmark
{

// ============================================================================
// Graph Statistics 
// ============================================================================

/**
 * @brief Complete graph statistics structure with in-degree and out-degree stats.
 */
struct GraphStats {
    // Basic stats
    size_t vertex_count = 0;       ///< Total number of vertices
    size_t edge_count = 0;         ///< Total number of edges
    uint32_t feature_dims = 0;     ///< Feature vector dimensions
    uint8_t edges_per_vertex = 0;  ///< Maximum edges per vertex (k)
    
    // Out-degree stats
    float avg_out_degree = 0.0f;
    uint32_t min_out_degree = 0;
    uint32_t max_out_degree = 0;
    
    // In-degree stats
    float avg_in_degree = 0.0f;
    uint32_t min_in_degree = 0;
    uint32_t max_in_degree = 0;
    uint32_t source_vertices = 0;           // Vertices with 0 in-degree
    
    // Quality metrics (optional, expensive to compute)
    float graph_quality = -1.0f;            // Graph quality (-1 means not computed)
    float search_reachability = -1.0f;      // search reach (-1 means not computed)
    float exploration_reachability = -1.0f; // exploration reach (-1 means not computed)
    
    // Memory
    size_t memory_bytes = 0;       ///< Estimated memory usage
};

/**
 * @brief Collect statistics from a search graph.
 * 
 * This collects statistics that are cheap to compute.
 * For expensive metrics (reachability, avg reach, graph quality), use compute_full_graph_stats().
 * 
 * @param graph The search graph to analyze
 * @return GraphStats with basic statistics filled in
 */
inline GraphStats collect_graph_stats(const deglib::search::SearchGraph& graph) {
    GraphStats stats;
    stats.vertex_count = graph.size();
    stats.feature_dims = graph.getFeatureSpace().dim();
    stats.edges_per_vertex = graph.getEdgesPerVertex();

    const auto graph_size = graph.size();
    const auto edges_per_vertex = graph.getEdgesPerVertex();

    // Count out-degrees
    size_t total_edges = 0;
    uint32_t min_out = (std::numeric_limits<uint32_t>::max)();
    uint32_t max_out = 0;

    for (uint32_t i = 0; i < graph_size; i++) {
        const auto neighbors = graph.getNeighborIndices(i);
        uint32_t valid_edges = 0;
        for (uint8_t j = 0; j < edges_per_vertex; j++) {
            if (neighbors[j] != (std::numeric_limits<uint32_t>::max)()) {
                valid_edges++;
            }
        }
        total_edges += valid_edges;
        if (valid_edges < min_out) min_out = valid_edges;
        if (valid_edges > max_out) max_out = valid_edges;
    }

    stats.edge_count = total_edges;
    stats.avg_out_degree = graph_size > 0 ? (float)total_edges / graph_size : 0.0f;
    stats.min_out_degree = graph_size > 0 ? min_out : 0;
    stats.max_out_degree = max_out;

    // Compute in-degree stats (inlined)
    auto in_degree_count = std::vector<uint32_t>(graph_size, 0);
    for (uint32_t v = 0; v < graph_size; v++) {
        const auto neighbor_indices = graph.getNeighborIndices(v);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_index = neighbor_indices[e];
            if (neighbor_index != (std::numeric_limits<uint32_t>::max)() && neighbor_index < graph_size) {
                in_degree_count[neighbor_index]++;
            }
        }
    }

    stats.min_in_degree = (std::numeric_limits<uint32_t>::max)();
    stats.max_in_degree = 0;
    uint64_t total_in_degree = 0;

    for (uint32_t v = 0; v < graph_size; v++) {
        const auto in_degree = in_degree_count[v];
        if (in_degree < stats.min_in_degree) stats.min_in_degree = in_degree;
        if (in_degree > stats.max_in_degree) stats.max_in_degree = in_degree;
        if (in_degree == 0) stats.source_vertices++;
        total_in_degree += in_degree;
    }

    stats.avg_in_degree = graph_size > 0 ? ((float)total_in_degree) / graph_size : 0.0f;
    if (graph_size == 0) stats.min_in_degree = 0;

    // Memory estimation
    // Per vertex: neighbors (k * 4 bytes) + weights (k * 4 bytes) + feature (dims * 4 bytes)
    stats.memory_bytes = stats.vertex_count * (stats.edges_per_vertex * 4 + stats.edges_per_vertex * 4 + stats.feature_dims * 4);

    return stats;
}

/**
 * @brief Log graph statistics.
 * 
 * @param stats The statistics to log
 */
inline void log_graph_stats(const GraphStats& stats) {
    log("Graph Statistics:\n");
    log("  Vertices: {}\n", stats.vertex_count);
    log("  Total edges: {}\n", stats.edge_count);
    log("  Feature dimensions: {}\n", stats.feature_dims);
    log("  Edges per vertex (k): {}\n", stats.edges_per_vertex);
    log("  Out-degree: avg={:.2f}, min={}, max={}\n", stats.avg_out_degree, stats.min_out_degree, stats.max_out_degree);
    log("  In-degree:  avg={:.2f}, min={}, max={}, source_vertices={}\n", 
        stats.avg_in_degree, stats.min_in_degree, stats.max_in_degree, stats.source_vertices);
    
    if (stats.graph_quality >= 0) {
        log("  Graph Quality (GQ): {:.4f}\n", stats.graph_quality);
    }
    if (stats.search_reachability > 0) {
        log("  Search Reachability: {:.2f}%\n", stats.search_reachability * 100);
    }
    if (stats.exploration_reachability >= 0) {
        log("  Exploration Reachability: {:.2f}%\n", stats.exploration_reachability * 100);
    }
    
    log("  Estimated memory: {:.2f} MB\n", stats.memory_bytes / (1024.0 * 1024.0));
}

// ============================================================================
// Search Reachability
// ============================================================================

/**
 * @brief Compute the seed reachability count.
 * 
 * Measures how many vertices can be reached from the graph's entry point via search.
 * This tests if the graph is well-connected for nearest neighbor search.
 * 
 * @param graph The search graph to analyze
 * @param thread_count Number of threads to use for parallel computation
 * @return Number of vertices reachable from entry points
 */
inline uint32_t compute_search_reachability(
    const deglib::search::SearchGraph& graph, 
    const uint32_t thread_count = std::thread::hardware_concurrency()) 
{
    auto stopw = StopW();
    const auto graph_size = (uint32_t)graph.size();
    const auto edges_per_vertex = graph.getEdgesPerVertex();
    const auto entry_vertices = graph.getEntryVertexIndices();
    
    std::atomic<uint32_t> reachable_count{0};
    std::atomic<uint32_t> tested_count{0};

    deglib::concurrent::parallel_for(0, graph_size, thread_count, [&](size_t id, size_t) {
        auto target_index = (uint32_t)id;
        auto target_feature = graph.getFeatureVector(target_index);
        bool found = false;

        // Try a simple search first to find the target vertex
        {
            auto result_queue = graph.search(entry_vertices, target_feature, 0.2f, 100);
            while (!result_queue.empty()) {
                if (result_queue.top().getInternalIndex() == target_index) {
                    found = true;
                    break;
                }
                result_queue.pop();
            }
        }

        // If the simple search was not successful, use flood fill to find the target vertex
        if (!found) {
            // Use the first entry vertex as starting point
            uint32_t start_vertex = entry_vertices.empty() ? 0 : entry_vertices[0];

            // Flood fill from the entry vertex
            auto checked_ids = std::vector<bool>(graph_size);
            auto check = std::vector<uint32_t>();
            auto check_next = std::vector<uint32_t>();

            checked_ids[start_vertex] = true;
            check.emplace_back(start_vertex);

            auto check_ptr = &check;
            auto check_next_ptr = &check_next;
            found = (start_vertex == target_index);

            while (check_ptr->size() > 0 && !found) {
                check_next_ptr->clear();

                for (size_t c = 0; c < check_ptr->size() && !found; c++) {
                    auto check_index = check_ptr->at(c);
                    auto neighbor_indices = graph.getNeighborIndices(check_index);

                    for (uint8_t e = 0; e < edges_per_vertex; e++) {
                        auto neighbor_index = neighbor_indices[e];
                        if (neighbor_index == (std::numeric_limits<uint32_t>::max)()) continue;

                        if (neighbor_index == target_index) {
                            found = true;
                            break;
                        }

                        if (!checked_ids[neighbor_index]) {
                            checked_ids[neighbor_index] = true;
                            check_next_ptr->emplace_back(neighbor_index);
                        }
                    }
                }

                std::swap(check_ptr, check_next_ptr);
            }
        }

        if (found) {
            reachable_count++;
        }

        uint32_t count = ++tested_count;
        if (count % 10000 == 0) {
            log("Seed reachability is {:7d} after checking {:7d} of {:7d} vertices after {:4d}s\n", 
                reachable_count.load(), count, graph_size, stopw.getElapsedTimeMicro() / 1000000);
        }
    });

    log("Seed Reachability is {} out of {}\n", reachable_count.load(), graph_size);
    return reachable_count.load();
}

/**
 * @brief Structure to store vertex reach information for caching during avg reach computation.
 */
struct VertexReach {
    uint32_t vertex_id;      ///< The vertex ID
    uint32_t reach_count;    ///< Number of vertices reachable from this vertex
    std::vector<bool> reachable_ids;  ///< Bitmap of reachable vertex IDs
    
    VertexReach(uint32_t id, uint32_t count, std::vector<bool>&& ids)
        : vertex_id(id), reach_count(count), reachable_ids(std::move(ids)) {}
};

/**
 * @brief Compute the average reach of the graph.
 * 
 * Measures the average number of vertices reachable from each vertex via graph exploration.
 * This indicates how well connected the graph is for exploration tasks.
 * 
 * @param graph The search graph to analyze
 * @return Average number of vertices reachable from each vertex
 */
inline float compute_exploration_reach(const deglib::search::SearchGraph& graph) {
    const auto graph_size = (uint32_t)graph.size();
    const auto edges_per_vertex = graph.getEdgesPerVertex();
    auto stopw = StopW();

    // Remember vertices with high reach for optimization
    uint32_t best_vertex_reach = 0;
    auto vertices_reach = std::vector<VertexReach>();
    auto index_of_vertex_reach = std::vector<uint32_t>(graph_size);
    std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), graph_size);

    uint64_t counter = 0;
    uint64_t exploration_reachability = 0;

    for (uint32_t entry_id = 0; entry_id < graph_size; entry_id++) {
        // Flood fill from this entry vertex
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<uint32_t>();
        auto check_next = std::vector<uint32_t>();

        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);

        // Try to speed up by reaching a vertex that can reach many others
        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;

        auto check_ptr = &check;
        auto check_next_ptr = &check_next;

        while (check_ptr->size() > 0 && best_reach_vertex_reach < graph_size) {
            check_next_ptr->clear();

            for (size_t c = 0; c < check_ptr->size() && best_reach_vertex_reach < graph_size; c++) {
                const auto check_index = check_ptr->at(c);
                const auto neighbor_indices = graph.getNeighborIndices(check_index);

                for (uint8_t e = 0; e < edges_per_vertex; e++) {
                    const auto neighbor_index = neighbor_indices[e];
                    if (neighbor_index == (std::numeric_limits<uint32_t>::max)()) continue;

                    if (!checked_ids[neighbor_index]) {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->emplace_back(neighbor_index);

                        // Check if neighbor is connected to a high-reach vertex
                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if (vertex_reach_index < graph_size) {
                            const auto& neighbor_reach = vertices_reach[vertex_reach_index];
                            
                            if (neighbor_reach.reach_count == graph_size) {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = graph_size;
                                break;
                            }

                            if (neighbor_reach.reach_count > best_reach_vertex_reach) {
                                best_reach_vertex_reach = neighbor_reach.reach_count;
                                best_reach_vertex_index = vertex_reach_index;

                                // Copy the reach of the best
                                const auto& best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < graph_size; b++) {
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                                }
                            }
                        }
                    }
                }
            }

            std::swap(check_ptr, check_next_ptr);
        }

        if (best_reach_vertex_reach == graph_size) {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            exploration_reachability += graph_size;
        } else {
            // Count how many nodes have been checked
            uint32_t reach_count = 0;
            for (size_t i = 0; i < graph_size; i++) {
                reach_count += checked_ids[i];
            }
            exploration_reachability += reach_count;

            if (best_vertex_reach < reach_count) {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = (uint32_t)vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            } else if (best_reach_vertex_reach > 0) {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            } else {
                index_of_vertex_reach[entry_id] = (uint32_t)vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            }
        }

        counter++;
        if (counter % 10000 == 0) {
            log("Avg reach is {:.2f} after checking {:7d} of {:7d} vertices after {:4d}s\n",
                ((float)exploration_reachability) / counter, counter, graph_size, stopw.getElapsedTimeMicro() / 1000000);
        }
    }

    log("Avg reach is {:.2f} after checking {:7d} of {:7d} vertices after {:4d}s\n",
        ((float)exploration_reachability) / counter, counter, graph_size, stopw.getElapsedTimeMicro() / 1000000);
    return ((float)exploration_reachability) / graph_size;
}

/**
 * @brief Compute the graph quality using exploration ground truth.
 * 
 * Graph quality (GQ) is the ratio of neighbors that are "perfect" - meaning they
 * appear in the ground truth top-k list for that vertex. Uses exploration ground
 * truth which is computed by finding true nearest neighbors for each vertex
 * within the graph itself.
 * 
 * @param graph The search graph to analyze
 * @param exploration_gt Exploration ground truth data (external labels)
 * @param exploration_gt_dims Number of neighbors per vertex in ground truth
 * @return Graph quality ratio in [0, 1]
 */
inline float compute_graph_quality(
    const deglib::search::SearchGraph& graph, 
    const uint32_t* exploration_gt, 
    const size_t exploration_gt_dims) 
{
    const auto graph_size = graph.size();
    const auto edges_per_vertex = graph.getEdgesPerVertex();

    if (exploration_gt_dims < edges_per_vertex) {
        log("Warning: Exploration GT size {} is smaller than edges per vertex {}\n", 
            exploration_gt_dims, edges_per_vertex);
    }

    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;

    for (uint32_t v = 0; v < graph_size; v++) {
        const auto neighbor_indices = graph.getNeighborIndices(v);
        
        // The exploration GT is indexed by vertex order in the graph
        const auto vertex_gt = exploration_gt + v * exploration_gt_dims;
        const auto compare_size = (std::min)((size_t)edges_per_vertex, exploration_gt_dims);

        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_index = neighbor_indices[e];
            if (neighbor_index == (std::numeric_limits<uint32_t>::max)()) continue;
            
            total_neighbor_count++;

            // Get the neighbor's external label
            const auto neighbor_label = graph.getExternalLabel(neighbor_index);

            // Check if neighbor is in the top-k list (ground truth contains external labels)
            for (size_t i = 0; i < compare_size; i++) {
                if (neighbor_label == vertex_gt[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
    }

    return total_neighbor_count > 0 ? ((float)perfect_neighbor_count) / total_neighbor_count : 0.0f;
}

/**
 * @brief Compute the graph quality using pre-computed ground truth sets.
 * 
 * Graph quality (GQ) is the ratio of neighbors that are "perfect" - meaning they
 * appear in the ground truth top-k list for that vertex.
 * 
 * @param graph The search graph to analyze
 * @param exploration_gt Pre-computed ground truth as vector of unordered_sets (one per vertex)
 * @return Graph quality ratio in [0, 1]
 */
inline float compute_graph_quality(
    const deglib::search::SearchGraph& graph, 
    const std::vector<std::unordered_set<uint32_t>>& exploration_gt) 
{
    const auto graph_size = graph.size();
    const auto edges_per_vertex = graph.getEdgesPerVertex();

    if (exploration_gt.size() < graph_size) {
        log("Warning: Exploration GT size {} is smaller than graph size {}\n", 
            exploration_gt.size(), graph_size);
        return 0.0f;
    }

    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;

    for (uint32_t v = 0; v < graph_size; v++) {
        const auto neighbor_indices = graph.getNeighborIndices(v);
        const auto& gt = exploration_gt[v];

        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_index = neighbor_indices[e];
            if (neighbor_index == (std::numeric_limits<uint32_t>::max)()) continue;
            
            total_neighbor_count++;

            // Get the neighbor's external label
            const auto neighbor_label = graph.getExternalLabel(neighbor_index);

            // Check if neighbor is in the ground truth set
            if (gt.find(neighbor_label) != gt.end()) {
                perfect_neighbor_count++;
            }
        }
    }

    return total_neighbor_count > 0 ? ((float)perfect_neighbor_count) / total_neighbor_count : 0.0f;
}

/**
 * @brief Compute all statistics including expensive ones.
 * 
 * @param graph The search graph to analyze
 * @param exploration_gt Exploration ground truth data (external labels), nullptr to skip GQ
 * @param exploration_gt_dims Number of neighbors per vertex in ground truth
 * @param compute_reachability Whether to compute seed reachability (expensive)
 * @param compute_reach Whether to compute average reach (expensive)
 * @param thread_count Number of threads for parallel computation
 * @return GraphStats with all requested statistics
 */
inline GraphStats compute_full_graph_stats(
    const deglib::search::SearchGraph& graph,
    const uint32_t* exploration_gt = nullptr,
    size_t exploration_gt_dims = 0,
    bool compute_reachability = true,
    bool compute_reach = true,
    uint32_t thread_count = std::thread::hardware_concurrency())
{
    auto stats = collect_graph_stats(graph);

    if (exploration_gt != nullptr && exploration_gt_dims > 0) {
        log("Computing graph quality...\n");
        stats.graph_quality = compute_graph_quality(graph, exploration_gt, exploration_gt_dims);
    }

    if (compute_reachability) {
        log("Computing seed reachability...\n");
        stats.search_reachability = compute_search_reachability(graph, thread_count);
    }

    if (compute_reach) {
        log("Computing average reach...\n");
        stats.exploration_reachability = compute_exploration_reach(graph);
    }

    return stats;
}

/**
 * @brief Compute all statistics including expensive ones, using pre-computed ground truth sets.
 * 
 * @param graph The search graph to analyze
 * @param exploration_gt Pre-computed ground truth as vector of unordered_sets
 * @param compute_reachability Whether to compute seed reachability (expensive)
 * @param compute_reach Whether to compute average reach (expensive)
 * @param thread_count Number of threads for parallel computation
 * @return GraphStats with all requested statistics
 */
inline GraphStats compute_full_graph_stats(
    const deglib::search::SearchGraph& graph,
    const std::vector<std::unordered_set<uint32_t>>& exploration_gt,
    bool compute_reachability = true,
    bool compute_reach = true,
    uint32_t thread_count = std::thread::hardware_concurrency())
{
    auto stats = collect_graph_stats(graph);

    if (!exploration_gt.empty()) {
        log("Computing graph quality...\n");
        stats.graph_quality = compute_graph_quality(graph, exploration_gt);
    }

    if (compute_reachability) {
        log("Computing seed reachability...\n");
        stats.search_reachability = compute_search_reachability(graph, thread_count);
    }

    if (compute_reach) {
        log("Computing average reach...\n");
        stats.exploration_reachability = compute_exploration_reach(graph);
    }

    return stats;
}

}  // namespace deglib::benchmark
