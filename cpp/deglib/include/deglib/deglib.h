#pragma once

#include "deglib/analysis.h"
#include "deglib/builder.h"
#include "deglib/config.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph.h"
#include "deglib/optimization.h"
#include "deglib/search.h"
#include "deglib/search/searcher.h"
#include "deglib/utils/memory.h"
#include "deglib/utils/random.h"

namespace deglib {
using builder::build_from_data;
using builder::EvenRegularGraphBuilder;
using builder::OptimizationTarget;

/**
 * Create an EvenRegularGraphBuilder for building and incrementally modifying a DynamicExplorationGraph.
 */
inline builder::EvenRegularGraphBuilder create_builder(
    DynamicExplorationGraph& graph,
    std::mt19937& rnd,
    const builder::OptimizationTarget optimization_target = builder::OptimizationTarget::LowLID,
    const uint8_t extend_k = 0,
    const float extend_eps = 0.1f,
    const uint8_t improve_k = 0,
    const float improve_eps = 0.001f,
    const uint8_t max_path_length = 5,
    const uint32_t improve_tries = 0
) {
    return builder::EvenRegularGraphBuilder(
        graph, rnd, optimization_target, extend_k, extend_eps, improve_k, improve_eps, max_path_length, improve_tries
    );
}

/**
 * Create an empty mutable DynamicExplorationGraph with the given capacity,
 * edges per vertex, and feature space (backed by SizeBoundedGraph).
 */
inline DynamicExplorationGraph create_empty(
    const uint32_t max_vertex_count,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space
) {
    auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(max_vertex_count, edges_per_vertex, feature_space);
    return DynamicExplorationGraph(std::move(graph));
}

/**
 * Create an empty mutable DynamicExplorationGraph with chunk-based dynamic memory allocation (backed by DynamicGraph).
 *
 * @param edges_per_vertex Number of edges per vertex (must be even).
 * @param feature_space The feature space defining dimensionality and metric.
 * @param chunk_size Target number of vertices per memory chunk (default = 1024).
 *                   Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
 */
inline DynamicExplorationGraph create_dynamic_empty(
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t chunk_size = 1024
) {
    auto graph = std::make_unique<deglib::graph::DynamicGraph>(edges_per_vertex, feature_space, chunk_size);
    return DynamicExplorationGraph(std::move(graph));
}

/**
 * Create a random exploration graph from the given feature data (backed by SizeBoundedGraph).
 *
 * @param feature_data Pointer to a contiguous array of feature vectors.
 *                     Each vector is feature_space.get_data_size() bytes.
 * @param vertex_count Number of vertices to insert.
 * @param edges_per_vertex Number of edges per vertex (must be even).
 * @param feature_space The feature space defining dimensionality and metric.
 * @param seed Random seed for deterministic graph construction.
 * @return A new DynamicExplorationGraph wrapping the created random graph.
 */
inline DynamicExplorationGraph create_random_graph(
    const std::byte* feature_data,
    const uint32_t vertex_count,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t seed = 7
) {
    auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(vertex_count, edges_per_vertex, feature_space);
    deglib::builder::populate_random_graph(*graph, feature_data, vertex_count, seed);
    return DynamicExplorationGraph(std::move(graph));
}

/**
 * Load a saved graph from disk as a compact, immutable DynamicExplorationGraph (backed by ReadOnlyGraph).
 */
inline DynamicExplorationGraph load_readonly_graph(const char* path_graph) {
    auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(deglib::graph::load_readonly_graph(path_graph));
    return DynamicExplorationGraph(std::move(graph));
}

/**
 * Load a saved graph from disk as a chunk-allocated mutable DynamicExplorationGraph (backed by DynamicGraph).
 */
inline DynamicExplorationGraph load_dynamic_graph(const char* path_graph, const uint32_t chunk_size = 1024) {
    auto graph = std::make_unique<deglib::graph::DynamicGraph>(deglib::graph::load_dynamic_graph(path_graph, chunk_size));
    return DynamicExplorationGraph(std::move(graph));
}

/**
 * Load a saved graph from disk as a fixed-capacity mutable DynamicExplorationGraph (backed by SizeBoundedGraph).
 *
 * @param new_max_size Optional capacity override. If 0, uses the graph's serialized size.
 */
inline DynamicExplorationGraph load_mutable_graph(const char* path_graph, const uint32_t new_max_size = 0) {
    auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(deglib::graph::load_sizebounded_graph(path_graph, new_max_size));
    return DynamicExplorationGraph(std::move(graph));
}

}  // namespace deglib