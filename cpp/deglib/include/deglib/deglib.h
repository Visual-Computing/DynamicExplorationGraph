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
#include "deglib/transformation.h"
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
    return builder::EvenRegularGraphBuilder(graph, rnd, optimization_target, extend_k, extend_eps, improve_k, improve_eps, max_path_length, improve_tries);
}

/**
 * Create an empty mutable DynamicExplorationGraph with the given capacity,
 * edges per vertex, and feature space (backed by SizeBoundedGraph).
 */
inline DynamicExplorationGraph create_empty(
    const uint32_t max_vertex_count,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(max_vertex_count, edges_per_vertex, feature_space);
    return DynamicExplorationGraph(std::move(graph), std::move(transformations));
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
    const uint32_t chunk_size = 1024,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    auto graph = std::make_unique<deglib::graph::DynamicGraph>(edges_per_vertex, feature_space, chunk_size);
    return DynamicExplorationGraph(std::move(graph), std::move(transformations));
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
    const uint32_t seed = 7,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(vertex_count, edges_per_vertex, feature_space);
    deglib::builder::populate_random_graph(*graph, feature_data, vertex_count, seed);
    return DynamicExplorationGraph(std::move(graph), std::move(transformations));
}

/**
 * Load a saved graph from disk as a compact, immutable DynamicExplorationGraph (backed by ReadOnlyGraph).
 */
inline DynamicExplorationGraph load_readonly_graph(const char* path_graph) { return DynamicExplorationGraph::load_readonly(path_graph); }

/**
 * Load a saved graph from disk as a chunk-allocated mutable DynamicExplorationGraph (backed by DynamicGraph).
 */
inline DynamicExplorationGraph load_dynamic_graph(const char* path_graph, const uint32_t chunk_size = 1024) {
    return DynamicExplorationGraph::load_dynamic(path_graph, chunk_size);
}

/**
 * Load a saved graph from disk as a fixed-capacity mutable DynamicExplorationGraph (backed by SizeBoundedGraph).
 *
 * @param new_max_size Optional capacity override. If 0, uses the graph's serialized size.
 */
inline DynamicExplorationGraph load_mutable_graph(const char* path_graph, const uint32_t new_max_size = 0) {
    return DynamicExplorationGraph::load_mutable(path_graph, new_max_size);
}

/**
 * Create a high-performance Searcher over a DynamicExplorationGraph facade.
 * Convenience overloads that forward to deglib::search::make_searcher using the graph's internal representation,
 * so end users can pass the public graph type directly.
 */
inline std::unique_ptr<deglib::search::Searcher> make_searcher(const DynamicExplorationGraph& graph) {
    return std::make_unique<deglib::search::Searcher>(graph.internal(), graph.getTransformations());
}

inline std::unique_ptr<deglib::search::Searcher> make_searcher(const DynamicExplorationGraph& graph, deglib::search::Reranker reranker) {
    return std::make_unique<deglib::search::Searcher>(graph.internal(), graph.getTransformations(), std::move(reranker));
}

}  // namespace deglib