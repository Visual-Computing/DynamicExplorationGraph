#pragma once

// Core InternalGraph Interface (0..N-1 internal index)
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/mutable_graph.h"
#include "deglib/graph/dynamic_graph.h"

// Visited List Pool Helper
#include "deglib/graph/visited_list_pool.h"

// Concrete Graph Implementations
#include "deglib/graph/readonly_graph.h"
#include "deglib/graph/sizebounded_graph.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace deglib {

/**
 * Public DynamicExplorationGraph Facade for End Users.
 * All public methods accept and return external_labels (User Object IDs).
 * Internal calls map transparently between external_label and internal_index.
 */
class DynamicExplorationGraph {
private:
    std::unique_ptr<deglib::graph::InternalGraph> owned_graph_;
    deglib::graph::InternalGraph* internal_graph_;

public:
    explicit DynamicExplorationGraph(std::unique_ptr<deglib::graph::InternalGraph> graph)
        : owned_graph_(std::move(graph)), internal_graph_(owned_graph_.get()) {
        if (!internal_graph_) {
            throw std::invalid_argument("DynamicExplorationGraph cannot be initialized with nullptr");
        }
    }

    explicit DynamicExplorationGraph(deglib::graph::InternalGraph& graph)
        : owned_graph_(nullptr), internal_graph_(&graph) {}

    DynamicExplorationGraph(DynamicExplorationGraph&&) noexcept = default;
    DynamicExplorationGraph& operator=(DynamicExplorationGraph&&) noexcept = default;

    DynamicExplorationGraph(const DynamicExplorationGraph&) = delete;
    DynamicExplorationGraph& operator=(const DynamicExplorationGraph&) = delete;

    /**
     * Create an empty mutable DynamicExplorationGraph with the given capacity,
     * edges per vertex, and feature space.
     */
    static DynamicExplorationGraph create_empty(
        const uint32_t max_vertex_count,
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace& feature_space)
    {
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(max_vertex_count, edges_per_vertex, feature_space);
        return DynamicExplorationGraph(std::move(graph));
    }

    /**
     * Create an empty mutable DynamicExplorationGraph with chunk-based dynamic memory allocation.
     *
     * @param edges_per_vertex Number of edges per vertex (must be even).
     * @param feature_space The feature space defining dimensionality and metric.
     * @param chunk_size Target number of vertices per memory chunk (default = 1024).
     *                   Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
     */
    static DynamicExplorationGraph create_dynamic_empty(
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace& feature_space,
        const uint32_t chunk_size = 1024)
    {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(edges_per_vertex, feature_space, chunk_size);
        return DynamicExplorationGraph(std::move(graph));
    }

   /**
    * Create a random exploration graph from the given feature data.
    *
    * @param feature_data Pointer to a contiguous array of feature vectors.
    *                     Each vector is feature_space.get_data_size() bytes.
    * @param vertex_count Number of vertices to insert.
    * @param edges_per_vertex Number of edges per vertex (must be even).
    * @param feature_space The feature space defining dimensionality and metric.
    * @param seed Random seed for deterministic graph construction.
    * @return A new DynamicExplorationGraph wrapping the created random graph.
    */
   static DynamicExplorationGraph create_random_graph(
       const std::byte* feature_data,
       const uint32_t vertex_count,
       const uint8_t edges_per_vertex,
       const deglib::distances::FloatSpace& feature_space,
       const uint32_t seed = 7)
   {
       auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(
           deglib::graph::SizeBoundedGraph::create_random_graph(feature_data, vertex_count, edges_per_vertex, feature_space, seed)
       );
       return DynamicExplorationGraph(std::move(graph));
   }

    const uint32_t size() const {
        return internal_graph_->size();
    }

    const uint8_t getEdgesPerVertex() const {
        return internal_graph_->getEdgesPerVertex();
    }

    const deglib::distances::FloatSpace& getFeatureSpace() const {
        return internal_graph_->getFeatureSpace();
    }

    bool hasVertex(const uint32_t external_label) const {
        return internal_graph_->hasVertex(external_label);
    }

    deglib::graph::InternalGraph& internal() {
        return *internal_graph_;
    }
    const deglib::graph::InternalGraph& internal() const {
        return *internal_graph_;
    }

    bool isMutable() const {
        return dynamic_cast<const deglib::graph::MutableGraph*>(internal_graph_) != nullptr;
    }
    /**
     * Search for similar feature vectors using query data.
     * Returns a ResultSet where internal indices are mapped to external_labels.
     */
    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0) const 
    {
        auto res = internal_graph_->search(query, k, eps, filter, max_distance_computation_count);

        // 1. Modify internal vertex IDs to external labels in-place
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }

        return res;
    }
   /**
    * Exploration starting at a specific external_label.
    * Maps entry external_label to internal_index, performs search, and maps result back to external_labels.
    *
    * @param entry_external_label The external label of the vertex to start exploration from.
    * @param k The number of nearest neighbors to return.
    * @param max_distance_computation_count Maximum number of distance calculations. 0 means unlimited.
    * @param eps Epsilon for search range expansion. 0.0 means exact search.
    * @param include_entry If true, the entry vertex is included in the result set.
    * @param filter Optional filter to restrict which vertices can be returned.
    * @return A ResultSet of ObjectDistance objects, with vertex IDs as external labels.
    */
    deglib::graph::ResultSet explore(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr) const
    {
        uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_label);
        auto res = internal_graph_->explore(internal_entry, k, max_distance_computation_count, eps, include_entry, filter);
        
        // 1. Modify internal vertex IDs to external labels in-place
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }

        return res;
    }

   /**
    * Get neighbors as external_labels for a given external_label.
    *
    * @param external_label The external label of the vertex to get neighbors for.
    * @return A vector of external labels of the neighbors.
    */
    std::vector<uint32_t> getNeighbors(const uint32_t external_label) const {
        uint32_t internal_idx = internal_graph_->getInternalIndex(external_label);
        const uint32_t* internal_neighbors = internal_graph_->getNeighborIndices(internal_idx);
        uint8_t edges_count = internal_graph_->getEdgesPerVertex();
        std::vector<uint32_t> external_neighbors(edges_count);
        for (size_t i = 0; i < edges_count; ++i) {
            external_neighbors[i] = internal_graph_->getExternalLabel(internal_neighbors[i]);
        }
        return external_neighbors;
    }

   /**
    * Convert this graph to a read-only graph.
    *
    * @return A new read-only DynamicExplorationGraph.
    */
    DynamicExplorationGraph to_readonly() const {
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(
            internal_graph_->size(),
            internal_graph_->getEdgesPerVertex(),
            internal_graph_->getFeatureSpace(),
            *internal_graph_
        );
        return DynamicExplorationGraph(std::move(graph));
    }

   /**
    * Convert this graph to a mutable SizeBoundedGraph by copying topology,
    * labels, features, and weights directly (fast path, no recalculation).
    *
    * @param new_max_size If > 0, sets capacity to max(new_max_size, input_graph.size()).
    *                     If 0, capacity equals input_graph.size().
    * @return A new mutable DynamicExplorationGraph wrapping a SizeBoundedGraph.
    */
    DynamicExplorationGraph to_mutable(const uint32_t new_max_size = 0) const {
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(
            deglib::graph::SizeBoundedGraph::from_graph(*internal_graph_, new_max_size)
        );
        return DynamicExplorationGraph(std::move(graph));
    }

   /**
    * Convert this graph to a mutable SizeBoundedGraph with a new feature space
    * and/or replacement features. All edge weights are recalculated using the new metric.
    *
    * @param custom_feature_space New feature space defining dimensionality and metric.
    * @param custom_features Optional pointer to replacement feature vectors. If null, features are copied from this graph.
    * @param new_max_size If > 0, sets capacity to max(new_max_size, input_graph.size()). If 0, capacity equals input_graph.size().
    * @return A new mutable DynamicExplorationGraph wrapping a SizeBoundedGraph.
    */
    DynamicExplorationGraph to_mutable(
        const deglib::distances::FloatSpace& custom_feature_space,
        const void* custom_features = nullptr,
        const uint32_t new_max_size = 0) const
    {
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(
            deglib::graph::SizeBoundedGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, new_max_size)
        );
        return DynamicExplorationGraph(std::move(graph));
    }

    /**
     * Convert this graph to a mutable DynamicGraph with chunk-based allocation.
     * 
     * @param chunk_size Target number of vertices per memory chunk (default = 1024).
     *                   Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
     */
    DynamicExplorationGraph to_dynamic(const uint32_t chunk_size = 1024) const {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(
            deglib::graph::DynamicGraph::from_graph(*internal_graph_, chunk_size)
        );
        return DynamicExplorationGraph(std::move(graph));
    }

    /**
     * Convert this graph to a mutable DynamicGraph with custom feature space.
     * 
     * @param custom_feature_space Target feature space (dimension and metric).
     * @param custom_features Optional raw byte pointer to new feature vectors.
     * @param chunk_size Target number of vertices per memory chunk (default = 1024).
     *                   Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
     */
    DynamicExplorationGraph to_dynamic(
        const deglib::distances::FloatSpace& custom_feature_space,
        const void* custom_features = nullptr,
        const uint32_t chunk_size = 1024) const
    {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(
            deglib::graph::DynamicGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, chunk_size)
        );
        return DynamicExplorationGraph(std::move(graph));
    }

   /**
    * Save the graph to a file.
    *
    * @param path The file path where the graph should be saved.
    * @return True if the graph was saved successfully.
    */
    bool saveGraph(const std::string& path) const {
        const auto* mutable_graph = dynamic_cast<const deglib::graph::MutableGraph*>(internal_graph_);
        if (mutable_graph == nullptr) {
            throw std::runtime_error("Graph must be mutable to save");
        }
        return mutable_graph->saveGraph(path.c_str());
    }
};

} // namespace deglib



