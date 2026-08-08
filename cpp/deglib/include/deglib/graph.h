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

#include <vector>
#include <span>
#include <stdexcept>
#include <memory>

namespace deglib {

/**
 * Public DynamicExplorationGraph Facade for End Users.
 * All public methods accept and return external_labels (User Object IDs).
 * Internal calls map transparently between external_label and internal_index.
 */
class DynamicExplorationGraph {
private:
    deglib::graph::InternalGraph& internal_graph_;

public:
   explicit DynamicExplorationGraph(deglib::graph::InternalGraph& graph) : internal_graph_(graph) {}

    const uint32_t size() const {
        return internal_graph_.size();
    }

    const uint8_t getEdgesPerVertex() const {
        return internal_graph_.getEdgesPerVertex();
    }

    const deglib::FloatSpace& getFeatureSpace() const {
        return internal_graph_.getFeatureSpace();
    }

    bool hasVertex(const uint32_t external_label) const {
        return internal_graph_.hasVertex(external_label);
    }

    deglib::graph::InternalGraph& internal() {
        return internal_graph_;
    }
    const deglib::graph::InternalGraph& internal() const {
        return internal_graph_;
    }

    bool isMutable() const {
        return dynamic_cast<const deglib::graph::MutableGraph*>(&internal_graph_) != nullptr;
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
        const deglib::graph::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0) const 
    {
        auto internal_res = internal_graph_.search(query, k, eps, filter, max_distance_computation_count);
        deglib::graph::ResultSet external_res;
        for (const auto& od : internal_res) {
            uint32_t ext_label = internal_graph_.getExternalLabel(od.getInternalIndex());
            external_res.push(deglib::graph::ObjectDistance(ext_label, od.getDistance()));
        }
        return external_res;
    }

    /**
     * Exploration starting at a specific external_label.
     * Maps entry external_label to internal_index, performs search, and maps result back to external_labels.
     */
    deglib::graph::ResultSet explore(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::graph::Filter* filter = nullptr) const
    {
        uint32_t internal_entry = internal_graph_.getInternalIndex(entry_external_label);
        auto internal_res = internal_graph_.explore(internal_entry, k, max_distance_computation_count, eps, include_entry, filter);
        deglib::graph::ResultSet external_res;
        for (const auto& od : internal_res) {
            uint32_t ext_label = internal_graph_.getExternalLabel(od.getInternalIndex());
            external_res.push(deglib::graph::ObjectDistance(ext_label, od.getDistance()));
        }
        return external_res;
    }

    /**
     * Get neighbors as external_labels for a given external_label.
     */
    std::vector<uint32_t> getNeighbors(const uint32_t external_label) const {
        uint32_t internal_idx = internal_graph_.getInternalIndex(external_label);
        const uint32_t* internal_neighbors = internal_graph_.getNeighborIndices(internal_idx);
        uint8_t edges_count = internal_graph_.getEdgesPerVertex();
        std::vector<uint32_t> external_neighbors(edges_count);
        for (size_t i = 0; i < edges_count; ++i) {
            external_neighbors[i] = internal_graph_.getExternalLabel(internal_neighbors[i]);
        }
        return external_neighbors;
    }
};

} // namespace deglib
