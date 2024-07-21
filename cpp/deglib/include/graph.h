#pragma once

#include "search.h"

namespace deglib::graph
{

class MutableGraph : public deglib::search::SearchGraph
{
  public:    

   /**
    * Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.
    * 
    * @return the internal index of the new vertex
    */
    virtual uint32_t addVertex(const uint32_t external_label, const std::byte* feature_vector) = 0;

   /**
    * Remove an existing vertex and returns its list of neighbors
    */
    virtual std::vector<uint32_t> removeVertex(const uint32_t external_labelr) = 0;

   /**
    * Swap a neighbor with another neighbor and its weight.
    * 
    * @param internal_index vertex index which neighbors should be changed
    * @param from_neighbor_index neighbor index to remove
    * @param to_neighbor_index neighbor index to add
    * @param to_neighbor_weight weight of the neighbor to add
    * @return true if the from_neighbor_index was found and changed
    */
    virtual bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) = 0;


    /**
     * Change all edges of a vertex.
     * The neighbor indices/weights and feature vectors will be copied.
     * The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
     * The indices in the neighbor_indices array must be sorted.
     */
    virtual void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) = 0;


    /**
     * 
     */
    virtual const float* getNeighborWeights(const uint32_t internal_index) const = 0;    

    virtual const float getEdgeWeight(const uint32_t from_neighbor_index, const uint32_t to_neighbor_index) const = 0;    

    virtual const bool saveGraph(const char* path_to_graph) const = 0;
};

}  // end namespace deglib::graph
