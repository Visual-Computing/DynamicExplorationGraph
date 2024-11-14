#pragma once

#include "search.h"

namespace deglib::graph
{

  inline void set_navigation_bit_at_index(std::byte* navigation_mask, std::size_t pos, bool value) {
    std::size_t byte_index = pos / 8;
    std::size_t bit_position = pos % 8;

    if (value) {
        navigation_mask[byte_index] |= (std::byte{1} << bit_position);
    } else {
        navigation_mask[byte_index] &= ~(std::byte{1} << bit_position);
    }
  }

  inline void move_bits(std::byte* navigation_mask, std::size_t from_index, std::size_t to_index, std::size_t bit_count) {
    if (from_index == to_index || bit_count == 0) {
        return; // Nichts zu tun, wenn die Indizes gleich sind oder keine Bits verschoben werden sollen
    }

    if (from_index < to_index && to_index < from_index + bit_count) {
        // Rückwärts kopieren, um Überlappungen zu vermeiden
        for (std::size_t i = bit_count; i > 0; --i) {
            std::size_t from_bit_position = from_index + i - 1;
            std::size_t to_bit_position = to_index + i - 1;
            bool bit_value = static_cast<bool>(navigation_mask[from_bit_position / 8] & (std::byte{1} << (from_bit_position % 8)));
            
            if (bit_value) {
                navigation_mask[to_bit_position / 8] |= (std::byte{1} << (to_bit_position % 8));
            } else {
                navigation_mask[to_bit_position / 8] &= ~(std::byte{1} << (to_bit_position % 8));
            }
            
            // Ursprungs-Bit auf 0 setzen, wenn verschoben
            navigation_mask[from_bit_position / 8] &= ~(std::byte{1} << (from_bit_position % 8));
        }
    } else {
        // Vorwärts kopieren (normales Szenario)
        for (std::size_t i = 0; i < bit_count; ++i) {
            std::size_t from_bit_position = from_index + i;
            std::size_t to_bit_position = to_index + i;
            bool bit_value = static_cast<bool>(navigation_mask[from_bit_position / 8] & (std::byte{1} << (from_bit_position % 8)));
            
            if (bit_value) {
                navigation_mask[to_bit_position / 8] |= (std::byte{1} << (to_bit_position % 8));
            } else {
                navigation_mask[to_bit_position / 8] &= ~(std::byte{1} << (to_bit_position % 8));
            }
            
            // Ursprungs-Bit auf 0 setzen, wenn verschoben
            navigation_mask[from_bit_position / 8] &= ~(std::byte{1} << (from_bit_position % 8));
        }
    }
  }

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
    * @param is_navigation_edge is the new edge an navigation edge
    * @return true if the from_neighbor_index was found and changed
    */
    virtual bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight, const bool is_navigation_edge = false) = 0;


    /**
     * Change all edges of a vertex.
     * The neighbor indices/weights and feature vectors will be copied.
     * The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
     * The indices in the neighbor_indices array must be sorted.
     */
    virtual void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) = 0;


    virtual const std::byte* getNavigationMask(const uint32_t internal_idx) const = 0;

    /**
     * 
     */
    virtual const float* getNeighborWeights(const uint32_t internal_index) const = 0;    

    virtual const float getEdgeWeight(const uint32_t from_neighbor_index, const uint32_t to_neighbor_index) const = 0;    

    virtual const bool saveGraph(const char* path_to_graph) const = 0;
};

}  // end namespace deglib::graph
