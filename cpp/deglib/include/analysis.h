#pragma once

#include <math.h>
#include <thread>
#include <algorithm>

#include "concurrent.h"
#include "search.h"
#include "graph.h"

namespace deglib::analysis
{
    /**
     * Check if the number of vertices and edges is consistent. 
     * The edges of a vertex should only contain unique neighbor indices in ascending order and no self-loops.
     * 
     * @param check_back_link checks if all edges are undirected (quite expensive)
     */
    static bool check_graph_regularity(const deglib::search::SearchGraph& graph, const uint32_t expected_vertices, const bool check_back_link = false) {

        // check vertex count
        auto vertex_count = graph.size();
        if(vertex_count != expected_vertices) {
            std::fprintf(stderr, "the graph has an unexpected number of vertices. expected %d got %d \n", expected_vertices, vertex_count);
            return false;
        }

        // skip if the graph is too small to check
        auto edges_per_vertex = graph.getEdgesPerVertex();
        if(vertex_count <= edges_per_vertex) {
            std::fprintf(stderr, "the graph was to small for checking validity \n");
            return true;
        }

        // check edges
        for (uint32_t n = 0; n < vertex_count; n++) {
            auto neighbor_indices = graph.getNeighborIndices(n);

            // check if the neighbor indizizes of the vertices are in ascending order and unique
            int64_t last_index = -1;
            for (int64_t e = 0; e < edges_per_vertex; e++) {
                auto neighbor_index = neighbor_indices[e];

                if(n == neighbor_index) {
                    std::fprintf(stderr, "vertex %u has a self-loop at position %lld \n", n, e);
                    return false;
                }

                if(last_index == neighbor_index) {
                    std::fprintf(stderr, "vertex %u has a duplicate neighbor at position %lld with the neighbor index %u \n", n, e, neighbor_index);
                    return false;
                }

                if(last_index > neighbor_index) {
                    std::fprintf(stderr, "the neighbor order for vertex %u is invalid: pos %lld has index %lld while pos %lld has index %u \n", n, e-1, last_index, e, neighbor_index);
                    return false;
                }

                if(check_back_link && graph.hasEdge(neighbor_index, n) == false) {
                    std::fprintf(stderr, "the neighbor %u of vertex %u does not have a back link to the vertex \n", neighbor_index, n);
                    return false;
                }

                last_index = neighbor_index;
            }
        }
        
        return true;
    }

    /**
     * Compute the graph quality be
     */
    static float calc_avg_edge_weight(const deglib::graph::MutableGraph& graph, const int scale = 1) {
        double total_distance = 0;
        uint64_t count = 0;

        const auto edges_per_vertex = graph.getEdgesPerVertex();
        const auto vertex_count = graph.size();
        for (uint32_t n = 0; n < vertex_count; n++) {
            const auto weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_vertex; e++)
                total_distance += weights[e];
            count += edges_per_vertex;
        }
        
        total_distance = total_distance * scale / count;
        return (float) total_distance ;
    }

    static auto calc_edge_weight_histogram(const deglib::graph::MutableGraph& graph, const bool sorted, const int scale = 1) {
 
        const auto edges_per_vertex = graph.getEdgesPerVertex();
        const auto vertex_count = graph.size();
        auto all_edge_weights = std::vector<float>();
        all_edge_weights.reserve(edges_per_vertex*vertex_count);
        for (uint32_t n = 0; n < vertex_count; n++) {
            const auto weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_vertex; e++)
                if(weights[e] != 0)
                    all_edge_weights.push_back(weights[e]);
        }

        if(sorted)
            std::sort(all_edge_weights.begin(), all_edge_weights.end());

        size_t bin_count = 10;
        auto bin_size = all_edge_weights.size() / bin_count;
        auto avg_edge_weights = std::vector<float>(10);
        for (size_t bin = 0; bin < bin_count; bin++) {
            float weight_sum = 0;
            for (size_t n = 0; n < bin_size; n++) 
                weight_sum += all_edge_weights[bin_size * bin + n];
            avg_edge_weights[bin] = weight_sum * scale / bin_size;
        }
        
        return avg_edge_weights;
    }

    /**
     * Check if the weights of the graph are still the same to the distance of the vertices
     */
    static auto check_graph_weights(const deglib::graph::MutableGraph& graph) {
        const auto& feature_space = graph.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();
        const auto feature_size = feature_space.get_data_size();
        const auto edges_per_vertex = graph.getEdgesPerVertex();
        const auto vertex_count = graph.size();

        for (uint32_t n = 0; n < vertex_count; n++) {
            const auto fv1 = graph.getFeatureVector(n);
            const auto neighborIds = graph.getNeighborIndices(n); 
            const auto neighborWeights = graph.getNeighborWeights(n); 
            deglib::memory::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(neighborIds[0])), feature_size);
            for (uint8_t e = 0; e < edges_per_vertex; e++) {
                deglib::memory::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(neighborIds[std::min(e + 1, edges_per_vertex - 1)])), feature_size);
                const auto fv2 = graph.getFeatureVector(neighborIds[e]);
                const auto dist = dist_func(fv1, fv2, dist_func_param);

                if(neighborWeights[e] != dist) {
                    std::fprintf(stderr, "Vertex %u at edge index %u has a weight of %f to vertex %u but its distance is %f \n", n, e, neighborWeights[e], neighborIds[e], dist);
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Is the vertex_index a RNG conform neighbor if it gets connected to target_index?
     * 
     * Does vertex_index has a neighbor which is connected to the target_index and has a lower weight?
     */
    static auto checkRNG(const deglib::graph::MutableGraph& graph, const uint32_t edges_per_vertex, const uint32_t vertex_index, const uint32_t target_index, const float vertex_target_weight) {
      const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
      const auto neighbor_weight = graph.getNeighborWeights(vertex_index);
      for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
        const auto neighbor_target_weight = graph.getEdgeWeight(neighbor_indices[edge_idx], target_index);  
        if(neighbor_target_weight >= 0 && vertex_target_weight > std::max(neighbor_weight[edge_idx], neighbor_target_weight)) {
          return false;
        }
      }
      return true;
    }

    static uint32_t calc_non_rng_edges(const deglib::graph::MutableGraph& graph) {
        const auto vertex_count = graph.size();
        const auto edge_per_vertex = graph.getEdgesPerVertex();

        const auto thread_count = std::thread::hardware_concurrency();
        auto removed_rng_edges_per_thread = std::vector<uint32_t>(thread_count);
        deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&] (size_t vertex_index, size_t thread_id) {
            uint32_t removed_rng_edges = 0;
            const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
            const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

            // find all none rng conform neighbors
            for (uint32_t n = 0; n < edge_per_vertex; n++) {
                const auto neighbor_index = neighbor_indices[n];
                const auto neighbor_weight = neighbor_weights[n];

                if(checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) 
                    removed_rng_edges++;
            }
            removed_rng_edges_per_thread[thread_id] += removed_rng_edges;
        });

        // aggregate
        uint32_t removed_rng_edges = 0;
        for (uint32_t i = 0; i < thread_count; i++) 
            removed_rng_edges += removed_rng_edges_per_thread[i];

        return removed_rng_edges;
    }

    /**
     * check if the graph is connected and contains only one graph component
     */
    static bool check_graph_connectivity(const deglib::search::SearchGraph& graph) {
        const auto vertex_count = graph.size();
        const auto edges_per_vertex = graph.getEdgesPerVertex();

        // already checked vertices
        auto checked_ids = std::vector<bool>(vertex_count);

        // vertex the check
        auto check = std::vector<uint32_t>();

        // start with the first vertex
        checked_ids[0] = true;
        check.emplace_back(0);

        // repeat as long as we have vertices to check
		while(check.size() > 0) {	

            // neighbors which will be checked next round
            auto check_next = std::vector<uint32_t>();

            // get the neighbors to check next
            for (auto &&internal_index : check) {
                auto neighbor_indizes = graph.getNeighborIndices(internal_index);
                for (size_t e = 0; e < edges_per_vertex; e++) {
                    auto neighbor_index = neighbor_indizes[e];

                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next.emplace_back(neighbor_index);
                    }
                }
            }

            check = std::move(check_next);
        }

        // how many vertices have been checked
        uint32_t checked_vertex_count = 0;
        for (size_t i = 0; i < vertex_count; i++)
            if(checked_ids[i])
                checked_vertex_count++;

        return checked_vertex_count == vertex_count;
    }
    
} // end namespace deglib::analysis
