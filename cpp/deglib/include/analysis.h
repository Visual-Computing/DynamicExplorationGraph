#pragma once

#include <omp.h>
#include <math.h>
#include <algorithm>
#include <fmt/core.h>

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
    static bool check_graph_validation(const deglib::search::SearchGraph& graph, const uint32_t expected_vertices, const bool check_back_link = false) {

        // check vertex count
        auto vertex_count = graph.size();
        if(vertex_count != expected_vertices) {
            fmt::print(stderr, "the graph has an unexpected number of vertices. expected {} got {} \n", expected_vertices, vertex_count);
            return false;
        }

        // skip if the graph is too small to check
        auto edges_per_vertex = graph.getEdgesPerVertex();
        if(vertex_count <= edges_per_vertex) 
            return true;

        // check edges
        for (uint32_t n = 0; n < vertex_count; n++) {
            auto neighbor_indices = graph.getNeighborIndices(n);

            // check if the neighbor indizizes of the vertices are in ascending order and unique
            int64_t last_index = -1;
            for (size_t e = 0; e < edges_per_vertex; e++) {
                auto neighbor_index = neighbor_indices[e];

                if(n == neighbor_index) {
                    fmt::print(stderr, "vertex {} has a self-loop at position {} \n", n, e);
                    return false;
                }

                if(last_index == neighbor_index) {
                    fmt::print(stderr, "vertex {} has a duplicate neighbor at position {} with the neighbor index {} \n", n, e, neighbor_index);
                    return false;
                }

                if(last_index > neighbor_index) {
                    fmt::print(stderr, "the neighbor order for vertex {} is invalid: pos {} has index {} while pos {} has index {} \n", n, e-1, last_index, e, neighbor_index);
                    return false;
                }

                if(check_back_link && graph.hasEdge(neighbor_index, n) == false) {
                    fmt::print(stderr, "the neighbor {} of vertex {} does not have a back link to the vertex \n", neighbor_index, n);
                    return false;
                }

                last_index = neighbor_index;
            }
        }
        
        return true;
    }

    /**
     * Compute the average graph quality of all vertices in the graph
     */
    static float calc_graph_quality(const deglib::search::SearchGraph& graph) {
        const auto& feature_space = graph.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();
        const auto edges_per_vertex = (int)graph.getEdgesPerVertex();
        const auto vertex_count = (int)graph.size();

        auto perfect_neighbor_counts = std::vector<double>(vertex_count);
        #pragma omp parallel for
        for (int n1 = 0; n1 < vertex_count; n1++) {
            const auto fv1 = graph.getFeatureVector(n1);
            const auto neighborIds = graph.getNeighborIndices(n1); 

            // compute the distance to any other vertex in the graph and sort the list
            auto perfect_neighbors = std::vector<std::pair<uint32_t,float>>(vertex_count);
            MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(0)));
            for (int n2 = 0; n2 < vertex_count; n2++) {
                MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(std::min(n2 + 1, vertex_count - 1))));
                const auto fv2 = graph.getFeatureVector(n2);
                const auto dist = dist_func(fv1, fv2, dist_func_param);
                perfect_neighbors[n2] = {n2,dist};
            }
            std::sort(perfect_neighbors.begin(), perfect_neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;});

            // find the rank of each neighbor in the perfect neighbor list
            auto perfect_neighbor_count = 0.0;
            for (int e1 = 0; e1 < edges_per_vertex; e1++) {
                const auto neighbor_id = neighborIds[e1];
                for (int r = 1; r < edges_per_vertex + 1; r++) {    // skip self reference
                    if(neighbor_id == perfect_neighbors[r].first) {
                        perfect_neighbor_count++;                   // rank 0 is always a self reference
                        break;
                    }
                }
            }

            #pragma omp critical
            {
                perfect_neighbor_counts[n1] = perfect_neighbor_count / std::min(vertex_count, edges_per_vertex);
            }
        }

        auto perfect_neighbor_count = 0.0;
        for (int n1 = 0; n1 < vertex_count; n1++) 
            perfect_neighbor_count += perfect_neighbor_counts[n1];
        return (float)(perfect_neighbor_count / vertex_count);
    }

    /**
     * Compute the average neighbor rank of all vertices in the graph
     */
    static float calc_avg_neighbor_rank(const deglib::search::SearchGraph& graph) {
        const auto& feature_space = graph.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();
        const auto edges_per_vertex = (int)graph.getEdgesPerVertex();
        const auto vertex_count = (int) graph.size();

        auto average_neighbor_ranks = std::vector<double>(vertex_count);
        #pragma omp parallel for
        for (int n1 = 0; n1 < vertex_count; n1++) {
            const auto fv1 = graph.getFeatureVector(n1);
            const auto neighborIds = graph.getNeighborIndices(n1); 

            // compute the distance to any other vertex in the graph and sort the list
            auto perfect_neighbors = std::vector<std::pair<uint32_t,float>>(vertex_count);
            MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(0)));
            for (int n2 = 0; n2 < vertex_count; n2++) {
                MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(std::min(n2 + 1, vertex_count - 1))));
                const auto fv2 = graph.getFeatureVector(n2);
                const auto dist = dist_func(fv1, fv2, dist_func_param);
                perfect_neighbors[n2] = {n2,dist};
            }
            std::sort(perfect_neighbors.begin(), perfect_neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;});

            // find the rank of each neighbor in the perfect neighbor list
            double average_neighbor_rank = 0;
            for (int e1 = 0; e1 < edges_per_vertex; e1++) {
                const auto neighbor_id = neighborIds[e1];
                for (int r = 1; r < vertex_count; r++) {           // skip self reference
                    if(neighbor_id == perfect_neighbors[r].first) {
                        average_neighbor_rank += (r - 1);               // rank 0 is always a self reference
                        break;
                    }
                }
            }

            #pragma omp critical
            {
                average_neighbor_ranks[n1] = average_neighbor_rank / std::min(vertex_count, edges_per_vertex);
            }
        }

        auto average_neighbor_rank = 0.0;
        for (int n1 = 0; n1 < vertex_count; n1++) 
            average_neighbor_rank += average_neighbor_ranks[n1];
        return (float)(average_neighbor_rank / vertex_count);
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

        auto bin_count = 10;
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
        const auto edges_per_vertex = graph.getEdgesPerVertex();
        const auto vertex_count = graph.size();

        for (uint32_t n = 0; n < vertex_count; n++) {
            const auto fv1 = graph.getFeatureVector(n);
            const auto neighborIds = graph.getNeighborIndices(n); 
            const auto neighborWeights = graph.getNeighborWeights(n); 
            MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(neighborIds[0])));
            for (uint8_t e = 0; e < edges_per_vertex; e++) {
                MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(neighborIds[std::min(e + 1, edges_per_vertex - 1)])));
                const auto fv2 = graph.getFeatureVector(neighborIds[e]);
                const auto dist = dist_func(fv1, fv2, dist_func_param);

                if(neighborWeights[e] != dist) {
                    fmt::print(stderr, "Vertex {} at edge index {} has a weight of {} to vertex {} but its distance is {} \n", n, e, neighborWeights[e], neighborIds[e], dist);
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

    /**
     * Is the vertex_index a RNG conform neighbor if it gets connected to target_index?
     */
    static auto checkRNG(const deglib::graph::MutableGraph& graph, const uint32_t vertex_index, const uint32_t target_index, const std::vector<std::pair<uint32_t, float>>& new_neighbors) {
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
      const float vertex_target_weight = dist_func(graph.getFeatureVector(vertex_index), graph.getFeatureVector(target_index), dist_func_param); 

      const auto neighbor_size = new_neighbors.size();
      for (size_t n = 0; n < neighbor_size; n++) {
        const auto& new_neighbor = new_neighbors[n];
        const auto neighbor_target_weight = graph.getEdgeWeight(new_neighbor.first, target_index);
        if(neighbor_target_weight >= 0 && vertex_target_weight > std::max(new_neighbor.second, neighbor_target_weight)) {
         return false;
        }
      }
      return true;
    }


    constexpr double kPi = 3.14159265358979323846264;

    /**
     * Has any vertex in new_neighbors a similar vector degree to the target_index as the given vertex_index?
     * If the degree between the vertex_index and any vertex in new_neighbors is below the threshold (default 60) than the method returns false
     */
    static auto check_SSG_RNG(const deglib::graph::MutableGraph& graph, const uint32_t vertex_index, const float vertex_target_weight, const float angle_threshold, const std::vector<std::pair<uint32_t, float>>& new_neighbors) {

      const auto neighbor_size = new_neighbors.size();
      if(neighbor_size > 0) {
        const float threshold = std::cos(angle_threshold / 180 * kPi);

        const auto dist_func = graph.getFeatureSpace().get_dist_func();
        const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
        const auto vertex_feature = graph.getFeatureVector(vertex_index);

        MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(new_neighbors[0].first)));
        for (size_t n = 0; n < neighbor_size; n++) {
            MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(new_neighbors[std::min(n + 1, neighbor_size - 1)].first)));

            const auto& new_neighbor = new_neighbors[n];
            const auto djk = dist_func(vertex_feature, graph.getFeatureVector(new_neighbor.first), dist_func_param);   // distance between vertex_index and new_neighbor
            const auto p_dist = new_neighbor.second;                                                                   // distance between new_neighbor and target_index
            const auto result_dist = vertex_target_weight;                                                             // distance between vertex_index and target_index

            const auto cos_ij = (p_dist + result_dist - djk) / 2 / std::sqrt(p_dist * result_dist);

            if (cos_ij > threshold) 
                return false;
        }
      }
      return true;
    }

    // NSW edge selection strategy for approximating RNG
    static auto check_NSW_RNG(const deglib::graph::MutableGraph& graph, const uint32_t edges_per_vertex, const uint32_t vertex_index, const uint32_t target_index, const float vertex_target_weight) {
        const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
        for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
          const auto neighbor_target_weight = graph.getEdgeWeight(neighbor_indices[edge_idx], target_index);
          if(neighbor_target_weight >= 0 && vertex_target_weight > neighbor_target_weight) { // incompleteRNG
            return false;
          }
        }
      return true;
    }

    static uint32_t calc_non_rng_edges(const deglib::graph::MutableGraph& graph) {
        const auto vertex_count = graph.size();
        const auto edge_per_vertex =graph.getEdgesPerVertex();

        uint32_t removed_rng_edges = 0;
        for (uint32_t i = 0; i < vertex_count; i++) {
            const auto vertex_index = i;

            const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
            const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

            // find all none rng conform neighbors
            for (uint32_t n = 0; n < edge_per_vertex; n++) {
                const auto neighbor_index = neighbor_indices[n];
                const auto neighbor_weight = neighbor_weights[n];

                if(checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) 
                    removed_rng_edges++;
            }
        }

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