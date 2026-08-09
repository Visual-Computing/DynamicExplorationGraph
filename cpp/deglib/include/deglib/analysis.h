#pragma once

#include <math.h>

#include <limits>

#include "deglib/concurrent.h"
#include "deglib/utils/memory.h"
#include "deglib/search.h"
#include "deglib/graph.h"

namespace deglib::analysis
{
    /**
     * Check if the number of vertices and edges is consistent. 
     * The edges of a vertex should only contain unique neighbor indices in ascending order and no self-loops.
     * 
     * @param check_back_link checks if all edges are undirected (quite expensive)
     */
    static bool check_graph_regularity(const deglib::graph::InternalGraph& graph, const uint32_t expected_vertices, const bool check_back_link = false) {

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
    static bool check_graph_connectivity(const deglib::graph::InternalGraph& graph) {
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

    // ============================================================================
    // Graph Reachability & Analysis
    // ============================================================================

   /**
    * Statistics structure returned by analyze_graph.
    */
   struct GraphStats {
       uint32_t vertex_count = 0;
       uint32_t edge_count = 0;
       uint32_t feature_dims = 0;
       uint8_t edges_per_vertex = 0;
       float avg_out_degree = 0.0f;
       uint32_t min_out_degree = 0;
       uint32_t max_out_degree = 0;
       float avg_in_degree = 0.0f;
       uint32_t min_in_degree = 0;
       uint32_t max_in_degree = 0;
       uint32_t source_vertices = 0;
       float search_reachability = 0.0f;
       float exploration_reachability = 0.0f;
       size_t memory_bytes = 0;
   };

   /**
    * Compute the seed reachability count.
    *
    * Measures how many vertices can be reached from the graph's entry points via BFS.
    * This tests if the graph is well-connected for nearest neighbor search.
    *
    * @param graph The search graph to analyze
    * @return Number of vertices reachable from entry points
    */
   static uint32_t calc_search_reachability(const deglib::graph::InternalGraph& graph) {
       const auto graph_size = (uint32_t)graph.size();
       const auto edges_per_vertex = graph.getEdgesPerVertex();
       const auto entry_vertices = graph.getEntryVertexIndices();

       std::vector<bool> visited(graph_size, false);
       std::vector<uint32_t> frontier;
       frontier.reserve(graph_size);

       for (const auto& s : entry_vertices) {
           if (s < graph_size && !visited[s]) {
               visited[s] = true;
               frontier.push_back(s);
           }
       }

       while (!frontier.empty()) {
           std::vector<uint32_t> next_frontier;
           for (const auto& v : frontier) {
               const auto neighbor_indices = graph.getNeighborIndices(v);
               for (uint8_t e = 0; e < edges_per_vertex; e++) {
                   const auto neighbor_index = neighbor_indices[e];
                   if (neighbor_index == (std::numeric_limits<uint32_t>::max)()) continue;

                   if (!visited[neighbor_index]) {
                       visited[neighbor_index] = true;
                       next_frontier.push_back(neighbor_index);
                   }
               }
           }
           frontier = std::move(next_frontier);
       }

       uint32_t count = 0;
       for (size_t i = 0; i < graph_size; i++) {
           if (visited[i]) count++;
       }

       return count;
   }

   /**
    * Structure to store vertex reach information for caching during avg reach computation.
    */
   struct VertexReach {
       uint32_t vertex_id;               ///< The vertex ID
       uint32_t reach_count;             ///< Number of vertices reachable from this vertex
       std::vector<bool> reachable_ids;  ///< Bitmap of reachable vertex IDs

       VertexReach(uint32_t id, uint32_t count, std::vector<bool>&& ids) : vertex_id(id), reach_count(count), reachable_ids(std::move(ids)) {}
   };

   /**
    * Compute the average exploration reach of the graph.
    *
    * Measures the average number of vertices reachable from any given vertex.
    * Uses a caching optimization where vertices processed earlier share their reachability set.
    *
    * @param graph The search graph to analyze
    * @return Average number of vertices reachable per vertex
    */
   static float calc_exploration_reach(const deglib::graph::InternalGraph& graph) {
       const auto graph_size = (uint32_t)graph.size();
       const auto edges_per_vertex = graph.getEdgesPerVertex();

       if (graph_size == 0) return 0.0f;

       // Remember vertices with high reach for optimization
       uint32_t best_vertex_reach = 0;
       auto vertices_reach = std::vector<VertexReach>();
       auto index_of_vertex_reach = std::vector<uint32_t>(graph_size);
       std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), graph_size);

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
      }

      return (graph_size > 0) ? (static_cast<float>(exploration_reachability) / static_cast<float>(graph_size)) : 0.0f;
   }

   /**
    * Analyze a search graph, compute all statistics, and return them.
    *
    * This is the main function for graph analysis. It computes basic stats,
    * reachability metrics, and degree statistics.
    *
    * @param graph The search graph to analyze
    * @return GraphStats with all computed statistics
    */
   static GraphStats analyze_graph(const deglib::graph::InternalGraph& graph) {
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

       // Compute in-degree stats
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
       stats.memory_bytes = stats.vertex_count * (stats.edges_per_vertex * 4 + stats.edges_per_vertex * 4 + stats.feature_dims * 4);

       // Reachability metrics
       uint32_t reachable = calc_search_reachability(graph);
       stats.search_reachability = graph_size > 0 ? (float)reachable / graph_size : 0.0f;

       float avg_reach = calc_exploration_reach(graph);
       stats.exploration_reachability = graph_size > 0 ? avg_reach / graph_size : 0.0f;

       return stats;
   }

} // end namespace deglib::analysis
