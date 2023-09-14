#pragma once

#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <span>
#include <array>
#include <atomic>

#include <fmt/core.h>

#include "analysis.h"
#include "graph.h"

namespace deglib::builder
{

/**
 * Task to add a vertex to the graph
 */
struct BuilderAddTask {
  uint32_t label;
  uint64_t manipulation_index;
  std::vector<std::byte> feature;
};

/**
 * Task to remove a vertex to the graph
 */
struct BuilderRemoveTask {
  uint32_t label;
  uint64_t manipulation_index;
};

/**
 * Every graph change can be document with this struct. Needed to eventually revert back same changed.
 */ 
struct BuilderChange {
  uint32_t internal_index;
  uint32_t from_neighbor_index;
  float from_neighbor_weight;
  uint32_t to_neighbor_index;
  float to_neighbor_weight;
};

/**
 * Status of the build process. 
 * The process performs within a so called "step" a series of changes.
 * A step is either a series of graph improvement tries or the 
 * addition/deletion of a vertex followed be the improvement tries. 
 * The build process can only be stopped between two steps.
 */
struct BuilderStatus {
  uint64_t step;      // number of graph manipulation steps
  uint64_t added;     // number of added vertices
  uint64_t deleted;   // number of deleted vertices
  uint64_t improved;  // number of successful improvement
  uint64_t tries;     // number of improvement tries
};

class EvenRegularGraphBuilder {

    const uint8_t extend_k_;            // k value for extending the graph
    const float extend_eps_;            // eps value for extending the graph
    const uint8_t improve_k_;           // k value for improving the graph
    const float improve_eps_;           // eps value for improving the graph
    const uint8_t max_path_length_;     // max amount of changes before canceling an improvement try
    const uint32_t swap_tries_;
    const uint32_t additional_swap_tries_;

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    std::atomic<uint64_t> manipulation_counter_;
    std::deque<BuilderAddTask> new_entry_queue_;
    std::queue<BuilderRemoveTask> remove_entry_queue_;

    // should the build loop run until the stop method is called
    bool stop_building_ = false;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, 
                            const uint8_t extend_k, const float extend_eps,  
                            const uint8_t improve_k, const float improve_eps, 
                            const uint8_t max_path_length = 10, const uint32_t swap_tries = 3, const uint32_t additional_swap_tries = 3) 
      : graph_(graph), rnd_(rnd), extend_k_(extend_k), extend_eps_(extend_eps),  
        improve_k_(improve_k), improve_eps_(improve_eps), 
        max_path_length_(max_path_length), swap_tries_(swap_tries), additional_swap_tries_(additional_swap_tries) {
    }

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, const uint32_t swaps) 
      : EvenRegularGraphBuilder(graph, rnd, 
                                graph.getEdgesPerNode(), 0.2f, 
                                graph.getEdgesPerNode(), 0.001f, 
                                5, swaps, swaps) {
    }

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd) 
      : EvenRegularGraphBuilder(graph, rnd, 1) {
    }

    /**
     * Provide the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(const uint32_t label, std::vector<std::byte> feature) {
      auto manipulation_index = manipulation_counter_.fetch_add(1);
      new_entry_queue_.emplace_back(label, manipulation_index, std::move(feature));
    }

    /**
     * Command the builder to remove a vertex from the graph as fast as possible.
     */ 
    void removeEntry(const uint32_t label) {
      auto manipulation_index = manipulation_counter_.fetch_add(1);
      remove_entry_queue_.emplace(label, manipulation_index);
    }

  private:

    /**
     * Convert the queue into a vector with ascending distance order
     **/
    static auto topListAscending(deglib::search::ResultSet& queue) {
      const auto size = (int32_t) queue.size();
      auto topList = std::vector<deglib::search::ObjectDistance>(size);
      for (int32_t i = size - 1; i >= 0; i--) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
      }
      return topList;
    }

    /**
     * Convert the queue into a vector with decending distance order
     **/
    static auto topListDescending(deglib::search::ResultSet& queue) {
      const auto size = queue.size();
      auto topList = std::vector<deglib::search::ObjectDistance>(size);
      for (size_t i = 0; i < size; i++) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
      }
      return topList;
    }

    /**
     * Extend the graph with a new vertex. Find good existing vertex to which this new vertex gets connected.
     */
    void extendGraph(const BuilderAddTask& add_task) {
      auto& graph = this->graph_;
      const auto external_label = add_task.label;

      // graph should not contain a vertex with the same label
      if(graph.hasNode(external_label)) {
        fmt::print(stderr, "graph contains vertex {} already. can not add it again \n", external_label);
        perror("");
        abort();
      }

      // fully connect all vertices
      const auto new_vertex_feature = add_task.feature.data();
      const auto edges_per_vertex = uint32_t(graph.getEdgesPerNode());
      if(graph.size() < edges_per_vertex+1) {

        // add an empty vertex to the graph (no neighbor information yet)
        const auto internal_index = graph.addNode(external_label, new_vertex_feature);

        // connect the new vertex to all other vertices in the graph
        const auto dist_func = graph.getFeatureSpace().get_dist_func();
        const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
        for (size_t i = 0; i < graph.size(); i++) {
          if(i != internal_index) {
            const auto dist = dist_func(new_vertex_feature, graph.getFeatureVector(i), dist_func_param);
            graph.changeEdge(i, i, internal_index, dist);
            graph.changeEdge(internal_index, internal_index, i, dist);
          }
        }

        return;
      }

      // find good neighbors for the new vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      const std::vector<uint32_t> entry_vertex_indices = { distrib(this->rnd_) };
      auto top_list = graph.search(entry_vertex_indices, new_vertex_feature, this->extend_eps_, std::max(uint32_t(this->extend_k_), edges_per_vertex));
      const auto results = topListAscending(top_list);

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(results.size() < edges_per_vertex) {
        fmt::print(stderr, "the graph search for the new vertex {} did only provided {} results \n", external_label, results.size());
        perror("");
        abort();
      }

      // add an empty vertex to the graph (no neighbor information yet)
      const auto internal_index = graph.addNode(external_label, new_vertex_feature);

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
     
      // adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
      int check_rng_phase = 1; // 1 = activated, 2 = deactived

      // remove an edge of the good neighbors and connect them with this new vertex
      auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
      while(new_neighbors.size() < edges_per_vertex) {
        for (size_t i = 0; i < results.size() && new_neighbors.size() < edges_per_vertex; i++) {
          const auto candidate_index = results[i].getInternalIndex();
          const auto candidate_weight = results[i].getDistance();

          // check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
          // since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
          if(graph.hasEdge(candidate_index, internal_index)) 
            continue;

          // // does the candidate has a neighbor which is connected to the new vertex and has a lower distance?
          if(check_rng_phase <= 1 && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
            continue;

          // This version is good for high LID datasets or small graphs with low distance count limit during ANNS
          uint32_t new_neighbor_index = 0;
          float new_neighbor_distance = -1;
          {

            // find the worst edge of the new neighbor
            float new_neighbor_weight = -1;
            float new_neighbor_weight_orig = -1;
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);

            for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              const auto neighbor_index = neighbor_indices[edge_idx];

              // the suggested neighbor might already be in the edge list of the new vertex
              if(graph.hasEdge(neighbor_index, internal_index))
                continue;

              const auto neighbor_weight = neighbor_weights[edge_idx];
              if(neighbor_weight > new_neighbor_weight) {
                new_neighbor_weight = neighbor_weight;
                new_neighbor_weight_orig = neighbor_weights[edge_idx];
                new_neighbor_index = neighbor_index;
              }
            }

            if(new_neighbor_weight == -1) {
              continue;
            }

            new_neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param); 
          }

          // place the new vertex in the edge list of the result-vertex
          graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_weight);
          new_neighbors.emplace_back(candidate_index, candidate_weight);

          // place the new vertex in the edge list of the best edge neighbor
          graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_distance);
          new_neighbors.emplace_back(new_neighbor_index, new_neighbor_distance);
        }
        
        check_rng_phase++;
      }

      if(new_neighbors.size() < edges_per_vertex) {
        fmt::print(stderr, "could find only {} good neighbors for the new vertex {} need {}\n", new_neighbors.size(), internal_index, edges_per_vertex);
        perror("");
        abort();
      }

      // sort the neighbors by their neighbor indices and store them in the new vertex
      {
        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indices = std::vector<uint32_t>(new_neighbors.size());
        auto neighbor_weights = std::vector<float>(new_neighbors.size());
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];
          neighbor_indices[i] = neighbor.first;
          neighbor_weights[i] = neighbor.second;
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());  
      }

      // try to improve some of the non-perfect edges (not part of the range-search)
      {
        auto nonperfect_neighbors = std::vector<std::pair<uint32_t, float>>();
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];

          // was the new neighbor found by the range-search or is just a neighbor of a neighbor
          bool perfect = false;
          for (size_t r = 0; r < results.size(); r++) {
            const auto& result = results[r];
            if(result.getInternalIndex() == neighbor.first) {
              perfect = true;
              break;
            }
          } 

          if(perfect == false && graph.hasEdge(internal_index, neighbor.first)) 
            nonperfect_neighbors.emplace_back(neighbor.first, neighbor.second);
        }

        std::sort(nonperfect_neighbors.begin(), nonperfect_neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;}); // low to high
        for (size_t i = 0; i < nonperfect_neighbors.size(); i++) {
          if(graph.hasEdge(internal_index, nonperfect_neighbors[i].first) && (i % 2 == 0)) { 
            improveEdges(internal_index, nonperfect_neighbors[i].first, nonperfect_neighbors[i].second); 
          }
        }
      }
    }

    /**
     * Removing a vertex from the graph.
     */
    void shrinkGraph(const BuilderRemoveTask& del_task) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = std::min(graph.size(), uint32_t(graph.getEdgesPerNode()));
      
      // 1 collect the vertices which are missing an edge if the vertex gets deleted
      const auto internal_index = graph.getInternalIndex(del_task.label);
      const auto involved_indices = std::vector<uint32_t>(graph.getNeighborIndices(internal_index), graph.getNeighborIndices(internal_index) + edges_per_vertex);

      // 1.1 remove from the edge list of the involved vertices the internal_index (vertex to remove)
      for (size_t n = 0; n < edges_per_vertex; n++) 
        graph.changeEdge(involved_indices[n], internal_index, involved_indices[n], 0); // add self-reference

      // 1.2 handle the use case where the graph does not have enough vertices to fulfill the edgesPerVertex requirement
      //     and just remove the vertex without reconnecting the involved vertices because they are all fully connected
      if((graph.size()-1) <= edges_per_vertex) {
        graph.removeNode(del_task.label);
        return;
      }

      // 2 find pairs or groups of vertices which can reach each other
      auto reachability = tsl::robin_map<uint32_t, std::shared_ptr<tsl::robin_set<uint32_t>>>();
  
      // 2.1 start with checking the adjacent neighbors of the involved vertices
      for (auto&& involved_index : involved_indices) {
        auto it = reachability.find(involved_index);
        if (it == reachability.end())
          it = reachability.emplace(involved_index, std::make_shared<tsl::robin_set<uint32_t>>(tsl::robin_set<uint32_t> { involved_index })).first;

        // is any of the adjacent neighbors of involved_index also in the sorted array of involved_indices
        auto reachable_indices_ptr = it->second;
        auto reachable_indices = reachable_indices_ptr.get();
        const auto neighbor_indices = graph.getNeighborIndices(involved_index);
        for (size_t n = 0; n < edges_per_vertex; n++) {
          const auto neighbor_index = neighbor_indices[n];
          const auto is_involved = std::binary_search(involved_indices.begin(), involved_indices.end(), neighbor_index);
          const auto is_loop = neighbor_index == involved_index; // is self reference from 1.2
          if(is_involved && is_loop == false && reachable_indices->contains(neighbor_index) == false) {

            // if this neighbor does not have a set of reachable vertices yet, share the current set reachableVertices
            const auto neighbor_reachability = reachability.find(neighbor_index);
            if (neighbor_reachability == reachability.end()) {
              reachable_indices->insert(neighbor_index);
              reachability[neighbor_index] = reachable_indices_ptr;
            } else {

              // if the neighbor already has a set of reachable vertices, copy them over and replace all their references to the new and bigger set
              const auto neighbor_reachable_indices = *neighbor_reachability->second;
              reachable_indices->insert(neighbor_reachable_indices.begin(), neighbor_reachable_indices.end());
              for (const auto& neighbor_reachable_index : neighbor_reachable_indices) 
                reachability[neighbor_reachable_index] = reachable_indices_ptr;
            }
          }
        }
      }
  
      // 2.2 use graph.hasPath(...) to find a path for every not paired but involved vertex, to any other involved vertex 
      for (auto vertex_reachability = reachability.begin(); vertex_reachability != reachability.end(); ++vertex_reachability) {
        const auto involved_index = vertex_reachability.key();

        // during 2.1 each vertex got a set of reachable vertices with at least one entry (the vertex itself)
				// all vertices containing only one element still need to find one other reachable vertex 
				if(vertex_reachability.value().get()->size() <= 1) {

          // is there a path from any of the other involved_indices to the lonely vertex?
          auto from_indices = std::vector<uint32_t>();
          std::copy_if(involved_indices.begin(), involved_indices.end(), std::back_inserter(from_indices), [involved_index](uint32_t value) { return value != involved_index; });
          std::vector<deglib::search::ObjectDistance> traceback = graph.hasPath(from_indices, involved_index, improve_eps_, improve_k_);
          if(traceback.size() == 0) {
            // TODO replace with flood fill to find an involved vertex without compute distances
            traceback = graph.hasPath(from_indices, involved_index, 1, graph.size());
          }

          // the last vertex in the traceback path must be one of the other involved vertices
          const auto reachable_index = traceback.back().getInternalIndex();
          auto reachable_indices_of_reachable_index_ptr = reachability[reachable_index];

        // add the involved_index to its reachable set and replace the reachable set of the involved_index 
          reachable_indices_of_reachable_index_ptr.get()->insert(involved_index);
          vertex_reachability.value() = reachable_indices_of_reachable_index_ptr;
        }
      }

      // 3 reconnect the groups
      auto new_edges = std::vector<BoostedEdge>();
		  {
        const auto& feature_space = graph.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        // 3.1 get all unique groups of reachable vertex indices
        auto unique_reachable_groups = std::vector<tsl::robin_set<uint32_t>>();
        {
          auto reachable_groups = std::vector<std::shared_ptr<tsl::robin_set<uint32_t>>>();
          reachable_groups.reserve(reachability.size());
          for (const auto& reachable_vertex : reachability) 
            reachable_groups.push_back(reachable_vertex.second);

          auto unique_groups = std::vector<std::shared_ptr<tsl::robin_set<uint32_t>>>();
          unique_groups.reserve(reachability.size());
          std::unique_copy(reachable_groups.begin(), reachable_groups.end(), std::back_inserter(unique_groups));

          for (const auto& unique_group : unique_groups) 
            unique_reachable_groups.push_back(*unique_group);
        }

      	// 3.2 find the biggest group and connect each of its vertices to one of the smaller groups
        //      Stop when all groups are connected or every vertex in the big group got an additional edge.
        //      In case of the later, repeat the process with the next biggest group.
        if(unique_reachable_groups.size() > 1) {

          // Define a custom comparison function based on the size of the sets
          auto compareBySize = [](const tsl::robin_set<uint32_t>& a, const tsl::robin_set<uint32_t>& b) {
              return a.size() < b.size();
          };

          // Sort the groups by size in ascending order
          std::sort(unique_reachable_groups.begin(), unique_reachable_groups.end(), compareBySize);

          // find the next biggest group
				  for (int g = 0, n = 1; g < unique_reachable_groups.size() && n < unique_reachable_groups.size(); g++) {
            const auto reachable_group = unique_reachable_groups[g];

            // iterate over all its entries to find a vertex which is still missing an edge
            next_vertex: for(auto it = reachable_group.begin(); it != reachable_group.end() && n < unique_reachable_groups.size(); ++it) {
              const auto reachable_index = it.key();

              // has reachable_index still an self-reference?
              if(graph.hasEdge(reachable_index, reachable_index)) {

                // find another vertex in a smaller group, also missing an edge			
                // the other vertex and reachable_index can not share an edge yet, otherwise they would be in the same group due to step 2.1
							  for (; n < unique_reachable_groups.size(); n++) {	
                  const auto other_group = unique_reachable_groups[n];
                  for(const auto& other_index : other_group) {
                    if(graph.hasEdge(other_index, other_index)) {

                      // connect reachable_index and other_index
                      const auto reachable_feature = graph.getFeatureVector(reachable_index);
                      const auto other_feature = graph.getFeatureVector(other_index);
                      const auto new_neighbor_dist = dist_func(reachable_feature, other_feature, dist_func_param);
                      graph.changeEdge(reachable_index, reachable_index, other_index, new_neighbor_dist);
                      graph.changeEdge(other_index, other_index, reachable_index, new_neighbor_dist);
                      new_edges.emplace_back(other_index, reachable_index, new_neighbor_dist, new_neighbor_dist, false);

                      // repeat until all small groups are connected
                      n++;
                      goto next_vertex;
                    }
                  }
                }
              }
            }
          }
        }

        // 3.3 now all groups are reachable but still some vertices are missing edge, try to connect them to each other.
        auto remaining_indices = std::vector<uint32_t>();
        remaining_indices.reserve(edges_per_vertex);
        for(const auto& reachable_group : unique_reachable_groups) 
          std::copy_if(reachable_group.begin(), reachable_group.end(), std::back_inserter(remaining_indices), [this](uint32_t value) { return graph_.hasEdge(value, value); });
          
        for (size_t i = 0; i < remaining_indices.size(); i++) {
          const auto index_A = remaining_indices[i];
          if(graph.hasEdge(index_A, index_A)) { // still missing an edge?

            // find a index_B with the smallest distance to index_A
            const auto feature_A = graph.getFeatureVector(index_A);
            auto best_index_B = -1;
            auto best_distance_AB = std::numeric_limits<float>::max();
            for (size_t j = i+1; j < remaining_indices.size(); j++) {
              const auto index_B = remaining_indices[j];
              if(graph.hasEdge(index_B, index_B) && graph.hasEdge(index_A, index_B) == false) {
                const auto new_neighbor_dist = dist_func(feature_A, graph.getFeatureVector(index_B), dist_func_param);
                if(new_neighbor_dist < best_distance_AB) {
                  best_distance_AB = new_neighbor_dist;
                  best_index_B = index_B;
                }
              }
            }

            // connect vertexA and vertexB
            if(best_index_B >= 0) {
              graph.changeEdge(index_A, index_A, best_index_B, best_distance_AB);
              graph.changeEdge(best_index_B, best_index_B, index_A, best_distance_AB);
            }
          }
        }

        // 3.4 the remaining vertices can not be connected to any of the other involved vertices, because they already have an edge to all of them.
        for (size_t i = 0; i < remaining_indices.size(); i++) {
          const auto index_A = remaining_indices[i];
          if(graph.hasEdge(index_A, index_A)) { // still missing an edge?

            // scan the neighbors of the adjacent vertices of A and find a vertex B with the smallest distance to A
            const auto feature_A = graph.getFeatureVector(index_A);
            uint32_t best_index_B = 0;
            auto best_distance_AB = std::numeric_limits<float>::max();
            const auto neighbors_A = graph.getNeighborIndices(index_A);
            for (size_t n = 0; n < edges_per_vertex; n++) {
              const auto potential_indices = graph.getNeighborIndices(neighbors_A[n]);
              for (size_t p = 0; p < edges_per_vertex; p++) {
                const auto index_B = potential_indices[p];
                if(index_A != index_B && graph.hasEdge(index_A, index_B) == false) {
                   const auto new_neighbor_dist = dist_func(feature_A, graph.getFeatureVector(index_B), dist_func_param);
                  if(new_neighbor_dist < best_distance_AB) {
                    best_distance_AB = new_neighbor_dist;
                    best_index_B = index_B;
                  }
                }
              }
            }

            // Get another vertex missing an edge called C and at this point sharing an edge with A (by definition of 3.2)
            for (size_t j = i+1; j < remaining_indices.size(); j++) {
              const auto index_C = remaining_indices[j];
              if(graph.hasEdge(index_C, index_C)) { // still missing an edge?
                const auto feature_C = graph.getFeatureVector(index_C);

                // check the neighborhood of B to find a vertex D not yet adjacent to C but with the smallest possible distance to C
                auto best_index_D = -1;
                auto best_distance_CD = std::numeric_limits<float>::max();
                const auto neighbors_B = graph.getNeighborIndices(best_index_B);
                for (size_t n = 0; n < edges_per_vertex; n++) {
                  const auto index_D = neighbors_B[n];
                  if(index_A != index_D && best_index_B != index_D && graph.hasEdge(index_C, index_D) == false) {
                    const auto new_neighbor_dist = dist_func(feature_C, graph.getFeatureVector(index_D), dist_func_param);
                    if(new_neighbor_dist < best_distance_CD) {
                      best_distance_CD = new_neighbor_dist;
                      best_index_D = index_D;
                    }
                  }
                }

                // replace edge between B and D, with one between A and B as well as C and D
                graph.changeEdge(best_index_B, best_index_D, index_A, best_distance_AB);
                graph.changeEdge(index_A, index_A, best_index_B, best_distance_AB);
                graph.changeEdge(best_index_D, best_index_B, index_C, best_distance_CD);
                graph.changeEdge(index_C, index_C, best_index_D, best_distance_CD);
                
                break;
              }
            }
          }
        }
      }

      // 4 remove the old vertex, which is no longer referenced by another vertex, from the graph
      graph.removeNode(del_task.label);

      // 5 try to improve some of the new edges
      for (size_t i = 0; i < new_edges.size(); i++) {
        const auto edge = new_edges[i];
        if(graph.hasEdge(edge.from_vertex, edge.to_vertex)) { 
          improveEdges(edge.from_vertex, edge.to_vertex, edge.weight); 
        }
      }
    }

    /**
     * Do not call this method directly instead call improve() to improve the graph.
     *  
     * This is the extended part of the optimization process.
     * The method takes an array where all graph changes will be documented.
	   * Node1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
     * Thru a series of edges swaps both subgraphs should be reconnected..
     * If those changes improve the graph this method returns true otherwise false. 
     * 
     * @return true if a good sequences of changes has been found
     */
    bool improveEdges(std::vector<deglib::builder::BuilderChange>& changes, uint32_t vertex1, uint32_t vertex2, uint32_t vertex3, uint32_t vertex4, float total_gain, const uint8_t steps) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = graph.getEdgesPerNode();

      // the settings are the same for the first two iterations
      const auto search_eps = this->improve_eps_; 
      const auto search_k = this->improve_k_;

      {
        // 1. Find an edge for vertex2 which connects to the subgraph of vertex3 and vertex4. 
        //    Consider only vertices of the approximate nearest neighbor search. Since the 
        //    search started from vertex3 and vertex4 all vertices in the result list are in 
        //    their subgraph and would therefore connect the two potential subgraphs.	
        {
          const auto vertex2_feature = graph.getFeatureVector(vertex2);
          const std::vector<uint32_t> entry_vertex_indices = { vertex3, vertex4 };
          auto top_list = graph.search(entry_vertex_indices, vertex2_feature, this->improve_eps_, improve_k_);

          // find a good new vertex3
          float best_gain = total_gain;
          float dist23 = -1;
          float dist34 = -1;

          // We use the descending order to find the worst swap combination with the best gain
          // Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
          for(auto&& result : topListDescending(top_list)) {
            const uint32_t new_vertex3 = result.getInternalIndex();

            // vertex1 and vertex2 got tested in the recursive call before and vertex4 got just disconnected from vertex2
            if(vertex1 != new_vertex3 && vertex2 != new_vertex3 && graph.hasEdge(vertex2, new_vertex3) == false) {

              // 1.1 When vertex2 and the new vertex 3 gets connected, the full graph connectivity is assured again, 
              //     but the subgraph between vertex1/vertex2 and vertex3/vertex4 might just have one edge(vertex2, vertex3).
              //     Furthermore Node 3 has now to many edges, find an good edge to remove to improve the overall graph distortion. 
              //     FYI: If the just selected vertex3 is the same as the old vertex3, this process might cut its connection to vertex4 again.
              //     This will be fixed in the next step or until the recursion reaches max_path_length.
              const auto neighbor_indices = graph.getNeighborIndices(new_vertex3);
              const auto neighbor_weights = graph.getNeighborWeights(new_vertex3);
              
              for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
                uint32_t new_vertex4 = neighbor_indices[edge_idx];

                // compute the gain of the graph distortion if this change would be applied
                const auto gain = total_gain - result.getDistance() + neighbor_weights[edge_idx];

                // do not remove the edge which was just added
                if(new_vertex4 != vertex2 && best_gain < gain) {
                  best_gain = gain;
                  vertex3 = new_vertex3;
                  vertex4 = new_vertex4;
                  dist23 = result.getDistance();
                  dist34 = neighbor_weights[edge_idx];    
                }
              }
            }
          }

          // no new vertex3 was found
          if(dist23 == -1)
            return false;

          // replace the temporary self-loop of vertex2 with a connection to vertex3. 
          total_gain = (total_gain - dist23) + dist34;
          graph.changeEdge(vertex2, vertex2, vertex3, dist23);
          changes.emplace_back(vertex2, vertex2, 0.f, vertex3, dist23);

          // 1.2 Remove the worst edge of vertex3 to vertex4 and replace it with the connection to vertex2
          //     Add a temporaty self-loop for vertex4 for the missing edge to vertex3
          graph.changeEdge(vertex3, vertex4, vertex2, dist23);
          changes.emplace_back(vertex3, vertex4, dist34, vertex2, dist23);
          graph.changeEdge(vertex4, vertex3, vertex4, 0.f);
          changes.emplace_back(vertex4, vertex3, dist34, vertex4, 0.f);
        }
      }

      // 2. Try to connect vertex1 with vertex4
      {
        const auto& feature_space = this->graph_.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        // 2.1a Node1 and vertex4 might be the same. This is quite the rare case, but would mean there are two edges missing.
        //     Proceed like extending the graph:
        //     Search for a good vertex to connect to, remove its worst edge and connect
        //     both vertices of the worst edge to the vertex4. Skip the edge any of the two
        //     two vertices are already connected to vertex4.
        if(vertex1 == vertex4) {

          // find a good (not yet connected) vertex for vertex1/vertex4
          const std::vector<uint32_t> entry_vertex_indices = { vertex2, vertex3 };
          const auto vertex4_feature = graph.getFeatureVector(vertex4);
          auto top_list = graph.search(entry_vertex_indices, vertex4_feature, this->improve_eps_, improve_k_);

          float best_gain = 0;
          uint32_t best_selected_neighbor = 0;
          float best_old_neighbor_dist = 0;
          float best_new_neighbor_dist = 0;
          uint32_t best_good_vertex = 0;
          float best_good_vertex_dist = 0;
          for(auto&& result : topListAscending(top_list)) {
            const auto good_vertex = result.getInternalIndex();

            // the new vertex should not be connected to vertex4 yet
            if(vertex4 != good_vertex && graph.hasEdge(vertex4, good_vertex) == false) {
              const auto good_vertex_dist = result.getDistance();

              // select any edge of the good vertex which improves the graph quality when replaced with a connection to vertex 4
              const auto neighbors_indices = graph.getNeighborIndices(good_vertex);
              const auto neighbor_weights = graph.getNeighborWeights(good_vertex);
              for (size_t i = 0; i < edges_per_vertex; i++) {
                const auto selected_neighbor = neighbors_indices[i];

                // ignore edges where the second vertex is already connect to vertex4
                if(vertex4 != selected_neighbor && graph.hasEdge(vertex4, selected_neighbor) == false) {
                  const auto factor = 1;
                  // const auto factor = deglib::analysis::checkRNG(graph, edges_per_vertex, good_vertex, selected_neighbor, neighbor_weights[i]) ? 1.0f : rng_factor_;
                  const auto old_neighbor_dist = neighbor_weights[i];
                  const auto new_neighbor_dist = dist_func(vertex4_feature, graph.getFeatureVector(selected_neighbor), dist_func_param);

                  // do all the changes improve the graph?
                  float new_gain = (total_gain + old_neighbor_dist * factor) - (good_vertex_dist + new_neighbor_dist);
                  if(best_gain < new_gain) {
                    best_gain = new_gain;
                    best_selected_neighbor = selected_neighbor;
                    best_old_neighbor_dist = old_neighbor_dist;
                    best_new_neighbor_dist = new_neighbor_dist;
                    best_good_vertex = good_vertex;
                    best_good_vertex_dist = good_vertex_dist;
                  }
                }
              }
            }
          }

          if(best_gain > 0)
          {

            // replace the two self-loops of vertex4/vertex1 with a connection to the good vertex and its selected neighbor
            graph.changeEdge(vertex4, vertex4, best_good_vertex, best_good_vertex_dist);
            changes.emplace_back(vertex4, vertex4, 0.f, best_good_vertex, best_good_vertex_dist);
            graph.changeEdge(vertex4, vertex4, best_selected_neighbor, best_new_neighbor_dist);
            changes.emplace_back(vertex4, vertex4, 0.f, best_selected_neighbor, best_new_neighbor_dist);

            // replace from good vertex the connection to the selected neighbor with one to vertex4
            graph.changeEdge(best_good_vertex, best_selected_neighbor, vertex4, best_good_vertex_dist);
            changes.emplace_back(best_good_vertex, best_selected_neighbor, best_old_neighbor_dist, vertex4, best_good_vertex_dist);

            // replace from the selected neighbor the connection to the good vertex with one to vertex4
            graph.changeEdge(best_selected_neighbor, best_good_vertex, vertex4, best_new_neighbor_dist);
            changes.emplace_back(best_selected_neighbor, best_good_vertex, best_old_neighbor_dist, vertex4, best_new_neighbor_dist);

            return true;
          }

        } else {

          // 2.1b If there is a way from vertex2 or vertex3, to vertex1 or vertex4 then ...
				  //      Try to connect vertex1 with vertex4
          //      Much more likely than 2.1a 
				  if(graph.hasEdge(vertex1, vertex4) == false) {

            // Is the total of all changes still beneficial?
            const auto dist14 = dist_func(graph.getFeatureVector(vertex1), graph.getFeatureVector(vertex4), dist_func_param);
            if((total_gain - dist14) > 0) {

              const std::vector<uint32_t> entry_vertex_indices = { vertex2, vertex3 }; 
              if(graph.hasPath(entry_vertex_indices, vertex1, this->improve_eps_, this->improve_k_).size() > 0 || graph.hasPath(entry_vertex_indices, vertex4, this->improve_eps_, improve_k_).size() > 0) {
                
                // replace the the self-loops of vertex1 with a connection to the vertex4
                graph.changeEdge(vertex1, vertex1, vertex4, dist14);
                changes.emplace_back(vertex1, vertex1, 0.f, vertex4, dist14);

                // replace the the self-loops of vertex4 with a connection to the vertex1
                graph.changeEdge(vertex4, vertex4, vertex1, dist14);
                changes.emplace_back(vertex4, vertex4, 0.f, vertex1, dist14);

                return true;
              }
            }
          }
        }
      }
      
      // 3. Maximum path length
      if(steps >= this->max_path_length_) {
        return false;
      }
      
      // 4. swap vertex1 and vertex4 every second round, to give each a fair chance
      if(steps % 2 == 1) {
        uint32_t b = vertex1;
        vertex1 = vertex4;
        vertex4 = b;
      }

      // 5. early stop
      if(total_gain < 0) {
        return false;
      }

      return improveEdges(changes, vertex1, vertex4, vertex2, vertex3, total_gain, steps + 1);
    }

    /**
     * Try to improve the edge of a random vertex to its worst neighbor
     * 
     * @return true if a change could be made otherwise false
     */
    bool improveEdges() {

      auto& graph = this->graph_;
      const auto edges_per_vertex = graph.getEdgesPerNode();

      // 1. remove the worst edge of a random vertex 

      // 1.1 select a random vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      uint32_t vertex1 = distrib(this->rnd_);

      // 1.2 find the worst edge of this vertex
      const auto neighbor_weights = graph.getNeighborWeights(vertex1);
      const auto neighbor_indices = graph.getNeighborIndices(vertex1);
      for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++)
        if(deglib::analysis::checkRNG(graph, edges_per_vertex, vertex1, neighbor_indices[edge_idx], neighbor_weights[edge_idx]) == false) 
          improveEdges(vertex1, neighbor_indices[edge_idx], neighbor_weights[edge_idx]);

      uint32_t bad_neighbor_index = 0;
      float bad_neighbor_weight = -1.0f;
      for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
        if(bad_neighbor_weight < neighbor_weights[edge_idx]) {
          bad_neighbor_index = neighbor_indices[edge_idx];
          bad_neighbor_weight = neighbor_weights[edge_idx];
        }     
      }

      // nothing found
      if(bad_neighbor_weight < 0)
        return false;

      return improveEdges(vertex1, bad_neighbor_index, bad_neighbor_weight);
    }

    /**
     * Try to improve the existing edge between the two vertices
     * 
     * @return true if a change could be made otherwise false
     */
    bool improveEdges(uint32_t vertex1, uint32_t vertex2, float dist12) {
      auto changes = std::vector<deglib::builder::BuilderChange>();

      // remove the edge between vertex 1 and vertex 2 (add temporary self-loops)
      auto& graph = this->graph_;
      graph.changeEdge(vertex1, vertex2, vertex1, 0.f);
      changes.emplace_back(vertex1, vertex2, dist12, vertex1, 0.f);
      graph.changeEdge(vertex2, vertex1, vertex2, 0.f);
      changes.emplace_back(vertex2, vertex1, dist12, vertex2, 0.f);

      if(improveEdges(changes, vertex1, vertex2, vertex1, vertex1, dist12, 0) == false) {

        // undo all changes, in reverse order
        const auto size = changes.size();
        for (size_t i = 0; i < size; i++) {
          auto c = changes[(size - 1) - i];
          this->graph_.changeEdge(c.internal_index, c.to_neighbor_index, c.from_neighbor_index, c.from_neighbor_weight);
        }

        return false;
      }

      return true;
    }

  public:

    /**
     * Build the graph. This could be run on a separate thread in an infinite loop.
     */
    auto& build(std::function<void(deglib::builder::BuilderStatus&)> callback, const bool infinite = false) {
      auto status = BuilderStatus{};
      const auto edge_per_vertex = this->graph_.getEdgesPerNode();

      // run a loop to add, delete and improve the graph
      do{

        // add or delete a vertex
        if(this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0) {
          auto add_task_manipulation_index = std::numeric_limits<uint64_t>::max();
          auto del_task_manipulation_index = std::numeric_limits<uint64_t>::max();

          if(this->new_entry_queue_.size() > 0) 
            add_task_manipulation_index = this->new_entry_queue_.front().manipulation_index;

          if(this->remove_entry_queue_.size() > 0) 
            del_task_manipulation_index = this->remove_entry_queue_.front().manipulation_index;

          if(add_task_manipulation_index < del_task_manipulation_index) {
            extendGraph(this->new_entry_queue_.front());
            status.added++;
            this->new_entry_queue_.pop_front();
          } else {
            shrinkGraph(this->remove_entry_queue_.front());
            status.deleted++;
            this->remove_entry_queue_.pop();
          }
        }

        //try to improve the graph
        if(improve_k_ > 0) {
          for (int64_t swap_try = 0; swap_try < int64_t(this->swap_tries_); swap_try++) {
            status.tries++;

            if(this->improveEdges()) {
              status.improved++;
              swap_try -= this->additional_swap_tries_;
            }
          }
        }
        
        status.step++;
        callback(status);
      }
      while(this->stop_building_ == false && (infinite || this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0));

      // return the finished graph
      return this->graph_;
    }

    /**
     * Stop the build process
     */
    void stop() {
      this->stop_building_ = true;
    }
};

} // end namespace deglib::builder