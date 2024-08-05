#pragma once

#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <functional>
#include <span>
#include <array>
#include <unordered_map>
#include <unordered_set>

#include "concurrent.h"
#include "analysis.h"
#include "graph.h"

namespace deglib::builder
{


/**
 * A UnionFind class to represent disjoint set.
 * https://www.tutorialspoint.com/cplusplus-program-to-implement-disjoint-set-data-structure
 **/
class UnionFind { 
  private:
    uint32_t default_value;
    std::unordered_map<uint32_t, uint32_t> parents; // TODO replace with robin_map

  public:

    /**
     * Reserves space in the internal map
     */
    UnionFind(int expected_size) {
      parents.reserve(expected_size);
      default_value = std::numeric_limits<uint32_t>::max();
    }

    /**
     * get the default value if an element in not in the unsion
     */
    uint32_t getDefaultValue() {
      return default_value;
    }

    /**
     * Find the root of the set in which element belongs
     */
    uint32_t Find(uint32_t l) const {
      auto it = parents.find(l);
      if(it == parents.end())
        return default_value;

      auto entry = it->second;
      if (entry == l) // if l is root
         return l;
      return Find(entry); // recurs for parent till we find root
    }

    /**
     * perform Union of two subsets element1 and element2  
     */
    void Union(uint32_t m, uint32_t n) {
      uint32_t x = Find(m);
      uint32_t y = Find(n);
      Update(x, y);
    }

    /**
     * If the parents are known via find this method can be called instead of union
     */
    void Update(uint32_t element, uint32_t parent) {
      parents[element] = parent;
    }
};

/**
 * A group of vertices which can reach each other. Some of them might be missing edges.
 * A vertex index is associated with this group to make it unique.
 */
struct ReachableGroup {
  uint32_t vertex_index_;
  std::unordered_set<uint32_t> missing_edges_;      // TODO replace with robin_set
  std::unordered_set<uint32_t> reachable_vertices_; // TODO replace with robin_set

  ReachableGroup(uint32_t vertex_index, uint32_t expected_size) : vertex_index_(vertex_index) {
    missing_edges_.reserve(expected_size);
    reachable_vertices_.reserve(expected_size);
    missing_edges_.insert(vertex_index);
    reachable_vertices_.insert(vertex_index);
  }

   /**
   * removed the element from the list of vertices with missing edges
   */
  void hasEdge(uint32_t element) {
    missing_edges_.erase(element);
  }

  /**
   * return the vertex associated with this group
   */
  uint32_t getVertexIndex() const {
    return vertex_index_;
  }

  /**
   * get the number of vertices which can be reached by this group
   */
  size_t size() const {
    return reachable_vertices_.size();
  }

  /**
   * get the number of vertices in this group which are missing an edge
   */
  size_t getMissingEdgeSize() const {
    return missing_edges_.size();
  }

  /**
   * get the vertices which are missing an edges
   */
  const auto& getMissingEdges() {
    return missing_edges_;
  }

  /**
   * Copy the data from the other group to this group
   */
  void copyFrom(ReachableGroup& otherGroup) {

	  // skip if both are the same object
		if(vertex_index_ == otherGroup.vertex_index_)
			return;

    missing_edges_.insert(otherGroup.missing_edges_.begin(), otherGroup.missing_edges_.end());
    reachable_vertices_.insert(otherGroup.reachable_vertices_.begin(), otherGroup.reachable_vertices_.end());
    // std::copy(otherGroup.missing_edges_.begin(), otherGroup.missing_edges_.end(), std::back_inserter(missing_edges_));
    // std::copy(otherGroup.reachable_vertices_.begin(), otherGroup.reachable_vertices_.end(), std::back_inserter(reachable_vertices_));
  }
};

/**
 * Information about an graph edge. The edge might be no longer part of the graph
 */
struct GraphEdge {
  uint32_t from_vertex;
  uint32_t to_vertex;
  float weight;

  GraphEdge(uint32_t from_vertex, uint32_t to_vertex, float weight)
   : from_vertex(from_vertex), to_vertex(to_vertex), weight(weight) {}
};

/**
 * Task to add a vertex to the graph
 */
struct BuilderAddTask {
  uint32_t label;
  uint64_t manipulation_index;
  std::vector<std::byte> feature;

  BuilderAddTask(uint32_t lbl, uint64_t index, std::vector<std::byte> feat)
    : label(lbl), manipulation_index(index), feature(std::move(feat)) {}
};

/**
 * Task to remove a vertex to the graph
 */
struct BuilderRemoveTask {
  uint32_t label;
  uint64_t manipulation_index;

  BuilderRemoveTask(uint32_t lbl, uint64_t index)
    : label(lbl), manipulation_index(index) {}
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

  BuilderChange(uint32_t internalIdx, uint32_t fromIdx, float fromWeight, uint32_t toIdx, float toWeight)
    : internal_index(internalIdx), from_neighbor_index(fromIdx), from_neighbor_weight(fromWeight),
      to_neighbor_index(toIdx), to_neighbor_weight(toWeight) {}
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

/**
 * Information about the data distribution help to switch between different graph extension strategies.
 * Below 15 = Low (schemeD in the paper)
 * Above 15 = High (schemeC in the paper)
 * For data with distribution Shifts Unknown is better. 
 */
enum LID { Unknown, High, Low };

class EvenRegularGraphBuilder {

    const LID lid_;
    const uint8_t extend_k_;            // k value for extending the graph
    const float extend_eps_;            // eps value for extending the graph

    const uint8_t improve_k_;           // k value for improving the graph
    const float improve_eps_;           // eps value for improving the graph
    const uint8_t max_path_length_;     // max amount of changes before canceling an improvement try
    const uint32_t swap_tries_;
    const uint32_t additional_swap_tries_;

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    BuilderStatus build_status_;
    std::atomic<uint64_t> manipulation_counter_;
    std::deque<BuilderAddTask> new_entry_queue_;
    std::queue<BuilderRemoveTask> remove_entry_queue_;

    // the batch_size should be thread_count * thread_task_count thread_task_size
    uint32_t extend_batch_size = 32;          // the overall number of elements per batch
    uint32_t extend_thread_count = 1;         // number of concurrent threads
    uint32_t extend_thread_task_size = 32;   // each thread processed 32 elements per task
    uint32_t extend_thread_task_count = 10;  // there are 10 tasks per thread per batch


    mutable std::mutex extend_mutex;

    // should the build loop run until the stop method is called
    bool stop_building_ = false;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, const LID lid,
                            const uint8_t extend_k, const float extend_eps, 
                            const uint8_t improve_k, const float improve_eps, 
                            const uint8_t max_path_length = 5, const uint32_t swap_tries = 0, const uint32_t additional_swap_tries = 0) 
      : lid_(lid),
        extend_k_(extend_k),
        extend_eps_(extend_eps),
        improve_k_(improve_k), 
        improve_eps_(improve_eps), 
        max_path_length_(max_path_length), 
        swap_tries_(swap_tries), 
        additional_swap_tries_(additional_swap_tries),
        rnd_(rnd),  
        graph_(graph),
        build_status_() { 

          // each core processes extend_thread_batch_size element per tasks, there are 10 tasks per threads
          extend_thread_count = std::thread::hardware_concurrency();
          extend_batch_size = extend_thread_count * extend_thread_task_count * extend_thread_task_size;
    }

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, const uint32_t swaps) 
      : EvenRegularGraphBuilder(graph, rnd, LID::Unknown,
                                graph.getEdgesPerVertex(), 0.2f, 
                                graph.getEdgesPerVertex(), 0.001f, 
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

    /**
     * Numbers of entries which will be added to the graph
     */
    size_t getNumNewEntries() {
      return new_entry_queue_.size();
    }

    /**
     * Numbers of entries which will be removed from the graph
     */
    size_t getNumRemoveEntries() {
      return remove_entry_queue_.size();
    }

    /**
     * Set the thread count
     */
    void setThreadCount(uint32_t thread_count) {
      extend_thread_count = thread_count;
    }

    /**
     * Set the batch size when adding multiple elements
     */
    void setBatchSize(uint32_t batch_size) {
      extend_batch_size = batch_size;
    }

  private:

    /**
     * Convert the queue into a vector with ascending distance order
     **/
    static auto topListAscending(deglib::search::ResultSet& queue) {
      const auto size = (int32_t) queue.size();
      auto topList = std::vector<deglib::search::ObjectDistance>(size);
      for (int32_t i = size - 1; i >= 0; i--) {
        topList[i] = queue.top();
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
    void extendGraph(const std::vector<BuilderAddTask>& add_tasks) {
      auto& graph = this->graph_;

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();

      // fully connect all vertices
      uint32_t index = 0;
      const auto edges_per_vertex = uint32_t(graph.getEdgesPerVertex());
      while(graph.size() < edges_per_vertex+1 && index < add_tasks.size()) {
        const auto& add_task = add_tasks[index++];        
        const auto external_label = add_task.label;

        // graph should not contain a vertex with the same label
        if(graph.hasVertex(external_label)) {
          std::fprintf(stderr, "graph contains vertex %u already. can not add it again \n", external_label);
          std::perror("");
          std::abort();
        }

        // add an empty vertex to the graph (no neighbor information yet)
        const auto new_vertex_feature = add_task.feature.data();
        const auto internal_index = graph.addVertex(external_label, new_vertex_feature);

        // connect the new vertex to all other vertices in the graph
        for (uint32_t i = 0; i < graph.size(); i++) {
          if(i != internal_index) {
            const auto dist = dist_func(new_vertex_feature, graph.getFeatureVector(i), dist_func_param);
            graph.changeEdge(i, i, internal_index, dist);
            graph.changeEdge(internal_index, internal_index, i, dist);
          }
        }
      }

      if(this->lid_ == Unknown) {
        while(index < add_tasks.size()) 
          extendGraphUnknownLID(add_tasks[index++]);
      } else {
        const auto remaining_add_tasks = std::vector<BuilderAddTask>(add_tasks.begin() + index, add_tasks.end());

        auto batchExtendGraphKnownLID = [&](const std::vector<BuilderAddTask>& tasks, size_t task_index) {
          const auto start_index = task_index * this->extend_thread_task_size;
          const auto end_index = std::min(tasks.size(), (task_index+1) * this->extend_thread_task_size);
          for (size_t i = start_index; i < end_index; i++) 
            extendGraphKnownLID(tasks[i]);
        };

        size_t task_count = (remaining_add_tasks.size() / extend_thread_task_size) + ((remaining_add_tasks.size() % extend_thread_task_size != 0) ? 1 : 0);  // +1, if n_queries % batch_size != 0
        deglib::concurrent::parallel_for(0, task_count, extend_thread_count, [&] (size_t task_index, size_t thread_id) {
          batchExtendGraphKnownLID(remaining_add_tasks, task_index);
        });
      }
    }

    

    /**
     * The LID of the dataset is unknown, add the new data one-by-one single threaded.
     */
    void extendGraphUnknownLID(const BuilderAddTask& add_task) {
      auto& graph = this->graph_;
      const auto external_label = add_task.label;
      const auto new_vertex_feature = add_task.feature.data();
      const auto edges_per_vertex = uint32_t(graph.getEdgesPerVertex());
      
      // find good neighbor candidates for the new vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      const std::vector<uint32_t> entry_vertex_indices = { distrib(this->rnd_) };
      auto top_list = graph.search(entry_vertex_indices, new_vertex_feature, this->extend_eps_, std::max(uint32_t(this->extend_k_), edges_per_vertex*2)); // need 2x otherwise it might lock during neighbor selection
      const auto candidates = topListAscending(top_list);

      // their should always be enough neighbors (search candidates), otherwise the graph would be broken
      if(candidates.size() < edges_per_vertex) {
        std::cerr << "the graph search for the new vertex " << external_label << "did only provide " << candidates.size() << " candidates" << std::endl;
        std::perror("");
        std::abort();
      }

      // add an empty vertex to the graph (no neighbor information yet)
      const auto internal_index = graph.addVertex(external_label, new_vertex_feature);
     
      // adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
      bool check_rng_phase = true; // true = activated, false = deactived

      // list of potential isolates vertices
      auto isolated_vertices = std::vector<uint32_t>();
      isolated_vertices.emplace_back(internal_index);  // self loop needed for restore phase

      // remove an edge of a good neighbor candidate and connect the candidate with the new vertex
      auto slots = (uint32_t) edges_per_vertex - 1; // the new vertex will get an additional neighbor during the restore phase
      while(slots > 0) {
        for (size_t i = 0; i < candidates.size() && slots > 0; i++) {
          const auto candidate_index = candidates[i].getInternalIndex();
          const auto candidate_weight = candidates[i].getDistance();

          // check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
          // since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
          if(graph.hasEdge(candidate_index, internal_index)) 
            continue;

          // does the candidate has a neighbor which is connected to the new vertex and has a lower distance?
          if(check_rng_phase && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
            continue;

          // the vertex is already missing an edge (one of its longer edges was removed during a previous iteration), 
          // just add an new edge between the candidate and the new vertex
          if(graph.hasEdge(candidate_index, candidate_index)) {
            graph.changeEdge(candidate_index, candidate_index, internal_index, candidate_weight);
            graph.changeEdge(internal_index, internal_index, candidate_index, candidate_weight);
            slots--;
            continue;
          }

          // This version is good for high LID datasets or small graphs with low distance count limit during ANNS
          uint32_t new_neighbor_index = 0;
          {
            // find the worst edge of the new neighbor
            float new_neighbor_weight = std::numeric_limits<float>::lowest();
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);

            for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              const auto neighbor_index = neighbor_indices[edge_idx];

              // the suggested neighbor might already be in the edge list of the new vertex
              if(graph.hasEdge(neighbor_index, internal_index))
                continue;

              // is the neighbor already missing an edge?
              if(graph.hasEdge(neighbor_index, neighbor_index)) 
                continue;

              // find heightest weighted neighbor
              const auto neighbor_weight = neighbor_weights[edge_idx];
              if(neighbor_weight > new_neighbor_weight) {
                new_neighbor_weight = neighbor_weight;
                new_neighbor_index = neighbor_index;
              }
            }

            // this should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the candidate-vertex and still has space for more
            if(new_neighbor_weight == std::numeric_limits<float>::lowest()) 
              continue;
          }

          // place the new vertex in the edge list of the candidate_index and the new vertex internal_index
          graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_weight);
          graph.changeEdge(internal_index, internal_index, candidate_index, candidate_weight);
          slots--;

          // replace the edge to the candidate_index from the edge list of new_neighbor_index with a self-reference
          graph.changeEdge(new_neighbor_index, candidate_index, new_neighbor_index, 0);
          isolated_vertices.emplace_back(new_neighbor_index);
        }
        
        check_rng_phase = false;
      }

      // get all vertices which are missing an edge
      isolated_vertices.erase(std::remove_if(isolated_vertices.begin(), isolated_vertices.end(), [&graph](int val) { return graph.hasEdge(val, val) == false; }), isolated_vertices.end());

      // restore the potential disconnected graph componenten
      restoreGraph(isolated_vertices, false);
    }

    /**
     * The LID of the dataset is known and defined, use multi threading to build the graph.
     */
    void extendGraphKnownLID(const BuilderAddTask& add_task) {
      auto& graph = this->graph_;
      const auto external_label = add_task.label;
      const auto new_vertex_feature = add_task.feature.data();
      const auto edges_per_vertex = uint32_t(graph.getEdgesPerVertex());

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();

      // find good neighbors for the new vertex
      //auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      //const std::vector<uint32_t> entry_vertex_indices = { distrib(this->rnd_) };
      const std::vector<uint32_t> entry_vertex_indices = { 0 };
      auto top_list = graph.search(entry_vertex_indices, new_vertex_feature, this->extend_eps_, std::max(uint32_t(this->extend_k_), edges_per_vertex));
      const auto results = topListAscending(top_list);

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(results.size() < edges_per_vertex) {
        std::fprintf(stderr, "the graph search for the new vertex %u did only provided %zu results \n", external_label, results.size());
        std::perror("");
        std::abort();
      }

      uint32_t internal_index = 0;
      {
        std::lock_guard<std::mutex> lock(this->extend_mutex);
        std::atomic_thread_fence(std::memory_order_acquire);
        
        // graph should not contain a vertex with the same label
        if(graph.hasVertex(external_label)) {
          std::fprintf(stderr, "graph contains vertex %u already. can not add it again\n", external_label);
          perror("");
          abort();
        }

        // add an empty vertex to the graph (no neighbor information yet)
        internal_index = graph.addVertex(external_label, new_vertex_feature);
        std::atomic_thread_fence(std::memory_order_release);
      }
     

      // adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
      bool check_rng_phase = true; // true = activated, false = deactived

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

          // does the candidate has a neighbor which is connected to the new vertex and has a lower distance?
          if(check_rng_phase && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
            continue;

          // SchemeC: This version is good for high LID datasets or small graphs with low distance count limit during ANNS
          uint32_t new_neighbor_index = 0;
          float new_neighbor_distance = std::numeric_limits<float>::lowest();
          if(this->lid_ == High) 
          {
            // find the worst edge of the new neighbor
            float new_neighbor_weight = std::numeric_limits<float>::lowest();
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);
            for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              const auto neighbor_index = neighbor_indices[edge_idx];

              // if another thread is building the candidate_index at the moment, than its neighbor list contains self references
              if(candidate_index == neighbor_index)
                continue;

              // the suggested neighbor might already be in the edge list of the new vertex
              if(graph.hasEdge(neighbor_index, internal_index))
                continue;

              // the weight of the neighbor might not be worst than the current worst one     
              const auto neighbor_weight = neighbor_weights[edge_idx];
              if(neighbor_weight > new_neighbor_weight) {
                new_neighbor_weight = neighbor_weight;
                new_neighbor_index = neighbor_index;
              }
            }

            // should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
            if(new_neighbor_weight == std::numeric_limits<float>::lowest()) 
              continue;

            new_neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param); 
          } 
          else
          {
            // find the edge which improves the distortion the most: (distance_new_edge1 + distance_new_edge2) - distance_removed_edge       
            float best_distortion = std::numeric_limits<float>::max();
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);
            for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              const auto neighbor_index = neighbor_indices[edge_idx];

              // if another thread is building the candidate_index at the moment, than its neighbor list contains self references
              if(candidate_index == neighbor_index)
                continue;

              // the suggested neighbor might already be in the edge list of the new vertex
              if(graph.hasEdge(neighbor_index, internal_index))
                continue;

              // take the neighbor with the best distance to the new vertex, which might already be in its edge list
              const auto neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(neighbor_index), dist_func_param);
              float distortion = (candidate_weight + neighbor_distance) - neighbor_weights[edge_idx];   // version D in the paper
              if(distortion < best_distortion) {
                best_distortion = distortion;
                new_neighbor_index = neighbor_index;
                new_neighbor_distance = neighbor_distance;
              }
            }
          }

          // this should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
          if(new_neighbor_distance == std::numeric_limits<float>::lowest()) 
            continue;
          
          // update all edges
          {
            std::lock_guard<std::mutex> lock(this->extend_mutex); 
            std::atomic_thread_fence(std::memory_order_acquire);

            // other threads might have already changed the edges of the new_neighbor_index
            if(graph.hasEdge(candidate_index, new_neighbor_index) && graph.hasEdge(new_neighbor_index, candidate_index) && 
               graph.hasEdge(internal_index, candidate_index) == false && graph.hasEdge(candidate_index, internal_index) == false &&
               graph.hasEdge(internal_index, new_neighbor_index) == false && graph.hasEdge(new_neighbor_index, internal_index) == false) {

              // update edge list of the new vertex
              graph.changeEdge(internal_index, internal_index, candidate_index, candidate_weight);
              graph.changeEdge(internal_index, internal_index, new_neighbor_index, new_neighbor_distance);
              new_neighbors.emplace_back(candidate_index, candidate_weight);
              new_neighbors.emplace_back(new_neighbor_index, new_neighbor_distance);

              // place the new vertex in the edge list of the result-vertex
              graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_weight);

              // place the new vertex in the edge list of the best edge neighbor
              graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_distance);
            }
            std::atomic_thread_fence(std::memory_order_release);
          }
        }

        check_rng_phase = false;
      }
      
      if(new_neighbors.size() < edges_per_vertex) {
        std::fprintf(stderr, "could find only %zu good neighbors for the new vertex %u need %u\n", new_neighbors.size(), internal_index, edges_per_vertex);
        std::perror("");
        std::abort();
      }
    }

    /**
     * Removing a vertex from the graph.
     */
    void reduceGraph(const BuilderRemoveTask& del_task) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = std::min(graph.size(), uint32_t(graph.getEdgesPerVertex()));
      
      // 1 remove the vertex and collect the vertices which are missing an edge
      const auto involved_indices = graph.removeVertex(del_task.label);

      // 1.1 handle the use case where the graph does not have enough vertices to fulfill the edgesPerVertex requirement
		  //     and just remove the vertex without reconnecting the involved vertices because they are all fully connected
      if(graph.size() <= edges_per_vertex) 
        return;

      restoreGraph(involved_indices, true);
    }

    /**
     * Reconnect the vertices indicated in the list of involved_indices.
     * All these vertices are missing an edge.
     */
    void restoreGraph(const std::vector<uint32_t>& involved_indices, bool improve_edges) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = std::min(graph.size(), uint32_t(graph.getEdgesPerVertex()));
      
      // 2 find pairs or groups of vertices which can reach each other		
		  auto unique_groups = std::unordered_set<std::shared_ptr<ReachableGroup>>();	
      {
        auto path_map = UnionFind(edges_per_vertex);
        auto reachable_groups = std::unordered_map<uint32_t, std::shared_ptr<ReachableGroup>>();	
        reachable_groups.reserve(edges_per_vertex);
        for (const auto involved_index : involved_indices) {
          reachable_groups.emplace(involved_index, std::make_shared<ReachableGroup>(involved_index, edges_per_vertex));
          path_map.Update(involved_index, involved_index);
        }

        // helper function to check if we need to find more connected components
        auto is_enough_free_connections = [](const std::vector<uint32_t>& vertices, const UnionFind& paths, const std::unordered_map<uint32_t, std::shared_ptr<ReachableGroup>>& groups) {
          size_t isolated_vertex_counter = 0;
          size_t available_connections_counter = 0;
          for(const auto& involved_vertex : vertices) {
            const auto reachable_Vertex = paths.Find(involved_vertex);
            if(involved_vertex == reachable_Vertex) {
              const auto& group = groups.at(reachable_Vertex);
              if(group->size() == 1)
                isolated_vertex_counter++;
              else if(group->getMissingEdgeSize() > 2)
                available_connections_counter += group->getMissingEdgeSize() - 2;
            }
          }
          return available_connections_counter < isolated_vertex_counter;
        };

        // 2.1 start with checking the adjacent neighbors
        size_t neighbor_check_depth = 0;
        auto check = std::unordered_set<uint32_t>(involved_indices.begin(), involved_indices.end());
        auto check_next = std::unordered_set<uint32_t>();
        while(is_enough_free_connections(involved_indices, path_map, reachable_groups)) {
          for(const auto check_vertex : check) {
            auto involved_vertex = path_map.Find(check_vertex);
            auto reachable_group = reachable_groups.at(involved_vertex);

            // check only involved vertices and vertices which can only reach 1 involved vertex
						// no need for big groups to find other groups at the expense of processing power
            if(neighbor_check_depth > 0 && reachable_group->size() > 1)
              continue;

            // check the neighbors of checkVertex if they can reach another reachableGroup
            auto neighbor_indices = graph.getNeighborIndices(check_vertex);
            for(uint32_t i = 0; i < edges_per_vertex; i++) {
              auto neighbor_index = neighbor_indices[i];

              // skip self references (loops)
              if(neighbor_index == check_vertex)
                continue;

              // which other involved vertex can be reached by this neighbor
              auto other_involved_vertex = path_map.Find(neighbor_index);

              // neighbor is not yet in the union find
              if(other_involved_vertex == path_map.getDefaultValue()) {
                path_map.Update(neighbor_index, involved_vertex);
                check_next.emplace(neighbor_index);
              }
              // the neighbor can reach another involved vertex
              else if(other_involved_vertex != involved_vertex) {
                path_map.Update(other_involved_vertex, involved_vertex);
                reachable_group->copyFrom(*reachable_groups.at(other_involved_vertex));
              }
            }
          }

          // prepare for the next iteration
          std::swap(check, check_next);
          check_next.clear();
          neighbor_check_depth++;
        }

        // copy the unique groups
        for (const auto involved_index : involved_indices) 
          unique_groups.emplace(reachable_groups.at(path_map.Find(involved_index)));
      }

      // 2.2 get all isolated vertices
      auto isolated_groups = std::unordered_set<std::shared_ptr<ReachableGroup>>();	
      for(const auto group : unique_groups)
        if(group->size() == 1)
          isolated_groups.emplace(group);

      // 2.3 find for every isolated vertex the best other involved vertex which is part of a unique group      
      auto new_edges = std::vector<GraphEdge>();
      const auto& feature_space = graph.getFeatureSpace();
      const auto dist_func = feature_space.get_dist_func();
      const auto dist_func_param = feature_space.get_dist_func_param();
      for(const auto isolated_group : isolated_groups) {

        // are you still isolated?
        if(isolated_group->size() > 1)
          continue;

        const auto isolated_vertex = isolated_group->getVertexIndex();
        const auto isolated_vertex_feature = graph.getFeatureVector(isolated_vertex);

        // check the reachable groups for good candidates which can connect to the isolated vertex
        uint32_t best_candidate_index = 0;
        float best_candidate_distance = std::numeric_limits<float>::max();
        deglib::builder::ReachableGroup* best_candidate_group = nullptr;
        for (const auto candidate_group : unique_groups) {

          // skip all groups which do not have enough vertices missing an edge
          const auto& missing_edges = candidate_group->getMissingEdges();
          if(missing_edges.size() <= 2)
            continue;

          // find the candidate with the best distance to the isolated vertex
          for (const auto candidate : missing_edges) {
            const auto candidate_feature = graph.getFeatureVector(candidate);
            const auto distance = dist_func(isolated_vertex_feature, candidate_feature, dist_func_param);
            if(distance < best_candidate_distance) {
              best_candidate_distance = distance;
              best_candidate_index = candidate;
              best_candidate_group = candidate_group.get();
            }
          }
        }

        // found a good candidate, add the isolated vertex to its reachable group and an edge between them
        graph.changeEdge(isolated_vertex, isolated_vertex, best_candidate_index, best_candidate_distance);
        graph.changeEdge(best_candidate_index, best_candidate_index, isolated_vertex, best_candidate_distance);
        new_edges.emplace_back(isolated_vertex, best_candidate_index, best_candidate_distance);

        // merge groups
        best_candidate_group->hasEdge(best_candidate_index);
        isolated_group->hasEdge(isolated_vertex);
        best_candidate_group->copyFrom(*isolated_group);

        unique_groups.erase(isolated_group);
      }

      // 3 reconnect the groups
      auto reachable_groups = std::vector(unique_groups.begin(), unique_groups.end());

      // Define a custom comparison function based on the size of the sets
      auto compareBySize = [](const std::shared_ptr<deglib::builder::ReachableGroup>& a, const std::shared_ptr<deglib::builder::ReachableGroup>& b) {
          return a->getMissingEdgeSize() < b->getMissingEdgeSize(); // < is ascending, > is descending
      };

      // Sort the groups by size in ascending order
      std::sort(reachable_groups.begin(), reachable_groups.end(), compareBySize);

      // 3.1 Find the biggest group and one of its vertices to one vertex of a smaller group. Repeat until only one group is left.
      while(reachable_groups.size() >= 2) {
        auto& reachable_group = *reachable_groups[reachable_groups.size()-1];
        auto& other_group = *reachable_groups[reachable_groups.size()-2];
        auto& reachable_vertices = reachable_group.getMissingEdges();
        auto& other_vertices = other_group.getMissingEdges();

        auto best_other_it = reachable_vertices.begin();
        auto best_reachable_it = reachable_vertices.begin();
        auto best_other_distance = std::numeric_limits<float>::max();

        // iterate over all its entries to find a vertex which is still missing an edge
        for(auto reachable_it = reachable_vertices.begin(); reachable_it != reachable_vertices.end(); ++reachable_it) {
          const auto reachable_index = *reachable_it;
          const auto reachable_feature = graph.getFeatureVector(reachable_index);

          // find another vertex in a smaller group, also missing an edge			
          // the other vertex and reachable_index can not share an edge yet, otherwise they would be in the same group due to step 2.1           
          for(auto other_it = other_vertices.begin(); other_it != other_vertices.end(); ++other_it) {
            const auto other_index = *other_it;
            const auto other_feature = graph.getFeatureVector(other_index);
            const auto candidate_dist = dist_func(reachable_feature, other_feature, dist_func_param);

            if(candidate_dist < best_other_distance) {
              best_other_it = other_it;
              best_reachable_it = reachable_it;
              best_other_distance = candidate_dist;
            }
          }
        }

        // connect reachable_index and other_index
        const auto reachable_index = *best_reachable_it;
        const auto other_index = *best_other_it;
        graph.changeEdge(reachable_index, reachable_index, other_index, best_other_distance);
        graph.changeEdge(other_index, other_index, reachable_index, best_other_distance);

        // move the element from the list of missing edges
        reachable_group.hasEdge(reachable_index);
        other_group.hasEdge(other_index);

        // merge both groups
	      other_group.copyFrom(reachable_group);

        // remove the current group from the list of group since its merged
        reachable_groups.pop_back();
      }

      // 3.4 now all groups are reachable but still some vertices are missing edge, try to connect them to each other.
      auto remaining_indices = std::vector<uint32_t>(reachable_groups[0]->getMissingEdges().begin(), reachable_groups[0]->getMissingEdges().end());
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

      // 3.5 the remaining vertices can not be connected to any of the other involved vertices, because they already have an edge to all of them.
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

      // improve some of the new edges which are not so good
      if(improve_edges && this->improve_k_ > 0) {

        // Define a custom comparison function based on the size of the sets
        auto compareByWeight = [](const GraphEdge& a, const GraphEdge& b) {
          return a.weight > b.weight; // < is ascending, > is descending
        };

        // Sort the groups by size in ascending order
        std::sort(new_edges.begin(), new_edges.end(), compareByWeight);

        // 4 try to improve some of the new edges
        for (size_t i = 0; i < new_edges.size(); i++) {
          const auto edge = new_edges[i];
          if(graph.hasEdge(edge.from_vertex, edge.to_vertex))
            improveEdges(edge.from_vertex, edge.to_vertex, edge.weight); 
        }
      }
    }

    /**
     * Do not call this method directly instead call improve() to improve the graph.
     *  
     * This is the extended part of the optimization process.
     * The method takes an array where all graph changes will be documented.
	   * Vertex1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
     * Thru a series of edges swaps both subgraphs should be reconnected..
     * If those changes improve the graph this method returns true otherwise false. 
     * 
     * @return true if a good sequences of changes has been found
     */
    bool improveEdges(std::vector<deglib::builder::BuilderChange>& changes, uint32_t vertex1, uint32_t vertex2, uint32_t vertex3, uint32_t vertex4, float total_gain, const uint8_t steps) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = graph.getEdgesPerVertex();
      
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
          float dist23 = std::numeric_limits<float>::lowest();
          float dist34 = std::numeric_limits<float>::lowest();

          // We use the descending order to find the worst swap combination with the best gain
          // Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
          for(auto&& result : topListDescending(top_list)) {
            const uint32_t new_vertex3 = result.getInternalIndex();

            // vertex1 and vertex2 got tested in the recursive call before and vertex4 got just disconnected from vertex2
            if(vertex1 != new_vertex3 && vertex2 != new_vertex3 && graph.hasEdge(vertex2, new_vertex3) == false) {

              // 1.1 When vertex2 and the new vertex 3 gets connected, the full graph connectivity is assured again, 
              //     but the subgraph between vertex1/vertex2 and vertex3/vertex4 might just have one edge(vertex2, vertex3).
              //     Furthermore Vertex 3 has now to many edges, find an good edge to remove to improve the overall graph distortion. 
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
          if(dist23 == std::numeric_limits<float>::lowest())
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

        // 2.1a Vertex1 and vertex4 might be the same. This is quite the rare case, but would mean there are two edges missing.
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
      const auto edges_per_vertex = graph.getEdgesPerVertex();

      // 1.1 select a random vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      uint32_t vertex1 = distrib(this->rnd_);

      // 1.2 find the worst edge of this vertex
      const auto neighbor_weights = graph.getNeighborWeights(vertex1);
      const auto neighbor_indices = graph.getNeighborIndices(vertex1);
      auto success = false;
      for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
        const auto vertex2 = neighbor_indices[edge_idx];
        if(graph.hasEdge(vertex1, vertex2) && deglib::analysis::checkRNG(graph, edges_per_vertex, vertex2, vertex1, neighbor_weights[edge_idx]) == false) 
          success |= improveEdges(vertex1, vertex2, neighbor_weights[edge_idx]);
      }

      return success;
    }

    /**
     * Try to improve the existing edge between the two vertices
     * 
     * @return true if a change could be made otherwise false
     */
    bool improveEdges(uint32_t vertex1, uint32_t vertex2, float dist12) {

      // improving edges is disabled
      if(improve_k_ <= 0)
        return false;

      // remove the edge between vertex 1 and vertex 2 (add temporary self-loops)
      auto changes = std::vector<deglib::builder::BuilderChange>();
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
      const auto edge_per_vertex = this->graph_.getEdgesPerVertex();

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

            // create batches
            auto batch = std::vector<BuilderAddTask>();
            batch.reserve(this->extend_batch_size);
            while(this->new_entry_queue_.size() > 0 && batch.size() < this->extend_batch_size && this->new_entry_queue_.front().manipulation_index < del_task_manipulation_index) {
              batch.push_back(std::move(this->new_entry_queue_.front()));
              this->new_entry_queue_.pop_front();
            }

            extendGraph(batch);
            this->build_status_.added+=batch.size();
          } else {
            reduceGraph(this->remove_entry_queue_.front());
            this->build_status_.deleted++;
            this->remove_entry_queue_.pop();
          }
        }

        //try to improve the graph
        if(graph_.size() > edge_per_vertex && improve_k_ > 0) {
          for (int64_t swap_try = 0; swap_try < int64_t(this->swap_tries_); swap_try++) {
            this->build_status_.tries++;

            if(this->improveEdges()) {
              this->build_status_.improved++;
              swap_try -= this->additional_swap_tries_;
            }
          }
        }
        
        this->build_status_.step++;
        callback(this->build_status_);
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

/**
 * Remove all edges which are not MRNG conform
 */
void remove_non_mrng_edges(deglib::graph::MutableGraph& graph) {

  const auto vertex_count = graph.size();
  const auto edge_per_vertex = graph.getEdgesPerVertex();

  const auto start = std::chrono::steady_clock::now();
  const auto thread_count = std::thread::hardware_concurrency();
  auto removed_rng_edges_per_thread = std::vector<uint32_t>(thread_count);
  deglib::concurrent::parallel_for(0, vertex_count, thread_count, [&] (size_t vertex_index, size_t thread_id) {
    uint32_t removed_rng_edges = 0;

    const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
    const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

    // find all none rng conform neighbors
    std::vector<uint32_t> remove_neighbor_ids;
    for (uint32_t n = 0; n < edge_per_vertex; n++) {
      const auto neighbor_index = neighbor_indices[n];
      const auto neighbor_weight = neighbor_weights[n];

      if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) {
        remove_neighbor_ids.emplace_back(neighbor_index);
      }
    }

    for (uint32_t n = 0; n < remove_neighbor_ids.size(); n++) {
      graph.changeEdge(vertex_index, remove_neighbor_ids[n], vertex_index, 0);
      removed_rng_edges++;
    }
    removed_rng_edges_per_thread[thread_id] += removed_rng_edges;
  });

  // aggregate
  uint32_t removed_rng_edges = 0;
  for (uint32_t i = 0; i < thread_count; i++) 
    removed_rng_edges += removed_rng_edges_per_thread[i];

  const auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());

	std::cout << "Removed " << removed_rng_edges << " edges in " << duration_ms << " ms. Final graph contains " << deglib::analysis::calc_non_rng_edges(graph) << " non-RNG edges\n";
}


/**
 * Optimize the edges of the graph.
 */
void optimze_edges(deglib::graph::MutableGraph& graph, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t iterations) {
    
  auto rnd = std::mt19937(7);                         // default 7

  // create a graph builder to add vertices to the new graph and improve its edges
	std::cout << "Start graph builder\n";
  auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, deglib::builder::Unknown, 0, 0.0f, k_opt, eps_opt, i_opt, 1, 0);
  
  // check the integrity of the graph during the graph build process
  auto start = std::chrono::steady_clock::now();
  uint64_t duration_ms = 0;
  const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
    const auto size = graph.size();

    if(status.step % (iterations/10) == 0) {    
      duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
      auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 100);
      auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
      auto connected = deglib::analysis::check_graph_connectivity(graph);

      auto duration = duration_ms / 1000;
	    std::cout << std::setw(7) << status.step << " step, " << std::setw(5) << duration << "s, AEW: " << std::fixed << std::setprecision(2) << std::setw(4) << avg_edge_weight << ", " << (connected ? "" : "not") << " connected, " << (valid_weights ? "valid" : "invalid") << "\n";
      start = std::chrono::steady_clock::now();
    }

    if(status.step > iterations)
      builder.stop();
  };

  // start the build process
  builder.build(improvement_callback, true);
}

} // end namespace deglib::builder
