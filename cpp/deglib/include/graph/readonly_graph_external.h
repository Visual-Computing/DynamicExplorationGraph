#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <vector>

#include "deglib.h"
#include "distances.h"
#include "search.h"
#include "visited_list_pool.h"

namespace deglib::graph
{

/**
 * An immutable, read-only n-regular graph whose topology (neighbor index lists
 * and external labels) is stored internally, but whose feature vectors are
 * referenced directly from an externally managed, pre-permuted array.
 *
 * Compared to ReadOnlyGraph + FP16Proxy:
 *  - ReadOnlyGraph stores features inline → getFeatureVector(i) = vertex_data + 0
 *  - ReadOnlyGraph + FP16Proxy stores a {ptr, dim} proxy inline →
 *      getFeatureVector(i) = proxy.ptr (two loads)
 *  - ReadOnlyGraphExternal stores NO features inline →
 *      getFeatureVector(i) = external_features + i * feature_byte_size_ (one pointer + offset, no proxy)
 *
 * Prerequisite: Before constructing this graph, the external features array
 * must be permuted in-place via reorder_features_inplace() so that element
 * [internal_index] of the array corresponds exactly to graph vertex
 * [internal_index].  This permutation is O(N) time and O(1) extra memory
 * (beyond one temporary feature vector).
 *
 * The graph is n-regular and undirected: if there is an edge A→B, B→A must
 * exist.  Internal indices run from 0..N-1.  External labels can be any
 * uint32_t value.
 */
class ReadOnlyGraphExternal : public deglib::search::SearchGraph {

  // -------------------------------------------------------------------------
  // Search / Explore function pointer types
  // -------------------------------------------------------------------------

  using SEARCHFUNC = deglib::search::ResultSet (*)(const ReadOnlyGraphExternal& graph,
                                                    const std::vector<uint32_t>& entry_vertex_indices,
                                                    const std::byte* query,
                                                    const float eps,
                                                    const uint32_t k,
                                                    const deglib::graph::Filter* filter,
                                                    const uint32_t max_distance_computation_count);

  using EXPLOREFUNC = deglib::search::ResultSet (*)(const ReadOnlyGraphExternal& graph,
                                                     const uint32_t entry_vertex_index,
                                                     const uint32_t k,
                                                     const bool include_entry,
                                                     const uint32_t max_distance_computation_count);

  // -------------------------------------------------------------------------
  // Static search dispatch templates
  // -------------------------------------------------------------------------

  template <deglib::distances::DistanceComparator COMPARATOR, bool use_max_distance_count, bool use_filter>
  static deglib::search::ResultSet search_wrapper(const ReadOnlyGraphExternal& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count) {
    return graph.searchImpl<COMPARATOR, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <deglib::distances::DistanceComparator COMPARATOR>
  static deglib::search::ResultSet explore_wrapper(const ReadOnlyGraphExternal& graph, const uint32_t entry_vertex_index, const uint32_t k, const bool include_entry, const uint32_t max_distance_computation_count) {
    return graph.exploreImpl<COMPARATOR>(entry_vertex_index, k, include_entry, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static SEARCHFUNC getSearchFunction(const deglib::FloatSpace& feature_space) {
    return deglib::distances::dispatch_distance(feature_space, []<typename COMPARATOR>() -> SEARCHFUNC {
      return &search_wrapper<COMPARATOR, use_max_distance_count, use_filter>;
    });
  }

  inline static EXPLOREFUNC getExploreFunction(const deglib::FloatSpace& feature_space) {
    return deglib::distances::dispatch_distance(feature_space, []<typename COMPARATOR>() -> EXPLOREFUNC {
      return &explore_wrapper<COMPARATOR>;
    });
  }


  // -------------------------------------------------------------------------
  // Vertex layout: [ neighbor_indices (edges_per_vertex * uint32_t) | external_label (uint32_t) ]
  // Features are stored outside in an external array, permuted so that
  //   external_features_[i * feature_byte_size_]  ==  features for internal index i.
  // -------------------------------------------------------------------------

  static uint32_t compute_vertex_byte_size(const uint8_t edges_per_vertex, const uint8_t alignment) {
    const uint32_t raw = uint32_t(edges_per_vertex) * sizeof(uint32_t) + sizeof(uint32_t);
    if (alignment == 0)
      return raw;
    return ((raw + alignment - 1) / alignment) * alignment;
  }

  static std::byte* compute_aligned_pointer(const std::unique_ptr<std::byte[]>& arr, const uint8_t alignment) {
    if (alignment == 0)
      return arr.get();
    void* ptr = arr.get();
    size_t space = std::numeric_limits<size_t>::max();
    std::align(alignment, 0, ptr, space);
    return static_cast<std::byte*>(ptr);
  }

  // Align vertices to 16 bytes (neighbor lists are pure uint32_t arrays).
  static const uint8_t object_alignment = 16;

  const uint32_t max_vertex_count_;
  const uint8_t  edges_per_vertex_;
  const uint32_t feature_byte_size_;  // bytes per feature vector in the external array

  const uint32_t byte_size_per_vertex_;  // size of one vertex record in vertices_ array
  const uint32_t external_label_offset_; // offset of the external_label field inside a vertex record

  // Internal topology: each vertex record holds neighbor indices + external label.
  std::unique_ptr<std::byte[]> vertices_;
  std::byte* vertices_memory_;

  // Pointer to the externally managed, pre-permuted feature array.
  // features[i * feature_byte_size_ .. (i+1) * feature_byte_size_ - 1]  →  features for internal index i.
  const void* external_features_;

  // label → internal index mapping
  std::unordered_map<uint32_t, uint32_t> label_to_index_;

  const SEARCHFUNC  search_func_;
  const EXPLOREFUNC explore_func_;

  const deglib::FloatSpace feature_space_;

  std::unique_ptr<VisitedListPool> visited_list_pool_;

  // -------------------------------------------------------------------------
  // Private accessors
  // -------------------------------------------------------------------------
  inline std::byte* vertex_by_index(const uint32_t internal_idx) const {
    return vertices_memory_ + size_t(internal_idx) * byte_size_per_vertex_;
  }

  inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<const uint32_t*>(vertex_by_index(internal_idx));
  }

  inline const uint32_t label_by_index(const uint32_t internal_idx) const {
    return *reinterpret_cast<const uint32_t*>(vertex_by_index(internal_idx) + external_label_offset_);
  }

  inline const std::byte* feature_by_index(const uint32_t internal_idx) const {
    return static_cast<const std::byte*>(external_features_) + size_t(internal_idx) * feature_byte_size_;
  }

public:

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  /**
   * Constructs a ReadOnlyGraphExternal from an existing SearchGraph, referencing
   * an externally managed features array.
   *
   * The features array MUST have been pre-permuted with reorder_features_inplace()
   * before calling this constructor so that element [internal_index] contains
   * the features for vertex [internal_index].
   *
   * @param feature_space  Feature space describing the external features.
   * @param input_graph    Source graph (topology is copied, features are NOT copied).
   * @param external_features  Pointer to the pre-permuted external feature array.
   *                           Must remain valid for the lifetime of this graph.
   */
  ReadOnlyGraphExternal(const deglib::FloatSpace feature_space,
                        deglib::search::SearchGraph& input_graph,
                        const void* external_features)
      : max_vertex_count_(input_graph.size()),
        edges_per_vertex_(input_graph.getEdgesPerVertex()),
        feature_byte_size_(static_cast<uint32_t>(feature_space.get_data_size())),

        byte_size_per_vertex_(compute_vertex_byte_size(input_graph.getEdgesPerVertex(), object_alignment)),
        external_label_offset_(uint32_t(input_graph.getEdgesPerVertex()) * sizeof(uint32_t)),

        vertices_(std::make_unique<std::byte[]>(size_t(input_graph.size()) * byte_size_per_vertex_ + object_alignment)),
        vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)),

        external_features_(external_features),

        search_func_(getSearchFunction(feature_space)),
        explore_func_(getExploreFunction(feature_space)),
        feature_space_(feature_space),
        visited_list_pool_(std::make_unique<VisitedListPool>(1, input_graph.size()))
  {
    if (edges_per_vertex_ % 2 != 0)
      throw std::invalid_argument("edges_per_vertex must be even.");

    label_to_index_.reserve(max_vertex_count_);

    const uint32_t N = max_vertex_count_;
    for (uint32_t i = 0; i < N; ++i) {
      std::byte* vtx = vertex_by_index(i);

      // Copy neighbor indices
      const auto* src_neighbors = input_graph.getNeighborIndices(i);
      std::memcpy(vtx, src_neighbors, size_t(edges_per_vertex_) * sizeof(uint32_t));

      // Store external label
      const uint32_t label = input_graph.getExternalLabel(i);
      std::memcpy(vtx + external_label_offset_, &label, sizeof(uint32_t));

      label_to_index_.emplace(label, i);
    }
  }

  // -------------------------------------------------------------------------
  // SearchGraph interface
  // -------------------------------------------------------------------------

  const uint32_t size() const override {
    return static_cast<uint32_t>(label_to_index_.size());
  }

  const uint8_t getEdgesPerVertex() const override {
    return edges_per_vertex_;
  }

  const deglib::FloatSpace& getFeatureSpace() const override {
    return feature_space_;
  }

  inline const uint32_t getInternalIndex(const uint32_t external_label) const override {
    return label_to_index_.find(external_label)->second;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override {
    return label_by_index(internal_idx);
  }

  /**
   * Returns a pointer directly into the pre-permuted external features array.
   * This is a single pointer + multiply — no proxy dereference required.
   */
  inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override {
    return feature_by_index(internal_idx);
  }

  inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override {
    return neighbors_by_index(internal_idx);
  }

  inline const bool hasVertex(const uint32_t external_label) const override {
    return label_to_index_.contains(external_label);
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    const auto* n = getNeighborIndices(internal_index);
    const auto* end = n + edges_per_vertex_;
    return std::binary_search(n, end, neighbor_index);
  }

  // -------------------------------------------------------------------------
  // reorder_features_inplace
  // -------------------------------------------------------------------------

  /**
   * Permutes the external features array in-place so that element [internal_index]
   * contains the features for graph vertex [internal_index].
   *
   * Before calling, features[external_label] must contain the features for that label.
   * Afterwards, features[internal_index] will contain the features for the vertex
   * whose internal index is internal_index.
   *
   * This is an O(N) cycle-following in-place permutation using one temporary
   * feature vector — no additional O(N) memory is needed.
   *
   * @param input_graph   The graph whose internal→external mapping is used.
   * @param features      Pointer to the feature array of size N * dims elements.
   * @param dims          Number of elements per feature vector (e.g. number of uint16_t for FP16).
   */
  template <typename T>
  static void reorder_features_inplace(const deglib::search::SearchGraph& input_graph,
                                        T* features,
                                        const size_t dims)
  {
    const size_t N = input_graph.size();
    // Track which positions have been placed correctly.
    // Uses bit-packing to keep it O(N/8) extra bytes.
    std::vector<bool> done(N, false);

    std::vector<T> temp(dims);

    for (size_t start = 0; start < N; ++start) {
      if (done[start]) continue;

      // Follow the permutation cycle beginning at 'start'.
      // The desired value for position 'i' comes from external_label(i),
      // because before permutation features[label] = features for label,
      // and after we want features[i] = features for vertex i,
      // i.e. features[i] = features_original[getExternalLabel(i)].
      //
      // Cycle: start → p0 → p1 → ... → start
      // where p_k = getExternalLabel(k) gives the source index for position k.

      size_t i = start;
      uint32_t src = input_graph.getExternalLabel(static_cast<uint32_t>(i));

      if (src == static_cast<uint32_t>(i)) {
        // Trivial 1-cycle: already in place.
        done[i] = true;
        continue;
      }

      // Save the element that will be displaced at the start of this cycle.
      std::memcpy(temp.data(), features + i * dims, dims * sizeof(T));

      while (src != static_cast<uint32_t>(start)) {
        // Move features[src] → features[i]
        std::memcpy(features + i * dims, features + src * dims, dims * sizeof(T));
        done[i] = true;
        i = src;
        src = input_graph.getExternalLabel(static_cast<uint32_t>(i));
      }

      // Place the saved element into the last position of the cycle.
      std::memcpy(features + i * dims, temp.data(), dims * sizeof(T));
      done[i] = true;
    }
  }

  // -------------------------------------------------------------------------
  // hasPath
  // -------------------------------------------------------------------------

  std::vector<deglib::search::ObjectDistance> hasPath(
      const std::vector<uint32_t>& entry_vertex_indices,
      const uint32_t to_vertex,
      const float eps,
      const uint32_t k) const override
  {
    const auto query = feature_by_index(to_vertex);
    const auto dist_func = feature_space_.get_dist_func();
    const auto dist_func_param = feature_space_.get_dist_func_param();
    const auto feature_size = feature_space_.get_data_size();

    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    auto next_vertices = deglib::search::UncheckedSet();
    auto trackback = std::unordered_map<uint32_t, deglib::search::ObjectDistance>();
    auto results = deglib::search::ResultSet();

    for (auto&& index : entry_vertex_indices) {
      if (checked_ids[index] != checked_ids_tag) {
        checked_ids[index] = checked_ids_tag;
        const auto feature = feature_by_index(index);
        const auto distance = dist_func(query, feature, dist_func_param);
        results.emplace(index, distance);
        next_vertices.emplace(index, distance);
        trackback.emplace(index, deglib::search::ObjectDistance(index, distance));
      }
    }

    auto radius = std::numeric_limits<float>::max();
    auto exploration_radius = radius;

    auto good_neighbors = std::array<uint32_t, 256>();
    while (!next_vertices.empty()) {
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      if (next_vertex.getDistance() > exploration_radius)
        break;

      size_t good_neighbor_count = 0;
      const auto* neighbor_indices = neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < edges_per_vertex_; ++i) {
        const auto neighbor_index = neighbor_indices[i];

        if (neighbor_index == to_vertex) {
          auto path = std::vector<deglib::search::ObjectDistance>();
          path.emplace_back(next_vertex.getInternalIndex(), next_vertex.getDistance());
          auto last_vertex = trackback.find(next_vertex.getInternalIndex());
          while (last_vertex != trackback.cend() &&
                 last_vertex->first != last_vertex->second.getInternalIndex()) {
            path.emplace_back(last_vertex->second.getInternalIndex(), last_vertex->second.getDistance());
            last_vertex = trackback.find(last_vertex->second.getInternalIndex());
          }
          return path;
        }

        if (checked_ids[neighbor_index] != checked_ids_tag) {
          checked_ids[neighbor_index] = checked_ids_tag;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      memory::prefetch(reinterpret_cast<const char*>(feature_by_index(good_neighbors[0])), feature_size);
      for (size_t i = 0; i < good_neighbor_count; ++i) {
        memory::prefetch(reinterpret_cast<const char*>(feature_by_index(good_neighbors[size_t(i) + 1 < good_neighbor_count - 1 ? size_t(i) + 1 : good_neighbor_count - 1])), feature_size);

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_distance = dist_func(query, feature_by_index(neighbor_index), dist_func_param);

        if (neighbor_distance <= exploration_radius) {
          next_vertices.emplace(neighbor_index, neighbor_distance);
          trackback.insert({neighbor_index, deglib::search::ObjectDistance(next_vertex.getInternalIndex(), next_vertex.getDistance())});

          if (neighbor_distance < radius) {
            results.emplace(neighbor_index, neighbor_distance);
            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
              exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
            }
          }
        }
      }
    }

    return {};
  }

  // -------------------------------------------------------------------------
  // search
  // -------------------------------------------------------------------------

  deglib::search::ResultSet search(
      const std::vector<uint32_t>& entry_vertex_indices,
      const std::byte* query,
      const float eps,
      const uint32_t k,
      const deglib::graph::Filter* filter = nullptr,
      const uint32_t max_distance_computation_count = 0) const override
  {
    if (filter) {
      if (max_distance_computation_count == 0)
        return getSearchFunction<false, true>(feature_space_)(*this, entry_vertex_indices, query, eps, k, filter, 0);
      else
        return getSearchFunction<true, true>(feature_space_)(*this, entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
    } else {
      if (max_distance_computation_count == 0)
        return search_func_(*this, entry_vertex_indices, query, eps, k, nullptr, 0);
      else
        return getSearchFunction<true, false>(feature_space_)(*this, entry_vertex_indices, query, eps, k, nullptr, max_distance_computation_count);
    }
  }

  /**
   * Templated search implementation. The result set contains internal indices.
   */
  template <typename COMPARATOR, bool use_max_distance_count, bool use_filter>
  deglib::search::ResultSet searchImpl(
      const std::vector<uint32_t>& entry_vertex_indices,
      const std::byte* query,
      const float eps,
      const uint32_t initial_k,
      const deglib::graph::Filter* filter,
      const uint32_t max_distance_computation_count) const
  {
    const auto dist_func_param = feature_space_.get_dist_func_param();
    const auto feature_size = feature_space_.get_data_size();
    const size_t degree = edges_per_vertex_;
    const size_t vertex_count = size();

    size_t k = std::min(vertex_count, static_cast<size_t>(initial_k));
    uint32_t distance_computation_count = 0;

    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k * degree);

    auto results = deglib::search::ResultSet();
    results.reserve(k);

    if constexpr (use_filter) {
      if (vertex_count < 1'000 ||
          (filter->get_inclusion_rate() * vertex_count) < 10'000 ||
          filter->get_inclusion_rate() < 0.10f)
      {
        auto radius = std::numeric_limits<float>::max();
        filter->for_each_valid_label([&](uint32_t valid_label) {
          const uint32_t valid_index = getInternalIndex(valid_label);
          const auto distance = COMPARATOR::compare(query, feature_by_index(valid_index), dist_func_param);
          if (distance < radius) {
            results.emplace(valid_index, distance);
            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
            }
          }
        });
        return results;
      }
    }

    for (auto&& index : entry_vertex_indices) {
      if (checked_ids[index] != checked_ids_tag) {
        checked_ids[index] = checked_ids_tag;
        const auto distance = COMPARATOR::compare(query, feature_by_index(index), dist_func_param);
        next_vertices.emplace(index, distance);
        if constexpr (use_filter) {
          if (filter->is_valid(label_by_index(index)))
            results.emplace(index, distance);
        } else {
          results.emplace(index, distance);
        }
        if constexpr (use_max_distance_count) {
          if (++distance_computation_count >= max_distance_computation_count)
            return results;
        }
      }
    }

    auto radius = std::numeric_limits<float>::max();
    auto exploration_radius = radius;

    auto good_neighbors = std::array<uint32_t, 256>();
    alignas(32) auto db_arr = std::array<const void*, 256>();
    alignas(32) auto dists = std::array<float, 256>();
    while (!next_vertices.empty()) {
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      if (next_vertex.getDistance() > exploration_radius)
        break;

      size_t good_neighbor_count = 0;
      const auto* neighbor_indices = neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < degree; ++i) {
        const auto neighbor_index = neighbor_indices[i];
        if (checked_ids[neighbor_index] != checked_ids_tag) {
          checked_ids[neighbor_index] = checked_ids_tag;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      // Cap the neighbor count based on the remaining distance budget
      if constexpr (use_max_distance_count) {
        if (distance_computation_count + good_neighbor_count > max_distance_computation_count) {
          good_neighbor_count = max_distance_computation_count - distance_computation_count;
        }
      }

      // Construct features pointer array
      for (size_t i = 0; i < good_neighbor_count; ++i) {
        db_arr[i] = feature_by_index(good_neighbors[i]);
        if (i < 8)
          memory::prefetch(db_arr[i]);
      }

      // Compute distances in batch
      deglib::distances::compare_batch<COMPARATOR>(query, db_arr.data(), good_neighbor_count, dist_func_param, dists.data());

      // Process results sequentially
      for (size_t i = 0; i < good_neighbor_count; ++i) {
        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_distance = dists[i];

        if (neighbor_distance <= exploration_radius) {
          next_vertices.emplace(neighbor_index, neighbor_distance);

          if (neighbor_distance < radius) {
            if constexpr (use_filter) {
              if (filter->is_valid(label_by_index(neighbor_index)))
                results.emplace(neighbor_index, neighbor_distance);
            } else {
              results.emplace(neighbor_index, neighbor_distance);
            }

            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
              exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
            }
          }
        }
      }

      if constexpr (use_max_distance_count) {
        distance_computation_count += good_neighbor_count;
        if (distance_computation_count >= max_distance_computation_count) {
          return results;
        }
      }
    }

    return results;
  }

  // -------------------------------------------------------------------------
  // explore
  // -------------------------------------------------------------------------

  deglib::search::ResultSet explore(
      const uint32_t entry_vertex_index,
      const uint32_t k,
      const bool include_entry,
      const uint32_t max_distance_computation_count) const override
  {
    return explore_func_(*this, entry_vertex_index, k, include_entry, max_distance_computation_count);
  }

  /**
   * Templated explore implementation. The result set contains internal indices.
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet exploreImpl(
      const uint32_t entry_vertex_index,
      const uint32_t k,
      const bool include_entry,
      const uint32_t max_distance_computation_count) const
  {
    uint32_t distance_computation_count = 0;
    const auto dist_func_param = feature_space_.get_dist_func_param();

    // set of checked vertex ids
    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k * edges_per_vertex_);

    // result set
    auto results = deglib::search::ResultSet();
    results.reserve(k);

    // add the entry vertex index to the vertices which gets checked next and ignore it for further checks
    checked_ids[entry_vertex_index] = checked_ids_tag;
    next_vertices.emplace(entry_vertex_index, 0);
    if (include_entry)
      results.emplace(entry_vertex_index, 0);
    const auto query = feature_by_index(entry_vertex_index);

    // search radius
    auto radius = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_vertices queue and max_calcs is not yet reached
    auto good_neighbors = std::array<uint32_t, 256>();  // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    alignas(32) auto db_arr = std::array<const void*, 256>();
    alignas(32) auto dists = std::array<float, 256>();
    while (!next_vertices.empty()) 
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      uint8_t good_neighbor_count = 0;
      const auto* neighbor_indices = neighbors_by_index(next_vertex.getInternalIndex());
      for (uint8_t i = 0; i < edges_per_vertex_; ++i) {
        const auto neighbor_index = neighbor_indices[i];
        if (checked_ids[neighbor_index] != checked_ids_tag) {
          checked_ids[neighbor_index] = checked_ids_tag;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      // Cap the neighbor count based on the remaining distance budget
      if (distance_computation_count + good_neighbor_count > max_distance_computation_count) {
        good_neighbor_count = max_distance_computation_count - distance_computation_count;
      }

      // Construct features pointer array
      for (size_t i = 0; i < good_neighbor_count; ++i) {
        db_arr[i] = feature_by_index(good_neighbors[i]);
        if(i < 8)
          memory::prefetch(db_arr[i]);
      }

      // Compute distances in batch
      deglib::distances::compare_batch<COMPARATOR>(query, db_arr.data(), good_neighbor_count, dist_func_param, dists.data());

      // Process results sequentially
      for (size_t i = 0; i < good_neighbor_count; ++i) {
        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_distance = dists[i];

        if (neighbor_distance < radius) {
          next_vertices.emplace(neighbor_index, neighbor_distance);
          results.emplace(neighbor_index, neighbor_distance);

          if (results.size() > k) {
            results.pop();
            radius = results.top().getDistance();
          }
        }
      }

      // early stop after to many computations
      distance_computation_count += good_neighbor_count;
      if (distance_computation_count >= max_distance_computation_count)
        return results;
    }

    return results;
  }
};

}  // namespace deglib::graph
