#pragma once

#include <array>
#include <cstdint> // for types like uint32_t
#include <limits>
#include <queue>
#include <math.h>
#include <filesystem>
#include <unordered_map>

#include "graph.h"
#include "repository.h"
#include "search.h"

namespace deglib::graph
{

/**
 * A size bounded undirected and weighted n-regular graph.
 * 
 * The vertex count and number of edges per vertices is bounded to a fixed value at 
 * construction time. The graph is therefore n-regular where n is the number of 
 * eddes per vertex.
 * 
 * Furthermode the graph is undirected, if there is connection from A to B than 
 * there musst be one from B to A. All connections are stored in the neighbor 
 * indices list of every vertex. The indices are based on the indices of their 
 * corresponding vertices. Each vertex has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of vertices. Where 
 * the external label can be any signed 32-bit integer. The indices in the 
 * neighbors list are ascending sorted.
 * 
 * Every edge contains of a neighbor vertex index and a weight. The weights and
 * neighbor indices are in separated list, but have the same order.
 * 
 * The number of vertices is limited to uint32.max
 */
class SizeBoundedGraph : public deglib::graph::MutableGraph {

  using SEARCHFUNC = deglib::search::ResultSet (*)(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count);

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Ext16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Ext8(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float8Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Ext4(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Ext16Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16ExtResiduals, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Ext4Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4ExtResiduals, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProduct(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProductExt16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProductExt8(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat8Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProductExt4(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4Ext, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProductExt16Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16ExtResiduals, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchInnerProductExt4Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4ExtResiduals, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Uint8(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Uint8, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Uint8Ext32(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Uint8Ext32, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static deglib::search::ResultSet searchL2Uint8Ext16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Uint8Ext16, use_max_distance_count, use_filter>(entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false, bool use_filter = false>
  inline static SEARCHFUNC getSearchFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext16<use_max_distance_count, use_filter>;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext8<use_max_distance_count, use_filter>;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext4<use_max_distance_count, use_filter>;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::searchL2Ext16Residual<use_max_distance_count, use_filter>;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::searchL2Ext4Residual<use_max_distance_count, use_filter>;
      else
        return deglib::graph::SizeBoundedGraph::searchL2<use_max_distance_count, use_filter>;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {
      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt16<use_max_distance_count, use_filter>;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt8<use_max_distance_count, use_filter>;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt4<use_max_distance_count, use_filter>;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt16Residual<use_max_distance_count, use_filter>;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt4Residual<use_max_distance_count, use_filter>;
      else
        return deglib::graph::SizeBoundedGraph::searchInnerProduct<use_max_distance_count, use_filter>;
    }
    else if(metric == deglib::Metric::L2_Uint8)
    {
      if (dim % 32 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Uint8Ext32<use_max_distance_count, use_filter>;
      else if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Uint8Ext16<use_max_distance_count, use_filter>;
      else
        return deglib::graph::SizeBoundedGraph::searchL2Uint8<use_max_distance_count, use_filter>;
    }

    std::fprintf(stderr, "Could not find metric %u for the sizebounded_graph search method \n", static_cast<int>(metric));
    std::perror("");
    std::abort();
  }


  using EXPLOREFUNC = deglib::search::ResultSet (*)(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count);

  inline static deglib::search::ResultSet exploreL2(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext8(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float8Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProduct(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt8(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat8Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Uint8(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Uint8>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Uint8Ext32(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Uint8Ext32>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Uint8Ext16(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Uint8Ext16>(entry_vertex_index, k, max_distance_computation_count);
  }


  inline static EXPLOREFUNC getExploreFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext16;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext8;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext4;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext16Residual;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext4Residual;
      else
        return deglib::graph::SizeBoundedGraph::exploreL2;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt16;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt8;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt4;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt16Residual;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt4Residual;
      else
        return deglib::graph::SizeBoundedGraph::exploreInnerProduct;
    }
    else if(metric == deglib::Metric::L2_Uint8)
    {

      if (dim % 32 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Uint8Ext32;
      else if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Uint8Ext16;
      else
        return deglib::graph::SizeBoundedGraph::exploreL2Uint8;
    }

    std::fprintf(stderr, "Could not find metric %u for the sizebounded_graph explore method \n", static_cast<int>(metric));
    std::perror("");
    std::abort();      
  }

  static uint32_t compute_aligned_byte_size_per_vertex(const uint8_t edges_per_vertex, const uint16_t feature_byte_size, const uint8_t alignment) {
    const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_vertex) * (sizeof(uint32_t) + sizeof(float)) + sizeof(uint32_t);
    if (alignment == 0)
      return byte_size;
    else {
      return ((byte_size + alignment - 1) / alignment) * alignment;
    }
  }

  static std::byte* compute_aligned_pointer(const std::unique_ptr<std::byte[]>& arr, const uint8_t alignment) {
    if (alignment == 0)
      return arr.get();
    else {
      void* ptr = arr.get();
      size_t space = std::numeric_limits<size_t>::max();
      std::align(alignment, 0, ptr, space);
      return static_cast<std::byte*>(ptr);
    }
  }

  // alignment of vertex information in bytes (all feature vectors will be 256bit aligned for faster SIMD processing)
  static const uint8_t object_alignment = 32; // deglib::memory::L1_CACHE_LINE_SIZE; // 32; // no effect on modern hardware

  const uint32_t max_vertex_count_;
  const uint8_t edges_per_vertex_;
  const uint16_t feature_byte_size_;

  const uint32_t byte_size_per_vertex_;
  const uint32_t neighbor_indices_offset_;
  const uint32_t neighbor_weights_offset_;
  const uint32_t external_label_offset_;

  // list of vertices (vertex: std::byte* feature vector, uint32_t* indices of neighbor vertices, float* weights of neighbor vertices, uint32_t external label)      
  std::unique_ptr<std::byte[]> vertices_;
  std::byte* vertices_memory_;

  // map from the label of a vertex to the internal vertex index
  std::unordered_map<uint32_t, uint32_t> label_to_index_;

  // internal search function with embedded distances function
  const SEARCHFUNC search_func_;
  const EXPLOREFUNC explore_func_;

  // distance calculation function between feature vectors of two graph vertices
  const deglib::FloatSpace feature_space_;

  std::unique_ptr<VisitedListPool> visited_list_pool_;

 public:
  SizeBoundedGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::FloatSpace feature_space)
      : max_vertex_count_(max_vertex_count),
        edges_per_vertex_(edges_per_vertex), 
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 

        byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)),
        neighbor_indices_offset_(uint32_t(feature_space.get_data_size())),
        neighbor_weights_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)),
        external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_vertex) * sizeof(float)), 

        vertices_(std::make_unique<std::byte[]>(size_t(max_vertex_count) * byte_size_per_vertex_ + object_alignment)), 
        vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)),

        search_func_(getSearchFunction(feature_space)), 
        explore_func_(getExploreFunction(feature_space)),
        feature_space_(feature_space),
        visited_list_pool_( std::make_unique<VisitedListPool>(1, max_vertex_count)) { 

    label_to_index_.reserve(max_vertex_count);  
  }
  //     : edges_per_vertex_(edges_per_vertex), 
  //       max_vertex_count_(max_vertex_count), 
  //       feature_space_(feature_space),
  //       search_func_(getSearchFunction(feature_space)), explore_func_(getExploreFunction(feature_space)),
  //       feature_byte_size_(uint16_t(feature_space.get_data_size())), 
  //       byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)), 
  //       neighbor_indices_offset_(uint32_t(feature_space.get_data_size())),
  //       neighbor_weights_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)),
  //       external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_vertex) * sizeof(float)), 
  //       vertices_(std::make_unique<std::byte[]>(size_t(max_vertex_count) * byte_size_per_vertex_ + object_alignment)), 
  //       vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)), 
  //       label_to_index_(max_vertex_count) {
  // }

  /**
   *  Load from file
   */
  SizeBoundedGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::FloatSpace feature_space, std::ifstream& ifstream, const uint32_t size)
      : SizeBoundedGraph(max_vertex_count, edges_per_vertex, std::move(feature_space)) {

    // copy the old data over
    uint32_t file_byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++) {
      ifstream.read(reinterpret_cast<char*>(this->vertex_by_index(i)), file_byte_size_per_vertex);
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }
  }

  /**
   * Current maximal capacity of vertices
   */ 
  const auto capacity() const {
    return this->max_vertex_count_;
  }

  /**
   * Number of vertices in the graph
   */
  const uint32_t size() const override {
    return (uint32_t) this->label_to_index_.size();
  }

  /**
   * Number of edges per vertex 
   */
  const uint8_t getEdgesPerVertex() const override {
    return this->edges_per_vertex_;
  }

  const deglib::SpaceInterface<float>& getFeatureSpace() const override {
    return this->feature_space_;
  }

private:  
  inline std::byte* vertex_by_index(const uint32_t internal_idx) const {
    return vertices_memory_ + size_t(internal_idx) * byte_size_per_vertex_;
  }

  inline const uint32_t label_by_index(const uint32_t internal_idx) const {
    return *reinterpret_cast<const int32_t*>(vertex_by_index(internal_idx) + external_label_offset_);
  }

  inline const std::byte* feature_by_index(const uint32_t internal_idx) const{
    return vertex_by_index(internal_idx);
  }

  inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(vertex_by_index(internal_idx) + neighbor_indices_offset_);
  }

  inline const float* weights_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<const float*>(vertex_by_index(internal_idx) + neighbor_weights_offset_);
  }

public:

  /**
   * convert an external label to an internal index
   */ 
  inline const uint32_t getInternalIndex(const uint32_t external_label) const override {
    return label_to_index_.find(external_label)->second;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override {
    return label_by_index(internal_idx);
  }

  inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override{
    return feature_by_index(internal_idx);
  }

  inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override {
    return neighbors_by_index(internal_idx);
  }

  inline const float* getNeighborWeights(const uint32_t internal_idx) const override {
    return weights_by_index(internal_idx);
  }

  inline const float getEdgeWeight(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    auto neighbor_indices = neighbors_by_index(internal_index);
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;  
    auto neighbor_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, neighbor_index); 
    if(*neighbor_ptr == neighbor_index) {
      auto weight_index = neighbor_ptr - neighbor_indices;
      return weights_by_index(internal_index)[weight_index];
    }
    return -1;
  }

  inline const bool hasVertex(const uint32_t external_label) const override {
    return label_to_index_.contains(external_label);
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    auto neighbor_indices = neighbors_by_index(internal_index);
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;  
    return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index);
  }

  const bool saveGraph(const char* path_to_graph) const override {
    
    // create parent dir
    std::filesystem::create_directories(std::filesystem::path(path_to_graph).parent_path());

    // check open file for write
    auto out = std::ofstream(path_to_graph, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
      std::fprintf(stderr, "Error in open file %s\n", path_to_graph);
      return false;
    }

    // store feature space information
    uint8_t metric_type = static_cast<uint8_t>(feature_space_.metric());
    out.write(reinterpret_cast<const char*>(&metric_type), sizeof(metric_type));
    uint16_t dim = uint16_t(this->feature_space_.dim());
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    // store graph information
    uint32_t size = uint32_t(this->size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(&this->edges_per_vertex_), sizeof(this->edges_per_vertex_));

    // store the existing vertices
    uint32_t byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++)
      out.write(reinterpret_cast<const char*>(this->vertex_by_index(i)), byte_size_per_vertex);    
    out.close();

    return true;
  }

  /**
   * Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.
   * 
   * @return the internal index of the new vertex
   */
  uint32_t addVertex(const uint32_t external_label, const std::byte* feature_vector) override {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    auto vertex_memory = vertex_by_index(new_internal_index);
    std::memcpy(vertex_memory, feature_vector, feature_byte_size_);
    std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), edges_per_vertex_, new_internal_index); // temporary self loop
    std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), edges_per_vertex_, float(0)); // 0 weight
    std::memcpy(vertex_memory + external_label_offset_, &external_label, sizeof(uint32_t));

    return new_internal_index;
  }

  /**
   * Remove an existing vertex.
   */
  std::vector<uint32_t> removeVertex(const uint32_t external_label) override {
    const auto internal_index = getInternalIndex(external_label);
    const auto last_internal_index = static_cast<uint32_t>(this->label_to_index_.size() - 1);

    // since the last_internal_index will be moved to the internal_index, 
    // update the current neighbor list if the last_internal_index is present
    if(hasEdge(internal_index, last_internal_index)) {
      changeEdge(internal_index, last_internal_index, internal_index, 0);
      changeEdge(last_internal_index, internal_index, last_internal_index, 0);
    }

    // copy the neighbor list to return it later
    const auto neighbor_indices = neighbors_by_index(internal_index);
    const auto involved_indices = std::vector<uint32_t>(neighbor_indices, neighbor_indices + this->edges_per_vertex_);

    // replace all references to the internal_index with a self-reference of the corresponding vertex
    for (size_t index = 0; index < this->edges_per_vertex_; index++) 
      changeEdge(neighbor_indices[index], internal_index, neighbor_indices[index], 0);

    // the last index will be moved to the internal_index position and overwrite its content
    if(internal_index != last_internal_index) {

      // update the neighbor list of the last vertex to reflex its new vertex index
      const auto last_neighbor_indices = neighbors_by_index(last_internal_index);
      const auto last_neighbor_weights = weights_by_index(last_internal_index);
      for (size_t index = 0; index < this->edges_per_vertex_; index++) 
        changeEdge(last_neighbor_indices[index], last_internal_index, internal_index, last_neighbor_weights[index]);
      
      // copy the last vertex to the vertex which gets removed
      std::memcpy(vertex_by_index(internal_index), vertex_by_index(last_internal_index), this->byte_size_per_vertex_);

      // update the index position of the last label
      const auto last_label = label_by_index(last_internal_index);
      label_to_index_[last_label] = internal_index;
    }

    // remove the external label from the hash map
    label_to_index_.erase(external_label);

    // return all neighbors of the deleted vertex
    return involved_indices;
  }

  /**
   * Swap a neighbor with another neighbor and its weight.
   * 
   * @param internal_index vertex index which neighbors should be changed
   * @param replace_index neighbor index to remove
   * @param new_index neighbor index to add
   * @param new_weight weight of the neighbor to add
   * @return true if the from_neighbor_index was found and changed
   */
  bool changeEdge(const uint32_t internal_index, const uint32_t replace_index, const uint32_t new_index, const float new_weight) override {
    auto vertex_memory = vertex_by_index(internal_index);

    // Find the position of the first index to be replaced
    auto neighbor_indices = reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_);    // list of neighbor indizizes
    auto neighbor_indices_end = neighbor_indices + edges_per_vertex_;                                 // end of the list
    uint32_t* replace_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, replace_index);
    size_t replace_idx = replace_pos - neighbor_indices;

    // Check if the replace index is found
    if (replace_pos == neighbor_indices_end || *replace_pos != replace_index) {
        std::cerr << "changeEdge: vertex " << internal_index << " does not have an edge to " << replace_index << " and therefore can not be swapped with " << new_index << " and distance " << new_weight << std::endl;
        return false;
    }

    // Find the position where the new index should be inserted
    uint32_t* insert_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, new_index);
    size_t insert_idx = insert_pos - neighbor_indices;

    // Handle the case where the insertion position is after the removal position
    auto neighbor_weights = reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_);         // list of neighbor weights
    if (insert_idx > replace_idx) {
        // Shift elements left from replace_idx to insert_idx - 1
        std::memmove(neighbor_indices + replace_idx, neighbor_indices + replace_idx + 1, (insert_idx - replace_idx - 1) * sizeof(uint32_t));
        std::memmove(neighbor_weights + replace_idx, neighbor_weights + replace_idx + 1, (insert_idx - replace_idx - 1) * sizeof(float));
        --insert_idx;
    } else if (insert_idx < replace_idx) {
        // Shift elements right from insert_idx to replace_idx
        std::memmove(neighbor_indices + insert_idx + 1, neighbor_indices + insert_idx, (replace_idx - insert_idx) * sizeof(uint32_t));
        std::memmove(neighbor_weights + insert_idx + 1, neighbor_weights + insert_idx, (replace_idx - insert_idx) * sizeof(float));
    }

    // Insert the new index and weight at the correct position
    neighbor_indices[insert_idx] = new_index;
    neighbor_weights[insert_idx] = new_weight;

    return true;
  }

  /**
   * Change all edges of a vertex.
   * The neighbor indices und weights will be copied.
   * The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
   * The indices in the neighbor_indices array must be sorted.
   */
  void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) override {
    auto vertex_memory = vertex_by_index(internal_index);
    std::memcpy(vertex_memory + neighbor_indices_offset_, neighbor_indices, uint32_t(edges_per_vertex_) * sizeof(uint32_t));
    std::memcpy(vertex_memory + neighbor_weights_offset_, neighbor_weights, uint32_t(edges_per_vertex_) * sizeof(float));
  }

  /**
   * Performan a search but stops when the to_vertex was found.
   */
  std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const override
  {
    const auto query = this->feature_by_index(to_vertex);
    const auto dist_func = this->feature_space_.get_dist_func();
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    const auto feature_size = this->feature_space_.get_data_size();

    // set of checked vertex ids
    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();

    // trackable information 
    auto trackback = std::unordered_map<uint32_t, deglib::search::ObjectDistance>();

    // result set
    auto results = deglib::search::ResultSet();   

    // copy the initial entry vertices and their distances to the query into the three containers
    for (auto&& index : entry_vertex_indices) {
      if(checked_ids[index] != checked_ids_tag) {
        checked_ids[index] = checked_ids_tag;

        const auto feature = this->feature_by_index(index);
        const auto distance = dist_func(query, feature, dist_func_param);
        results.emplace(index, distance);
        next_vertices.emplace(index, distance);
        trackback.emplace(index, deglib::search::ObjectDistance(index, distance));
      }
    }

    // search radius
    auto radius = std::numeric_limits<float>::max();
    auto exploration_radius = radius;

    // iterate as long as good elements are in the next_vertices queue     
    auto good_neighbors = std::array<uint32_t, 256>();
    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // max distance reached
      if (next_vertex.getDistance() > exploration_radius) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_vertex_; i++) {
        const auto neighbor_index = neighbor_indices[i];

        // found our target vertex, create a path back to the entry vertex
        if(neighbor_index == to_vertex) {
          auto path = std::vector<deglib::search::ObjectDistance>();
          path.emplace_back(to_vertex, 0.f);
          path.emplace_back(next_vertex.getInternalIndex(), next_vertex.getDistance());

          auto last_vertex = trackback.find(next_vertex.getInternalIndex());
          while(last_vertex != trackback.cend() && last_vertex->first != last_vertex->second.getInternalIndex()) {
            path.emplace_back(last_vertex->second.getInternalIndex(), last_vertex->second.getDistance());
            last_vertex = trackback.find(last_vertex->second.getInternalIndex());
          }

          return path;
        }

        // collect 
        if(checked_ids[neighbor_index] != checked_ids_tag) {
          checked_ids[neighbor_index] = checked_ids_tag;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])), feature_size);
      for (size_t i = 0; i < good_neighbor_count; i++) {
        memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])), feature_size);

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = dist_func(query, neighbor_feature_vector, dist_func_param);
             
        // check the neighborhood of this vertex later, if its good enough
        if (neighbor_distance <= exploration_radius) {
          next_vertices.emplace(neighbor_index, neighbor_distance);
          trackback.insert({neighbor_index, deglib::search::ObjectDistance(next_vertex.getInternalIndex(), next_vertex.getDistance())});

          // remember the vertex, if its better than the worst in the result list
          if (neighbor_distance < radius) {
            results.emplace(neighbor_index, neighbor_distance);

            // update the search radius
            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
              exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
            }
          }
        }
      }
    }

    // there is no path
    return std::vector<deglib::search::ObjectDistance>();
  }

  /**
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet search(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) const override
  {
    if(filter) {
      if(max_distance_computation_count == 0) {
        const auto limited_search_func = getSearchFunction<false, true>(this->feature_space_);
        return limited_search_func(*this, entry_vertex_indices, query, eps, k, filter, 0);
      } else {
        const auto limited_search_func = getSearchFunction<true, true>(this->feature_space_);
        return limited_search_func(*this, entry_vertex_indices, query, eps, k, filter, max_distance_computation_count);
      }
    } else {
      if(max_distance_computation_count == 0) {
        return search_func_(*this, entry_vertex_indices, query, eps, k, nullptr, 0);
      } else {
        const auto limited_search_func = getSearchFunction<true, false>(this->feature_space_);
        return limited_search_func(*this, entry_vertex_indices, query, eps, k, nullptr, max_distance_computation_count);
      }
    }
  }

  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR, bool use_max_distance_count, bool use_filter>
  deglib::search::ResultSet searchImpl(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t initial_k, const deglib::graph::Filter* filter, const uint32_t max_distance_computation_count) const
  {
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    const auto feature_size = this->feature_space_.get_data_size();
    uint32_t distance_computation_count = 0;
    const size_t vertex_count = this->size();
    size_t k = std::min(vertex_count, static_cast<size_t>(initial_k));

    // set of checked vertex ids
    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k*this->edges_per_vertex_);

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();   
    results.reserve(k+1);

    // if the filter only contains few valid ids brute force them all
    if constexpr (use_filter) {
      if (vertex_count < 1'000 || (filter->get_inclusion_rate() * vertex_count) < 10'000 || filter->get_inclusion_rate() < 0.10f) {
        auto radius = std::numeric_limits<float>::max();
        filter->for_each_valid_label([&](uint32_t valid_label) {
          auto valid_index = this->getInternalIndex(valid_label);
          const auto feature = reinterpret_cast<const float*>(this->feature_by_index(valid_index));
          const auto distance = COMPARATOR::compare(query, feature, dist_func_param);

          // remember the vertex, if its better than the worst in the result list
          if (distance < radius) {
            results.emplace(valid_index, distance);

            // update the search radius
            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
            }
          }
        });
        return results;
      }
    }

    // copy the initial entry vertices and their distances to the query into the three containers
    for (auto&& index : entry_vertex_indices) {
      if(checked_ids[index] != checked_ids_tag) {
        checked_ids[index] = checked_ids_tag;

        const auto feature = this->feature_by_index(index);
        const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
        next_vertices.emplace(index, distance);
        if constexpr (use_filter) {
          if(filter->is_valid(this->label_by_index(index)))  {
            results.emplace(index, distance);
          }
        } else {
          results.emplace(index, distance);
        }

        // early stop after to many computations
        if constexpr (use_max_distance_count) {
          if(distance_computation_count++ >= max_distance_computation_count)
            return results;
        }
      }
    }

    // search radius
    auto radius = std::numeric_limits<float>::max();
    auto exploration_radius = radius;

    // iterate as long as good elements are in the next_vertices queue
    auto good_neighbors = std::array<uint32_t, 256>(); 
    while (next_vertices.empty() == false)
    {

      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // max distance reached
      if (next_vertex.getDistance() > exploration_radius) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_vertex_; i++) {
        const auto neighbor_index = neighbor_indices[i];
        if(checked_ids[neighbor_index] != checked_ids_tag) {
          checked_ids[neighbor_index] = checked_ids_tag;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])), feature_size);
      for (size_t i = 0; i < good_neighbor_count; i++) {
        memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])), feature_size);

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);

             
        // check the neighborhood of this vertex later, if its good enough
        if (neighbor_distance <= exploration_radius) {
            next_vertices.emplace(neighbor_index, neighbor_distance);

          // remember the vertex, if its better than the worst in the result list
          if (neighbor_distance < radius) {
            if constexpr (use_filter) {
              if(filter->is_valid(this->label_by_index(neighbor_index)))  {
                results.emplace(neighbor_index, neighbor_distance);
              }
            } else {
              results.emplace(neighbor_index, neighbor_distance);
            }

            // update the search radius
            if (results.size() > k) {
              results.pop();
              radius = results.top().getDistance();
              exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
            }
          }
        }
      }

      // early stop after to many computations
      if constexpr (use_max_distance_count) {
        if(distance_computation_count++ >= max_distance_computation_count)
          return results;
      }
    }

    return results;
  }

  /**
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet explore(const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count) const override
  {
    return explore_func_(*this, entry_vertex_index, k, max_distance_computation_count);
  }

  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet exploreImpl(const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    uint32_t distance_computation_count = 0;
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    const auto feature_size = this->feature_space_.get_data_size();

    // set of checked vertex ids
    const auto vl = visited_list_pool_->getFreeVisitedList();
    auto* checked_ids = vl->get_visited();
    const auto checked_ids_tag = vl->get_tag();

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k*this->edges_per_vertex_);

    // result set
    auto results = deglib::search::ResultSet();   
    results.reserve(k);

    // add the entry vertex index to the vertices which gets checked next and ignore it for further checks
    checked_ids[entry_vertex_index] = checked_ids_tag;
    next_vertices.emplace(entry_vertex_index, 0);
    const auto query = this->feature_by_index(entry_vertex_index);

    // search radius
    auto radius = std::numeric_limits<float>::max();

    // experimental: eps replacement parameter
    const auto eps = std::log10(float(max_distance_computation_count)/k);
    auto exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));

    // iterate as long as good elements are in the next_vertices queue and max_calcs is not yet reached
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // if no weight of this neighbor would survive the distance estimation check, stop here
      if (next_vertex.getDistance() > exploration_radius)
        break;

      uint8_t good_neighbor_count = 0;
      {
        const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
        const auto neighbor_weights = this->weights_by_index(next_vertex.getInternalIndex());
        memory::prefetch(reinterpret_cast<const char*>(neighbor_indices));
        memory::prefetch(reinterpret_cast<const char*>(neighbor_weights));
        for (uint8_t i = 0; i < this->edges_per_vertex_; i++) {
          const auto neighbor_index = neighbor_indices[i];

          if (checked_ids[neighbor_index] != checked_ids_tag)  {
            checked_ids[neighbor_index] = checked_ids_tag;

            // distance estimation check: allow only edges with a worst case distance < r
            // this produces slighly better results and brings the sizebound graph on par with the readonly graph when comparing speed vs quality
            if(next_vertex.getDistance() + neighbor_weights[i] < exploration_radius)
              good_neighbors[good_neighbor_count++] = neighbor_index;
          }
        }
      }

      if (good_neighbor_count == 0)
        continue;

      memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])), feature_size);
      for (uint8_t i = 0; i < good_neighbor_count; i++) {
        memory::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])), feature_size);

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);

        if (neighbor_distance < radius) {

          // check the neighborhood of this vertex later
          next_vertices.emplace(neighbor_index, neighbor_distance);

          // remember the vertex, if its better than the worst in the result list
          results.emplace(neighbor_index, neighbor_distance);

          // update the search radius
          if (results.size() > k) {
            results.pop();
            radius = results.top().getDistance();
            exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
          }
        }

        // early stop after to many computations
        if(distance_computation_count++ >= max_distance_computation_count)
          return results;
      }
    }

    return results;
  }  
};

/**
 * Load the graph
 */
auto load_sizebounded_graph(const char* path_graph, uint32_t new_max_size = 0)
{
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(path_graph, ec);
  if (ec != std::error_code{})
  {
    std::fprintf(stderr, "error when accessing test file, size is: %ju message: %s \n", file_size, ec.message().c_str());
    perror("");
    abort();
  }

  auto ifstream = std::ifstream(path_graph, std::ios::binary);
  if (!ifstream.is_open())
  {
    std::fprintf(stderr, "could not open %s\n", path_graph);
    perror("");
    abort();
  }

  // create feature space
  uint8_t metric_type;
  ifstream.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
  uint16_t dim;
  ifstream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  const auto feature_space = deglib::FloatSpace(dim, static_cast<deglib::Metric>(metric_type));

  // create the graph
  uint32_t size;
  ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
  uint8_t edges_per_vertex;
  ifstream.read(reinterpret_cast<char*>(&edges_per_vertex), sizeof(edges_per_vertex));

  // if no new max size is set use the size of the graph from disk
  if(new_max_size == 0) 
    new_max_size = size;
  
  // if there is a max size is should be higher than the needed graph size from disk
  if(new_max_size < size) {
    std::fprintf(stderr, "The graph in the %s file has %u vertices but the new max size is %u\n", path_graph, size, new_max_size);
    perror("");
    abort();
  }

  auto graph = deglib::graph::SizeBoundedGraph(new_max_size, edges_per_vertex, std::move(feature_space), ifstream, size);
  ifstream.close();

  return graph;
}

}  // namespace deglib::graph