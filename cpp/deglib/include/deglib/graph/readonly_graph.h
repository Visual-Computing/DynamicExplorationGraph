#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <math.h>
#include <unordered_map>
#include <array>
#include <unordered_set>
#include <random>
#include <fstream>
#include <filesystem>

#include "deglib/utils/memory.h"
#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/visited_list_pool.h"

namespace deglib::graph
{

/**
 * A immutable simple undirected n-regular graph. This version is prefered 
 * to use for existing graphs where only the search performance is important.
 * 
 * The vertex count and number of edges per vertices is known at construction time.
 * While the content of a vertex can be mutated after construction, it is not 
 * recommended. See SizeBoundedGraphs for a mutable version or understand the 
 * inner workings of the search function and memory layout, to make safe changes. 
 * The graph is n-regular where n is the number of eddes per vertex.
 * 
 * Furthermode the graph is undirected, if there is connection from A to B than 
 * there musst be one from B to A. All connections are stored in the neighbor 
 * indices list of every vertex. The indices are based on the indices of their 
 * corresponding vertices. Each vertex has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of vertices. Where 
 * the external label can be any signed 32-bit integer.
 * 
 * The number of vertices is limited to uint32.max
 */
class ReadOnlyGraph : public deglib::graph::InternalGraph {
  friend class deglib::graph::InternalGraph;

  static uint32_t compute_aligned_byte_size_per_vertex(const uint8_t edges_per_vertex, const uint16_t feature_byte_size, const uint8_t alignment) {
      const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_vertex) * sizeof(uint32_t) + sizeof(uint32_t);
    if (alignment == 0)
      return  byte_size;
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
  const uint32_t external_label_offset_;

  // list of vertices (vertex: feature vector, indices of neighbor vertices, external label)
  std::unique_ptr<std::byte[]> vertices_;
  std::byte* vertices_memory_;

  // map from the label of a vertex to the internal vertex index
  std::unordered_map<uint32_t, uint32_t> label_to_index_;

  // distance calculation function between feature vectors of two graph vertices
  const deglib::distances::FloatSpace feature_space_;

  std::unique_ptr<VisitedListPool> visited_list_pool_;

public:
  ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space)
      : max_vertex_count_(max_vertex_count),
        edges_per_vertex_(edges_per_vertex), 
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 

        byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)), 
        neighbor_indices_offset_(uint32_t(feature_space.get_data_size())), 
        external_label_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)), 

        vertices_(std::make_unique<std::byte[]>(size_t(max_vertex_count) * byte_size_per_vertex_ + object_alignment)), 
        vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)),

        feature_space_(feature_space),
        visited_list_pool_( std::make_unique<VisitedListPool>(1, max_vertex_count)) { 

    if (edges_per_vertex % 2 != 0) 
      throw std::invalid_argument("edges_per_vertex must be even.");

    label_to_index_.reserve(max_vertex_count);
  }

  /**
   *  Load from file
   */
  ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space, std::ifstream& ifstream)
      : ReadOnlyGraph(max_vertex_count, edges_per_vertex, feature_space) {

    // copy the old data over
    uint32_t vertex_without_external = uint32_t(feature_space.get_data_size()) + uint32_t(edges_per_vertex) * sizeof(uint32_t);
    for (uint32_t i = 0; i < max_vertex_count; i++) {
      auto vertex = reinterpret_cast<char*>(this->vertex_by_index(i));
      ifstream.read(vertex, vertex_without_external);                     // read the feature vector and neighbor indices
      ifstream.ignore(uint32_t(edges_per_vertex) * sizeof(float));        // skip the weights
      ifstream.read(vertex + vertex_without_external, sizeof(uint32_t));  // read the external label
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }
  }

  /**
   *  Copy from input graph (with optional custom features buffer)
   */
  ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space, const deglib::graph::InternalGraph& input_graph, const void* custom_features = nullptr)
      : ReadOnlyGraph(max_vertex_count, edges_per_vertex, feature_space) {

    const auto custom_feature_bytes = reinterpret_cast<const std::byte*>(custom_features);

    for (uint32_t i = 0; i < max_vertex_count; i++) {
        auto vertex = reinterpret_cast<char*>(this->vertex_by_index(i));
        const auto label = input_graph.getExternalLabel(i);

        if (custom_features != nullptr) {
            const auto feature = custom_feature_bytes + size_t(label) * feature_space.get_data_size();
            std::memcpy(vertex, feature, feature_space.get_data_size());
        } else {
            const auto feature = input_graph.getFeatureVector(i);
            std::memcpy(vertex, feature, feature_space.get_data_size());
        }
        vertex += feature_space.get_data_size();

        const auto neighbor_indices = input_graph.getNeighborIndices(i);
        std::memcpy(vertex, neighbor_indices, sizeof(uint32_t) * uint32_t(edges_per_vertex));
        vertex += sizeof(uint32_t) * uint32_t(edges_per_vertex);

        std::memcpy(vertex, &label, sizeof(uint32_t));
        label_to_index_.emplace(label, i);
    }
  }

  /**
   * Current maximal capacity of vertices
   */ 
  const auto capacity() const {
    return max_vertex_count_;
  }

  /**
   * Number of vertices in the graph
   */
  const uint32_t size() const override {
    return (uint32_t) label_to_index_.size();
  }

  /**
   * Number of edges per vertex 
   */
  const uint8_t getEdgesPerVertex() const override {
    return edges_per_vertex_;
  }

  const deglib::distances::FloatSpace& getFeatureSpace() const override {
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

  inline const bool hasVertex(const uint32_t external_label) const override {
    return label_to_index_.contains(external_label);
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    auto neighbor_indices = getNeighborIndices(internal_index);
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;  
    return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index); 
  }

  /**
   * Perform a search but stops when the to_vertex was found.
   */
  std::vector<deglib::graph::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const override
  {
    return hasPathImpl(*this, entry_vertex_indices, to_vertex, eps, k);
  }

protected:
  deglib::graph::ResultSet search_intern(
      const std::vector<uint32_t>& entry_vertex_indices,
      const std::byte* query,
      const uint32_t k,
      const float eps = 0.0f,
      const bool include_entry = true,
      const deglib::search::Filter* filter = nullptr,
      const uint32_t max_distance_computation_count = 0) const override
  {
    return searchInternImpl(*this, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count);
  }
};


/**
 * Load the graph
 */
inline auto load_readonly_graph(const char* path_graph)
{
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(path_graph, ec);
  if (ec != std::error_code{})
  {
    std::fprintf(stderr, "error when accessing graph file %s, size is: %ju message: %s \n", path_graph, file_size, ec.message().c_str());
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
  const auto feature_space = deglib::distances::FloatSpace(dim, static_cast<deglib::distances::Metric>(metric_type));
  
  // create the graph
  uint32_t size;
  ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
  uint8_t edges_per_vertex;
  ifstream.read(reinterpret_cast<char*>(&edges_per_vertex), sizeof(edges_per_vertex));

  auto graph = deglib::graph::ReadOnlyGraph(size, edges_per_vertex, std::move(feature_space), ifstream);
  ifstream.close();

  return graph;
}

/**
 * Convert the given graph to a readonly graph
 */
inline auto convert_to_readonly_graph(const deglib::graph::InternalGraph& input_graph)
{
  auto size = input_graph.size();
  auto edges_per_vertex = input_graph.getEdgesPerVertex();
  return deglib::graph::ReadOnlyGraph(size, edges_per_vertex, input_graph.getFeatureSpace(), input_graph);
}

/**
 * Convert the given graph to a readonly graph, overriding feature space and feature vectors.
 */
inline auto convert_to_readonly_graph(const deglib::graph::InternalGraph& input_graph, const deglib::distances::FloatSpace feature_space, const void* custom_features = nullptr)
{
  auto size = input_graph.size();
  auto edges_per_vertex = input_graph.getEdgesPerVertex();
  return deglib::graph::ReadOnlyGraph(size, edges_per_vertex, feature_space, input_graph, custom_features);
}
}  // namespace deglib::graph



