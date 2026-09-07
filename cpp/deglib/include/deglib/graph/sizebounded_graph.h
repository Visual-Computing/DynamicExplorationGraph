#pragma once

#include "deglib/graph/mutable_graph.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/search.h"
#include "deglib/utils/memory.h"
#include "deglib/utils/random.h"

#include <math.h>

#include <algorithm>
#include <array>
#include <cstdint>  // for types like uint32_t
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace deglib::graph {

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
    friend class deglib::graph::InternalGraph;

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
    static const uint8_t object_alignment = 32;  // deglib::memory::L1_CACHE_LINE_SIZE; // 32; // no effect on modern hardware

    static constexpr uint32_t INVALID_LABEL = std::numeric_limits<uint32_t>::max();

    const uint32_t max_vertex_count_;
    const uint8_t edges_per_vertex_;
    const uint16_t feature_byte_size_;

    const uint32_t byte_size_per_vertex_;
    const uint32_t neighbor_indices_offset_;
    const uint32_t neighbor_weights_offset_;
    const uint32_t external_label_offset_;

    // list of vertices (vertex: std::byte* feature vector, uint32_t* indices of neighbor vertices, float* weights of neighbor vertices, uint32_t external
    // label)
    std::unique_ptr<std::byte[]> vertices_;
    std::byte* vertices_memory_;

    // map from the label of a vertex to the internal vertex index
    std::unordered_map<uint32_t, uint32_t> label_to_index_;

    // distance calculation function between feature vectors of two graph vertices
    const deglib::distances::FloatSpace feature_space_;

    std::unique_ptr<VisitedListPool> visited_list_pool_;

  public:
    SizeBoundedGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space)
        : max_vertex_count_(max_vertex_count),
          edges_per_vertex_(edges_per_vertex),
          feature_byte_size_(uint16_t(feature_space.get_data_size())),

          byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)),
          neighbor_indices_offset_(uint32_t(feature_space.get_data_size())),
          neighbor_weights_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)),
          external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_vertex) * sizeof(float)),

          vertices_(std::make_unique<std::byte[]>(size_t(max_vertex_count) * byte_size_per_vertex_ + object_alignment)),
          vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)),

          feature_space_(feature_space),
          visited_list_pool_(std::make_unique<VisitedListPool>(1, max_vertex_count)) {
        if (edges_per_vertex % 2 != 0) throw std::invalid_argument("edges_per_vertex must be even.");

        label_to_index_.reserve(max_vertex_count);

        // Mark all vertex slots with INVALID_LABEL initially
        for (uint32_t i = 0; i < max_vertex_count; ++i) {
            uint32_t inv = INVALID_LABEL;
            std::memcpy(vertex_by_index(i) + external_label_offset_, &inv, sizeof(uint32_t));
        }
    }

    /**
     *  Load from file
     */
    SizeBoundedGraph(
        const uint32_t max_vertex_count,
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace feature_space,
        std::ifstream& ifstream,
        const uint32_t size
    )
        : SizeBoundedGraph(max_vertex_count, edges_per_vertex, std::move(feature_space)) {
        // copy the data over
        uint32_t file_byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
        for (uint32_t i = 0; i < size; i++) {
            ifstream.read(reinterpret_cast<char*>(this->vertex_by_index(i)), file_byte_size_per_vertex);
            const uint32_t lbl = this->getExternalLabel(i);
            if (lbl != INVALID_LABEL) {
                label_to_index_.emplace(lbl, i);
            }
        }
    }

    /**
     * Copy from an existing InternalGraph, optionally replacing the feature space
     * and feature vectors, and recalculating all edge weights with the new metric.
     * If the input graph is a MutableGraph with the same feature space and no custom
     * features are provided, edge weights are copied directly without recalculation.
     *
     * @param input_graph The source graph to copy topology, labels, and (optionally) features from.
     * @param feature_space The feature space for the new graph (may differ from the input).
     * @param custom_features Optional pointer to replacement feature vectors. If non-null,
     *                        features are read from this buffer indexed by external label.
     *                        If null, features are copied from input_graph.
     * @param new_max_size If > 0, sets the capacity to max(new_max_size, input_graph.size()).
     *                     If 0, capacity equals input_graph.size().
     */
    SizeBoundedGraph(
        const deglib::graph::InternalGraph& input_graph,
        const deglib::distances::FloatSpace feature_space,
        const void* custom_features = nullptr,
        const uint32_t new_max_size = 0
    )
        : SizeBoundedGraph(
              (new_max_size > 0 ? std::max(new_max_size, input_graph.size()) : input_graph.size()),
              input_graph.getEdgesPerVertex(),
              std::move(feature_space)
          ) {
        const auto custom_feature_bytes = reinterpret_cast<const std::byte*>(custom_features);
        const auto dist_func = this->feature_space_.get_dist_func();
        const auto dist_func_param = this->feature_space_.get_dist_func_param();
        const auto feature_data_size = this->feature_space_.get_data_size();
        const auto edges_per_vertex = this->edges_per_vertex_;

        const uint32_t graph_size = input_graph.size();

        // Check if we can directly copy neighbor weights from a mutable graph
        const auto* src_mutable = dynamic_cast<const deglib::graph::MutableGraph*>(&input_graph);
        const bool can_direct_copy_weights = (src_mutable != nullptr) && (custom_features == nullptr) &&
                                             (this->feature_space_.metric() == input_graph.getFeatureSpace().metric()) &&
                                             (this->feature_space_.dim() == input_graph.getFeatureSpace().dim());

        for (uint32_t i = 0; i < graph_size; i++) {
            const auto label = input_graph.getExternalLabel(i);
            auto vertex_memory = vertex_by_index(i);

            // Copy feature vector
            if (custom_features != nullptr) {
                const auto feature = custom_feature_bytes + size_t(label) * feature_data_size;
                std::memcpy(vertex_memory, feature, feature_data_size);
            } else {
                const auto feature = input_graph.getFeatureVector(i);
                std::memcpy(vertex_memory, feature, feature_data_size);
            }

            // Copy neighbor indices
            const auto input_neighbor_indices = input_graph.getNeighborIndices(i);
            auto neighbor_indices = reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_);
            std::memcpy(neighbor_indices, input_neighbor_indices, sizeof(uint32_t) * edges_per_vertex);

            // Copy or recalculate edge weights
            auto neighbor_weights = reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_);
            if (can_direct_copy_weights) {
                std::memcpy(neighbor_weights, src_mutable->getNeighborWeights(i), sizeof(float) * edges_per_vertex);
            } else {
                for (uint8_t e = 0; e < edges_per_vertex; e++) {
                    const auto neighbor_internal_index = input_neighbor_indices[e];
                    const auto neighbor_feature = custom_features != nullptr
                                                      ? custom_feature_bytes + size_t(input_graph.getExternalLabel(neighbor_internal_index)) * feature_data_size
                                                      : input_graph.getFeatureVector(neighbor_internal_index);
                    neighbor_weights[e] = dist_func(vertex_memory, neighbor_feature, dist_func_param);
                }
            }

            // Copy external label
            std::memcpy(vertex_memory + external_label_offset_, &label, sizeof(uint32_t));

            // Register label
            label_to_index_.emplace(label, i);
        }
    }

    /**
     * Create an empty SizeBoundedGraph with the given capacity, edges per vertex, and feature space.
     * The graph starts with zero vertices; vertices can be added via addVertex().
     */
    static SizeBoundedGraph create_empty(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space) {
        return SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space);
    }

    /**
     * Create a SizeBoundedGraph by copying topology, labels, and (optionally) features
     * from an existing InternalGraph, recalculating all edge weights with the given
     * feature space (or copying them directly if the input graph is a MutableGraph with the same metric).
     *
     * @param input_graph The source graph to copy from.
     * @param new_max_size If > 0, sets capacity to max(new_max_size, input_graph.size()).
     *                     If 0, capacity equals input_graph.size().
     * @return A new SizeBoundedGraph with copied topology and appropriate edge weights.
     */
    static SizeBoundedGraph from_graph(const deglib::graph::InternalGraph& input_graph, const uint32_t new_max_size = 0) {
        return SizeBoundedGraph(input_graph, input_graph.getFeatureSpace(), nullptr, new_max_size);
    }

    /**
     * Create a SizeBoundedGraph by copying topology, labels, and (optionally) features
     * from an existing InternalGraph, recalculating all edge weights with the given
     * feature space.
     *
     * @param input_graph The source graph to copy from.
     * @param feature_space The feature space for the new graph (may differ from the input).
     * @param custom_features Optional pointer to replacement feature vectors. If non-null,
     *                        features are read from this buffer indexed by external label.
     *                        If null, features are copied from input_graph.
     * @param new_max_size If > 0, sets the capacity to max(new_max_size, input_graph.size()).
     *                     If 0, capacity equals input_graph.size().
     * @return A new SizeBoundedGraph with copied topology and recalculated edge weights.
     */
    static SizeBoundedGraph from_graph(
        const deglib::graph::InternalGraph& input_graph,
        const deglib::distances::FloatSpace feature_space,
        const void* custom_features = nullptr,
        const uint32_t new_max_size = 0
    ) {
        return SizeBoundedGraph(input_graph, feature_space, custom_features, new_max_size);
    }

    /**
     * Current maximal capacity of vertices
     */
    const auto capacity() const { return this->max_vertex_count_; }

    /**
     * Number of vertices in the graph
     */
    const uint32_t size() const override { return (uint32_t)this->label_to_index_.size(); }

    /**
     * Number of edges per vertex
     */
    const uint8_t getEdgesPerVertex() const override { return this->edges_per_vertex_; }

    const deglib::distances::FloatSpace& getFeatureSpace() const override { return this->feature_space_; }

  private:
    inline std::byte* vertex_by_index(const uint32_t internal_idx) const { return vertices_memory_ + size_t(internal_idx) * byte_size_per_vertex_; }

    inline const uint32_t label_by_index(const uint32_t internal_idx) const {
        return *reinterpret_cast<const int32_t*>(vertex_by_index(internal_idx) + external_label_offset_);
    }

    inline const std::byte* feature_by_index(const uint32_t internal_idx) const { return vertex_by_index(internal_idx); }

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
    inline const uint32_t getInternalIndex(const uint32_t external_label) const override { return label_to_index_.find(external_label)->second; }

    inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override { return label_by_index(internal_idx); }

    inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override { return feature_by_index(internal_idx); }

    inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override { return neighbors_by_index(internal_idx); }

    inline const float* getNeighborWeights(const uint32_t internal_idx) const override { return weights_by_index(internal_idx); }

    inline const float getEdgeWeight(const uint32_t internal_index, const uint32_t neighbor_index) const override {
        auto neighbor_indices = neighbors_by_index(internal_index);
        auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;
        auto neighbor_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, neighbor_index);
        if (*neighbor_ptr == neighbor_index) {
            auto weight_index = neighbor_ptr - neighbor_indices;
            return weights_by_index(internal_index)[weight_index];
        }
        return -1;
    }

    inline const bool hasVertex(const uint32_t external_label) const override { return label_to_index_.find(external_label) != label_to_index_.end(); }

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
        uint8_t metric_type = static_cast<uint8_t>(feature_space_.metric().value);
        out.write(reinterpret_cast<const char*>(&metric_type), sizeof(metric_type));
        uint16_t dim = uint16_t(this->feature_space_.dim());
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        // store graph information
        uint32_t size = uint32_t(this->size());
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(reinterpret_cast<const char*>(&this->edges_per_vertex_), sizeof(this->edges_per_vertex_));

        // store the existing vertices up to size
        uint32_t byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
        for (uint32_t i = 0; i < size; i++) out.write(reinterpret_cast<const char*>(this->vertex_by_index(i)), byte_size_per_vertex);
        out.close();

        return true;
    }

    /**
     * Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.
     *
     * @return the internal index of the new vertex
     */
    uint32_t addVertex(const uint32_t external_label, const std::byte* feature_vector) override {
        const auto new_internal_index = static_cast<uint32_t>(this->label_to_index_.size());
        if (new_internal_index >= max_vertex_count_) {
            throw std::runtime_error("SizeBoundedGraph is full: capacity exhausted.");
        }
        label_to_index_.emplace(external_label, new_internal_index);

        auto vertex_memory = vertex_by_index(new_internal_index);
        std::memcpy(vertex_memory, feature_vector, feature_byte_size_);
        std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), edges_per_vertex_, new_internal_index);  // temporary self loop
        std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), edges_per_vertex_, float(0));               // 0 weight
        std::memcpy(vertex_memory + external_label_offset_, &external_label, sizeof(uint32_t));

        return new_internal_index;
    }

    /**
     * Remove an existing vertex by swapping with the last vertex in the graph.
     */
    std::vector<uint32_t> removeVertex(const uint32_t external_label) override {
        const auto internal_index = getInternalIndex(external_label);
        const auto last_internal_index = static_cast<uint32_t>(this->label_to_index_.size() - 1);

        // since the last_internal_index will be moved to the internal_index,
        // update the current neighbor list if the last_internal_index is present
        if (hasEdge(internal_index, last_internal_index)) {
            changeEdge(internal_index, last_internal_index, internal_index, 0);
            changeEdge(last_internal_index, internal_index, last_internal_index, 0);
        }

        // copy the neighbor list to return it later
        const auto neighbor_indices = neighbors_by_index(internal_index);
        const auto involved_indices = std::vector<uint32_t>(neighbor_indices, neighbor_indices + this->edges_per_vertex_);

        // replace all references to the internal_index with a self-reference of the corresponding vertex
        for (size_t index = 0; index < this->edges_per_vertex_; index++) changeEdge(neighbor_indices[index], internal_index, neighbor_indices[index], 0);

        // the last index will be moved to the internal_index position and overwrite its content
        if (internal_index != last_internal_index) {
            // update the neighbor list of the last vertex to reflect its new vertex index
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

        // mark external label as INVALID_LABEL at the last_internal_index slot
        uint32_t inv = INVALID_LABEL;
        std::memcpy(vertex_by_index(last_internal_index) + external_label_offset_, &inv, sizeof(uint32_t));

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
        auto neighbor_indices = reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_);  // list of neighbor indizizes
        auto neighbor_indices_end = neighbor_indices + edges_per_vertex_;                               // end of the list
        uint32_t* replace_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, replace_index);
        size_t replace_idx = replace_pos - neighbor_indices;

        // Check if the replace index is found
        if (replace_pos == neighbor_indices_end || *replace_pos != replace_index) {
            std::cerr << "changeEdge: vertex " << internal_index << " does not have an edge to " << replace_index << " and therefore can not be swapped with "
                      << new_index << " and distance " << new_weight << std::endl;
            return false;
        }

        // Find the position where the new index should be inserted
        uint32_t* insert_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, new_index);
        size_t insert_idx = insert_pos - neighbor_indices;

        // Handle the case where the insertion position is after the removal position
        auto neighbor_weights = reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_);  // list of neighbor weights
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
     * Perform a search but stops when the to_vertex was found.
     */
    std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const override {
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
        const uint32_t max_distance_computation_count = 0
    ) const override {
        return searchInternImpl(*this, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count);
    }
};

/**
 * Load the graph
 */
inline auto load_sizebounded_graph(const char* path_graph, uint32_t new_max_size = 0) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(path_graph, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing test file, size is: %ju message: %s \n", file_size, ec.message().c_str());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(path_graph, std::ios::binary);
    if (!ifstream.is_open()) {
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

    // if no new max size is set use the size of the graph from disk
    if (new_max_size == 0) new_max_size = size;

    // if there is a max size is should be higher than the needed graph size from disk
    if (new_max_size < size) {
        std::fprintf(stderr, "The graph in the %s file has %u vertices but the new max size is %u\n", path_graph, size, new_max_size);
        perror("");
        abort();
    }

    auto graph = deglib::graph::SizeBoundedGraph(new_max_size, edges_per_vertex, std::move(feature_space), ifstream, size);
    ifstream.close();

    return graph;
}

/**
 * Convert the given graph to a SizeBoundedGraph, copying topology, labels,
 * and recalculating all edge weights with the given feature space.
 */
inline auto convert_to_sizebounded_graph(const deglib::graph::InternalGraph& input_graph) {
    return deglib::graph::SizeBoundedGraph::from_graph(input_graph, input_graph.getFeatureSpace());
}

/**
 * Convert the given graph to a SizeBoundedGraph, overriding feature space and feature vectors.
 */
inline auto convert_to_sizebounded_graph(
    const deglib::graph::InternalGraph& input_graph,
    const deglib::distances::FloatSpace feature_space,
    const void* custom_features = nullptr,
    const uint32_t new_max_size = 0
) {
    return deglib::graph::SizeBoundedGraph::from_graph(input_graph, feature_space, custom_features, new_max_size);
}

}  // namespace deglib::graph