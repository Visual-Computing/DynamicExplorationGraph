#pragma once

#include "deglib/graph/mutable_graph.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/search.h"
#include "deglib/utils/memory.h"
#include "deglib/utils/random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

namespace deglib::graph {

/**
 * A dynamic, undirected and weighted n-regular graph with chunk-based memory management.
 *
 * Unlike SizeBoundedGraph, DynamicGraph does not require a fixed maximum vertex count at construction.
 * Memory is allocated in chunks of vertices (rounded up to the nearest power of two). Looking up a vertex
 * by internal ID is performed with bit-shifts and bitwise AND operations for O(1) addressing.
 *
 * When vertices are removed, the last vertex in the last chunk is moved into the freed slot
 * (swap-with-last), maintaining a compact, contiguous index space [0, size - 1].
 * Empty chunks are automatically freed.
 */
class DynamicGraph : public deglib::graph::MutableGraph {
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

    static uint32_t compute_chunk_shift(uint32_t chunk_size) {
        if (chunk_size == 0) chunk_size = 1024;
        uint32_t shift = 0;
        while ((1u << shift) < chunk_size && shift < 24) {
            ++shift;
        }
        return std::max(1u, shift);
    }

    static const uint8_t object_alignment = 32;

    const uint8_t edges_per_vertex_;
    const uint16_t feature_byte_size_;
    const uint32_t chunk_shift_;
    const uint32_t chunk_capacity_;
    const uint32_t chunk_mask_;

    const uint32_t byte_size_per_vertex_;
    const uint32_t neighbor_indices_offset_;
    const uint32_t neighbor_weights_offset_;
    const uint32_t external_label_offset_;
    const size_t chunk_byte_size_;

    // List of allocated chunks and aligned access pointers
    std::vector<std::unique_ptr<std::byte[]>> chunks_;
    std::vector<std::byte*> chunks_memory_;

    // Map from external label to internal index (0..size-1)
    std::unordered_map<uint32_t, uint32_t> label_to_index_;

    // Distance calculation function between feature vectors
    const deglib::distances::FloatSpace feature_space_;

    mutable size_t visited_list_pool_capacity_{0};
    mutable std::unique_ptr<VisitedListPool> visited_list_pool_;

    void allocate_chunk() {
        auto chunk = std::make_unique<std::byte[]>(chunk_byte_size_);
        auto* aligned_ptr = compute_aligned_pointer(chunk, object_alignment);
        chunks_.emplace_back(std::move(chunk));
        chunks_memory_.emplace_back(aligned_ptr);

        const auto cap = capacity();
        if (!visited_list_pool_ || cap > visited_list_pool_capacity_) {
            visited_list_pool_capacity_ = cap;
            visited_list_pool_ = std::make_unique<VisitedListPool>(1, static_cast<int>(visited_list_pool_capacity_));
        }
    }

  public:
    /**
     * Construct an empty DynamicGraph with chunk-based dynamic memory allocation.
     *
     * @param edges_per_vertex Number of edges per vertex (must be even).
     * @param feature_space Distance metric and vector dimensionality.
     * @param chunk_size Target number of vertices per memory chunk (default = 1024).
     *                   Will be automatically rounded up to the nearest power of 2 (e.g., 600 -> 1024)
     *                   to enable fast bit-shift indexing (2 <= chunk_size <= 2^24).
     *                   Guidelines:
     *                   - Small (64 - 512): Low memory overhead for small graphs or frequent shrinking.
     *                   - Medium (1024 - 4096): Balanced performance and allocation frequency (recommended).
     *                   - Large (8192 - 65536): High throughput for very large static/semi-static graphs (>1M vertices).
     */
    DynamicGraph(const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space, const uint32_t chunk_size = 1024)
        : edges_per_vertex_(edges_per_vertex),
          feature_byte_size_(uint16_t(feature_space.get_data_size())),
          chunk_shift_(compute_chunk_shift(chunk_size)),
          chunk_capacity_(1u << chunk_shift_),
          chunk_mask_(chunk_capacity_ - 1),

          byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)),
          neighbor_indices_offset_(uint32_t(feature_space.get_data_size())),
          neighbor_weights_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)),
          external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_vertex) * sizeof(float)),
          chunk_byte_size_(size_t(chunk_capacity_) * byte_size_per_vertex_ + object_alignment),

          feature_space_(feature_space),
          visited_list_pool_capacity_(chunk_capacity_),
          visited_list_pool_(std::make_unique<VisitedListPool>(1, static_cast<int>(chunk_capacity_))) {
        if (edges_per_vertex % 2 != 0) throw std::invalid_argument("edges_per_vertex must be even.");
    }

    /**
     * Load from an open binary file stream.
     *
     * @param edges_per_vertex Number of edges per vertex.
     * @param feature_space Distance metric and vector dimensionality.
     * @param ifstream Open binary input stream positioned at the vertex data.
     * @param size Total number of vertices to read from the stream.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    DynamicGraph(
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace feature_space,
        std::ifstream& ifstream,
        const uint32_t size,
        const uint32_t chunk_size = 1024
    )
        : DynamicGraph(edges_per_vertex, std::move(feature_space), chunk_size) {
        label_to_index_.reserve(size);
        const uint32_t file_byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);

        for (uint32_t i = 0; i < size; i++) {
            if (i >= capacity()) {
                allocate_chunk();
            }
            ifstream.read(reinterpret_cast<char*>(this->vertex_by_index(i)), file_byte_size_per_vertex);
            const uint32_t lbl = this->getExternalLabel(i);
            label_to_index_.emplace(lbl, i);
        }
    }

    /**
     * Copy from an existing InternalGraph, optionally recalculating edge weights or replacing features.
     *
     * @param input_graph Source graph to copy topology and labels from.
     * @param feature_space Target feature space (dimension and metric).
     * @param custom_features Optional raw byte pointer to new feature vectors.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    DynamicGraph(
        const deglib::graph::InternalGraph& input_graph,
        const deglib::distances::FloatSpace feature_space,
        const void* custom_features = nullptr,
        const uint32_t chunk_size = 1024
    )
        : DynamicGraph(input_graph.getEdgesPerVertex(), std::move(feature_space), chunk_size) {
        const auto custom_feature_bytes = reinterpret_cast<const std::byte*>(custom_features);
        const auto dist_func = this->feature_space_.get_dist_func();
        const auto dist_func_param = this->feature_space_.get_dist_func_param();
        const auto feature_data_size = this->feature_space_.get_data_size();
        const auto edges_per_vertex = this->edges_per_vertex_;

        const uint32_t graph_size = input_graph.size();
        label_to_index_.reserve(graph_size);

        const auto* src_mutable = dynamic_cast<const deglib::graph::MutableGraph*>(&input_graph);
        const bool can_direct_copy_weights = (src_mutable != nullptr) && (custom_features == nullptr) &&
                                             (this->feature_space_.metric() == input_graph.getFeatureSpace().metric()) &&
                                             (this->feature_space_.dim() == input_graph.getFeatureSpace().dim());

        for (uint32_t i = 0; i < graph_size; i++) {
            if (i >= capacity()) {
                allocate_chunk();
            }

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
     * Create an empty DynamicGraph with the given edges per vertex and feature space.
     *
     * @param edges_per_vertex Number of edges per vertex (must be even).
     * @param feature_space Distance metric and vector dimensionality.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    static DynamicGraph create_empty(const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t chunk_size = 1024) {
        return DynamicGraph(edges_per_vertex, feature_space, chunk_size);
    }

    /**
     * Create a DynamicGraph by copying topology, labels, and features from an existing InternalGraph.
     *
     * @param input_graph Source graph to copy from.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    static DynamicGraph from_graph(const deglib::graph::InternalGraph& input_graph, const uint32_t chunk_size = 1024) {
        return DynamicGraph(input_graph, input_graph.getFeatureSpace(), nullptr, chunk_size);
    }

    /**
     * Create a DynamicGraph by copying topology and labels from an existing InternalGraph,
     * recalculating edge weights with the given feature space.
     *
     * @param input_graph Source graph to copy from.
     * @param feature_space Target feature space (dimension and metric).
     * @param custom_features Optional raw byte pointer to new feature vectors.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    static DynamicGraph from_graph(
        const deglib::graph::InternalGraph& input_graph,
        const deglib::distances::FloatSpace feature_space,
        const void* custom_features = nullptr,
        const uint32_t chunk_size = 1024
    ) {
        return DynamicGraph(input_graph, feature_space, custom_features, chunk_size);
    }

    /**
     * Create a random exploration graph from the given feature data.
     *
     * @param feature_data Pointer to contiguous feature vectors.
     * @param vertex_count Number of vertices to add.
     * @param edges_per_vertex Number of edges per vertex (must be even).
     * @param feature_space Distance metric and vector dimensionality.
     * @param seed Random seed.
     * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
     */
    static DynamicGraph create_random_graph(
        const std::byte* feature_data,
        const uint32_t vertex_count,
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace& feature_space,
        const uint32_t seed = 7,
        const uint32_t chunk_size = 1024
    ) {
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        auto graph = DynamicGraph(edges_per_vertex, feature_space, chunk_size);

        // add the initial vertices (edges_per_vertex + 1)
        {
            const auto size = static_cast<uint32_t>(edges_per_vertex + 1);
            for (uint32_t y = 0; y < size; y++) {
                const auto query = feature_data + size_t(y) * feature_space.get_data_size();
                const auto internal_index = graph.addVertex(y, query);

                auto neighbor_indices = std::vector<uint32_t>();
                auto neighbor_weights = std::vector<float>();
                for (uint32_t x = 0; x < size; x++) {
                    if (x == internal_index) continue;
                    neighbor_indices.emplace_back(x);
                    neighbor_weights.emplace_back(dist_func(query, feature_data + size_t(x) * feature_space.get_data_size(), dist_func_param));
                }
                graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
            }
        }

        // random order of vertices
        auto rnd = std::mt19937(seed);
        auto rnd_neighbor = deglib::random::DeterministicUniformIntDistribution<uint32_t>(0, edges_per_vertex - 1);

        // add the remaining vertices
        for (uint32_t label = edges_per_vertex + 1; label < vertex_count; label++) {
            const auto new_vertex_feature = feature_data + size_t(label) * feature_space.get_data_size();
            const auto internal_index = graph.addVertex(label, new_vertex_feature);
            auto top_list = deglib::random::DeterministicUniformIntDistribution<uint32_t>(0, label - 1);

            auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
            while (new_neighbors.size() < edges_per_vertex) {
                const auto candidate_index = static_cast<uint32_t>(top_list(rnd));

                if (graph.hasEdge(candidate_index, internal_index)) continue;

                uint32_t new_neighbor_index = 0;
                float new_neighbor_weight = 0;
                bool found = false;
                const auto neighbor_weights = graph.getNeighborWeights(candidate_index);
                const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
                while (!found) {
                    const auto edge_idx = static_cast<uint32_t>(rnd_neighbor(rnd));
                    const auto neighbor_index = neighbor_indices[edge_idx];
                    const auto neighbor_weight = neighbor_weights[edge_idx];

                    if (graph.hasEdge(neighbor_index, internal_index) == false) {
                        new_neighbor_index = neighbor_index;
                        new_neighbor_weight = neighbor_weight;
                        found = true;
                    }
                }

                const auto candidate_dist = dist_func(new_vertex_feature, graph.getFeatureVector(candidate_index), dist_func_param);
                graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_dist);
                new_neighbors.emplace_back(candidate_index, candidate_dist);

                const auto new_neighbor_dist = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param);
                graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_dist);
                new_neighbors.emplace_back(new_neighbor_index, new_neighbor_dist);
            }

            std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            auto neighbor_indices = std::vector<uint32_t>();
            auto neighbor_weights = std::vector<float>();
            for (auto&& neighbor : new_neighbors) {
                neighbor_indices.emplace_back(neighbor.first);
                neighbor_weights.emplace_back(neighbor.second);
            }
            graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
        }

        return graph;
    }

    /**
     * Current allocated capacity across all chunks.
     */
    uint32_t capacity() const { return static_cast<uint32_t>(chunks_.size()) * chunk_capacity_; }

    /**
     * Number of allocated chunks.
     */
    uint32_t chunk_count() const { return static_cast<uint32_t>(chunks_.size()); }

    /**
     * Capacity of an individual chunk.
     */
    uint32_t chunk_capacity() const { return chunk_capacity_; }

    /**
     * Number of vertices in the graph.
     */
    const uint32_t size() const override { return static_cast<uint32_t>(this->label_to_index_.size()); }

    const uint8_t getEdgesPerVertex() const override { return this->edges_per_vertex_; }

    const deglib::distances::FloatSpace& getFeatureSpace() const override { return this->feature_space_; }

  private:
    inline std::byte* vertex_by_index(const uint32_t internal_idx) const {
        const auto chunk_idx = internal_idx >> chunk_shift_;
        const auto idx_in_chunk = internal_idx & chunk_mask_;
        return chunks_memory_[chunk_idx] + size_t(idx_in_chunk) * byte_size_per_vertex_;
    }

    inline const uint32_t label_by_index(const uint32_t internal_idx) const {
        return *reinterpret_cast<const uint32_t*>(vertex_by_index(internal_idx) + external_label_offset_);
    }

    inline const std::byte* feature_by_index(const uint32_t internal_idx) const { return vertex_by_index(internal_idx); }

    inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
        return reinterpret_cast<uint32_t*>(vertex_by_index(internal_idx) + neighbor_indices_offset_);
    }

    inline const float* weights_by_index(const uint32_t internal_idx) const {
        return reinterpret_cast<const float*>(vertex_by_index(internal_idx) + neighbor_weights_offset_);
    }

  public:
    inline const uint32_t getInternalIndex(const uint32_t external_label) const override { return label_to_index_.find(external_label)->second; }

    inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override { return label_by_index(internal_idx); }

    inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override { return feature_by_index(internal_idx); }

    inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override { return neighbors_by_index(internal_idx); }

    inline const float* getNeighborWeights(const uint32_t internal_idx) const override { return weights_by_index(internal_idx); }

    inline const float getEdgeWeight(const uint32_t internal_index, const uint32_t neighbor_index) const override {
        auto neighbor_indices = neighbors_by_index(internal_index);
        auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;
        auto neighbor_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, neighbor_index);
        if (neighbor_ptr != neighbor_indices_end && *neighbor_ptr == neighbor_index) {
            auto weight_index = neighbor_ptr - neighbor_indices;
            return weights_by_index(internal_index)[weight_index];
        }
        return -1.0f;
    }

    inline const bool hasVertex(const uint32_t external_label) const override { return label_to_index_.find(external_label) != label_to_index_.end(); }

    inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
        auto neighbor_indices = neighbors_by_index(internal_index);
        auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;
        return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index);
    }

    const bool saveGraph(const char* path_to_graph) const override {
        std::filesystem::create_directories(std::filesystem::path(path_to_graph).parent_path());

        auto out = std::ofstream(path_to_graph, std::ios::out | std::ios::binary);
        if (!out.is_open()) {
            std::fprintf(stderr, "Error opening file for write: %s\n", path_to_graph);
            return false;
        }

        uint8_t metric_type = static_cast<uint8_t>(feature_space_.metric().value);
        out.write(reinterpret_cast<const char*>(&metric_type), sizeof(metric_type));
        uint16_t dim = uint16_t(this->feature_space_.dim());
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        uint32_t graph_size = uint32_t(this->size());
        out.write(reinterpret_cast<const char*>(&graph_size), sizeof(graph_size));
        out.write(reinterpret_cast<const char*>(&this->edges_per_vertex_), sizeof(this->edges_per_vertex_));

        uint32_t file_byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
        for (uint32_t i = 0; i < graph_size; i++) out.write(reinterpret_cast<const char*>(this->vertex_by_index(i)), file_byte_size_per_vertex);
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
        if (new_internal_index >= capacity()) {
            allocate_chunk();
        }
        label_to_index_.emplace(external_label, new_internal_index);

        auto vertex_memory = vertex_by_index(new_internal_index);
        std::memcpy(vertex_memory, feature_vector, feature_byte_size_);
        std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), edges_per_vertex_, new_internal_index);
        std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), edges_per_vertex_, 0.0f);
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

        // remove the external label from the hash map
        label_to_index_.erase(external_label);

        // If the removed element was the only vertex in the last chunk, free the chunk
        if ((last_internal_index & chunk_mask_) == 0) {
            chunks_.pop_back();
            chunks_memory_.pop_back();
        }

        // return all neighbors of the deleted vertex
        return involved_indices;
    }

    /**
     * Swap a neighbor with another neighbor and its weight.
     */
    bool changeEdge(const uint32_t internal_index, const uint32_t replace_index, const uint32_t new_index, const float new_weight) override {
        auto vertex_memory = vertex_by_index(internal_index);

        auto neighbor_indices = reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_);
        auto neighbor_indices_end = neighbor_indices + edges_per_vertex_;
        uint32_t* replace_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, replace_index);
        size_t replace_idx = replace_pos - neighbor_indices;

        if (replace_pos == neighbor_indices_end || *replace_pos != replace_index) {
            std::cerr << "changeEdge: vertex " << internal_index << " does not have an edge to " << replace_index << " and therefore can not be swapped with "
                      << new_index << " and distance " << new_weight << std::endl;
            return false;
        }

        uint32_t* insert_pos = std::lower_bound(neighbor_indices, neighbor_indices_end, new_index);
        size_t insert_idx = insert_pos - neighbor_indices;

        auto neighbor_weights = reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_);
        if (insert_idx > replace_idx) {
            std::memmove(neighbor_indices + replace_idx, neighbor_indices + replace_idx + 1, (insert_idx - replace_idx - 1) * sizeof(uint32_t));
            std::memmove(neighbor_weights + replace_idx, neighbor_weights + replace_idx + 1, (insert_idx - replace_idx - 1) * sizeof(float));
            --insert_idx;
        } else if (insert_idx < replace_idx) {
            std::memmove(neighbor_indices + insert_idx + 1, neighbor_indices + insert_idx, (replace_idx - insert_idx) * sizeof(uint32_t));
            std::memmove(neighbor_weights + insert_idx + 1, neighbor_weights + insert_idx, (replace_idx - insert_idx) * sizeof(float));
        }

        neighbor_indices[insert_idx] = new_index;
        neighbor_weights[insert_idx] = new_weight;

        return true;
    }

    /**
     * Change all edges of a vertex.
     */
    void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) override {
        auto vertex_memory = vertex_by_index(internal_index);
        std::memcpy(vertex_memory + neighbor_indices_offset_, neighbor_indices, uint32_t(edges_per_vertex_) * sizeof(uint32_t));
        std::memcpy(vertex_memory + neighbor_weights_offset_, neighbor_weights, uint32_t(edges_per_vertex_) * sizeof(float));
    }

    /**
     * Perform a search stopping when to_vertex was found.
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
 * Load a DynamicGraph from disk.
 *
 * @param path_graph File path to the saved graph file.
 * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
 */
inline auto load_dynamic_graph(const char* path_graph, uint32_t chunk_size = 1024) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(path_graph, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file, size is: %ju message: %s \n", file_size, ec.message().c_str());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(path_graph, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", path_graph);
        perror("");
        abort();
    }

    uint8_t metric_type;
    ifstream.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
    uint16_t dim;
    ifstream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    const auto feature_space = deglib::distances::FloatSpace(dim, static_cast<deglib::distances::Metric>(metric_type));

    uint32_t size;
    ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
    uint8_t edges_per_vertex;
    ifstream.read(reinterpret_cast<char*>(&edges_per_vertex), sizeof(edges_per_vertex));

    auto graph = deglib::graph::DynamicGraph(edges_per_vertex, std::move(feature_space), ifstream, size, chunk_size);
    ifstream.close();

    return graph;
}

/**
 * Convert the given graph to a DynamicGraph, copying topology, labels,
 * and recalculating all edge weights with the given feature space.
 *
 * @param input_graph Source graph to convert from.
 * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
 */
inline auto convert_to_dynamic_graph(const deglib::graph::InternalGraph& input_graph, const uint32_t chunk_size = 1024) {
    return deglib::graph::DynamicGraph::from_graph(input_graph, chunk_size);
}

/**
 * Convert the given graph to a DynamicGraph, overriding feature space and feature vectors.
 *
 * @param input_graph Source graph to convert from.
 * @param feature_space Target feature space (dimension and metric).
 * @param custom_features Optional raw byte pointer to new feature vectors.
 * @param chunk_size Target number of vertices per memory chunk (rounded up to nearest power of 2, default = 1024).
 */
inline auto convert_to_dynamic_graph(
    const deglib::graph::InternalGraph& input_graph,
    const deglib::distances::FloatSpace feature_space,
    const void* custom_features = nullptr,
    const uint32_t chunk_size = 1024
) {
    return deglib::graph::DynamicGraph::from_graph(input_graph, feature_space, custom_features, chunk_size);
}

}  // namespace deglib::graph
