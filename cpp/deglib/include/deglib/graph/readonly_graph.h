#pragma once

#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/search/result_list.h"
#include "deglib/utils/memory.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace deglib::graph {

/**
 * An immutable, simple, undirected n-regular graph optimized exclusively for read-only search queries.
 *
 * The vertex count and number of edges per vertex are fixed at construction time.
 * The graph is strictly read-only and cannot be mutated. For mutable graphs, see SizeBoundedGraph or DynamicGraph.
 */
class ReadOnlyGraph : public deglib::graph::InternalGraph {
    friend class deglib::graph::InternalGraph;

    static const uint8_t alignment = 64;  // 64-byte alignment for SIMD vector access

    const uint32_t max_vertex_count_;
    const uint8_t edges_per_vertex_;
    const uint16_t feature_byte_size_;
    const size_t feature_stride_;

    // Features: 64-byte aligned contiguous buffer, stride padded to multiple of 64 bytes
    std::unique_ptr<std::byte[]> features_storage_;
    std::byte* features_ = nullptr;

    // Neighbors: contiguous array of edges_per_vertex_ entries per vertex
    std::vector<uint32_t> neighbors_;

    // Labels: contiguous array of external labels
    std::vector<uint32_t> labels_;

    std::unordered_map<uint32_t, uint32_t> label_to_index_;
    const deglib::distances::FloatSpace feature_space_;
    std::unique_ptr<VisitedListPool> visited_list_pool_;

  public:
    ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space)
        : max_vertex_count_(max_vertex_count),
          edges_per_vertex_(edges_per_vertex),
          feature_byte_size_(uint16_t(feature_space.get_data_size())),
          feature_stride_(((size_t(feature_space.get_data_size()) + 63) / 64) * 64),
          neighbors_(size_t(max_vertex_count) * size_t(edges_per_vertex)),
          labels_(max_vertex_count),
          feature_space_(feature_space),
          visited_list_pool_(std::make_unique<VisitedListPool>(1, max_vertex_count)) {
        if (edges_per_vertex % 2 != 0) throw std::invalid_argument("edges_per_vertex must be even.");
        label_to_index_.reserve(max_vertex_count);

        if (max_vertex_count > 0) {
            const size_t total_feat_bytes = size_t(max_vertex_count) * feature_stride_;
            features_storage_ = std::make_unique<std::byte[]>(total_feat_bytes + alignment);
            void* ptr = features_storage_.get();
            size_t space = std::numeric_limits<size_t>::max();
            std::align(alignment, total_feat_bytes, ptr, space);
            features_ = static_cast<std::byte*>(ptr);
            std::memset(features_, 0, total_feat_bytes);
        }
    }

    /**
     *  Load from file (file format stores per vertex: features, neighbors, weights [ignored], label)
     */
    ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space, std::ifstream& ifstream)
        : ReadOnlyGraph(max_vertex_count, edges_per_vertex, feature_space) {
        const size_t feat_size = feature_space_.get_data_size();
        const size_t neighbor_bytes = size_t(edges_per_vertex_) * sizeof(uint32_t);
        const size_t weight_bytes = size_t(edges_per_vertex_) * sizeof(float);

        for (uint32_t i = 0; i < max_vertex_count_; i++) {
            ifstream.read(reinterpret_cast<char*>(features_ + size_t(i) * feature_stride_), feat_size);
            ifstream.read(reinterpret_cast<char*>(neighbors_.data() + size_t(i) * size_t(edges_per_vertex_)), neighbor_bytes);
            ifstream.ignore(weight_bytes);
            uint32_t lbl = 0;
            ifstream.read(reinterpret_cast<char*>(&lbl), sizeof(uint32_t));
            labels_[i] = lbl;
            label_to_index_.emplace(lbl, i);
        }
    }

    /**
     *  Copy from input graph (with optional custom features buffer)
     */
    ReadOnlyGraph(
        const uint32_t max_vertex_count,
        const uint8_t edges_per_vertex,
        const deglib::distances::FloatSpace feature_space,
        const deglib::graph::InternalGraph& input_graph,
        const void* custom_features = nullptr
    )
        : ReadOnlyGraph(max_vertex_count, edges_per_vertex, feature_space) {
        const auto custom_feature_bytes = reinterpret_cast<const std::byte*>(custom_features);
        const size_t feat_size = feature_space_.get_data_size();
        const size_t neighbor_bytes = size_t(edges_per_vertex_) * sizeof(uint32_t);

        for (uint32_t i = 0; i < max_vertex_count_; i++) {
            const auto label = input_graph.getExternalLabel(i);
            labels_[i] = label;
            label_to_index_.emplace(label, i);

            if (custom_features != nullptr) {
                const auto feat_src = custom_feature_bytes + size_t(label) * feat_size;
                std::memcpy(features_ + size_t(i) * feature_stride_, feat_src, feat_size);
            } else {
                const auto feat_src = input_graph.getFeatureVector(i);
                std::memcpy(features_ + size_t(i) * feature_stride_, feat_src, feat_size);
            }

            const auto neighbor_indices = input_graph.getNeighborIndices(i);
            std::memcpy(neighbors_.data() + size_t(i) * size_t(edges_per_vertex_), neighbor_indices, neighbor_bytes);
        }
    }

    const auto capacity() const { return max_vertex_count_; }
    const uint32_t size() const override { return (uint32_t)label_to_index_.size(); }
    const uint8_t getEdgesPerVertex() const override { return edges_per_vertex_; }
    const deglib::distances::FloatSpace& getFeatureSpace() const override { return this->feature_space_; }

  private:
    inline const uint32_t label_by_index(const uint32_t internal_idx) const { return labels_[internal_idx]; }

    inline const std::byte* feature_by_index(const uint32_t internal_idx) const { return features_ + size_t(internal_idx) * feature_stride_; }

    inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
        return neighbors_.data() + size_t(internal_idx) * size_t(edges_per_vertex_);
    }

  public:
    inline const uint32_t getInternalIndex(const uint32_t external_label) const override { return label_to_index_.find(external_label)->second; }
    inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override { return label_by_index(internal_idx); }
    inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override { return feature_by_index(internal_idx); }
    inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override { return neighbors_by_index(internal_idx); }
    inline const bool hasVertex(const uint32_t external_label) const override { return label_to_index_.contains(external_label); }

    inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
        auto neighbor_indices = getNeighborIndices(internal_index);
        auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;
        return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index);
    }

    std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const override {
        return hasPathImpl(*this, entry_vertex_indices, to_vertex, eps, k);
    }

    std::vector<deglib::graph::ObjectDistance> search_ef_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const override {
        return searchEfInternImpl(*this, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count);
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
inline auto load_readonly_graph(const char* path_graph) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(path_graph, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing graph file %s, size is: %ju message: %s\n", path_graph, file_size, ec.message().c_str());
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

    auto graph = deglib::graph::ReadOnlyGraph(size, edges_per_vertex, std::move(feature_space), ifstream);
    ifstream.close();

    return graph;
}

/**
 * Convert the given graph to a readonly graph
 */
inline auto convert_to_readonly_graph(const deglib::graph::InternalGraph& input_graph) {
    auto size = input_graph.size();
    auto edges_per_vertex = input_graph.getEdgesPerVertex();
    return deglib::graph::ReadOnlyGraph(size, edges_per_vertex, input_graph.getFeatureSpace(), input_graph);
}

/**
 * Convert the given graph to a readonly graph, overriding feature space and feature vectors.
 */
inline auto convert_to_readonly_graph(
    const deglib::graph::InternalGraph& input_graph,
    const deglib::distances::FloatSpace feature_space,
    const void* custom_features = nullptr
) {
    auto size = input_graph.size();
    auto edges_per_vertex = input_graph.getEdgesPerVertex();
    return deglib::graph::ReadOnlyGraph(size, edges_per_vertex, feature_space, input_graph, custom_features);
}
}  // namespace deglib::graph
