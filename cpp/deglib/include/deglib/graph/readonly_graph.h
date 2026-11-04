#pragma once

#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/search/result_list.h"
#include "deglib/serialization.h"
#include "deglib/transformation.h"
#include "deglib/utils/memory.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
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

  public:
    ReadOnlyGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace feature_space)
        : max_vertex_count_(max_vertex_count),
          edges_per_vertex_(edges_per_vertex),
          feature_byte_size_(uint16_t(feature_space.get_data_size())),
          neighbor_indices_offset_(((size_t(feature_space.get_data_size()) + 63) / 64) * 64),
          vertex_stride_(((neighbor_indices_offset_ + size_t(edges_per_vertex) * sizeof(uint32_t)) + 63) / 64 * 64),
          labels_(max_vertex_count),
          feature_space_(feature_space),
          visited_list_pool_(std::make_unique<VisitedListPool>(1, max_vertex_count)) {
        if (edges_per_vertex % 2 != 0) throw std::invalid_argument("edges_per_vertex must be even.");
        label_to_index_.reserve(max_vertex_count);

        if (max_vertex_count > 0) {
            const size_t total_vertex_bytes = size_t(max_vertex_count) * vertex_stride_;
            vertices_storage_ = std::make_unique<std::byte[]>(total_vertex_bytes + alignment);
            void* ptr = vertices_storage_.get();
            size_t space = std::numeric_limits<size_t>::max();
            std::align(alignment, total_vertex_bytes, ptr, space);
            vertices_ = static_cast<std::byte*>(ptr);
            std::memset(vertices_, 0, total_vertex_bytes);
        }
    }

    /**
     * Create a ReadOnlyGraph from an existing InternalGraph, copying topology and optionally
     * replacing or directly transforming features into the strided ReadOnlyGraph memory layout.
     *
     * @param input_graph Source graph to copy topology and labels from.
     * @param target_feature_space Optional target feature space (if nullopt, derived from transformations or input_graph).
     * @param custom_features Optional raw byte pointer to new feature vectors in the graph's internal index order (0 .. size()-1, see
     * input_graph.getExternalLabels()).
     * @param transformations Optional sequence of transformations applied directly into destination memory.
     */
    static ReadOnlyGraph from_graph(
        const deglib::graph::InternalGraph& input_graph,
        std::optional<deglib::distances::FloatSpace> target_feature_space = std::nullopt,
        const void* custom_features = nullptr,
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    ) {
        auto resolve_space = [&]() -> deglib::distances::FloatSpace {
            if (target_feature_space.has_value()) {
                return *target_feature_space;
            } else if (!transformations.empty()) {
                const uint32_t target_dim = deglib::transformation::get_output_dim(transformations);
                const auto target_metric = deglib::transformation::get_target_metric(transformations, input_graph.getFeatureSpace().metric());
                return deglib::distances::FloatSpace(target_dim, target_metric);
            }
            return input_graph.getFeatureSpace();
        };

        const auto feature_space = resolve_space();
        const uint32_t count = input_graph.size();
        const uint8_t edges = input_graph.getEdgesPerVertex();
        ReadOnlyGraph target(count, edges, feature_space);

        const auto custom_feature_bytes = reinterpret_cast<const std::byte*>(custom_features);
        const size_t feat_size = target.feature_space_.get_data_size();
        const size_t neighbor_bytes = size_t(edges) * sizeof(uint32_t);

        const auto src_metric = input_graph.getFeatureSpace().metric();
        const uint32_t in_dim = input_graph.getFeatureSpace().dim();
        std::vector<float> f_buf;
        if (!transformations.empty() && src_metric.get_data_type() == deglib::distances::MetricDataType::FP16) {
            f_buf.resize(in_dim);
        }

        for (uint32_t i = 0; i < count; ++i) {
            const auto label = input_graph.getExternalLabel(i);
            target.labels_[i] = label;
            target.label_to_index_.emplace(label, i);

            // Feature vector: direct transformation or copy into 64-byte aligned vertex memory
            std::byte* vertex_ptr = target.vertices_ + size_t(i) * target.vertex_stride_;
            std::byte* dst_feat = vertex_ptr;
            if (!transformations.empty() && (src_metric != target.feature_space_.metric())) {
                const auto* src_feat = input_graph.getFeatureVector(i);
                if (src_metric.get_data_type() == deglib::distances::MetricDataType::FP32) {
                    deglib::transformation::apply_database(transformations, reinterpret_cast<const float*>(src_feat), dst_feat, 1);
                } else if (src_metric.get_data_type() == deglib::distances::MetricDataType::FP16) {
                    deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src_feat), f_buf.data(), in_dim);
                    deglib::transformation::apply_database(transformations, f_buf.data(), dst_feat, 1);
                } else {
                    throw std::invalid_argument("ReadOnlyGraph::from_graph: cannot self-transform from non-float base features");
                }
            } else if (custom_features != nullptr) {
                const auto feat_src = custom_feature_bytes + size_t(i) * feat_size;
                std::memcpy(dst_feat, feat_src, feat_size);
            } else {
                const auto feat_src = input_graph.getFeatureVector(i);
                std::memcpy(dst_feat, feat_src, feat_size);
            }

            // Copy topology directly adjacent to feature vector
            const auto neighbor_indices = input_graph.getNeighborIndices(i);
            std::memcpy(vertex_ptr + target.neighbor_indices_offset_, neighbor_indices, neighbor_bytes);
        }

        target.setPrefetch(input_graph.getPo(), input_graph.getPl(), input_graph.getNl());
        target.setEntryVertexIndices(input_graph.getEntryVertexIndices());
        return target;
    }

    static ReadOnlyGraph loadGraph(const char* path_graph) {
        auto [is, header] = deglib::serialization::open_graph_file(path_graph);
        ReadOnlyGraph graph(header.size, header.edges_per_vertex, *header.feature_space);
        const size_t feat_size = header.feature_space->get_data_size();
        const size_t neighbor_bytes = size_t(header.edges_per_vertex) * sizeof(uint32_t);
        const size_t weight_bytes = size_t(header.edges_per_vertex) * sizeof(float);
        const size_t file_record_size = feat_size + neighbor_bytes + weight_bytes + sizeof(uint32_t);

        constexpr size_t BATCH_SIZE = 256;
        std::vector<char> batch_buf(BATCH_SIZE * file_record_size);

        for (uint32_t offset = 0; offset < header.size; offset += BATCH_SIZE) {
            const uint32_t current_batch = std::min<uint32_t>(BATCH_SIZE, header.size - offset);
            const size_t bytes_to_read = current_batch * file_record_size;
            if (!is.read(batch_buf.data(), bytes_to_read)) {
                throw std::runtime_error("ReadOnlyGraph::loadGraph: unexpected end of stream reading vertices");
            }

            const char* ptr = batch_buf.data();
            for (uint32_t b = 0; b < current_batch; ++b) {
                const uint32_t i = offset + b;
                std::byte* vertex_ptr = graph.vertices_ + size_t(i) * graph.vertex_stride_;
                std::memcpy(vertex_ptr, ptr, feat_size);
                ptr += feat_size;
                std::memcpy(vertex_ptr + graph.neighbor_indices_offset_, ptr, neighbor_bytes);
                ptr += neighbor_bytes;
                ptr += weight_bytes;  // skip weights
                uint32_t lbl = 0;
                std::memcpy(&lbl, ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                graph.labels_[i] = lbl;
                graph.label_to_index_.emplace(lbl, i);
            }
        }

        graph.setPrefetch(header.po, header.pl, header.nl);
        graph.setEntryVertexIndices(header.entry_vertex_indices);
        return graph;
    }

    // --- Capacity & Metadata ---
    const auto capacity() const { return max_vertex_count_; }
    const uint32_t size() const override { return static_cast<uint32_t>(label_to_index_.size()); }
    const uint8_t getEdgesPerVertex() const override { return edges_per_vertex_; }
    const deglib::distances::FloatSpace& getFeatureSpace() const override { return this->feature_space_; }

    // --- Vertex & Topology Accessors ---
    inline const uint32_t getInternalIndex(const uint32_t external_label) const override { return label_to_index_.find(external_label)->second; }
    inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override { return label_by_index(internal_idx); }

    /**
     * @brief Returns all external labels ordered by their internal index (0 .. size()-1).
     *
     * In ReadOnlyGraph, this directly returns the immutable labels_ array, which reflects the
     * builder's insertion order. Use this to align external feature arrays (e.g. for custom_features
     * or Searcher rerank_data) into the contiguous internal order expected by C++ methods.
     */
    std::vector<uint32_t> getExternalLabels() const override { return labels_; }

    inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override { return feature_by_index(internal_idx); }
    inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override { return neighbors_by_index(internal_idx); }
    inline const bool hasVertex(const uint32_t external_label) const override { return label_to_index_.contains(external_label); }

    inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
        auto neighbor_indices = getNeighborIndices(internal_index);
        auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;
        return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index);
    }

    // --- Traversal & Search ---
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

  private:
    static const uint8_t alignment = 64;  // 64-byte alignment for SIMD vector access

    inline const uint32_t label_by_index(const uint32_t internal_idx) const { return labels_[internal_idx]; }
    inline const std::byte* feature_by_index(const uint32_t internal_idx) const { return vertices_ + size_t(internal_idx) * vertex_stride_; }
    inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
        return reinterpret_cast<const uint32_t*>(vertices_ + size_t(internal_idx) * vertex_stride_ + neighbor_indices_offset_);
    }

    const uint32_t max_vertex_count_;
    const uint8_t edges_per_vertex_;
    const uint16_t feature_byte_size_;
    const size_t neighbor_indices_offset_;
    const size_t vertex_stride_;

    // Interleaved vertices: 64-byte aligned contiguous buffer, each vertex holds [Features | Neighbors | Padding]
    std::unique_ptr<std::byte[]> vertices_storage_;
    std::byte* vertices_ = nullptr;

    // Labels: contiguous array of external labels
    std::vector<uint32_t> labels_;

    std::unordered_map<uint32_t, uint32_t> label_to_index_;
    const deglib::distances::FloatSpace feature_space_;
    std::unique_ptr<VisitedListPool> visited_list_pool_;
};

/**
 * Load the graph (supports both DEG v2 Container format and legacy v1 format)
 */
inline auto load_readonly_graph(const char* path_graph) { return ReadOnlyGraph::loadGraph(path_graph); }

/**
 * Convert the given graph to a readonly graph
 */
inline auto convert_to_readonly_graph(const deglib::graph::InternalGraph& input_graph) { return deglib::graph::ReadOnlyGraph::from_graph(input_graph); }

/**
 * Convert the given graph to a readonly graph, overriding feature space and feature vectors.
 *
 * @param input_graph Source graph to convert from.
 * @param feature_space Target feature space (dimension and metric).
 * @param custom_features Optional raw byte pointer to new feature vectors in the graph's internal index order (0 .. size()-1, see
 * input_graph.getExternalLabels()).
 */
inline auto convert_to_readonly_graph(
    const deglib::graph::InternalGraph& input_graph,
    const deglib::distances::FloatSpace feature_space,
    const void* custom_features = nullptr
) {
    return deglib::graph::ReadOnlyGraph::from_graph(input_graph, feature_space, custom_features);
}
}  // namespace deglib::graph
