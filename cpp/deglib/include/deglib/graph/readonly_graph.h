#pragma once

#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/serialization.h"
#include "deglib/transformation.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace deglib::graph {

/**
 * Read-only graph with a split storage layout:
 *
 * - features are stored in one contiguous array with a padded stride
 * - edges are stored in one contiguous int32_t array
 *
 * The layout keeps distance access separate from neighbor metadata. Feature rows start at
 * the graph feature base plus id * feature stride.
 */
class ReadOnlyGraph : public InternalGraph {
  public:
    friend class InternalGraph;

    ReadOnlyGraph() = default;

    ReadOnlyGraph(uint32_t count, uint8_t edges_per_vertex, const deglib::distances::FloatSpace& space) {
        if (count == 0 || edges_per_vertex == 0) {
            throw std::invalid_argument("ReadOnlyGraph requires count and edges_per_vertex > 0");
        }

        const size_t feature_bytes = space.get_data_size();
        if (feature_bytes == 0) {
            throw std::invalid_argument("ReadOnlyGraph feature size must be > 0");
        }

        vertex_count_ = count;
        edges_per_vertex_ = edges_per_vertex;
        feature_dim_ = static_cast<int32_t>(space.dim());
        feature_byte_size_ = feature_bytes;
        feature_stride_ = deglib::memory::align_up(feature_bytes, deglib::memory::kCacheLineAlignment);

        const size_t feature_total = size_t(vertex_count_) * feature_stride_;
        const size_t edge_total = size_t(vertex_count_) * size_t(edges_per_vertex_) * sizeof(int32_t);

        deglib::memory::allocate_aligned(features_storage_, features_, feature_total);
        deglib::memory::allocate_aligned(edges_storage_, edges_, edge_total);

        std::memset(features_, 0, feature_total);
        std::memset(edges_, 0xff, edge_total);

        feature_space_ = space;
        labels_.resize(count);
        std::iota(labels_.begin(), labels_.end(), uint32_t{0});
        label_to_index_.clear();
        label_to_index_.reserve(count);
        for (const auto label : labels_) {
            label_to_index_.emplace(label, label);
        }
        entry_vertex_indices_ = {0};
        visited_list_pool_ = std::make_unique<VisitedListPool>(1, count);
        po_ = 4;
        pl_ = 2;
        nl_ = 1;
    }

    ReadOnlyGraph(const ReadOnlyGraph&) = delete;
    ReadOnlyGraph& operator=(const ReadOnlyGraph&) = delete;

    ReadOnlyGraph(ReadOnlyGraph&& other) noexcept = default;
    ReadOnlyGraph& operator=(ReadOnlyGraph&& other) noexcept = default;

    [[nodiscard]] static ReadOnlyGraph from_graph(
        const InternalGraph& input_graph,
        std::optional<deglib::distances::FloatSpace> target_feature_space = std::nullopt,
        const void* custom_features = nullptr,
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t num_threads = 0
    ) {
        const auto resolved_feature_space = deglib::transformation::resolve_graph_conversion_space(
            input_graph.getFeatureSpace(), target_feature_space, custom_features, transformations
        );

        ReadOnlyGraph graph(input_graph.size(), input_graph.getEdgesPerVertex(), resolved_feature_space);

        const size_t target_feature_bytes = graph.feature_space_.get_data_size();
        const std::byte* custom_source_features = custom_features == nullptr ? nullptr : static_cast<const std::byte*>(custom_features);

        const auto source_metric = input_graph.getFeatureSpace().metric();
        const uint32_t source_dim = input_graph.getFeatureSpace().dim();
        const bool self_transform = (custom_features == nullptr) && !transformations.empty() && (source_metric != graph.feature_space_.metric());

        graph.label_to_index_.clear();
        graph.label_to_index_.reserve(graph.vertex_count_);
        for (uint32_t i = 0; i < graph.vertex_count_; ++i) {
            const auto label = input_graph.getExternalLabel(i);
            graph.labels_[i] = label;
            graph.label_to_index_.emplace(label, i);

            if (!self_transform) {
                std::byte* target_features = graph.features_ + size_t(i) * graph.feature_stride_;
                const std::byte* source_features =
                    custom_source_features == nullptr ? input_graph.getFeatureVector(i) : custom_source_features + size_t(i) * target_feature_bytes;
                std::memcpy(target_features, source_features, target_feature_bytes);
            }

            const uint32_t* neighbor_indices = input_graph.getNeighborIndices(i);
            int32_t* edge_row = graph.edges_ + size_t(i) * graph.edges_per_vertex_;
            for (uint32_t j = 0; j < graph.edges_per_vertex_; ++j) {
                const uint32_t neighbor_index = neighbor_indices[j];
                edge_row[j] = static_cast<int32_t>(neighbor_index < graph.vertex_count_ ? neighbor_index : i);
            }
        }

        if (self_transform && graph.vertex_count_ > 0) {
            const bool is_fp16 = (source_metric.get_data_type() == deglib::distances::MetricDataType::FP16);
            auto transform_vertex = [&](uint32_t i, std::vector<float>& float_buffer) {
                std::byte* target_features = graph.features_ + size_t(i) * graph.feature_stride_;
                const auto* input_features = input_graph.getFeatureVector(i);
                if (!is_fp16) {
                    deglib::transformation::apply_on_database(
                        transformations,
                        std::span<const float>(reinterpret_cast<const float*>(input_features), source_dim),
                        std::span<std::byte>(target_features, target_feature_bytes),
                        1
                    );
                } else {
                    deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(input_features), float_buffer.data(), source_dim);
                    deglib::transformation::apply_on_database(
                        transformations,
                        std::span<const float>(float_buffer),
                        std::span<std::byte>(target_features, target_feature_bytes),
                        1
                    );
                }
            };

            if (num_threads == 1 || graph.vertex_count_ == 1) {
                std::vector<float> float_buffer(is_fp16 ? source_dim : 0);
                for (uint32_t i = 0; i < graph.vertex_count_; ++i) {
                    transform_vertex(i, float_buffer);
                }
            } else {
                deglib::concurrent::parallel_batch_for(
                    uint32_t{0}, graph.vertex_count_, num_threads,
                    [&](uint32_t begin, uint32_t end, size_t) {
                        std::vector<float> float_buffer(is_fp16 ? source_dim : 0);
                        for (uint32_t i = begin; i < end; ++i) {
                            transform_vertex(i, float_buffer);
                        }
                    },
                    64
                );
            }
        }

        graph.setEntryVertexIndices(input_graph.getEntryVertexIndices());
        graph.setPrefetch(input_graph.getPo(), input_graph.getPl(), input_graph.getNl());
        return graph;
    }

    [[nodiscard]] static ReadOnlyGraph loadGraph(const char* path_graph) {
        auto [is, header] = deglib::serialization::open_graph_file(path_graph);
        ReadOnlyGraph graph(header.size, header.edges_per_vertex, *header.feature_space);
        graph.label_to_index_.clear();
        graph.label_to_index_.reserve(header.size);

        const size_t feature_bytes = graph.feature_space_.get_data_size();
        const size_t neighbor_bytes = size_t(header.edges_per_vertex) * sizeof(uint32_t);
        const size_t weight_bytes = size_t(header.edges_per_vertex) * sizeof(float);
        const size_t record_bytes = feature_bytes + neighbor_bytes + weight_bytes + sizeof(uint32_t);

        constexpr size_t batch_size = 256;
        std::vector<char> batch(batch_size * record_bytes);

        for (uint32_t offset = 0; offset < header.size; offset += batch_size) {
            const uint32_t current_batch = std::min<uint32_t>(batch_size, header.size - offset);
            const size_t read_bytes = size_t(current_batch) * record_bytes;
            if (!is.read(batch.data(), read_bytes)) {
                throw std::runtime_error("ReadOnlyGraph::loadGraph: unexpected end of stream reading vertices");
            }

            const char* ptr = batch.data();
            for (uint32_t b = 0; b < current_batch; ++b) {
                const uint32_t i = offset + b;
                std::memcpy(graph.features_ + size_t(i) * graph.feature_stride_, ptr, feature_bytes);
                ptr += feature_bytes;

                int32_t* edge_row = graph.edges_ + size_t(i) * graph.edges_per_vertex_;
                const auto* neighbor_indices = reinterpret_cast<const uint32_t*>(ptr);
                for (uint32_t j = 0; j < graph.edges_per_vertex_; ++j) {
                    const uint32_t neighbor_index = neighbor_indices[j];
                    edge_row[j] = static_cast<int32_t>(neighbor_index < graph.vertex_count_ ? neighbor_index : i);
                }
                ptr += neighbor_bytes;
                ptr += weight_bytes;

                uint32_t label = 0;
                std::memcpy(&label, ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                graph.labels_[i] = label;
                graph.label_to_index_.emplace(label, i);
            }
        }

        graph.setPrefetch(header.po, header.pl, header.nl);
        graph.setEntryVertexIndices(header.entry_vertex_indices);
        return graph;
    }

    [[nodiscard]] uint32_t size() const noexcept override { return vertex_count_; }
    [[nodiscard]] uint8_t getEdgesPerVertex() const noexcept override { return edges_per_vertex_; }
    [[nodiscard]] const deglib::distances::FloatSpace& getFeatureSpace() const noexcept override { return feature_space_; }

    [[nodiscard]] const std::byte* getFeatureVector(uint32_t id) const noexcept override { return feature_by_index(id); }
    [[nodiscard]] const uint32_t* getNeighborIndices(uint32_t id) const noexcept override { return neighbors_by_index(id); }
    [[nodiscard]] uint32_t getExternalLabel(uint32_t id) const noexcept override { return labels_[id]; }
    [[nodiscard]] uint32_t getInternalIndex(uint32_t label) const noexcept override {
        if (label < vertex_count_ && labels_[label] == label) [[likely]] {
            return label;
        }
        return label_to_index_.find(label)->second;
    }
    [[nodiscard]] std::vector<uint32_t> getExternalLabels() const override { return labels_; }

    [[nodiscard]] bool hasVertex(uint32_t label) const noexcept override {
        if (label < vertex_count_ && labels_[label] == label) [[likely]] {
            return true;
        }
        return label_to_index_.find(label) != label_to_index_.end();
    }
    [[nodiscard]] bool hasEdge(uint32_t internal_index, uint32_t neighbor_index) const noexcept override {
        const auto neighbor_indices = std::span(getNeighborIndices(internal_index), edges_per_vertex_);
        return std::ranges::binary_search(neighbor_indices, neighbor_index);
    }

    [[nodiscard]] const int32_t* edges(uint32_t id) const noexcept { return edges_ + size_t(id) * edges_per_vertex_; }
    [[nodiscard]] int32_t at(uint32_t id, uint32_t pos) const noexcept { return edges_[size_t(id) * edges_per_vertex_ + pos]; }

    [[nodiscard]] std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_vertex_indices, uint32_t to_vertex, float eps, uint32_t k) const override {
        return hasPathImpl(*this, entry_vertex_indices, to_vertex, eps, k);
    }

  protected:
    std::vector<deglib::graph::ObjectDistance> search_ef_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        uint32_t k,
        uint32_t ef,
        bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const override {
        return searchEfInternImpl(*this, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count);
    }

    deglib::graph::ResultSet search_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        uint32_t k,
        float eps = 0.0f,
        bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const override {
        return searchInternImpl(*this, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count);
    }

  private:
    [[nodiscard]] uint32_t label_by_index(uint32_t id) const noexcept { return labels_[id]; }

    [[nodiscard]] const std::byte* feature_by_index(uint32_t id) const noexcept { return features_ + size_t(id) * feature_stride_; }

    [[nodiscard]] const uint32_t* neighbors_by_index(uint32_t id) const noexcept {
        return reinterpret_cast<const uint32_t*>(edges_ + size_t(id) * edges_per_vertex_);
    }

    uint32_t vertex_count_ = 0;
    uint8_t edges_per_vertex_ = 0;
    int32_t feature_dim_ = 0;
    size_t feature_byte_size_ = 0;
    size_t feature_stride_ = 0;

    std::unique_ptr<std::byte[]> features_storage_;
    std::unique_ptr<std::byte[]> edges_storage_;

    std::byte* features_ = nullptr;
    int32_t* edges_ = nullptr;

    // Labels: contiguous array of external labels
    std::vector<uint32_t> labels_;
    std::unordered_map<uint32_t, uint32_t> label_to_index_;

    deglib::distances::FloatSpace feature_space_;
    std::unique_ptr<VisitedListPool> visited_list_pool_;
};

}  // namespace deglib::graph
