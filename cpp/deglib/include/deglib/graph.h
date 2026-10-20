#pragma once

// Core InternalGraph Interface (0..N-1 internal index)
#include "deglib/graph/dynamic_graph.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/graph/mutable_graph.h"

// Visited List Pool Helper
#include "deglib/graph/visited_list_pool.h"

// Concrete Graph Implementations
#include "deglib/graph/readonly_graph.h"
#include "deglib/graph/sizebounded_graph.h"

#include "deglib/serialization.h"
#include "deglib/transformation.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace deglib {

/**
 * Public DynamicExplorationGraph Facade for End Users.
 * All public methods accept and return external_labels (User Object IDs).
 * Internal calls map transparently between external_label and internal_index.
 */
class DynamicExplorationGraph {
  private:
    std::unique_ptr<deglib::graph::InternalGraph> owned_graph_;
    deglib::graph::InternalGraph* internal_graph_;
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations_;

  public:
    /**
     * @brief Constructs a DynamicExplorationGraph by taking ownership of an InternalGraph.
     *
     * @param graph Unique pointer to the internal graph implementation (e.g. ReadOnlyGraph, SizeBoundedGraph, DynamicGraph).
     * @param transformations Optional sequence of feature transformations/quantizers to apply to query vectors.
     * @throws std::invalid_argument if graph is nullptr.
     */
    explicit DynamicExplorationGraph(
        std::unique_ptr<deglib::graph::InternalGraph> graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    )
        : owned_graph_(std::move(graph)), internal_graph_(owned_graph_.get()), transformations_(std::move(transformations)) {
        if (!internal_graph_) {
            throw std::invalid_argument("DynamicExplorationGraph cannot be initialized with nullptr");
        }
    }

    /**
     * @brief Constructs a non-owning DynamicExplorationGraph viewing an existing InternalGraph.
     *
     * The referenced graph must outlive this DynamicExplorationGraph instance.
     *
     * @param graph Reference to an existing InternalGraph.
     * @param transformations Optional sequence of feature transformations/quantizers to apply to query vectors.
     */
    explicit DynamicExplorationGraph(
        deglib::graph::InternalGraph& graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    )
        : owned_graph_(nullptr), internal_graph_(&graph), transformations_(std::move(transformations)) {}

    DynamicExplorationGraph(DynamicExplorationGraph&&) noexcept = default;
    DynamicExplorationGraph& operator=(DynamicExplorationGraph&&) noexcept = default;

    DynamicExplorationGraph(const DynamicExplorationGraph&) = delete;
    DynamicExplorationGraph& operator=(const DynamicExplorationGraph&) = delete;

    /**
     * @brief Returns the total number of vertices currently stored in the graph.
     * @return Number of vertices.
     */
    const uint32_t size() const { return internal_graph_->size(); }

    /**
     * @brief Returns the fixed regular out-degree (number of outgoing edges) per vertex.
     * @return Edge count per vertex.
     */
    const uint8_t getEdgesPerVertex() const { return internal_graph_->getEdgesPerVertex(); }

    /**
     * @brief Returns the underlying feature space describing data dimensionality and distance metric.
     * @return Const reference to the FloatSpace.
     */
    const deglib::distances::FloatSpace& getFeatureSpace() const { return internal_graph_->getFeatureSpace(); }

    /**
     * @brief Checks whether a vertex with the specified external label exists in the graph.
     * @param external_label External user label / object ID.
     * @return True if the vertex exists, false otherwise.
     */
    bool hasVertex(const uint32_t external_label) const { return internal_graph_->hasVertex(external_label); }

    /**
     * @brief Returns the list of query feature transformations configured for this graph.
     * @return Const reference to vector of shared pointers to FeatureTransformation.
     */
    const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& getTransformations() const noexcept { return transformations_; }

    /**
     * @brief Checks whether any query feature transformations or quantizers are attached.
     * @return True if transformations exist, false otherwise.
     */
    bool hasTransformations() const noexcept { return !transformations_.empty(); }

    /**
     * @brief Returns the expected input dimension of uncompressed query vectors.
     * If transformations are present, returns the input dimension of the first transformation;
     * otherwise returns the feature space dimension.
     * @return Query feature dimensionality.
     */
    size_t getInputDim() const noexcept { return transformations_.empty() ? internal_graph_->getFeatureSpace().dim() : transformations_.front()->input_dim(); }

    /**
     * @brief Access the underlying low-level InternalGraph instance.
     * @return Reference to InternalGraph.
     */
    deglib::graph::InternalGraph& internal() { return *internal_graph_; }

    /**
     * @brief Access the underlying low-level InternalGraph instance (const).
     * @return Const reference to InternalGraph.
     */
    const deglib::graph::InternalGraph& internal() const { return *internal_graph_; }

    /**
     * @brief Checks whether the underlying graph implementation supports mutations (is a MutableGraph).
     * @return True if mutable, false if read-only.
     */
    bool isMutable() const { return dynamic_cast<const deglib::graph::MutableGraph*>(internal_graph_) != nullptr; }

    /**
     * @brief Casts and returns the underlying graph as a MutableGraph reference.
     * @return Reference to MutableGraph.
     * @throws std::invalid_argument if the graph is not mutable (e.g. ReadOnlyGraph).
     */
    deglib::graph::MutableGraph& asMutable() {
        auto* mutable_ptr = dynamic_cast<deglib::graph::MutableGraph*>(internal_graph_);
        if (!mutable_ptr) {
            throw std::invalid_argument("DynamicExplorationGraph must wrap a MutableGraph");
        }
        return *mutable_ptr;
    }

    /**
     * @brief Casts and returns the underlying graph as a MutableGraph const reference.
     * @return Const reference to MutableGraph.
     * @throws std::invalid_argument if the graph is not mutable (e.g. ReadOnlyGraph).
     */
    const deglib::graph::MutableGraph& asMutable() const {
        const auto* mutable_ptr = dynamic_cast<const deglib::graph::MutableGraph*>(internal_graph_);
        if (!mutable_ptr) {
            throw std::invalid_argument("DynamicExplorationGraph must wrap a MutableGraph");
        }
        return *mutable_ptr;
    }

    /**
     * @brief Searches for the k approximate nearest neighbors of a query vector using dynamic exploration radius (eps).
     *
     * Traverses the graph starting from the pre-configured entry vertices. The exploration radius dynamically
     * scales by (1 + eps) relative to the k-th best distance in the candidate set.
     * If feature transformations are configured, the input query is transformed before traversal.
     *
     * @tparam T Element type of the query vector (e.g., float, uint16_t for FP16, or std::byte).
     * @param query Contiguous span of query vector elements.
     * @param k The number of nearest neighbors to return.
     * @param eps Relative exploration expansion margin (default 0.0f). Higher values widen the exploration radius.
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations allowed (0 = unlimited).
     * @return deglib::graph::ResultSet Max-heap of nearest ObjectDistance entries, where identifiers are external labels.
     */
    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                if (query.size() != getInputDim()) {
                    throw std::invalid_argument(std::format("Dimension mismatch: Transformations expect {} dims, query has {}", getInputDim(), query.size()));
                }
                thread_local deglib::transformation::QueryWorkspace workspace;
                const size_t out_bytes = deglib::transformation::get_output_bytes(transformations_);
                std::byte* q_buf = workspace.get_buffer_a(out_bytes);

                if constexpr (std::is_same_v<T, float>) {
                    deglib::transformation::apply_query(transformations_, query.data(), q_buf, 1);
                } else {
                    float* f_query = workspace.get_float_buffer(query.size());
                    deglib::distances::fp16::fp16_to_floats(query.data(), f_query, query.size());
                    deglib::transformation::apply_query(transformations_, f_query, q_buf, 1);
                }

                auto res = internal_graph_->search(std::span<const std::byte>(q_buf, out_bytes), k, eps, filter, max_distance_computation_count);
                for (auto& od : res) {
                    uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                    od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
                }
                return res;
            }
        }

        auto res = internal_graph_->search(query, k, eps, filter, max_distance_computation_count);
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }
        return res;
    }

    /**
     * @brief Searches for the k approximate nearest neighbors of a query vector using a fixed candidate pool size (ef).
     *
     * Traverses the graph maintaining a beam of size ef (beam search). Results are returned in ascending distance order.
     * If feature transformations are configured, the input query is transformed before traversal.
     *
     * @tparam T Element type of the query vector (e.g., float, uint16_t for FP16, or std::byte).
     * @param query Contiguous span of query vector elements.
     * @param k The number of nearest neighbors to return.
     * @param ef Fixed candidate pool / beam width (clamped to at least k).
     * @param include_entry If true, the entry vertex itself is eligible in the result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations allowed (0 = unlimited).
     * @return std::vector<deglib::graph::ObjectDistance> Vector of nearest neighbors sorted ascending by distance (external labels).
     */
    template <typename T>
    std::vector<deglib::graph::ObjectDistance> search_ef(
        std::span<const T> query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                if (query.size() != getInputDim()) {
                    throw std::invalid_argument(std::format("Dimension mismatch: Transformations expect {} dims, query has {}", getInputDim(), query.size()));
                }
                thread_local deglib::transformation::QueryWorkspace workspace;
                const size_t out_bytes = deglib::transformation::get_output_bytes(transformations_);
                std::byte* q_buf = workspace.get_buffer_a(out_bytes);

                if constexpr (std::is_same_v<T, float>) {
                    deglib::transformation::apply_query(transformations_, query.data(), q_buf, 1);
                } else {
                    float* f_query = workspace.get_float_buffer(query.size());
                    deglib::distances::fp16::fp16_to_floats(query.data(), f_query, query.size());
                    deglib::transformation::apply_query(transformations_, f_query, q_buf, 1);
                }

                auto res =
                    internal_graph_->search_ef(std::span<const std::byte>(q_buf, out_bytes), k, ef, include_entry, filter, max_distance_computation_count);
                for (auto& od : res) {
                    uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                    od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
                }
                return res;
            }
        }

        auto res = internal_graph_->search_ef(query, k, ef, include_entry, filter, max_distance_computation_count);
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }
        return res;
    }

    /**
     * @brief Batch nearest neighbor search with dynamic exploration radius (eps), returning a vector of ResultSets.
     *
     * Processes count queries in parallel using worker threads. If feature transformations are configured,
     * queries are transformed in parallel across worker threads with zero per-query heap allocations.
     *
     * @tparam T Element type of the query vectors (e.g., float, uint16_t for FP16, or std::byte).
     * @param queries Contiguous flat span containing all query vectors concatenated (size = count * stride).
     * @param count Number of query vectors in the batch.
     * @param k The number of nearest neighbors to return per query.
     * @param eps Relative exploration expansion margin (default 0.0f).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @return std::vector<deglib::graph::ResultSet> Vector of size count, each containing a max-heap of nearest ObjectDistance entries (external labels).
     * @throws std::invalid_argument if queries.size() != count * stride.
     */
    template <typename T>
    std::vector<deglib::graph::ResultSet> search_batch(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1
    ) const {
        if (count == 0) return {};
        const size_t in_stride = hasTransformations() ? getInputDim() : (internal_graph_->getFeatureSpace().get_data_size() / sizeof(T));
        if (queries.size() != count * in_stride) {
            throw std::invalid_argument(
                std::format("Batch query size mismatch: expected {} * {} = {}, got {}", count, in_stride, count * in_stride, queries.size())
            );
        }

        std::vector<deglib::graph::ResultSet> results(count);

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                const size_t out_stride = deglib::transformation::get_output_bytes(transformations_);

                deglib::concurrent::parallel_for(0, count, threads, [&](size_t i, size_t) {
                    alignas(64) std::byte stack_buf[512];
                    std::byte* q_buf = stack_buf;
                    if (out_stride > sizeof(stack_buf)) {
                        thread_local deglib::transformation::QueryWorkspace ws;
                        q_buf = ws.get_buffer_a(out_stride);
                    }

                    if constexpr (std::is_same_v<T, float>) {
                        deglib::transformation::apply_query(transformations_, queries.data() + i * in_stride, q_buf, 1);
                    } else {
                        alignas(64) float stack_f16_buf[256];
                        float* f16_buf = stack_f16_buf;
                        if (in_stride > sizeof(stack_f16_buf) / sizeof(float)) {
                            thread_local deglib::transformation::QueryWorkspace ws;
                            f16_buf = ws.get_float_buffer(in_stride);
                        }
                        deglib::distances::fp16::fp16_to_floats(queries.data() + i * in_stride, f16_buf, in_stride);
                        deglib::transformation::apply_query(transformations_, f16_buf, q_buf, 1);
                    }

                    std::span<const std::byte> single_q(q_buf, out_stride);
                    auto res = internal_graph_->search(single_q, k, eps, filter, max_distance_computation_count);
                    for (auto& od : res) {
                        uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                        od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
                    }
                    results[i] = std::move(res);
                });
                return results;
            }
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t i, size_t) {
            std::span<const T> single_q(queries.data() + i * in_stride, in_stride);
            auto res = internal_graph_->search(single_q, k, eps, filter, max_distance_computation_count);
            for (auto& od : res) {
                uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
            }
            results[i] = std::move(res);
        });
        return results;
    }

    /**
     * @brief High-performance batch search into pre-allocated flat output buffers using dynamic exploration radius (eps).
     *
     * Avoids dynamic memory allocations per query. Outputs external labels and optionally float distances directly
     * into caller-provided spans.
     *
     * @tparam T Element type of the query vectors (e.g., float, uint16_t for FP16, or std::byte).
     * @param queries Contiguous flat span containing all query vectors concatenated (size = count * stride).
     * @param count Number of query vectors in the batch.
     * @param k The number of nearest neighbors to return per query.
     * @param out_indices Pre-allocated destination span of size >= count * k to receive external labels.
     * @param out_distances Optional destination span of size >= count * k to receive distances. If empty, distances are omitted.
     * @param eps Relative exploration expansion margin (default 0.0f).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param unsorted If false (default), results per query are sorted ascending by distance (nearest first).
     *                 If true, results remain in heap order for maximum throughput.
     * @throws std::invalid_argument if queries, out_indices, or out_distances do not meet size requirements.
     *
     * @note If fewer than k results are found for a query, the remaining slots in that query's block
     *       are padded with std::numeric_limits<uint32_t>::max() in out_indices and
     *       std::numeric_limits<float>::quiet_NaN() in out_distances.
     */
    template <typename T>
    void search_batch(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1,
        bool unsorted = false
    ) const {
        if (count == 0) return;
        if (out_indices.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::search_batch: out_indices size ({}) is smaller than count * k ({})", out_indices.size(), count * k)
            );
        }
        const bool return_distances = !out_distances.empty();
        if (return_distances && out_distances.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::search_batch: out_distances size ({}) is smaller than count * k ({})", out_distances.size(), count * k)
            );
        }
        const size_t in_stride = hasTransformations() ? getInputDim() : (internal_graph_->getFeatureSpace().get_data_size() / sizeof(T));
        if (queries.size() != count * in_stride) {
            throw std::invalid_argument(
                std::format("Batch query size mismatch: expected {} * {} = {}, got {}", count, in_stride, count * in_stride, queries.size())
            );
        }

        auto process_results = [&](size_t q, deglib::graph::ResultSet& res) {
            while (res.size() > k) {
                res.pop();
            }
            const size_t count_res = res.size();
            const size_t base_offset = q * k;
            if (unsorted) {
                for (size_t idx = 0; idx < count_res; ++idx) {
                    out_indices[base_offset + idx] = internal_graph_->getExternalLabel(res[idx].getIdentifier());
                    if (return_distances) {
                        out_distances[base_offset + idx] = res[idx].getDistance();
                    }
                }
            } else {
                for (size_t i = count_res; i > 0; --i) {
                    auto next_result = res.top();
                    out_indices[base_offset + i - 1] = internal_graph_->getExternalLabel(next_result.getIdentifier());
                    if (return_distances) {
                        out_distances[base_offset + i - 1] = next_result.getDistance();
                    }
                    res.pop();
                }
            }
            for (size_t idx = count_res; idx < k; ++idx) {
                out_indices[base_offset + idx] = std::numeric_limits<uint32_t>::max();
                if (return_distances) {
                    out_distances[base_offset + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        };

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                const size_t out_stride = deglib::transformation::get_output_bytes(transformations_);

                deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
                    alignas(64) std::byte stack_buf[512];
                    std::byte* q_buf = stack_buf;
                    if (out_stride > sizeof(stack_buf)) {
                        thread_local deglib::transformation::QueryWorkspace ws;
                        q_buf = ws.get_buffer_a(out_stride);
                    }

                    if constexpr (std::is_same_v<T, float>) {
                        deglib::transformation::apply_query(transformations_, queries.data() + q * in_stride, q_buf, 1);
                    } else {
                        alignas(64) float stack_f16_buf[256];
                        float* f16_buf = stack_f16_buf;
                        if (in_stride > sizeof(stack_f16_buf) / sizeof(float)) {
                            thread_local deglib::transformation::QueryWorkspace ws;
                            f16_buf = ws.get_float_buffer(in_stride);
                        }
                        deglib::distances::fp16::fp16_to_floats(queries.data() + q * in_stride, f16_buf, in_stride);
                        deglib::transformation::apply_query(transformations_, f16_buf, q_buf, 1);
                    }

                    std::span<const std::byte> single_q(q_buf, out_stride);
                    auto res = internal_graph_->search(single_q, k, eps, filter, max_distance_computation_count);
                    process_results(q, res);
                });
                return;
            }
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
            std::span<const T> single_q(queries.data() + q * in_stride, in_stride);
            auto res = internal_graph_->search(single_q, k, eps, filter, max_distance_computation_count);
            process_results(q, res);
        });
    }

    /**
     * @brief High-performance batch search into pre-allocated flat output buffers using fixed beam width (ef).
     *
     * Avoids dynamic memory allocations per query. Outputs external labels and optionally float distances directly
     * into caller-provided spans. Results per query are sorted in ascending distance order.
     *
     * @tparam T Element type of the query vectors (e.g., float, uint16_t for FP16, or std::byte).
     * @param queries Contiguous flat span containing all query vectors concatenated (size = count * stride).
     * @param count Number of query vectors in the batch.
     * @param k The number of nearest neighbors to return per query.
     * @param ef Fixed candidate pool / beam width per query.
     * @param out_indices Pre-allocated destination span of size >= count * k to receive external labels.
     * @param out_distances Optional destination span of size >= count * k to receive distances. If empty, distances are omitted.
     * @param include_entry Whether entry vertex candidates should be eligible in results (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param unsorted If false (default), results per query are ordered nearest first.
     * @throws std::invalid_argument if queries, out_indices, or out_distances do not meet size requirements.
     *
     * @note If fewer than k results are found for a query, unfilled slots are padded with
     *       std::numeric_limits<uint32_t>::max() in out_indices and
     *       std::numeric_limits<float>::quiet_NaN() in out_distances.
     */
    template <typename T>
    void search_batch_ef(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1,
        bool unsorted = false
    ) const {
        if (count == 0) return;
        if (out_indices.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::search_batch_ef: out_indices size ({}) is smaller than count * k ({})", out_indices.size(), count * k)
            );
        }
        const bool return_distances = !out_distances.empty();
        if (return_distances && out_distances.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::search_batch_ef: out_distances size ({}) is smaller than count * k ({})", out_distances.size(), count * k)
            );
        }
        const size_t in_stride = hasTransformations() ? getInputDim() : (internal_graph_->getFeatureSpace().get_data_size() / sizeof(T));
        if (queries.size() != count * in_stride) {
            throw std::invalid_argument(
                std::format("Batch query size mismatch: expected {} * {} = {}, got {}", count, in_stride, count * in_stride, queries.size())
            );
        }

        auto process_results = [&](size_t q, const std::vector<deglib::graph::ObjectDistance>& res) {
            const size_t count_res = std::min(static_cast<size_t>(k), res.size());
            const size_t base_offset = q * k;
            for (size_t idx = 0; idx < count_res; ++idx) {
                out_indices[base_offset + idx] = internal_graph_->getExternalLabel(res[idx].getIdentifier());
                if (return_distances) {
                    out_distances[base_offset + idx] = res[idx].getDistance();
                }
            }
            for (size_t idx = count_res; idx < k; ++idx) {
                out_indices[base_offset + idx] = std::numeric_limits<uint32_t>::max();
                if (return_distances) {
                    out_distances[base_offset + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        };

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                const size_t out_stride = deglib::transformation::get_output_bytes(transformations_);

                deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
                    alignas(64) std::byte stack_buf[512];
                    std::byte* q_buf = stack_buf;
                    if (out_stride > sizeof(stack_buf)) {
                        thread_local deglib::transformation::QueryWorkspace ws;
                        q_buf = ws.get_buffer_a(out_stride);
                    }

                    if constexpr (std::is_same_v<T, float>) {
                        deglib::transformation::apply_query(transformations_, queries.data() + q * in_stride, q_buf, 1);
                    } else {
                        alignas(64) float stack_f16_buf[256];
                        float* f16_buf = stack_f16_buf;
                        if (in_stride > sizeof(stack_f16_buf) / sizeof(float)) {
                            thread_local deglib::transformation::QueryWorkspace ws;
                            f16_buf = ws.get_float_buffer(in_stride);
                        }
                        deglib::distances::fp16::fp16_to_floats(queries.data() + q * in_stride, f16_buf, in_stride);
                        deglib::transformation::apply_query(transformations_, f16_buf, q_buf, 1);
                    }

                    std::span<const std::byte> single_q(q_buf, out_stride);
                    auto res = internal_graph_->search_ef(single_q, k, ef, include_entry, filter, max_distance_computation_count);
                    process_results(q, res);
                });
                return;
            }
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
            std::span<const T> single_q(queries.data() + q * in_stride, in_stride);
            auto res = internal_graph_->search_ef(single_q, k, ef, include_entry, filter, max_distance_computation_count);
            process_results(q, res);
        });
    }

    /**
     * @brief Batch nearest neighbor search with fixed beam width (ef), returning a vector of result vectors.
     *
     * Processes count queries in parallel across worker threads.
     *
     * @tparam T Element type of the query vectors (e.g., float, uint16_t for FP16, or std::byte).
     * @param queries Contiguous flat span containing all query vectors concatenated (size = count * stride).
     * @param count Number of query vectors in the batch.
     * @param k The number of nearest neighbors to return per query.
     * @param ef Fixed candidate pool / beam width per query.
     * @param include_entry Whether entry vertex candidates should be eligible in results (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @return std::vector<std::vector<deglib::graph::ObjectDistance>> A vector of size count, each containing nearest neighbors sorted by distance with
     * external labels.
     * @throws std::invalid_argument if queries.size() does not match count * expected_dimension.
     */
    template <typename T>
    std::vector<std::vector<deglib::graph::ObjectDistance>> search_batch_ef(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1
    ) const {
        if (count == 0) return {};
        std::vector<std::vector<deglib::graph::ObjectDistance>> results(count);
        const size_t in_stride = hasTransformations() ? getInputDim() : (internal_graph_->getFeatureSpace().get_data_size() / sizeof(T));
        if (queries.size() != count * in_stride) {
            throw std::invalid_argument(
                std::format("Batch query size mismatch: expected {} * {} = {}, got {}", count, in_stride, count * in_stride, queries.size())
            );
        }

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                const size_t out_stride = deglib::transformation::get_output_bytes(transformations_);

                deglib::concurrent::parallel_for(0, count, threads, [&](size_t i, size_t) {
                    alignas(64) std::byte stack_buf[512];
                    std::byte* q_buf = stack_buf;
                    if (out_stride > sizeof(stack_buf)) {
                        thread_local deglib::transformation::QueryWorkspace ws;
                        q_buf = ws.get_buffer_a(out_stride);
                    }

                    if constexpr (std::is_same_v<T, float>) {
                        deglib::transformation::apply_query(transformations_, queries.data() + i * in_stride, q_buf, 1);
                    } else {
                        alignas(64) float stack_f16_buf[256];
                        float* f16_buf = stack_f16_buf;
                        if (in_stride > sizeof(stack_f16_buf) / sizeof(float)) {
                            thread_local deglib::transformation::QueryWorkspace ws;
                            f16_buf = ws.get_float_buffer(in_stride);
                        }
                        deglib::distances::fp16::fp16_to_floats(queries.data() + i * in_stride, f16_buf, in_stride);
                        deglib::transformation::apply_query(transformations_, f16_buf, q_buf, 1);
                    }

                    std::span<const std::byte> single_q(q_buf, out_stride);
                    auto res = internal_graph_->search_ef(single_q, k, ef, include_entry, filter, max_distance_computation_count);
                    for (auto& od : res) {
                        uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                        od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
                    }
                    results[i] = std::move(res);
                });
                return results;
            }
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t i, size_t) {
            std::span<const T> single_q(queries.data() + i * in_stride, in_stride);
            auto res = internal_graph_->search_ef(single_q, k, ef, include_entry, filter, max_distance_computation_count);
            for (auto& od : res) {
                uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
                od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
            }
            results[i] = std::move(res);
        });
        return results;
    }

    /**
     * @brief Explores the graph starting at a specific stored vertex using dynamic exploration radius (eps).
     *
     * Traverses the graph from the given entry vertex using its stored feature vector as the query.
     * Maps the entry external label to its internal index and maps all result IDs back to external labels.
     *
     * @param entry_external_label The external label of the vertex to start exploration from.
     * @param k The number of nearest neighbors to return.
     * @param max_distance_computation_count Maximum number of distance calculations allowed (0 = unlimited).
     * @param eps Relative exploration expansion margin (default 0.0f).
     * @param include_entry If true, the entry vertex itself is eligible in the result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @return deglib::graph::ResultSet Max-heap of nearest ObjectDistance entries with external labels.
     */
    deglib::graph::ResultSet explore(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_label);
        auto res = internal_graph_->explore(internal_entry, k, max_distance_computation_count, eps, include_entry, filter);

        // 1. Modify internal vertex IDs to external labels in-place
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }

        return res;
    }

    /**
     * @brief Explores the graph starting at a specific stored vertex using fixed beam width (ef).
     *
     * Traverses the graph from the given entry vertex using its stored feature vector as the query.
     * Maps the entry external label to its internal index and maps all result IDs back to external labels.
     *
     * @param entry_external_label The external label of the vertex to start exploration from.
     * @param k The number of nearest neighbors to return.
     * @param ef Fixed candidate pool / beam width (clamped to at least k).
     * @param max_distance_computation_count Maximum number of distance calculations allowed (0 = unlimited).
     * @param include_entry If true, the entry vertex itself is eligible in the result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @return std::vector<deglib::graph::ObjectDistance> Vector of nearest neighbors sorted ascending by distance with external labels.
     */
    std::vector<deglib::graph::ObjectDistance> explore_ef(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t ef,
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_label);
        auto res = internal_graph_->explore_ef(internal_entry, k, ef, max_distance_computation_count, include_entry, filter);
        for (auto& od : res) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }
        return res;
    }

    /**
     * @brief Batch exploration into pre-allocated flat output buffers using dynamic exploration radius (eps).
     *
     * Starts independent graph traversals from each external label in entry_external_labels in parallel,
     * writing external labels and optionally distances directly into pre-allocated contiguous spans.
     *
     * @param entry_external_labels Contiguous span of starting vertex external labels.
     * @param k The number of nearest neighbors to return per starting vertex.
     * @param out_indices Pre-allocated destination span of size >= entry_external_labels.size() * k.
     * @param out_distances Optional destination span of size >= entry_external_labels.size() * k. If empty, distances are omitted.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param eps Relative exploration expansion margin (default 0.0f).
     * @param include_entry If true, each entry vertex is eligible in its result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param unsorted If false (default), results per query are sorted ascending by distance (nearest first).
     *                 If true, results remain in heap order.
     * @throws std::invalid_argument if out_indices or out_distances do not meet size requirements.
     *
     * @note If fewer than k results are found for an entry, unfilled slots are padded with
     *       std::numeric_limits<uint32_t>::max() in out_indices and
     *       std::numeric_limits<float>::quiet_NaN() in out_distances.
     */
    void explore_batch(
        std::span<const uint32_t> entry_external_labels,
        const uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const {
        const size_t count = entry_external_labels.size();
        if (count == 0) return;
        if (out_indices.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::explore_batch: out_indices size ({}) is smaller than count * k ({})", out_indices.size(), count * k)
            );
        }
        const bool return_distances = !out_distances.empty();
        if (return_distances && out_distances.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::explore_batch: out_distances size ({}) is smaller than count * k ({})", out_distances.size(), count * k)
            );
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
            uint32_t entry_label = entry_external_labels[q];
            auto res = explore(entry_label, k, max_distance_computation_count, eps, include_entry, filter);
            while (res.size() > k) {
                res.pop();
            }
            const size_t count_res = res.size();
            const size_t base_offset = q * k;
            if (unsorted) {
                for (size_t idx = 0; idx < count_res; ++idx) {
                    out_indices[base_offset + idx] = res[idx].getIdentifier();
                    if (return_distances) {
                        out_distances[base_offset + idx] = res[idx].getDistance();
                    }
                }
            } else {
                for (size_t i = count_res; i > 0; --i) {
                    auto next_result = res.top();
                    out_indices[base_offset + i - 1] = next_result.getIdentifier();
                    if (return_distances) {
                        out_distances[base_offset + i - 1] = next_result.getDistance();
                    }
                    res.pop();
                }
            }
            for (size_t idx = count_res; idx < k; ++idx) {
                out_indices[base_offset + idx] = std::numeric_limits<uint32_t>::max();
                if (return_distances) {
                    out_distances[base_offset + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        });
    }

    /**
     * @brief Batch exploration returning std::vector buffers of indices and distances with dynamic exploration radius (eps).
     *
     * Convenient container-allocating wrapper around flat buffer explore_batch.
     *
     * @param entry_external_labels Contiguous span of starting vertex external labels.
     * @param k The number of nearest neighbors to return per starting vertex.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param eps Relative exploration expansion margin (default 0.0f).
     * @param include_entry If true, each entry vertex is eligible in its result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param return_distances If true, calculates and returns distances in the second vector; if false, second vector is empty.
     * @param unsorted If false (default), results per query are sorted ascending by distance.
     * @return std::pair<std::vector<uint32_t>, std::vector<float>> Pair containing flat indices (size = count * k) and distances.
     */
    std::pair<std::vector<uint32_t>, std::vector<float>> explore_batch(
        std::span<const uint32_t> entry_external_labels,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
    ) const {
        const size_t count = entry_external_labels.size();
        std::vector<uint32_t> indices(count * k);
        std::vector<float> distances;
        if (return_distances) {
            distances.resize(count * k);
        }
        explore_batch(
            entry_external_labels, k, std::span<uint32_t>(indices), return_distances ? std::span<float>(distances) : std::span<float>{},
            max_distance_computation_count, eps, include_entry, filter, threads, unsorted
        );
        return {std::move(indices), std::move(distances)};
    }

    /**
     * @brief Batch exploration into pre-allocated flat output buffers using fixed beam width (ef).
     *
     * Starts independent graph traversals from each external label in entry_external_labels in parallel,
     * writing external labels and optionally distances directly into pre-allocated contiguous spans.
     *
     * @param entry_external_labels Contiguous span of starting vertex external labels.
     * @param k The number of nearest neighbors to return per starting vertex.
     * @param ef Fixed candidate pool / beam width (clamped to at least k).
     * @param out_indices Pre-allocated destination span of size >= entry_external_labels.size() * k.
     * @param out_distances Optional destination span of size >= entry_external_labels.size() * k. If empty, distances are omitted.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param include_entry If true, each entry vertex is eligible in its result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param unsorted If false (default), results per query are ordered nearest first.
     * @throws std::invalid_argument if out_indices or out_distances do not meet size requirements.
     *
     * @note If fewer than k results are found for an entry, unfilled slots are padded with
     *       std::numeric_limits<uint32_t>::max() in out_indices and
     *       std::numeric_limits<float>::quiet_NaN() in out_distances.
     */
    void explore_batch_ef(
        std::span<const uint32_t> entry_external_labels,
        const uint32_t k,
        const uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const {
        const size_t count = entry_external_labels.size();
        if (count == 0) return;
        if (out_indices.size() < count * k) {
            throw std::invalid_argument(
                std::format("DynamicExplorationGraph::explore_batch_ef: out_indices size ({}) is smaller than count * k ({})", out_indices.size(), count * k)
            );
        }
        const bool return_distances = !out_distances.empty();
        if (return_distances && out_distances.size() < count * k) {
            throw std::invalid_argument(
                std::format(
                    "DynamicExplorationGraph::explore_batch_ef: out_distances size ({}) is smaller than count * k ({})", out_distances.size(), count * k
                )
            );
        }

        deglib::concurrent::parallel_for(0, count, threads, [&](size_t q, size_t) {
            uint32_t entry_label = entry_external_labels[q];
            auto res = explore_ef(entry_label, k, ef, max_distance_computation_count, include_entry, filter);
            const size_t count_res = std::min(static_cast<size_t>(k), res.size());
            const size_t base_offset = q * k;
            for (size_t idx = 0; idx < count_res; ++idx) {
                out_indices[base_offset + idx] = res[idx].getIdentifier();
                if (return_distances) {
                    out_distances[base_offset + idx] = res[idx].getDistance();
                }
            }
            for (size_t idx = count_res; idx < k; ++idx) {
                out_indices[base_offset + idx] = std::numeric_limits<uint32_t>::max();
                if (return_distances) {
                    out_distances[base_offset + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        });
    }

    /**
     * @brief Batch exploration returning std::vector buffers of indices and distances with fixed beam width (ef).
     *
     * Convenient container-allocating wrapper around flat buffer explore_batch_ef.
     *
     * @param entry_external_labels Contiguous span of starting vertex external labels.
     * @param k The number of nearest neighbors to return per starting vertex.
     * @param ef Fixed candidate pool / beam width (clamped to at least k).
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param include_entry If true, each entry vertex is eligible in its result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param threads Number of worker threads for parallel batch execution (1 = sequential).
     * @param return_distances If true, calculates and returns distances in the second vector; if false, second vector is empty.
     * @param unsorted If false (default), results per query are sorted ascending by distance.
     * @return std::pair<std::vector<uint32_t>, std::vector<float>> Pair containing flat indices (size = count * k) and distances.
     */
    std::pair<std::vector<uint32_t>, std::vector<float>> explore_batch_ef(
        std::span<const uint32_t> entry_external_labels,
        const uint32_t k,
        const uint32_t ef,
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
    ) const {
        const size_t count = entry_external_labels.size();
        std::vector<uint32_t> indices(count * k);
        std::vector<float> distances;
        if (return_distances) {
            distances.resize(count * k);
        }
        explore_batch_ef(
            entry_external_labels, k, ef, std::span<uint32_t>(indices), return_distances ? std::span<float>(distances) : std::span<float>{},
            max_distance_computation_count, include_entry, filter, threads, unsorted
        );
        return {std::move(indices), std::move(distances)};
    }

    /**
     * Greedy best-first reachability search towards a stored vertex.
     *
     * Traverses the graph from the given entry vertices, using the target vertex's own feature vector as
     * the query, and backtracks the chain of vertices that reached the target. With eps = 0.0 and k = 1
     * this degenerates into a greedy descent; larger eps widen the exploration radius past local optima.
     *
     * @param entry_external_labels The external labels of the vertices to start the traversal from.
     * @param to_external_label The external label of the vertex the path should end at.
     * @param eps Exploration factor widening the search radius. 0.0 restricts the traversal to a greedy descent.
     * @param k Size of the result list driving the exploration radius.
     * @return A vector of ObjectDistance objects ordered from the target back to the entry, holding external
     *         labels and distances to the target. Empty if the target was not reached within the budget.
     */
    std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_external_labels, const uint32_t to_external_label, const float eps = 0.0f, const uint32_t k = 1) const {
        std::vector<uint32_t> entry_indices;
        entry_indices.reserve(entry_external_labels.size());
        for (const uint32_t entry_external_label : entry_external_labels) {
            entry_indices.push_back(internal_graph_->getInternalIndex(entry_external_label));
        }

        auto path = internal_graph_->hasPath(entry_indices, internal_graph_->getInternalIndex(to_external_label), eps, k);

        // Modify internal vertex IDs to external labels in-place
        for (auto& od : path) {
            uint32_t ext_label = internal_graph_->getExternalLabel(od.getIdentifier());
            od = deglib::graph::ObjectDistance(ext_label, od.getDistance());
        }

        return path;
    }

    /**
     * @brief Retrieves the neighbor external labels of a given vertex.
     *
     * @param external_label The external label of the vertex whose neighbors to retrieve.
     * @return std::vector<uint32_t> Vector containing the external labels of the vertex's neighbors.
     */
    std::vector<uint32_t> getNeighbors(const uint32_t external_label) const {
        uint32_t internal_idx = internal_graph_->getInternalIndex(external_label);
        const uint32_t* internal_neighbors = internal_graph_->getNeighborIndices(internal_idx);
        uint8_t edges_count = internal_graph_->getEdgesPerVertex();
        std::vector<uint32_t> external_neighbors(edges_count);
        for (size_t i = 0; i < edges_count; ++i) {
            external_neighbors[i] = internal_graph_->getExternalLabel(internal_neighbors[i]);
        }
        return external_neighbors;
    }

    /**
     * @brief Batch retrieval of neighbor external labels into a pre-allocated flat span.
     *
     * Retrieves neighbors for multiple vertices in parallel without allocating memory.
     *
     * @param external_labels Contiguous span of query vertex external labels.
     * @param out_neighbors Flat span of size >= external_labels.size() * getEdgesPerVertex() to receive neighbor labels.
     * @param threads Number of worker threads for parallel processing (1 = sequential).
     * @throws std::invalid_argument if out_neighbors.size() < external_labels.size() * getEdgesPerVertex().
     */
    void getNeighbors_batch(std::span<const uint32_t> external_labels, std::span<uint32_t> out_neighbors, size_t threads = 1) const {
        const size_t count = external_labels.size();
        if (count == 0) return;
        const uint8_t edges_count = internal_graph_->getEdgesPerVertex();
        if (out_neighbors.size() < count * edges_count) {
            throw std::invalid_argument(
                std::format(
                    "DynamicExplorationGraph::getNeighbors_batch: out_neighbors size ({}) is smaller than count * edges ({})", out_neighbors.size(),
                    count * edges_count
                )
            );
        }

        deglib::concurrent::parallel_batch_for(0, count, threads, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                uint32_t internal_idx = internal_graph_->getInternalIndex(external_labels[i]);
                const uint32_t* internal_neighbors = internal_graph_->getNeighborIndices(internal_idx);
                const size_t base_offset = i * edges_count;
                for (size_t j = 0; j < edges_count; ++j) {
                    out_neighbors[base_offset + j] = internal_graph_->getExternalLabel(internal_neighbors[j]);
                }
            }
        });
    }

    /**
     * @brief Batch retrieval of neighbor external labels returning a flat std::vector.
     *
     * @param external_labels Contiguous span of query vertex external labels.
     * @param threads Number of worker threads for parallel processing (1 = sequential).
     * @return std::vector<uint32_t> Flat vector of size external_labels.size() * getEdgesPerVertex() with neighbor external labels.
     */
    std::vector<uint32_t> getNeighbors_batch(std::span<const uint32_t> external_labels, size_t threads = 1) const {
        const uint8_t edges_count = internal_graph_->getEdgesPerVertex();
        std::vector<uint32_t> result(external_labels.size() * edges_count);
        getNeighbors_batch(external_labels, std::span<uint32_t>(result), threads);
        return result;
    }

    /**
     * @brief Converts this graph to a compact read-only graph optimized for search.
     *
     * @return A new read-only DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_readonly() const {
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(
            internal_graph_->size(), internal_graph_->getEdgesPerVertex(), internal_graph_->getFeatureSpace(), *internal_graph_
        );
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a compact read-only graph with a custom feature space and features buffer.
     *
     * Replaces feature representation (e.g. uncompressed float to quantized bytes or FP16).
     *
     * @param custom_feature_space The new target feature space.
     * @param custom_features Pointer to replacement raw feature memory (size = size() * custom_feature_space.get_data_size()).
     * @return A new read-only DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_readonly(const deglib::distances::FloatSpace& custom_feature_space, const void* custom_features = nullptr) const {
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(
            internal_graph_->size(), internal_graph_->getEdgesPerVertex(), custom_feature_space, *internal_graph_, custom_features
        );
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a mutable SizeBoundedGraph with a fixed capacity.
     *
     * @param new_max_size Desired vertex capacity. If 0 or less than size(), capacity is set to size().
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_mutable(const uint32_t new_max_size = 0) const {
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(deglib::graph::SizeBoundedGraph::from_graph(*internal_graph_, new_max_size));
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a mutable SizeBoundedGraph with a custom feature space and features buffer.
     *
     * Replaces feature representation (e.g. uncompressed float to quantized bytes or FP16) and recomputes all edge weights.
     *
     * @param custom_feature_space The new target feature space.
     * @param custom_features Pointer to replacement raw feature memory (size = size() * custom_feature_space.get_data_size()).
     * @param new_max_size Desired vertex capacity. If 0 or less than size(), capacity is set to size().
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph
    to_mutable(const deglib::distances::FloatSpace& custom_feature_space, const void* custom_features = nullptr, const uint32_t new_max_size = 0) const {
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(
            deglib::graph::SizeBoundedGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, new_max_size)
        );
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a mutable DynamicGraph with chunked dynamic memory allocation.
     *
     * Allows unbounded growth and vertex removals using chunked vertex buffers.
     *
     * @param chunk_size Number of vertices allocated per memory chunk (default 1024). Rounded to power of 2.
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_dynamic(const uint32_t chunk_size = 1024) const {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(deglib::graph::DynamicGraph::from_graph(*internal_graph_, chunk_size));
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a mutable DynamicGraph with a custom feature space and chunked allocation.
     *
     * @param custom_feature_space The new target feature space.
     * @param custom_features Pointer to replacement raw feature memory.
     * @param chunk_size Number of vertices allocated per memory chunk (default 1024). Rounded to power of 2.
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph
    to_dynamic(const deglib::distances::FloatSpace& custom_feature_space, const void* custom_features = nullptr, const uint32_t chunk_size = 1024) const {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(
            deglib::graph::DynamicGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, chunk_size)
        );
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Returns the list of internal entry vertex indices used as starting points for global searches.
     * @return Const reference to internal entry vertex indices.
     */
    const std::vector<uint32_t>& getEntryVertexIndices() const { return internal_graph_->getEntryVertexIndices(); }

    /**
     * @brief Sets the internal entry vertex indices used as starting points for global searches.
     * @param indices Vector of internal entry vertex indices.
     */
    void setEntryVertexIndices(std::vector<uint32_t> indices) { internal_graph_->setEntryVertexIndices(std::move(indices)); }

    /**
     * @brief Returns the feature vector prefetch offset (candidates prefetched ahead during search).
     */
    int32_t getPo() const { return internal_graph_->getPo(); }

    /**
     * @brief Returns the feature vector prefetch look-ahead (number of cache lines per feature).
     */
    int32_t getPl() const { return internal_graph_->getPl(); }

    /**
     * @brief Returns the neighbor index prefetch look-ahead.
     */
    int32_t getNl() const { return internal_graph_->getNl(); }

    /**
     * @brief Sets the feature vector prefetch offset.
     * @param po Prefetch offset (must be > 0).
     */
    void setPo(int32_t po) { internal_graph_->setPo(po); }

    /**
     * @brief Sets the feature vector prefetch look-ahead.
     * @param pl Cache lines to prefetch (must be > 0).
     */
    void setPl(int32_t pl) { internal_graph_->setPl(pl); }

    /**
     * @brief Sets the neighbor index prefetch look-ahead.
     * @param nl Neighbor prefetch look-ahead (must be > 0).
     */
    void setNl(int32_t nl) { internal_graph_->setNl(nl); }

    /**
     * @brief Configures software prefetching parameters for graph traversal.
     *
     * @param po Prefetch offset for candidate feature vectors.
     * @param pl Cache line look-ahead for feature vectors.
     * @param nl Look-ahead for neighbor list prefetching (default 3).
     */
    void setPrefetch(int32_t po, int32_t pl, int32_t nl = 3) { internal_graph_->setPrefetch(po, pl, nl); }

    /**
     * @brief Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by empirically timing
     * traversal over sampled vertices. The graph self-samples its own stored features as queries,
     * so no external query buffer is needed. Forwards to the internal graph's optimize().
     *
     * @param sample_count  Number of vertices sampled as queries (capped at the graph size).
     * @param k             Result count per traversal (the expected query operating point).
     * @param ef            Beam width per traversal; clamped to at least k.
     * @param seed          Random seed for query sampling.
     */
    void optimize(size_t sample_count = 50, uint32_t k = 100, uint32_t ef = 200, uint32_t seed = 7) const {
        internal_graph_->optimize(sample_count, k, ef, seed);
    }

    /**
     * @brief Saves the graph and any attached transformations to a binary file.
     *
     * @param path File path to write to (C-string).
     * @return True if save succeeded, false otherwise.
     */
    bool saveGraph(const char* path) const {
        return deglib::serialization::save_graph(path, *internal_graph_, transformations_.empty() ? nullptr : &transformations_);
    }

    /**
     * @brief Saves the graph and any attached transformations to a binary file.
     *
     * @param path File path to write to (std::string).
     * @return True if save succeeded, false otherwise.
     */
    bool saveGraph(const std::string& path) const { return saveGraph(path.c_str()); }

    /**
     * @brief Loads a graph from a binary file into a read-only DynamicExplorationGraph.
     *
     * Reads graph topology, feature space, stored features, entry vertices, prefetch parameters,
     * and any embedded feature transformations/quantizers.
     *
     * @param path File path to read from (C-string).
     * @return Loaded DynamicExplorationGraph.
     */
    static DynamicExplorationGraph loadGraph(const char* path) {
        auto data = deglib::serialization::read_graph_file(path);
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(data.size, data.edges_per_vertex, std::move(*data.feature_space), data.vertex_bytes);
        graph->setPrefetch(data.po, data.pl, data.nl);
        graph->setEntryVertexIndices(std::move(data.entry_vertex_indices));
        return DynamicExplorationGraph(std::move(graph), std::move(data.transformations));
    }

    /**
     * @brief Loads a graph from a binary file into a read-only DynamicExplorationGraph.
     *
     * @param path File path to read from (std::string).
     * @return Loaded DynamicExplorationGraph.
     */
    static DynamicExplorationGraph loadGraph(const std::string& path) { return loadGraph(path.c_str()); }
};

}  // namespace deglib
