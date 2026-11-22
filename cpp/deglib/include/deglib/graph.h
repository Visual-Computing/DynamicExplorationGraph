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

#include "deglib/search/recall_table.h"
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
     * @brief Returns all external labels ordered by their internal index (0 .. size()-1).
     *
     * In static graphs, this matches the builder's insertion order. In dynamic graphs where vertices
     * have been deleted, this reflects the current internal layout after compacting swaps.
     *
     * Use this vector to align external feature arrays (e.g. for to_readonly custom_features or
     * Searcher rerank_data) into the contiguous internal order expected by C++ methods.
     *
     * @return Vector of external labels of size size(), where element i is at internal index i.
     */
    std::vector<uint32_t> getExternalLabels() const { return internal_graph_->getExternalLabels(); }

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
     * @brief Set or replace the query feature transformations for this graph.
     */
    void setTransformations(std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations) {
        transformations_ = std::move(transformations);
    }

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
     * @param return_distances If true (default), includes distances in the returned `SearchResults`.
     * @param unsorted If true, leaves candidates in heap order instead of sorting ascending by distance.
     * @return deglib::search::SearchResults Neighbor external labels, optional distances, and traversal telemetry.
     */
    template <typename T>
    deglib::search::SearchResults search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        const bool return_distances = true,
        const bool unsorted = false,
        const bool pad_missing = false
    ) const {
        thread_local deglib::graph::SearchWorkspace search_ws;
        run_with_prepared_query(query, [&](auto prepared_q) {
            internal_graph_->search(search_ws, prepared_q, k, eps, filter, max_distance_computation_count);
        });
        return drain_to_single_results(search_ws, k, return_distances, unsorted, pad_missing);
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
     * @param return_distances If true (default), includes distances in the returned `SearchResults`.
     * @return deglib::search::SearchResults Neighbor external labels, optional distances, and traversal telemetry.
     */
    template <typename T>
    deglib::search::SearchResults search_ef(
        std::span<const T> query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        const bool return_distances = true,
        const bool pad_missing = false
    ) const {
        thread_local deglib::graph::SearchWorkspace search_ws;
        run_with_prepared_query(query, [&](auto prepared_q) {
            internal_graph_->search_ef(search_ws, prepared_q, k, ef, include_entry, filter, max_distance_computation_count);
        });
        return drain_to_single_results(search_ws, k, return_distances, /*unsorted=*/false, pad_missing);
    }

    /**
     * @brief Batch nearest neighbor search with dynamic exploration radius (eps), returning `SearchResults`.
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
     * @param threads Number of worker threads for parallel batch execution (0 = hardware concurrency, 1 = sequential).
     * @param return_distances If true (default), includes distances in `SearchResults`.
     * @param unsorted If true, leaves candidates in heap order instead of sorting ascending.
     * @return deglib::search::SearchResults Flat batch indices, optional distances, and aggregated telemetry.
     * @throws std::invalid_argument if queries.size() != count * stride.
     */
    template <typename T>
    deglib::search::SearchResults search_batch(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1,
        const bool return_distances = true,
        const bool unsorted = false
    ) const {
        deglib::search::SearchResults res;
        res.n_queries = count;
        res.k = k;
        res.is_batch = true;
        res.with_distances = return_distances;
        res.indices.resize(count * k);
        if (return_distances) {
            res.distances.resize(count * k);
        }
        execute_search_batch_impl<T, false>(
            queries, count, k, eps, 0, std::span<uint32_t>(res.indices), return_distances ? std::span<float>(res.distances) : std::span<float>{}, true, filter,
            max_distance_computation_count, threads, unsorted, res.graph_distance_count, res.graph_hop_count
        );
        return res;
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
     * @param threads Number of worker threads for parallel batch execution (0 = hardware concurrency, 1 = sequential).
     * @param unsorted If false (default), results per query are sorted ascending by distance (nearest first).
     *                 If true, results remain in heap order for maximum throughput.
     * @throws std::invalid_argument if queries, out_indices, or out_distances do not meet size requirements.
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
        uint64_t dummy_dists = 0;
        uint64_t dummy_hops = 0;
        execute_search_batch_impl<T, false>(
            queries, count, k, eps, 0, out_indices, out_distances, true, filter, max_distance_computation_count, threads, unsorted, dummy_dists, dummy_hops
        );
    }

    /**
     * @brief High-performance batch search into pre-allocated flat output buffers using fixed beam width (ef).
     *
     * @copydetails search_batch(std::span<const T>, size_t, uint32_t, std::span<uint32_t>, std::span<float>, float, const deglib::search::Filter*, uint32_t,
     * size_t, bool) const
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
        uint64_t dummy_dists = 0;
        uint64_t dummy_hops = 0;
        execute_search_batch_impl<T, true>(
            queries, count, k, 0.0f, ef, out_indices, out_distances, include_entry, filter, max_distance_computation_count, threads, unsorted, dummy_dists,
            dummy_hops
        );
    }

    /**
     * @brief Batch nearest neighbor search with fixed beam width (ef), returning `SearchResults`.
     *
     * @tparam T Element type of the query vectors (e.g., float, uint16_t for FP16, or std::byte).
     * @param queries Contiguous flat span containing all query vectors concatenated (size = count * stride).
     * @param count Number of query vectors in the batch.
     * @param k The number of nearest neighbors to return per query.
     * @param ef Fixed candidate pool / beam width per query.
     * @param include_entry Whether entry vertex candidates should be eligible in results (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param max_distance_computation_count Maximum number of distance calculations per query (0 = unlimited).
     * @param threads Number of worker threads for parallel batch execution (0 = hardware concurrency, 1 = sequential).
     * @param return_distances If true (default), includes distances in `SearchResults`.
     * @param unsorted If false (default), results are sorted ascending by distance.
     * @return deglib::search::SearchResults Flat batch indices, optional distances, and aggregated telemetry.
     */
    template <typename T>
    deglib::search::SearchResults search_batch_ef(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0,
        size_t threads = 1,
        const bool return_distances = true,
        const bool unsorted = false
    ) const {
        deglib::search::SearchResults res;
        res.n_queries = count;
        res.k = k;
        res.is_batch = true;
        res.with_distances = return_distances;
        res.indices.resize(count * k);
        if (return_distances) {
            res.distances.resize(count * k);
        }
        execute_search_batch_impl<T, true>(
            queries, count, k, 0.0f, ef, std::span<uint32_t>(res.indices), return_distances ? std::span<float>(res.distances) : std::span<float>{},
            include_entry, filter, max_distance_computation_count, threads, unsorted, res.graph_distance_count, res.graph_hop_count
        );
        return res;
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
     * @param return_distances If true (default), includes distances in `SearchResults`.
     * @param unsorted If true, leaves candidates in heap order instead of sorting ascending.
     * @return deglib::search::SearchResults Nearest neighbor external labels, optional distances, and traversal telemetry.
     */
    deglib::search::SearchResults explore(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const bool return_distances = true,
        const bool unsorted = false
    ) const {
        const uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_label);
        thread_local deglib::graph::SearchWorkspace search_ws;
        internal_graph_->explore(search_ws, internal_entry, k, max_distance_computation_count, eps, include_entry, filter);
        return drain_to_single_results(search_ws, k, return_distances, unsorted);
    }

    /**
     * @brief Explores the graph starting at a specific stored vertex using fixed beam width (ef).
     *
     * @param entry_external_label The external label of the vertex to start exploration from.
     * @param k The number of nearest neighbors to return.
     * @param ef Fixed candidate pool / beam width (clamped to at least k).
     * @param max_distance_computation_count Maximum number of distance calculations allowed (0 = unlimited).
     * @param include_entry If true, the entry vertex itself is eligible in the result set (default true).
     * @param filter Optional predicate filter to restrict eligible vertices.
     * @param return_distances If true (default), includes distances in `SearchResults`.
     * @return deglib::search::SearchResults Nearest neighbor external labels, optional distances, and traversal telemetry.
     */
    deglib::search::SearchResults explore_ef(
        const uint32_t entry_external_label,
        const uint32_t k,
        const uint32_t ef,
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const bool return_distances = true
    ) const {
        const uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_label);
        thread_local deglib::graph::SearchWorkspace search_ws;
        internal_graph_->explore_ef(search_ws, internal_entry, k, ef, max_distance_computation_count, include_entry, filter);
        return drain_to_single_results(search_ws, k, return_distances, /*unsorted=*/false);
    }

    /**
     * @brief Batch exploration into pre-allocated flat output buffers using dynamic exploration radius (eps).
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
        uint64_t dummy_dists = 0;
        uint64_t dummy_hops = 0;
        execute_explore_batch_impl<false>(
            entry_external_labels, k, eps, 0, out_indices, out_distances, max_distance_computation_count, include_entry, filter, threads, unsorted, dummy_dists,
            dummy_hops
        );
    }

    /**
     * @brief Batch exploration returning `SearchResults` with dynamic exploration radius (eps).
     */
    deglib::search::SearchResults explore_batch(
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
        deglib::search::SearchResults res;
        res.n_queries = count;
        res.k = k;
        res.is_batch = true;
        res.with_distances = return_distances;
        res.indices.resize(count * k);
        if (return_distances) {
            res.distances.resize(count * k);
        }
        execute_explore_batch_impl<false>(
            entry_external_labels, k, eps, 0, std::span<uint32_t>(res.indices), return_distances ? std::span<float>(res.distances) : std::span<float>{},
            max_distance_computation_count, include_entry, filter, threads, unsorted, res.graph_distance_count, res.graph_hop_count
        );
        return res;
    }

    /**
     * @brief Batch exploration into pre-allocated flat output buffers using fixed beam width (ef).
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
        uint64_t dummy_dists = 0;
        uint64_t dummy_hops = 0;
        execute_explore_batch_impl<true>(
            entry_external_labels, k, 0.0f, ef, out_indices, out_distances, max_distance_computation_count, include_entry, filter, threads, unsorted,
            dummy_dists, dummy_hops
        );
    }

    /**
     * @brief Batch exploration returning `SearchResults` with fixed beam width (ef).
     */
    deglib::search::SearchResults explore_batch_ef(
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
        deglib::search::SearchResults res;
        res.n_queries = count;
        res.k = k;
        res.is_batch = true;
        res.with_distances = return_distances;
        res.indices.resize(count * k);
        if (return_distances) {
            res.distances.resize(count * k);
        }
        execute_explore_batch_impl<true>(
            entry_external_labels, k, 0.0f, ef, std::span<uint32_t>(res.indices), return_distances ? std::span<float>(res.distances) : std::span<float>{},
            max_distance_computation_count, include_entry, filter, threads, unsorted, res.graph_distance_count, res.graph_hop_count
        );
        return res;
    }

  private:
    template <typename T, typename Fn>
    void run_with_prepared_query(std::span<const T> query, Fn&& fn) const {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
            if (hasTransformations()) {
                if (query.size() != getInputDim()) {
                    throw std::invalid_argument(std::format("Dimension mismatch: Transformations expect {} dims, query has {}", getInputDim(), query.size()));
                }
                thread_local deglib::transformation::TransformationWorkspace workspace;
                const size_t out_bytes =
                    std::max(deglib::transformation::get_output_byte_size(transformations_), internal_graph_->getFeatureSpace().get_data_size());
                std::byte* q_buf = workspace.get_buffer_a(out_bytes);

                if constexpr (std::is_same_v<T, float>) {
                    deglib::transformation::apply_on_query(transformations_, query, std::span<std::byte>(q_buf, out_bytes));
                } else {
                    float* f_query = workspace.get_float_buffer(query.size());
                    deglib::distances::fp16::fp16_to_floats(query.data(), f_query, query.size());
                    deglib::transformation::apply_on_query(
                        transformations_, std::span<const float>(f_query, query.size()), std::span<std::byte>(q_buf, out_bytes)
                    );
                }

                fn(std::span<const std::byte>(q_buf, out_bytes));
                return;
            }
        }
        fn(query);
    }

    deglib::search::SearchResults
    drain_to_single_results(deglib::graph::SearchWorkspace& ws, uint32_t k, bool return_distances, bool unsorted, bool pad_missing = false) const {
        deglib::search::SearchResults res;
        res.n_queries = 1;
        res.k = k;
        res.is_batch = false;
        res.with_distances = return_distances;
        res.graph_distance_count = ws.graph_distance_count;
        res.graph_hop_count = ws.graph_hop_count;

        const uint32_t avail = static_cast<uint32_t>(ws.result_count(k));
        if (avail == 0 && !pad_missing) return res;

        const uint32_t alloc_count = pad_missing ? k : avail;
        res.indices.resize(alloc_count);
        if (return_distances) res.distances.resize(alloc_count);
        const uint32_t written =
            ws.drain_into(*internal_graph_, k, res.indices.data(), return_distances ? res.distances.data() : nullptr, unsorted, pad_missing);
        if (!pad_missing) {
            res.indices.resize(written);
            if (return_distances) res.distances.resize(written);
        }
        return res;
    }

    template <typename T, bool UseEf>
    void execute_search_batch_impl(
        std::span<const T> queries,
        size_t count,
        const uint32_t k,
        const float eps,
        const uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count,
        size_t threads,
        bool unsorted,
        uint64_t& out_graph_distance_count,
        uint64_t& out_graph_hop_count
    ) const {
        out_graph_distance_count = 0;
        out_graph_hop_count = 0;
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
        if (k == 0) return;

        const size_t resolved_threads = (threads == 0) ? std::max<size_t>(1, std::thread::hardware_concurrency()) : threads;
        const size_t actual_threads = std::min(resolved_threads, count);

        deglib::search::detail::ThreadSearchCounters single_counter{};
        std::vector<deglib::search::detail::ThreadSearchCounters> heap_counters;
        deglib::search::detail::ThreadSearchCounters* counters_ptr = &single_counter;
        if (actual_threads > 1) {
            heap_counters.resize(actual_threads);
            counters_ptr = heap_counters.data();
        }

        const bool use_xfrms = (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) && hasTransformations();
        const size_t out_stride =
            use_xfrms ? std::max(deglib::transformation::get_output_byte_size(transformations_), internal_graph_->getFeatureSpace().get_data_size()) : 0;

        deglib::concurrent::parallel_batch_for(
            0, count, actual_threads,
            [&](size_t begin, size_t end, size_t thread_id) {
                thread_local deglib::graph::SearchWorkspace search_ws;
                alignas(64) std::byte stack_buf[512];
                std::byte* q_buf = stack_buf;
                if (use_xfrms && out_stride > sizeof(stack_buf)) {
                    thread_local deglib::transformation::TransformationWorkspace xfrm_ws;
                    q_buf = xfrm_ws.get_buffer_a(out_stride);
                }

                uint64_t local_dists = 0;
                uint64_t local_hops = 0;
                for (size_t q = begin; q < end; ++q) {
                    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint16_t>) {
                        if (use_xfrms) {
                            if constexpr (std::is_same_v<T, float>) {
                                deglib::transformation::apply_on_query(
                                    transformations_, queries.subspan(q * in_stride, in_stride), std::span<std::byte>(q_buf, out_stride)
                                );
                            } else {
                                alignas(64) float stack_f16_buf[256];
                                float* f16_buf = stack_f16_buf;
                                if (in_stride > sizeof(stack_f16_buf) / sizeof(float)) {
                                    thread_local deglib::transformation::TransformationWorkspace xfrm_ws;
                                    f16_buf = xfrm_ws.get_float_buffer(in_stride);
                                }
                                deglib::distances::fp16::fp16_to_floats(queries.data() + q * in_stride, f16_buf, in_stride);
                                deglib::transformation::apply_on_query(
                                    transformations_, std::span<const float>(f16_buf, in_stride), std::span<std::byte>(q_buf, out_stride)
                                );
                            }

                            std::span<const std::byte> single_q(q_buf, out_stride);
                            if constexpr (UseEf) {
                                internal_graph_->search_ef(search_ws, single_q, k, ef, include_entry, filter, max_distance_computation_count);
                            } else {
                                internal_graph_->search(search_ws, single_q, k, eps, filter, max_distance_computation_count);
                            }
                        } else {
                            std::span<const T> single_q(queries.data() + q * in_stride, in_stride);
                            if constexpr (UseEf) {
                                internal_graph_->search_ef(search_ws, single_q, k, ef, include_entry, filter, max_distance_computation_count);
                            } else {
                                internal_graph_->search(search_ws, single_q, k, eps, filter, max_distance_computation_count);
                            }
                        }
                    } else {
                        std::span<const T> single_q(queries.data() + q * in_stride, in_stride);
                        if constexpr (UseEf) {
                            internal_graph_->search_ef(search_ws, single_q, k, ef, include_entry, filter, max_distance_computation_count);
                        } else {
                            internal_graph_->search(search_ws, single_q, k, eps, filter, max_distance_computation_count);
                        }
                    }

                    search_ws.drain_into(
                        *internal_graph_, k, out_indices.data() + q * k, return_distances ? (out_distances.data() + q * k) : nullptr, unsorted,
                        /*pad_missing=*/true
                    );
                    local_dists += search_ws.graph_distance_count;
                    local_hops += search_ws.graph_hop_count;
                }

                counters_ptr[thread_id].graph_distance_count += local_dists;
                counters_ptr[thread_id].graph_hop_count += local_hops;
            },
            16
        );

        for (size_t t = 0; t < actual_threads; ++t) {
            out_graph_distance_count += counters_ptr[t].graph_distance_count;
            out_graph_hop_count += counters_ptr[t].graph_hop_count;
        }
    }

    template <bool UseEf>
    void execute_explore_batch_impl(
        std::span<const uint32_t> entry_external_labels,
        const uint32_t k,
        const float eps,
        const uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances,
        const uint32_t max_distance_computation_count,
        const bool include_entry,
        const deglib::search::Filter* filter,
        size_t threads,
        bool unsorted,
        uint64_t& out_graph_distance_count,
        uint64_t& out_graph_hop_count
    ) const {
        out_graph_distance_count = 0;
        out_graph_hop_count = 0;
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
        if (k == 0) return;

        const size_t resolved_threads = (threads == 0) ? std::max<size_t>(1, std::thread::hardware_concurrency()) : threads;
        const size_t actual_threads = std::min(resolved_threads, count);

        deglib::search::detail::ThreadSearchCounters single_counter{};
        std::vector<deglib::search::detail::ThreadSearchCounters> heap_counters;
        deglib::search::detail::ThreadSearchCounters* counters_ptr = &single_counter;
        if (actual_threads > 1) {
            heap_counters.resize(actual_threads);
            counters_ptr = heap_counters.data();
        }

        deglib::concurrent::parallel_batch_for(
            0, count, actual_threads,
            [&](size_t begin, size_t end, size_t thread_id) {
                thread_local deglib::graph::SearchWorkspace search_ws;
                uint64_t local_dists = 0;
                uint64_t local_hops = 0;
                for (size_t q = begin; q < end; ++q) {
                    const uint32_t internal_entry = internal_graph_->getInternalIndex(entry_external_labels[q]);
                    if constexpr (UseEf) {
                        internal_graph_->explore_ef(search_ws, internal_entry, k, ef, max_distance_computation_count, include_entry, filter);
                    } else {
                        internal_graph_->explore(search_ws, internal_entry, k, max_distance_computation_count, eps, include_entry, filter);
                    }
                    search_ws.drain_into(
                        *internal_graph_, k, out_indices.data() + q * k, return_distances ? (out_distances.data() + q * k) : nullptr, unsorted,
                        /*pad_missing=*/true
                    );
                    local_dists += search_ws.graph_distance_count;
                    local_hops += search_ws.graph_hop_count;
                }
                counters_ptr[thread_id].graph_distance_count += local_dists;
                counters_ptr[thread_id].graph_hop_count += local_hops;
            },
            16
        );

        for (size_t t = 0; t < actual_threads; ++t) {
            out_graph_distance_count += counters_ptr[t].graph_distance_count;
            out_graph_hop_count += counters_ptr[t].graph_hop_count;
        }
    }

  public:
    /**
     * Greedy best-first reachability search towards a stored vertex.
     *
     * Traverses the graph from the given entry vertices, using the target vertex's own feature vector as
     * the query, and backtracks the chain of vertices that reached the target. With eps = 0.0 and k = 1
     * this degenerates into a greedy descent; larger eps widen the exploration radius past local optima.
     *
     * @param from_external_labels The external labels of the vertices to start the traversal from.
     * @param to_external_label The external label of the vertex the path should end at.
     * @param eps Exploration factor widening the search radius. 0.0 restricts the traversal to a greedy descent.
     * @param k Size of the result list driving the exploration radius.
     * @return A vector of ObjectDistance objects ordered from the target back to the entry, holding external
     *         labels and distances to the target. Empty if the target was not reached within the budget.
     */
    std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& from_external_labels, const uint32_t to_external_label, const float eps = 0.0f, const uint32_t k = 1) const {
        std::vector<uint32_t> entry_indices;
        entry_indices.reserve(from_external_labels.size());
        for (const uint32_t from_external_label : from_external_labels) {
            entry_indices.push_back(internal_graph_->getInternalIndex(from_external_label));
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
    /**
     * @brief Converts this graph to a compact, immutable ReadOnlyGraph optimized for search.
     *
     * Supports four conversion modes:
     * 1. Simple copy (all arguments omitted): copies topology and features as-is.
     * 2. In-place self-transformation (transformations non-empty, custom_features == nullptr):
     *    transforms/quantizes directly from stored FP32/FP16 features into the new graph
     *    without allocating an intermediate buffer (requires target metric != source metric).
     * 3. Custom feature buffer (custom_features != nullptr): copies pre-transformed features
     *    aligned with getExternalLabels() (custom_features is NOT passed through transformations).
     * 4. Query-only transformation attachment (transformations keeps source metric, custom_features == nullptr):
     *    copies features as-is and attaches transformations for query preprocessing.
     *
     * @param custom_feature_space Optional target feature space. Derived automatically from transformations if omitted.
     * @param custom_features Optional pointer to replacement raw feature memory in internal index order (0 .. size()-1).
     * @param transformations Optional sequence of fitted feature transformations/quantizers to apply/attach.
     * @param num_threads Number of worker threads for parallel self-transformation (0 = hardware concurrency).
     * @return A new read-only DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_readonly(
        std::optional<deglib::distances::FloatSpace> custom_feature_space = std::nullopt,
        const void* custom_features = nullptr,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t num_threads = 0
    ) const {
        auto xfrms = !transformations.empty() ? std::move(transformations) : transformations_;
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(
            deglib::graph::ReadOnlyGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, xfrms, num_threads)
        );
        return DynamicExplorationGraph(std::move(graph), std::move(xfrms));
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
     * @brief Converts this graph to a mutable SizeBoundedGraph with optional feature/space conversion.
     *
     * Supports the same four conversion modes as to_readonly() (simple copy, in-place
     * self-transformation when custom_features == nullptr and target metric != source metric,
     * custom feature buffer, or query-only transformation attachment). Edge weights are recalculated.
     *
     * @param custom_feature_space Optional target feature space (derived from transformations if omitted).
     * @param custom_features Optional pointer to replacement raw feature memory in internal index order (0 .. size()-1).
     * @param new_max_size Desired vertex capacity. If 0 or less than size(), capacity is set to size().
     * @param transformations Optional sequence of fitted feature transformations/quantizers to apply/attach.
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_mutable(
        std::optional<deglib::distances::FloatSpace> custom_feature_space,
        const void* custom_features = nullptr,
        const uint32_t new_max_size = 0,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    ) const {
        auto xfrms = !transformations.empty() ? std::move(transformations) : transformations_;
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(
            deglib::graph::SizeBoundedGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, xfrms, new_max_size)
        );
        return DynamicExplorationGraph(std::move(graph), std::move(xfrms));
    }

    /**
     * @brief Converts this graph to a mutable DynamicGraph with chunked dynamic memory allocation.
     *
     * @param chunk_size Number of vertices allocated per memory chunk (default 1024). Rounded to power of 2.
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_dynamic(const uint32_t chunk_size = 1024) const {
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(deglib::graph::DynamicGraph::from_graph(*internal_graph_, chunk_size));
        return DynamicExplorationGraph(std::move(graph), transformations_);
    }

    /**
     * @brief Converts this graph to a mutable DynamicGraph with chunked allocation and optional feature/space conversion.
     *
     * Supports the same four conversion modes as to_readonly() (simple copy, in-place
     * self-transformation when custom_features == nullptr and target metric != source metric,
     * custom feature buffer, or query-only transformation attachment). Edge weights are recalculated.
     *
     * @param custom_feature_space Optional target feature space (derived from transformations if omitted).
     * @param custom_features Optional pointer to replacement raw feature memory in internal index order (0 .. size()-1).
     * @param chunk_size Number of vertices allocated per memory chunk (default 1024). Rounded to power of 2.
     * @param transformations Optional sequence of fitted feature transformations/quantizers to apply/attach.
     * @return A new mutable DynamicExplorationGraph.
     */
    DynamicExplorationGraph to_dynamic(
        std::optional<deglib::distances::FloatSpace> custom_feature_space,
        const void* custom_features = nullptr,
        const uint32_t chunk_size = 1024,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    ) const {
        auto xfrms = !transformations.empty() ? std::move(transformations) : transformations_;
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(
            deglib::graph::DynamicGraph::from_graph(*internal_graph_, custom_feature_space, custom_features, xfrms, chunk_size)
        );
        return DynamicExplorationGraph(std::move(graph), std::move(xfrms));
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
     * @brief Auto-tunes the graph's traversal prefetch parameters (po, pl, nl) by empirically timing
     * traversal over sampled vertices.
     *
     * The graph self-samples its own stored features as queries, so no external query buffer is needed.
     * Forwards to `InternalGraph::optimize()`.
     *
     * @param sample_count  Number of vertices sampled as queries (capped at the graph size; default 300).
     * @param median_subset Subset size for median entry point selection (`<= 1` keeps native entries).
     */
    void optimize(size_t sample_count = 300, size_t median_subset = 4096) const { internal_graph_->optimize(sample_count, median_subset); }

    /**
     * @brief Calibrates a `RecallTable` mapping target recall@k values to `ef` and `eps` search parameters.
     *
     * Self-samples up to `sample_count` stored vertices as queries, computes exact ground truth in the
     * graph's native feature space, and solves for each target recall in `recall_range` (step `recall_step`)
     * guarded by an estimated brute-force time budget.
     */
    [[nodiscard]] deglib::search::RecallTable to_recall_table(
        uint32_t k,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
        float recall_step = deglib::search::RecallTable::kDefaultRecallStep,
        size_t threads = 0
    ) const {
        return deglib::search::build_recall_table(*internal_graph_, k, sample_count, recall_range, recall_step, threads);
    }

    /**
     * @brief Calibrates a `RecallTable` for an explicit sequence of target recall@k values.
     */
    [[nodiscard]] deglib::search::RecallTable to_recall_table(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        size_t threads = 0
    ) const {
        return deglib::search::build_recall_table(*internal_graph_, k, target_recalls, sample_count, threads);
    }

    /**
     * @brief Saves the graph and any attached transformations to a binary file.
     * Graph must be mutable (SizeBoundedGraph or DynamicGraph) to be saved.
     *
     * @param path File path to write to (C-string).
     * @return True if save succeeded, false otherwise.
     */
    bool saveGraph(const char* path) const {
        const auto* mutable_graph = dynamic_cast<const deglib::graph::MutableGraph*>(internal_graph_);
        if (mutable_graph == nullptr) {
            throw std::runtime_error("Graph must be mutable to save; read-only graphs do not store edge weights.");
        }
        return deglib::serialization::save_graph(path, *mutable_graph, transformations_.empty() ? nullptr : &transformations_);
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
     * @param path File path to read from (C-string).
     * @return Loaded DynamicExplorationGraph.
     */
    static DynamicExplorationGraph load_readonly(const char* path) {
        auto transformations = deglib::serialization::read_transformations(path);
        auto graph = std::make_unique<deglib::graph::ReadOnlyGraph>(deglib::graph::ReadOnlyGraph::loadGraph(path));
        return DynamicExplorationGraph(std::move(graph), std::move(transformations));
    }

    static DynamicExplorationGraph load_readonly(const std::string& path) { return load_readonly(path.c_str()); }

    /**
     * @brief Loads a graph from a binary file directly into a mutable SizeBoundedGraph.
     *
     * @param path File path to read from (C-string).
     * @param new_max_size Optional capacity override. If 0, uses the graph's serialized size.
     * @return Loaded DynamicExplorationGraph.
     */
    static DynamicExplorationGraph load_mutable(const char* path, const uint32_t new_max_size = 0) {
        auto transformations = deglib::serialization::read_transformations(path);
        auto graph = std::make_unique<deglib::graph::SizeBoundedGraph>(deglib::graph::SizeBoundedGraph::loadGraph(path, new_max_size));
        return DynamicExplorationGraph(std::move(graph), std::move(transformations));
    }

    static DynamicExplorationGraph load_mutable(const std::string& path, const uint32_t new_max_size = 0) { return load_mutable(path.c_str(), new_max_size); }

    /**
     * @brief Loads a graph from a binary file directly into a mutable chunk-allocated DynamicGraph.
     *
     * @param path File path to read from (C-string).
     * @param chunk_size Target number of vertices per memory chunk (default = 1024).
     * @return Loaded DynamicExplorationGraph.
     */
    static DynamicExplorationGraph load_dynamic(const char* path, const uint32_t chunk_size = 1024) {
        auto transformations = deglib::serialization::read_transformations(path);
        auto graph = std::make_unique<deglib::graph::DynamicGraph>(deglib::graph::DynamicGraph::loadGraph(path, chunk_size));
        return DynamicExplorationGraph(std::move(graph), std::move(transformations));
    }

    static DynamicExplorationGraph load_dynamic(const std::string& path, const uint32_t chunk_size = 1024) { return load_dynamic(path.c_str(), chunk_size); }
};

}  // namespace deglib
