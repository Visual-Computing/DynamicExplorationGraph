#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/search/kmeans.h"
#include "deglib/search/reranker.h"
#include "deglib/search/result_list.h"
#include "deglib/transformation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace deglib::search {

/**
 * Concrete, high-performance stateless search pipeline coordinator.
 *
 * Coordinates:
 * 1. Query preparation (FP32/FP16 conversion, on-the-fly quantization via attached FeatureTransformations).
 * 2. Dynamic entry point routing (via KMeansEntrySelector).
 * 3. Graph traversal (relative epsilon or fixed-pool ef exploration).
 * 4. Post-graph candidate reranking (via Reranker against full-precision base vectors).
 * 5. Full-stack hardware cache and routing auto-tuning (optimize).
 */
class Searcher {
  private:
    // Thread-local scratch workspace for query transformations, graph traversal, and candidate reranking
    struct QueryWorkspace : deglib::graph::SearchWorkspace {
        std::vector<float> norm_buf;
        std::vector<float> f32_buf;
        std::vector<uint16_t> f16_buf;
        std::vector<std::byte> quant_buf1;
        std::vector<std::byte> quant_buf2;
        std::vector<uint32_t> entry_buf;  // reusable entry-point buffer (avoids per-query allocation)
        std::vector<uint32_t> cand_buf;   // reusable candidate-ID buffer for reranking (> 256 candidates)
        uint32_t rerank_distance_count = 0;
    };

    const deglib::graph::InternalGraph* graph_ = nullptr;
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations_;
    KMeansEntrySelector entry_selector_;
    std::shared_ptr<Reranker> reranker_;
    bool normalize_query_ = false;

    /**
     * Prepares a float32 query for graph traversal by running any attached transformations
     * or converting to graph-native format.
     */
    inline const std::byte* prepare_query(const float* query, uint32_t dim, QueryWorkspace& ws, std::span<std::byte> stack_buf = {}) const {
        if (transformations_.empty()) {
            const auto& space = graph_->getFeatureSpace();
            if (space.metric().get_data_type() == deglib::distances::MetricDataType::FP32) {
                return reinterpret_cast<const std::byte*>(query);
            } else if (space.metric().get_data_type() == deglib::distances::MetricDataType::FP16) {
                const size_t need_bytes = static_cast<size_t>(dim) * sizeof(uint16_t);
                if (need_bytes <= stack_buf.size()) {
                    deglib::distances::fp16::floats_to_fp16(query, reinterpret_cast<uint16_t*>(stack_buf.data()), dim);
                    return stack_buf.data();
                }
                ws.f16_buf.resize(dim);
                deglib::distances::fp16::floats_to_fp16(query, ws.f16_buf.data(), dim);
                return reinterpret_cast<const std::byte*>(ws.f16_buf.data());
            }
        }

        const float* current_f32 = query;
        for (size_t t = 0; t < transformations_.size(); ++t) {
            const auto& trans = transformations_[t];
            const bool is_last = (t + 1 == transformations_.size());

            const size_t out_bytes = std::max(std::max(trans->query_byte_size(), trans->output_byte_size()), graph_->getFeatureSpace().get_data_size());
            const size_t cur_dim = trans->input_dim();
            std::span<const float> in_span(current_f32, cur_dim);
            if (is_last) {
                std::byte* dst = nullptr;
                if (out_bytes <= stack_buf.size()) {
                    dst = stack_buf.data();
                } else {
                    ws.quant_buf1.resize(out_bytes);
                    dst = ws.quant_buf1.data();
                }
                trans->transform_query(in_span, std::span<std::byte>(dst, out_bytes));
                return dst;
            } else {
                ws.quant_buf2.resize(out_bytes);
                trans->transform_query(in_span, std::span<std::byte>(ws.quant_buf2.data(), out_bytes));
                current_f32 = reinterpret_cast<const float*>(ws.quant_buf2.data());
            }
        }

        return reinterpret_cast<const std::byte*>(query);
    }

    /**
     * Prepares a float16 query for graph traversal.
     */
    inline const std::byte* prepare_query(const uint16_t* query, uint32_t dim, QueryWorkspace& ws, std::span<std::byte> stack_buf = {}) const {
        if (transformations_.empty()) {
            const auto& space = graph_->getFeatureSpace();
            if (space.metric().get_data_type() == deglib::distances::MetricDataType::FP16) {
                return reinterpret_cast<const std::byte*>(query);
            } else if (space.metric().get_data_type() == deglib::distances::MetricDataType::FP32) {
                ws.f32_buf.resize(dim);
                deglib::distances::fp16::fp16_to_floats(query, ws.f32_buf.data(), dim);
                return reinterpret_cast<const std::byte*>(ws.f32_buf.data());
            }
        }

        ws.f32_buf.resize(dim);
        deglib::distances::fp16::fp16_to_floats(query, ws.f32_buf.data(), dim);
        return prepare_query(ws.f32_buf.data(), dim, ws, stack_buf);
    }

    /**
     * @brief Internal single-query execution engine.
     *
     * Handles optional query normalization, query transformation/quantization,
     * graph traversal (`search_ef` or `search`) reusing `ws`, optional full-precision reranking,
     * and internal-to-external label mapping.
     *
     * @tparam QueryT                        Input query element type (`float` or `uint16_t`).
     * @param  query                         Pointer to the input query vector of length `dim`.
     * @param  k                             Number of nearest neighbors to return.
     * @param  eps                           Relative distance margin (used when `use_ef` is false).
     * @param  ef                            Fixed exploration pool size (used when `use_ef` is true).
     * @param  use_ef                        Selects `search_ef` (`true`) or `search` (`false`).
     * @param  rerank_factor                 Candidate expansion factor for exact reranking.
     * @param  out_indices                   Caller-allocated output buffer of at least `k` labels.
     * @param  out_distances                 Optional output buffer of at least `k` distances (may be null).
     * @param  return_distances              Whether to write distances to `out_distances`.
     * @param  unsorted                      If true, skip final ascending distance sort where supported.
     * @param  max_distance_computation_count Hard per-query budget for graph distance computations (`0` = unlimited).
     * @param  ws                            Reusable thread-local workspace buffers and telemetry counters.
     * @param  pad_missing                   If true, pad unfilled trailing slots up to `k` with `UINT32_MAX` / `NaN`.
     * @return Number of neighbors written to `out_indices` (at most `k`).
     */
    template <typename QueryT>
    inline uint32_t execute_search_single(
        const QueryT* query,
        uint32_t k,
        float eps,
        uint32_t ef,
        bool use_ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances,
        bool return_distances,
        bool unsorted,
        uint32_t max_distance_computation_count,
        QueryWorkspace& ws,
        bool pad_missing = false
    ) const {
        ws.graph_distance_count = 0;
        ws.graph_hop_count = 0;
        ws.rerank_distance_count = 0;

        if (!graph_ || graph_->size() == 0 || k == 0 || query == nullptr || out_indices == nullptr) {
            if (pad_missing && out_indices != nullptr) {
                for (uint32_t i = 0; i < k; ++i) {
                    out_indices[i] = std::numeric_limits<uint32_t>::max();
                    if (return_distances && out_distances) {
                        out_distances[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            return 0;
        }

        const uint32_t dim = transformations_.empty() ? graph_->getFeatureSpace().dim() : transformations_.front()->input_dim();
        const QueryT* effective_query = query;

        if constexpr (std::is_same_v<QueryT, float>) {
            if (normalize_query_) {
                if (ws.norm_buf.size() < dim) ws.norm_buf.resize(dim);
                deglib::transformation::detail::normalize_vector_l2(query, ws.norm_buf.data(), dim);
                effective_query = ws.norm_buf.data();
            }
        }

        alignas(64) std::byte stack_qbuf[512];
        const std::byte* graph_query_ptr = prepare_query(effective_query, dim, ws, stack_qbuf);
        const auto query_span = std::span<const std::byte>(graph_query_ptr, graph_->getFeatureSpace().get_data_size());

        // Select entry point: query routing medoid or direct reference to static graph anchor (zero copy).
        const auto& entry_vertices = entry_selector_.has_entries() ? [&]() -> const std::vector<uint32_t>& {
            entry_selector_.top_entries(query_span, 2, ws.entry_buf);
            return ws.entry_buf;
        }()
            : graph_->getEntryVertexIndices();

        const uint32_t num_candidates = reranker_ ? static_cast<uint32_t>(std::ceil(static_cast<float>(k) * std::max(1.0f, rerank_factor))) : k;
        const bool should_rerank = static_cast<bool>(reranker_) && (num_candidates > k);

        if (use_ef) {
            graph_->search_ef(ws, query_span, entry_vertices, num_candidates, ef, true, nullptr, max_distance_computation_count);
        } else {
            graph_->search(ws, query_span, entry_vertices, num_candidates, eps, nullptr, max_distance_computation_count);
        }

        const uint32_t found_cands = static_cast<uint32_t>(ws.result_count(num_candidates));
        if (found_cands == 0) {
            if (pad_missing) {
                for (uint32_t i = 0; i < k; ++i) {
                    out_indices[i] = std::numeric_limits<uint32_t>::max();
                    if (return_distances && out_distances) {
                        out_distances[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            return 0;
        }

        if (should_rerank) {
            uint32_t stack_cand_indices[256];
            uint32_t* cand_indices = stack_cand_indices;
            if (found_cands > 256) {
                if (ws.cand_buf.size() < found_cands) {
                    ws.cand_buf.resize(found_cands);
                }
                cand_indices = ws.cand_buf.data();
            }

            ws.extract_internal_ids(found_cands, cand_indices);
            ws.rerank_distance_count = found_cands;

            const uint32_t found = reranker_->rerank(
                std::span<const QueryT>(effective_query, dim), std::span<const uint32_t>(cand_indices, found_cands), k, std::span<uint32_t>(out_indices, k),
                return_distances ? std::span<float>(out_distances, k) : std::span<float>{}, return_distances, unsorted
            );
            for (size_t i = 0; i < found; ++i) {
                out_indices[i] = graph_->getExternalLabel(out_indices[i]);
            }
            if (pad_missing) {
                for (uint32_t i = found; i < k; ++i) {
                    out_indices[i] = std::numeric_limits<uint32_t>::max();
                    if (return_distances && out_distances) {
                        out_distances[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            return found;
        }

        return ws.drain_into(*graph_, k, out_indices, return_distances ? out_distances : nullptr, unsorted, pad_missing);
    }

  public:
    /**
     * Constructs a Searcher without transformations or reranking.
     *
     * @param graph Underlying graph to search. Results are mapped back to the graph's external labels.
     */
    explicit Searcher(const deglib::graph::InternalGraph& graph) : graph_(&graph), entry_selector_(graph) {}

    /**
     * Constructs a Searcher with attached query transformations/quantizers.
     *
     * @param graph           Underlying graph to search. Results are mapped back to external labels.
     * @param transformations Feature transformations (e.g. quantizers) applied to incoming query vectors.
     */
    Searcher(const deglib::graph::InternalGraph& graph, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations)
        : graph_(&graph), transformations_(std::move(transformations)), entry_selector_(graph) {}

    /**
     * Constructs a Searcher with candidate reranking.
     *
     * @param graph    Underlying graph to search. Results are mapped back to external labels.
     * @param reranker Reranker holding full-precision features. Its base_vectors must follow the
     *                 graph's internal index order (0 .. size()-1, see graph.getExternalLabels()).
     */
    Searcher(const deglib::graph::InternalGraph& graph, std::shared_ptr<Reranker> reranker)
        : graph_(&graph), entry_selector_(graph), reranker_(std::move(reranker)) {}

    Searcher(const deglib::graph::InternalGraph& graph, Reranker reranker)
        : graph_(&graph), entry_selector_(graph), reranker_(std::make_shared<Reranker>(std::move(reranker))) {}

    /**
     * Constructs a Searcher with query transformations and candidate reranking.
     *
     * @param graph           Underlying graph to search. Results are mapped back to external labels.
     * @param transformations Feature transformations (e.g. quantizers) applied to incoming query vectors.
     * @param reranker        Reranker holding full-precision features. Its base_vectors must follow the
     *                        graph's internal index order (0 .. size()-1, see graph.getExternalLabels()).
     */
    Searcher(
        const deglib::graph::InternalGraph& graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        std::shared_ptr<Reranker> reranker
    )
        : graph_(&graph), transformations_(std::move(transformations)), entry_selector_(graph), reranker_(std::move(reranker)) {}

    Searcher(
        const deglib::graph::InternalGraph& graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        Reranker reranker
    )
        : graph_(&graph), transformations_(std::move(transformations)), entry_selector_(graph), reranker_(std::make_shared<Reranker>(std::move(reranker))) {}

    bool normalize_query() const noexcept { return normalize_query_; }
    void set_normalize_query(bool val) noexcept { normalize_query_ = val; }

    /**
     * @brief Shared parallel batch-search execution helper.
     *
     * Dispatches contiguous query rows across worker threads via `parallel_batch_for`,
     * reusing a `thread_local QueryWorkspace` per worker thread and aggregating telemetry
     * lock-free via per-thread cache-line-aligned slots.
     */
    template <typename QueryT>
    inline void execute_search_batch(
        const QueryT* queries,
        size_t n_queries,
        uint32_t k,
        float eps_or_ef,
        float rerank_factor,
        uint32_t* out_indices,
        float* out_distances,
        size_t threads,
        bool return_distances,
        bool unsorted,
        uint32_t max_distance_computation_count,
        uint64_t& out_graph_distance_count,
        uint64_t& out_rerank_distance_count,
        uint64_t& out_graph_hop_count
    ) const {
        out_graph_distance_count = 0;
        out_rerank_distance_count = 0;
        out_graph_hop_count = 0;
        if (!graph_ || n_queries == 0 || k == 0 || queries == nullptr || out_indices == nullptr) return;

        const uint32_t dim = transformations_.empty() ? graph_->getFeatureSpace().dim() : transformations_.front()->input_dim();
        const bool use_ef = (eps_or_ef >= 1.0f);
        const uint32_t ef = use_ef ? static_cast<uint32_t>(std::lround(eps_or_ef)) : 0;
        const float eps = use_ef ? 0.0f : eps_or_ef;

        const size_t resolved_threads = (threads == 0) ? std::max<size_t>(1, std::thread::hardware_concurrency()) : threads;
        const size_t actual_threads = std::min(resolved_threads, n_queries);

        deglib::search::detail::ThreadSearchCounters single_counter{};
        std::vector<deglib::search::detail::ThreadSearchCounters> heap_counters;
        deglib::search::detail::ThreadSearchCounters* counters_ptr = &single_counter;
        if (actual_threads > 1) {
            heap_counters.resize(actual_threads);
            counters_ptr = heap_counters.data();
        }

        deglib::concurrent::parallel_batch_for(
            0, n_queries, actual_threads,
            [&](size_t begin, size_t end, size_t thread_id) {
                thread_local QueryWorkspace ws;
                uint64_t local_graph_dists = 0;
                uint64_t local_rerank_dists = 0;
                uint64_t local_hops = 0;
                for (size_t q = begin; q < end; ++q) {
                    const QueryT* q_ptr = queries + q * dim;
                    uint32_t* idx_ptr = out_indices + q * k;
                    float* dist_ptr = return_distances ? (out_distances + q * k) : nullptr;
                    execute_search_single(
                        q_ptr, k, eps, ef, use_ef, rerank_factor, idx_ptr, dist_ptr, return_distances, unsorted, max_distance_computation_count, ws,
                        /*pad_missing=*/true
                    );
                    local_graph_dists += ws.graph_distance_count;
                    local_rerank_dists += ws.rerank_distance_count;
                    local_hops += ws.graph_hop_count;
                }
                auto& tc = counters_ptr[thread_id];
                tc.graph_distance_count += local_graph_dists;
                tc.rerank_distance_count += local_rerank_dists;
                tc.graph_hop_count += local_hops;
            },
            16
        );

        for (size_t t = 0; t < actual_threads; ++t) {
            out_graph_distance_count += counters_ptr[t].graph_distance_count;
            out_rerank_distance_count += counters_ptr[t].rerank_distance_count;
            out_graph_hop_count += counters_ptr[t].graph_hop_count;
        }
    }

  public:
    // --- Modern C++20 Search API with eps_or_ef ---

    /**
     * @brief Searches a single query into caller-provided `std::span` output buffers.
     *
     * @tparam T                             Query element type (`float` or `uint16_t`).
     * @param  query                         Span covering the query vector (`dim` elements).
     * @param  k                             Number of nearest neighbors to return.
     * @param  out_indices                   Destination span of at least `k` elements.
     * @param  out_distances                 Optional destination span of at least `k` elements when `return_distances` is true.
     * @param  eps_or_ef                     Exploration parameter: values `>= 1.0f` select fixed pool `ef = lround(eps_or_ef)`,
     *                                       values `< 1.0f` select relative distance margin `eps`.
     * @param  rerank_factor                 Candidate expansion factor for exact reranking (`1.0f` disables expansion).
     * @param  return_distances              If true, writes distances to `out_distances`.
     * @param  unsorted                      If true, skips final ascending distance sort where supported.
     * @param  max_distance_computation_count Hard per-query limit on graph distance computations (`0` = unlimited).
     * @return Number of neighbors written to `out_indices` (at most `k`).
     * @throws std::invalid_argument         If `out_indices` or required `out_distances` is smaller than `k`.
     */
    template <typename T>
    uint32_t search(
        std::span<const T> query,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t>, "Unsupported query type: must be float or uint16_t");
        if (out_indices.size() < k) {
            throw std::invalid_argument("Searcher::search: out_indices span smaller than k");
        }
        if (return_distances && out_distances.size() < k) {
            throw std::invalid_argument("Searcher::search: return_distances is true but out_distances span smaller than k");
        }
        float* d_ptr = return_distances ? out_distances.data() : nullptr;
        const bool use_ef = (eps_or_ef >= 1.0f);
        const uint32_t ef = use_ef ? static_cast<uint32_t>(std::lround(eps_or_ef)) : 0;
        const float eps = use_ef ? 0.0f : eps_or_ef;

        thread_local QueryWorkspace ws;
        return execute_search_single(
            query.data(), k, eps, ef, use_ef, rerank_factor, out_indices.data(), d_ptr, return_distances, unsorted, max_distance_computation_count, ws
        );
    }

    /**
     * @brief Searches a single query and returns a `SearchResults` holding indices, optional distances, and telemetry.
     *
     * @copydetails search(std::span<const T>, uint32_t, std::span<uint32_t>, std::span<float>, float, float, bool, bool, uint32_t) const
     */
    template <typename T>
    SearchResults search(
        std::span<const T> query,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t>, "Unsupported query type: must be float or uint16_t");
        SearchResults res;
        res.n_queries = 1;
        res.k = k;
        res.is_batch = false;
        res.with_distances = return_distances;
        res.indices.resize(k);
        if (return_distances) res.distances.resize(k);

        const bool use_ef = (eps_or_ef >= 1.0f);
        const uint32_t ef = use_ef ? static_cast<uint32_t>(std::lround(eps_or_ef)) : 0;
        const float eps = use_ef ? 0.0f : eps_or_ef;

        thread_local QueryWorkspace ws;
        const uint32_t count = execute_search_single(
            query.data(), k, eps, ef, use_ef, rerank_factor, res.indices.data(), return_distances ? res.distances.data() : nullptr, return_distances, unsorted,
            max_distance_computation_count, ws
        );
        res.indices.resize(count);
        if (return_distances) res.distances.resize(count);
        res.graph_distance_count = ws.graph_distance_count;
        res.rerank_distance_count = ws.rerank_distance_count;
        res.graph_hop_count = ws.graph_hop_count;
        return res;
    }

    /**
     * @brief Searches a batch of queries in parallel into caller-provided `std::span` output buffers.
     *
     * @tparam T                             Query element type (`float` or `uint16_t`).
     * @param  queries                       Span covering `n_queries * dim` elements in row-major order.
     * @param  n_queries                     Number of queries in the batch.
     * @param  k                             Number of nearest neighbors per query.
     * @param  out_indices                   Destination span of at least `n_queries * k` elements.
     * @param  out_distances                 Optional destination span of at least `n_queries * k` elements when `return_distances` is true.
     * @param  eps_or_ef                     Exploration parameter: `>= 1.0f` uses fixed pool `ef`, `< 1.0f` uses margin `eps`.
     * @param  rerank_factor                 Candidate expansion factor for exact reranking (`1.0f` disables expansion).
     * @param  threads                       Number of worker threads (`0` = hardware concurrency, `1` = sequential).
     * @param  return_distances              If true, writes distances to `out_distances`.
     * @param  unsorted                      If true, skips final ascending distance sort where supported.
     * @param  max_distance_computation_count Hard per-query limit on graph distance computations (`0` = unlimited).
     * @throws std::invalid_argument         If `out_indices` or required `out_distances` is smaller than `n_queries * k`.
     */
    template <typename T>
    void search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t>, "Unsupported query type: must be float or uint16_t");
        if (out_indices.size() < n_queries * k) {
            throw std::invalid_argument("Searcher::search_batch: out_indices span smaller than n_queries * k");
        }
        if (return_distances && out_distances.size() < n_queries * k) {
            throw std::invalid_argument("Searcher::search_batch: return_distances is true but out_distances span smaller than n_queries * k");
        }
        uint64_t dummy_graph_dists = 0;
        uint64_t dummy_rerank_dists = 0;
        uint64_t dummy_hops = 0;
        execute_search_batch<T>(
            queries.data(), n_queries, k, eps_or_ef, rerank_factor, out_indices.data(), return_distances ? out_distances.data() : nullptr, threads,
            return_distances, unsorted, max_distance_computation_count, dummy_graph_dists, dummy_rerank_dists, dummy_hops
        );
    }

    /**
     * @brief Searches a batch of queries in parallel and returns a `SearchResults` holding indices, optional distances, and telemetry.
     *
     * @copydetails search_batch(std::span<const T>, size_t, uint32_t, std::span<uint32_t>, std::span<float>, float, float, size_t, bool, bool, uint32_t) const
     */
    template <typename T>
    SearchResults search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t>, "Unsupported query type: must be float or uint16_t");
        SearchResults result;
        result.n_queries = n_queries;
        result.k = k;
        result.is_batch = true;
        result.with_distances = return_distances;
        result.indices.resize(n_queries * k);
        if (return_distances) {
            result.distances.resize(n_queries * k);
        }
        execute_search_batch<T>(
            queries.data(), n_queries, k, eps_or_ef, rerank_factor, result.indices.data(), return_distances ? result.distances.data() : nullptr, threads,
            return_distances, unsorted, max_distance_computation_count, result.graph_distance_count, result.rerank_distance_count, result.graph_hop_count
        );
        return result;
    }

    // --- Optimization / Auto-Tuning ---

    /**
     * @brief Calibrates the search pipeline for the host CPU and graph.
     *
     * 1. Trains K-Means medoids for query routing (when `use_routing` is true).
     * 2. Auto-tunes candidate reranker prefetch lookahead (`po`, `pl`).
     * 3. Auto-tunes graph traversal prefetch (`po`, `pl`, `nl`).
     *
     * @param n_clusters  Number of entry medoids to select (maximum 512).
     * @param n_iter      Number of K-Means refinement iterations.
     * @param sample_size Number of rows sampled for K-Means (`0` selects 3% of graph size, min 1000).
     * @param threads     Worker thread count (`0` selects `hardware_concurrency()`).
     * @param use_routing If true, builds the K-Means entry routing graph; otherwise keeps static entry points.
     */
    void optimize(uint32_t n_clusters = 128, uint32_t n_iter = 15, size_t sample_size = 0, size_t threads = 0, bool use_routing = true) {
        if (!graph_ || graph_->size() == 0) return;
        constexpr uint32_t seed = 42;  // fixed sampling seed (K-Means candidate shuffle + init): not caller-facing.
        const size_t n = graph_->size();
        if (threads == 0) threads = std::max<size_t>(1, std::thread::hardware_concurrency());

        // 1. K-Means medoids query-routing selection (skippable via use_routing=false -> static entry).
        if (use_routing) {
            const size_t actual_sample = sample_size == 0 ? std::min<size_t>(n, std::max<size_t>(1000, n * 3 / 100)) : std::min(sample_size, n);
            std::vector<uint32_t> candidate_ids(n);
            std::iota(candidate_ids.begin(), candidate_ids.end(), 0);
            std::mt19937 rng(seed);
            std::shuffle(candidate_ids.begin(), candidate_ids.end(), rng);
            candidate_ids.resize(actual_sample);

            auto fit_candidates = [&](std::span<const float* const> rows, uint32_t dim, bool is_l2) {
                entry_selector_.fit(rows, candidate_ids, dim, is_l2, n_clusters, n_iter, seed, threads);
            };

            const bool use_reranker = reranker_ && reranker_->getBaseVectors() != nullptr && reranker_->getNumBaseVectors() >= n;
            const auto& space = use_reranker ? reranker_->getSpace() : graph_->getFeatureSpace();
            const uint32_t dim = space.dim();
            const bool is_l2 = space.metric().get_distance_kind() == deglib::distances::MetricDistanceKind::L2;
            const auto data_type = space.metric().get_data_type();

            auto get_raw_ptr = [&](uint32_t id) -> const void* {
                return use_reranker ? static_cast<const void*>(reranker_->getBaseVector(id)) : graph_->getFeatureVector(id);
            };

            if (data_type == deglib::distances::MetricDataType::FP32) {
                std::vector<const float*> rows(actual_sample);
                for (size_t i = 0; i < actual_sample; ++i) {
                    rows[i] = static_cast<const float*>(get_raw_ptr(candidate_ids[i]));
                }
                fit_candidates(rows, dim, is_l2);
            } else {
                std::vector<float> decoded(actual_sample * dim);
                std::vector<const float*> rows(actual_sample);

                auto decode_row = [&](const void* src, float* dst) {
                    if (data_type == deglib::distances::MetricDataType::FP16) {
                        deglib::distances::fp16::fp16_to_floats(static_cast<const uint16_t*>(src), dst, dim);
                    } else if (!transformations_.empty()) {
                        transformations_.back()->dequantize({static_cast<const std::byte*>(src), space.get_data_size()}, {dst, dim});
                    } else if (data_type == deglib::distances::MetricDataType::Int8) {
                        std::copy_n(static_cast<const int8_t*>(src), dim, dst);
                    } else if (data_type == deglib::distances::MetricDataType::Uint8) {
                        std::copy_n(static_cast<const uint8_t*>(src), dim, dst);
                    }
                };

                deglib::concurrent::parallel_batch_for(
                    0, actual_sample, threads,
                    [&](size_t begin, size_t end, size_t) {
                        for (size_t i = begin; i < end; ++i) {
                            decode_row(get_raw_ptr(candidate_ids[i]), decoded.data() + i * dim);
                        }
                    },
                    16
                );

                for (size_t i = 0; i < actual_sample; ++i) {
                    rows[i] = decoded.data() + i * dim;
                }
                fit_candidates(rows, dim, is_l2);
            }
        }

        // 2. Candidate reranker prefetch auto-tuning (library default: 300 sample queries = noise-stable).
        if (reranker_) {
            reranker_->optimize(300, 150);
        }

        // 3. Graph traversal prefetch tuning.
        graph_->optimize(300);
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

inline std::unique_ptr<Searcher> make_searcher(const deglib::graph::InternalGraph& graph) { return std::make_unique<Searcher>(graph); }

inline std::unique_ptr<Searcher> make_searcher(const deglib::graph::InternalGraph& graph, std::shared_ptr<Reranker> reranker) {
    return std::make_unique<Searcher>(graph, std::move(reranker));
}

inline std::unique_ptr<Searcher> make_searcher(const deglib::graph::InternalGraph& graph, Reranker reranker) {
    return std::make_unique<Searcher>(graph, std::move(reranker));
}

// Convenience wrapper accepting concrete FeatureTransformation instances directly
template <typename QuantT>
    requires(
        !std::is_same_v<std::decay_t<QuantT>, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>> &&
        !std::is_same_v<std::decay_t<QuantT>, Reranker> && !std::is_same_v<std::decay_t<QuantT>, std::shared_ptr<Reranker>>
    )
inline std::unique_ptr<Searcher> make_searcher(const deglib::graph::InternalGraph& graph, QuantT&& quantizer) {
    using Q = std::decay_t<QuantT>;
    if constexpr (std::is_base_of_v<deglib::transformation::FeatureTransformation, Q>) {
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transforms;
        transforms.push_back(std::make_shared<Q>(std::forward<QuantT>(quantizer)));
        return std::make_unique<Searcher>(graph, std::move(transforms));
    } else {
        return std::make_unique<Searcher>(graph);
    }
}

template <typename QuantT>
    requires(!std::is_same_v<std::decay_t<QuantT>, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>>)
inline std::unique_ptr<Searcher> make_searcher(const deglib::graph::InternalGraph& graph, QuantT&& quantizer, Reranker reranker) {
    using Q = std::decay_t<QuantT>;
    if constexpr (std::is_base_of_v<deglib::transformation::FeatureTransformation, Q>) {
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transforms;
        transforms.push_back(std::make_shared<Q>(std::forward<QuantT>(quantizer)));
        return std::make_unique<Searcher>(graph, std::move(transforms), std::move(reranker));
    } else {
        return std::make_unique<Searcher>(graph, std::move(reranker));
    }
}

}  // namespace deglib::search
