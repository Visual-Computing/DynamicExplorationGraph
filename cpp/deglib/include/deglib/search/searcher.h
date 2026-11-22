#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/search/kmeans.h"
#include "deglib/search/pareto_line.h"
#include "deglib/search/recall_table.h"
#include "deglib/search/reranker.h"
#include "deglib/search/result_list.h"
#include "deglib/transformation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
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

    // --- Recall Calibration & Pareto Optimization ---

    static constexpr std::array<float, 4> kDefaultRerankFactors = {1.0f, 1.25f, 1.5f, 2.0f};

    /**
     * Calibrates a single `RecallTable` (`Ef` or `Eps`) using sample queries drawn from the collection.
     *
     * @param k             Target neighbor count per query (`>= 1`).
     * @param mode          Parameter family (`RecallTable::Mode::Ef` or `RecallTable::Mode::Eps`).
     * @param rerank_factor Candidate expansion factor (`>= 1.0`, collapsed to `1.0` when no reranker is configured).
     * @param sample_count  Number of distinct collection vectors sampled as calibration queries (`0` uses default).
     * @param recall_range  Inclusive `[min_recall, max_recall]` range for evenly spaced targets.
     * @param recall_step   Step size across `recall_range`.
     * @param threads       Worker thread count (`0` uses hardware concurrency).
     */
    [[nodiscard]] RecallTable to_recall_table(
        uint32_t k,
        RecallTable::Mode mode,
        float rerank_factor = 1.0f,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall},
        float recall_step = RecallTable::kDefaultRecallStep,
        size_t threads = 0
    ) const {
        const auto resolved_targets = RecallTable::resolve_targets_from_range(recall_range, recall_step);
        const std::array<RecallTable::Mode, 1> modes{mode};
        const std::array<float, 1> rfs{rerank_factor};
        return build_tables_impl(k, modes, resolved_targets, rfs, sample_count, threads).front();
    }

    /**
     * Calibrates a single `RecallTable` (`Ef` or `Eps`) at explicit ascending target recall values.
     */
    [[nodiscard]] RecallTable to_recall_table(
        uint32_t k,
        RecallTable::Mode mode,
        std::span<const float> target_recalls,
        float rerank_factor = 1.0f,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        size_t threads = 0
    ) const {
        const auto resolved_targets = RecallTable::resolve_explicit_targets(target_recalls);
        const std::array<RecallTable::Mode, 1> modes{mode};
        const std::array<float, 1> rfs{rerank_factor};
        return build_tables_impl(k, modes, resolved_targets, rfs, sample_count, threads).front();
    }

    /**
     * Calibrates both `Ef` and `Eps` `RecallTable`s across `rerank_factors` in a shared single-pass evaluation.
     *
     * When `rerank_factors` is empty and a reranker is present, defaults to `{1.0, 1.25, 1.5, 2.0}`.
     * Without a reranker, calibrates `[Ef, Eps]` for `rerank_factor = 1.0`.
     */
    [[nodiscard]] std::vector<RecallTable> to_recall_tables(
        uint32_t k,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall},
        float recall_step = RecallTable::kDefaultRecallStep,
        std::span<const float> rerank_factors = {},
        size_t threads = 0
    ) const {
        const auto resolved_targets = RecallTable::resolve_targets_from_range(recall_range, recall_step);
        constexpr std::array<RecallTable::Mode, 2> kBothModes{RecallTable::Mode::Ef, RecallTable::Mode::Eps};
        return build_tables_impl(k, kBothModes, resolved_targets, rerank_factors, sample_count, threads);
    }

    /**
     * Calibrates both `Ef` and `Eps` `RecallTable`s across `rerank_factors` at explicit target recalls.
     */
    [[nodiscard]] std::vector<RecallTable> to_recall_tables(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::span<const float> rerank_factors = {},
        size_t threads = 0
    ) const {
        const auto resolved_targets = RecallTable::resolve_explicit_targets(target_recalls);
        constexpr std::array<RecallTable::Mode, 2> kBothModes{RecallTable::Mode::Ef, RecallTable::Mode::Eps};
        return build_tables_impl(k, kBothModes, resolved_targets, rerank_factors, sample_count, threads);
    }

    /**
     * Builds the unified end-to-end recall-latency `ParetoLine` from precomputed `RecallTable`s.
     */
    [[nodiscard]] ParetoLine
    to_pareto_line(std::span<const RecallTable> recall_tables, size_t sample_count = 0, size_t rounds = ParetoLine::kDefaultRounds) const {
        const uint32_t n = graph_ ? graph_->size() : 0;
        auto [line, q_count] = ParetoLine::validate_and_init(n, recall_tables, sample_count, rounds);
        if (q_count == 0) {
            return line;
        }

        const uint32_t effective_k = (std::min)(line.k(), n);
        const bool use_reranker = has_usable_reranker();
        const uint32_t dim = input_dim();
        const CalibQueryMode query_mode = determine_query_mode(use_reranker, dim);
        const std::vector<uint32_t> sample_indices = RecallTable::sample_distinct_query_indices(n, q_count, collect_excluded_entries());

        std::vector<float> f32_queries;
        if (query_mode == CalibQueryMode::F32) {
            f32_queries.resize(q_count * dim);
            for (size_t q = 0; q < q_count; ++q) {
                decode_base_to_f32(sample_indices[q], use_reranker, dim, f32_queries.data() + q * dim);
            }
        }

        const size_t graph_bytes = graph_->getFeatureSpace().get_data_size();
        const size_t rr_bytes = use_reranker ? reranker_->getSpace().get_data_size() : 0;
        QueryWorkspace ws;
        std::vector<uint32_t> out_indices(effective_k);
        std::vector<uint32_t> cand_ids;

        auto run_single_pass = [&](const ParetoLine::CandidateSpec& cand, size_t count) -> ParetoLine::TimingSample {
            const uint32_t ef = cand.use_ef ? static_cast<uint32_t>(cand.param) : 0;
            const float eps = cand.use_ef ? 0.0f : (static_cast<float>(cand.param) / 10000.0f);
            uint64_t total_graph_dists = 0;
            uint64_t total_rerank_dists = 0;
            size_t anti_dce = 0;

            const auto t_start = std::chrono::steady_clock::now();
            for (size_t q = 0; q < count; ++q) {
                if (query_mode == CalibQueryMode::F32) {
                    anti_dce += execute_search_single<float>(
                        f32_queries.data() + q * dim, effective_k, eps, ef, cand.use_ef, cand.rerank_factor, out_indices.data(), nullptr, false, false, 0, ws,
                        false
                    );
                } else if (query_mode == CalibQueryMode::F16) {
                    const auto* q_f16 = reinterpret_cast<const uint16_t*>(reranker_->getBaseVector(sample_indices[q]));
                    anti_dce += execute_search_single<uint16_t>(
                        q_f16, effective_k, eps, ef, cand.use_ef, cand.rerank_factor, out_indices.data(), nullptr, false, false, 0, ws, false
                    );
                } else {
                    ws.graph_distance_count = 0;
                    ws.graph_hop_count = 0;
                    ws.rerank_distance_count = 0;
                    const uint32_t v = sample_indices[q];
                    const std::span<const std::byte> q_span(graph_->getFeatureVector(v), graph_bytes);
                    const auto& entry_vertices = entry_selector_.has_entries() ? [&]() -> const std::vector<uint32_t>& {
                        entry_selector_.top_entries(q_span, 2, ws.entry_buf);
                        return ws.entry_buf;
                    }()
                        : graph_->getEntryVertexIndices();

                    if (cand.use_ef) {
                        graph_->search_ef(ws, q_span, entry_vertices, cand.num_candidates, ef, true, nullptr, 0);
                    } else {
                        graph_->search(ws, q_span, entry_vertices, cand.num_candidates, eps, nullptr, 0);
                    }

                    const uint32_t found_cands = static_cast<uint32_t>(ws.result_count(cand.num_candidates));
                    if (found_cands > 0 && use_reranker && cand.num_candidates > effective_k) {
                        if (cand_ids.size() < found_cands) {
                            cand_ids.resize(found_cands);
                        }
                        ws.extract_internal_ids(found_cands, cand_ids.data());
                        ws.rerank_distance_count = found_cands;
                        const uint32_t found = reranker_->rerank(
                            std::span<const std::byte>(reranker_->getBaseVector(v), rr_bytes), std::span<const uint32_t>(cand_ids.data(), found_cands),
                            effective_k, std::span<uint32_t>(out_indices.data(), effective_k), {}, false, false
                        );
                        for (uint32_t i = 0; i < found; ++i) {
                            out_indices[i] = graph_->getExternalLabel(out_indices[i]);
                        }
                        anti_dce += found;
                    } else if (found_cands > 0) {
                        anti_dce += ws.drain_into(*graph_, effective_k, out_indices.data(), nullptr, false, false);
                    }
                }
                total_graph_dists += ws.graph_distance_count;
                total_rerank_dists += ws.rerank_distance_count;
            }
            const double total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
            (void)anti_dce;
            const double denom = static_cast<double>((std::max)(count, size_t{1}));
            return {(std::max)(total_us / denom, 1e-6), static_cast<double>(total_graph_dists) / denom, static_cast<double>(total_rerank_dists) / denom};
        };

        return ParetoLine::run_pareto_selection(std::move(line), n, rounds, use_reranker, run_single_pass);
    }

    /**
     * Convenience overload constructing a `ParetoLine` from a single `RecallTable`.
     */
    [[nodiscard]] ParetoLine to_pareto_line(const RecallTable& recall_table, size_t sample_count = 0, size_t rounds = ParetoLine::kDefaultRounds) const {
        const std::array<RecallTable, 1> single{recall_table};
        return to_pareto_line(std::span<const RecallTable>(single), sample_count, rounds);
    }

    /**
     * Calibrates `(Ef, Eps) x rerank_factors` recall tables and selects the Pareto-optimal frontier.
     */
    [[nodiscard]] ParetoLine to_pareto_line(
        uint32_t k,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall},
        float recall_step = RecallTable::kDefaultRecallStep,
        std::span<const float> rerank_factors = {},
        size_t threads = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const {
        if (rounds == 0) {
            throw std::invalid_argument("to_pareto_line: rounds must be >= 1");
        }
        const auto tables = to_recall_tables(k, sample_count, recall_range, recall_step, rerank_factors, threads);
        return to_pareto_line(tables, sample_count, rounds);
    }

    /**
     * Calibrates `(Ef, Eps) x rerank_factors` recall tables at explicit targets and selects the Pareto frontier.
     */
    [[nodiscard]] ParetoLine to_pareto_line(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::span<const float> rerank_factors = {},
        size_t threads = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const {
        if (rounds == 0) {
            throw std::invalid_argument("to_pareto_line: rounds must be >= 1");
        }
        const auto tables = to_recall_tables(k, target_recalls, sample_count, rerank_factors, threads);
        return to_pareto_line(tables, sample_count, rounds);
    }

  private:
    enum class CalibQueryMode : uint8_t { F32, F16, RawBytes };

    struct CalibrationContext {
        bool use_reranker = false;
        CalibQueryMode query_mode = CalibQueryMode::RawBytes;
        uint32_t dim = 0;
        size_t graph_bytes = 0;
        size_t rr_bytes = 0;
        std::vector<uint32_t> sample_indices;
        std::vector<std::byte> graph_queries;
        std::vector<std::byte> rr_queries;
        std::vector<const void*> gt_ptrs;
        bool use_kmeans_entries = false;
        std::vector<uint32_t> entry_ids;
        std::vector<uint32_t> entry_counts;
        std::vector<float> gt_thresholds;
        double bf_batch_us = 10000.0;

        [[nodiscard]] std::span<const std::byte> graph_query_span(size_t q) const noexcept { return {graph_queries.data() + q * graph_bytes, graph_bytes}; }

        [[nodiscard]] std::span<const uint32_t> entry_vertices(size_t q, const deglib::graph::InternalGraph& graph) const noexcept {
            if (use_kmeans_entries) {
                return {entry_ids.data() + q * 2, entry_counts[q]};
            }
            return graph.getEntryVertexIndices();
        }
    };

    [[nodiscard]] bool has_usable_reranker() const noexcept {
        return graph_ != nullptr && reranker_ != nullptr && reranker_->getBaseVectors() != nullptr && reranker_->getNumBaseVectors() >= graph_->size();
    }

    [[nodiscard]] uint32_t input_dim() const noexcept {
        return transformations_.empty() ? graph_->getFeatureSpace().dim() : transformations_.front()->input_dim();
    }

    [[nodiscard]] CalibQueryMode determine_query_mode(bool use_reranker, uint32_t dim) const {
        const auto g_dt = graph_->getFeatureSpace().metric().get_data_type();
        const bool graph_accepts_float =
            !transformations_.empty() || g_dt == deglib::distances::MetricDataType::FP32 || g_dt == deglib::distances::MetricDataType::FP16;
        if (use_reranker) {
            if (!graph_accepts_float) {
                return CalibQueryMode::RawBytes;
            }
            const auto& rr_space = reranker_->getSpace();
            const auto rr_dt = rr_space.metric().get_data_type();
            const auto& rr_xfrms = reranker_->getTransformations();
            if (rr_space.dim() == dim &&
                ((rr_dt == deglib::distances::MetricDataType::FP32 && rr_xfrms.empty()) ||
                 (rr_dt == deglib::distances::MetricDataType::FP16 && rr_xfrms.size() == 1 && rr_xfrms[0]->tag() == deglib::transformation::TAG_SQ16))) {
                return CalibQueryMode::F32;
            }
            if (rr_space.dim() == dim && rr_dt == deglib::distances::MetricDataType::FP16 && rr_xfrms.empty()) {
                return CalibQueryMode::F16;
            }
            return CalibQueryMode::RawBytes;
        }
        if (transformations_.empty() && (g_dt == deglib::distances::MetricDataType::FP32 || g_dt == deglib::distances::MetricDataType::FP16)) {
            return CalibQueryMode::F32;
        }
        return CalibQueryMode::RawBytes;
    }

    [[nodiscard]] std::vector<float> resolve_rerank_factors(std::span<const float> requested, bool use_reranker) const {
        if (requested.empty()) {
            if (use_reranker) {
                return std::vector<float>(kDefaultRerankFactors.begin(), kDefaultRerankFactors.end());
            }
            return {1.0f};
        }
        for (const float rf : requested) {
            if (!std::isfinite(rf) || rf < 1.0f) {
                throw std::invalid_argument(std::format("rerank_factor must be finite and >= 1.0, got {}", rf));
            }
        }
        if (!use_reranker) {
            return {1.0f};
        }
        std::vector<float> sorted(requested.begin(), requested.end());
        std::sort(sorted.begin(), sorted.end());
        std::vector<float> unique;
        unique.reserve(sorted.size());
        for (const float rf : sorted) {
            if (unique.empty() || (rf - unique.back()) > 1e-6f) {
                unique.push_back(rf);
            }
        }
        return unique;
    }

    [[nodiscard]] std::vector<uint32_t> collect_excluded_entries() const {
        const auto& static_entries = graph_->getEntryVertexIndices();
        if (!entry_selector_.has_entries()) {
            return static_entries;
        }
        const auto& km_entries = entry_selector_.entries();
        std::vector<uint32_t> combined;
        combined.reserve(static_entries.size() + km_entries.size());
        combined.insert(combined.end(), static_entries.begin(), static_entries.end());
        combined.insert(combined.end(), km_entries.begin(), km_entries.end());
        return combined;
    }

    void decode_base_to_f32(uint32_t vertex_id, bool use_reranker, uint32_t dim, float* dst) const {
        if (use_reranker) {
            const auto& rr_xfrms = reranker_->getTransformations();
            if (rr_xfrms.empty()) {
                std::memcpy(dst, reranker_->getBaseVector(vertex_id), static_cast<size_t>(dim) * sizeof(float));
            } else {
                const size_t rr_bytes = reranker_->getSpace().get_data_size();
                rr_xfrms[0]->dequantize({reranker_->getBaseVector(vertex_id), rr_bytes}, {dst, dim});
            }
            return;
        }
        const auto g_dt = graph_->getFeatureSpace().metric().get_data_type();
        if (g_dt == deglib::distances::MetricDataType::FP32) {
            std::memcpy(dst, graph_->getFeatureVector(vertex_id), static_cast<size_t>(dim) * sizeof(float));
        } else {
            deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(graph_->getFeatureVector(vertex_id)), dst, dim);
        }
    }

    [[nodiscard]] CalibrationContext build_calibration_context(size_t q_count, uint32_t effective_k, size_t actual_threads, size_t eval_threads) const {
        const uint32_t n = graph_->size();
        CalibrationContext ctx;
        ctx.use_reranker = has_usable_reranker();
        ctx.dim = input_dim();
        ctx.query_mode = determine_query_mode(ctx.use_reranker, ctx.dim);
        ctx.graph_bytes = graph_->getFeatureSpace().get_data_size();
        ctx.rr_bytes = ctx.use_reranker ? reranker_->getSpace().get_data_size() : 0;

        ctx.sample_indices = RecallTable::sample_distinct_query_indices(n, q_count, collect_excluded_entries());
        ctx.graph_queries.resize(q_count * ctx.graph_bytes);
        if (ctx.use_reranker) {
            ctx.rr_queries.resize(q_count * ctx.rr_bytes);
        }
        ctx.gt_ptrs.resize(q_count);

        QueryWorkspace ws;
        alignas(64) std::byte stack_qbuf[512];
        std::vector<float> f32_raw(ctx.query_mode == CalibQueryMode::F32 ? ctx.dim : 0);
        std::vector<float> f32_norm(normalize_query_ && ctx.query_mode == CalibQueryMode::F32 ? ctx.dim : 0);

        for (size_t q = 0; q < q_count; ++q) {
            const uint32_t v = ctx.sample_indices[q];
            if (ctx.query_mode == CalibQueryMode::F32) {
                decode_base_to_f32(v, ctx.use_reranker, ctx.dim, f32_raw.data());
                const float* eff_q = f32_raw.data();
                if (normalize_query_) {
                    deglib::transformation::detail::normalize_vector_l2(eff_q, f32_norm.data(), ctx.dim);
                    eff_q = f32_norm.data();
                }
                const std::byte* g_ptr = prepare_query(eff_q, ctx.dim, ws, stack_qbuf);
                std::memcpy(ctx.graph_queries.data() + q * ctx.graph_bytes, g_ptr, ctx.graph_bytes);

                if (ctx.use_reranker) {
                    const auto& rr_xfrms = reranker_->getTransformations();
                    std::byte* rr_dst = ctx.rr_queries.data() + q * ctx.rr_bytes;
                    if (rr_xfrms.empty()) {
                        std::memcpy(rr_dst, eff_q, ctx.rr_bytes);
                    } else {
                        rr_xfrms[0]->transform_query(std::span<const float>(eff_q, ctx.dim), std::span<std::byte>(rr_dst, ctx.rr_bytes));
                    }
                }
            } else if (ctx.query_mode == CalibQueryMode::F16) {
                const auto* f16_ptr = reinterpret_cast<const uint16_t*>(reranker_->getBaseVector(v));
                const std::byte* g_ptr = prepare_query(f16_ptr, ctx.dim, ws, stack_qbuf);
                std::memcpy(ctx.graph_queries.data() + q * ctx.graph_bytes, g_ptr, ctx.graph_bytes);
                std::memcpy(ctx.rr_queries.data() + q * ctx.rr_bytes, reranker_->getBaseVector(v), ctx.rr_bytes);
            } else {
                std::memcpy(ctx.graph_queries.data() + q * ctx.graph_bytes, graph_->getFeatureVector(v), ctx.graph_bytes);
                if (ctx.use_reranker) {
                    std::memcpy(ctx.rr_queries.data() + q * ctx.rr_bytes, reranker_->getBaseVector(v), ctx.rr_bytes);
                }
            }
            ctx.gt_ptrs[q] = ctx.use_reranker ? static_cast<const void*>(ctx.rr_queries.data() + q * ctx.rr_bytes)
                                              : static_cast<const void*>(ctx.graph_queries.data() + q * ctx.graph_bytes);
        }

        if (entry_selector_.has_entries()) {
            ctx.use_kmeans_entries = true;
            ctx.entry_ids.resize(q_count * 2, 0);
            ctx.entry_counts.resize(q_count, 0);
            std::vector<uint32_t> top_buf;
            for (size_t q = 0; q < q_count; ++q) {
                entry_selector_.top_entries(ctx.graph_query_span(q), 2, top_buf);
                const size_t cnt = (std::min)(top_buf.size(), size_t{2});
                ctx.entry_counts[q] = static_cast<uint32_t>(cnt);
                for (size_t i = 0; i < cnt; ++i) {
                    ctx.entry_ids[q * 2 + i] = top_buf[i];
                }
            }
        }

        if (ctx.use_reranker) {
            ctx.gt_thresholds = RecallTable::compute_exact_gt_thresholds(
                n, reranker_->getSpace(), [this](uint32_t id) -> const void* { return reranker_->getBaseVector(id); }, ctx.gt_ptrs, effective_k, actual_threads
            );
            ctx.bf_batch_us = RecallTable::estimate_brute_force_batch_us(
                n, reranker_->getSpace(), [this](uint32_t id) -> const void* { return reranker_->getBaseVector(id); }, reranker_->getPo(),
                static_cast<size_t>(reranker_->getPl()) * deglib::memory::L1_CACHE_LINE_SIZE, ctx.gt_ptrs, eval_threads
            );
        } else {
            ctx.gt_thresholds = RecallTable::compute_exact_gt_thresholds(
                n, graph_->getFeatureSpace(), [this](uint32_t id) -> const void* { return graph_->getFeatureVector(id); }, ctx.gt_ptrs, effective_k,
                actual_threads
            );
            ctx.bf_batch_us = RecallTable::estimate_brute_force_batch_us(
                n, graph_->getFeatureSpace(), [this](uint32_t id) -> const void* { return graph_->getFeatureVector(id); }, graph_->getPo(),
                static_cast<size_t>(graph_->getPl()) * deglib::memory::L1_CACHE_LINE_SIZE, ctx.gt_ptrs, eval_threads
            );
        }

        return ctx;
    }

    [[nodiscard]] std::vector<RecallTable> build_tables_impl(
        uint32_t k,
        std::span<const RecallTable::Mode> modes,
        std::span<const float> resolved_targets,
        std::span<const float> requested_rfs,
        size_t sample_count,
        size_t threads
    ) const {
        RecallTable::validate_common_args(k, sample_count);
        const bool use_reranker = has_usable_reranker();
        const std::vector<float> rfs = resolve_rerank_factors(requested_rfs, use_reranker);

        const uint32_t n = graph_ ? graph_->size() : 0;
        if (n == 0) {
            std::vector<RecallTable> empty_tables;
            empty_tables.reserve(rfs.size() * modes.size());
            for (const float rf : rfs) {
                for (const RecallTable::Mode mode : modes) {
                    RecallTable t;
                    t.k_ = k;
                    t.sample_count_ = 0;
                    t.mode_ = mode;
                    t.rerank_factor_ = rf;
                    empty_tables.push_back(std::move(t));
                }
            }
            return empty_tables;
        }

        const size_t q_count = (std::min)(sample_count, static_cast<size_t>(n));
        const uint32_t effective_k = (std::min)(k, n);
        const size_t actual_threads = (threads == 0) ? (std::max)(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency())) : threads;
        const size_t eval_threads = (std::min)(actual_threads, q_count);

        const CalibrationContext ctx = build_calibration_context(q_count, effective_k, actual_threads, eval_threads);
        const double soft_budget_us = 0.75 * ctx.bf_batch_us;
        const double hard_abort_us = 1.0 * ctx.bf_batch_us;

        const size_t num_rfs = rfs.size();
        std::vector<uint32_t> cand_counts(num_rfs);
        for (size_t r = 0; r < num_rfs; ++r) {
            cand_counts[r] = ParetoLine::compute_num_candidates(effective_k, rfs[r], ctx.use_reranker);
        }
        const uint32_t max_cands = cand_counts.back();

        const auto* rr_space_ptr = ctx.use_reranker ? &reranker_->getSpace() : nullptr;
        const auto rr_batch_dist = ctx.use_reranker ? rr_space_ptr->get_batch_dist_func() : nullptr;
        const void* rr_dist_param = ctx.use_reranker ? rr_space_ptr->get_dist_func_param() : nullptr;
        const int32_t rr_po = ctx.use_reranker ? reranker_->getPo() : 0;
        const size_t rr_prefetch_bytes = ctx.use_reranker ? static_cast<size_t>(reranker_->getPl()) * deglib::memory::L1_CACHE_LINE_SIZE : 0;
        const double total_expected = static_cast<double>(q_count) * static_cast<double>(effective_k);

        auto eval_ef_all = [&](int32_t ef, double hard_limit_us) -> std::vector<RecallTable::EvalResult> {
            std::atomic<bool> aborted{false};
            std::vector<uint64_t> thread_hits(eval_threads * num_rfs, 0);

            const auto t_start = std::chrono::steady_clock::now();
            deglib::concurrent::parallel_batch_for(
                size_t{0}, q_count, eval_threads,
                [&](size_t begin, size_t end, size_t thread_id) {
                    thread_local deglib::graph::SearchWorkspace ws;
                    thread_local std::vector<const void*> rr_ptrs;
                    thread_local std::vector<float> rr_dists;
                    uint64_t* local_hits = thread_hits.data() + thread_id * num_rfs;

                    for (size_t q = begin; q < end; ++q) {
                        if (((q - begin) & 3u) == 0u) {
                            if (aborted.load(std::memory_order_relaxed)) {
                                break;
                            }
                            const double elapsed_so_far = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
                            if (elapsed_so_far >= hard_limit_us) {
                                aborted.store(true, std::memory_order_relaxed);
                                break;
                            }
                        }

                        graph_->search_ef(ws, ctx.graph_query_span(q), ctx.entry_vertices(q, *graph_), max_cands, static_cast<uint32_t>(ef), true, nullptr, 0);
                        const float threshold = ctx.gt_thresholds[q];

                        if (!ctx.use_reranker) {
                            const size_t found = (std::min)(static_cast<size_t>(effective_k), ws.pool.size());
                            uint32_t hits = 0;
                            for (size_t i = 0; i < found; ++i) {
                                if (ws.pool[i].getDistance() <= threshold) {
                                    ++hits;
                                }
                            }
                            const uint32_t capped = (std::min)(effective_k, hits);
                            for (size_t r = 0; r < num_rfs; ++r) {
                                local_hits[r] += capped;
                            }
                        } else {
                            const size_t found = (std::min)(static_cast<size_t>(max_cands), ws.pool.size());
                            if (found == 0) {
                                continue;
                            }
                            if (rr_ptrs.size() < found) {
                                rr_ptrs.resize(found);
                                rr_dists.resize(found);
                            }
                            for (size_t i = 0; i < found; ++i) {
                                rr_ptrs[i] = reranker_->getBaseVector(ws.pool[i].getIdentifier());
                            }
                            rr_batch_dist(ctx.gt_ptrs[q], rr_ptrs.data(), found, rr_dist_param, rr_dists.data(), rr_po, rr_prefetch_bytes);

                            size_t r = 0;
                            uint32_t hits = 0;
                            for (size_t j = 0; j < found; ++j) {
                                if (rr_dists[j] <= threshold) {
                                    ++hits;
                                }
                                while (r < num_rfs && (j + 1) >= cand_counts[r]) {
                                    local_hits[r] += (std::min)(effective_k, hits);
                                    ++r;
                                }
                            }
                            while (r < num_rfs) {
                                local_hits[r] += (std::min)(effective_k, hits);
                                ++r;
                            }
                        }
                    }
                },
                4
            );

            const double elapsed_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
            if (aborted.load(std::memory_order_relaxed)) {
                return std::vector<RecallTable::EvalResult>(num_rfs, RecallTable::EvalResult{0.0f, elapsed_us, true});
            }

            std::vector<RecallTable::EvalResult> results(num_rfs);
            for (size_t r = 0; r < num_rfs; ++r) {
                uint64_t total_hits = 0;
                for (size_t tid = 0; tid < eval_threads; ++tid) {
                    total_hits += thread_hits[tid * num_rfs + r];
                }
                results[r] = RecallTable::EvalResult{static_cast<float>(static_cast<double>(total_hits) / total_expected), elapsed_us, false};
            }
            return results;
        };

        auto eval_eps = [&](uint32_t cand_count, float eps, double hard_limit_us) -> RecallTable::EvalResult {
            std::atomic<bool> aborted{false};
            std::vector<uint64_t> thread_hits(eval_threads, 0);

            const auto t_start = std::chrono::steady_clock::now();
            deglib::concurrent::parallel_batch_for(
                size_t{0}, q_count, eval_threads,
                [&](size_t begin, size_t end, size_t thread_id) {
                    thread_local deglib::graph::SearchWorkspace ws;
                    thread_local std::vector<const void*> rr_ptrs;
                    thread_local std::vector<float> rr_dists;
                    uint64_t local_hits = 0;

                    for (size_t q = begin; q < end; ++q) {
                        if (((q - begin) & 3u) == 0u) {
                            if (aborted.load(std::memory_order_relaxed)) {
                                break;
                            }
                            const double elapsed_so_far = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
                            if (elapsed_so_far >= hard_limit_us) {
                                aborted.store(true, std::memory_order_relaxed);
                                break;
                            }
                        }

                        graph_->search(ws, ctx.graph_query_span(q), ctx.entry_vertices(q, *graph_), cand_count, eps, nullptr, 0);
                        const float threshold = ctx.gt_thresholds[q];

                        if (!ctx.use_reranker) {
                            while (ws.results.size() > effective_k) {
                                ws.results.pop();
                            }
                            uint32_t hits = 0;
                            for (size_t i = 0; i < ws.results.size(); ++i) {
                                if (ws.results[i].getDistance() <= threshold) {
                                    ++hits;
                                }
                            }
                            local_hits += (std::min)(effective_k, hits);
                        } else {
                            while (ws.results.size() > cand_count) {
                                ws.results.pop();
                            }
                            const size_t found = ws.result_count(cand_count);
                            if (found == 0) {
                                continue;
                            }
                            if (rr_ptrs.size() < found) {
                                rr_ptrs.resize(found);
                                rr_dists.resize(found);
                            }
                            for (size_t i = 0; i < found; ++i) {
                                rr_ptrs[i] = reranker_->getBaseVector(ws.results[i].getIdentifier());
                            }
                            rr_batch_dist(ctx.gt_ptrs[q], rr_ptrs.data(), found, rr_dist_param, rr_dists.data(), rr_po, rr_prefetch_bytes);

                            uint32_t hits = 0;
                            for (size_t i = 0; i < found; ++i) {
                                if (rr_dists[i] <= threshold) {
                                    ++hits;
                                }
                            }
                            local_hits += (std::min)(effective_k, hits);
                        }
                    }

                    thread_hits[thread_id] += local_hits;
                },
                4
            );

            const double elapsed_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
            if (aborted.load(std::memory_order_relaxed)) {
                return {0.0f, elapsed_us, true};
            }
            uint64_t total_hits = 0;
            for (const uint64_t h : thread_hits) {
                total_hits += h;
            }
            return {static_cast<float>(static_cast<double>(total_hits) / total_expected), elapsed_us, false};
        };

        const bool want_ef = std::find(modes.begin(), modes.end(), RecallTable::Mode::Ef) != modes.end();
        const bool want_eps = std::find(modes.begin(), modes.end(), RecallTable::Mode::Eps) != modes.end();

        std::vector<RecallTable::CalibratedAxisResult> ef_results(want_ef ? num_rfs : 0);
        if (want_ef) {
            std::unordered_map<int32_t, std::vector<RecallTable::EvalResult>> ef_probe_cache;
            for (size_t r = 0; r < num_rfs; ++r) {
                ef_results[r] = RecallTable::calibrate_axis(
                    RecallTable::kEfLadder, resolved_targets, soft_budget_us, hard_abort_us,
                    [&](int32_t ef, double limit_us) -> RecallTable::EvalResult {
                        if (const auto it = ef_probe_cache.find(ef); it != ef_probe_cache.end()) {
                            return it->second[r];
                        }
                        auto all_res = eval_ef_all(ef, limit_us);
                        if (!all_res.front().aborted) {
                            ef_probe_cache.emplace(ef, all_res);
                        }
                        return all_res[r];
                    },
                    /*plateau_min_param=*/80
                );
            }
        }

        std::vector<RecallTable::CalibratedAxisResult> eps_results(want_eps ? num_rfs : 0);
        if (want_eps) {
            std::unordered_map<uint32_t, RecallTable::CalibratedAxisResult> eps_results_by_cands;
            for (size_t r = 0; r < num_rfs; ++r) {
                const uint32_t cand_count = cand_counts[r];
                if (const auto it = eps_results_by_cands.find(cand_count); it != eps_results_by_cands.end()) {
                    eps_results[r] = it->second;
                    continue;
                }
                auto res = RecallTable::calibrate_axis(
                    RecallTable::kEpsLadder, resolved_targets, soft_budget_us, hard_abort_us,
                    [&](int32_t eps_scaled, double limit_us) { return eval_eps(cand_count, static_cast<float>(eps_scaled) / 10000.0f, limit_us); },
                    /*plateau_min_param=*/400
                );
                eps_results_by_cands.emplace(cand_count, res);
                eps_results[r] = std::move(res);
            }
        }

        std::vector<RecallTable> tables;
        tables.reserve(num_rfs * modes.size());
        for (size_t r = 0; r < num_rfs; ++r) {
            for (const RecallTable::Mode mode : modes) {
                if (mode == RecallTable::Mode::Ef) {
                    tables.push_back(RecallTable::make_from_axis(k, q_count, RecallTable::Mode::Ef, rfs[r], ef_results[r]));
                } else {
                    tables.push_back(RecallTable::make_from_axis(k, q_count, RecallTable::Mode::Eps, rfs[r], eps_results[r]));
                }
            }
        }
        return tables;
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
