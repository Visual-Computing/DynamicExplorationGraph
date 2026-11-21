#pragma once

#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/utils/memory.h"
#include "deglib/search/result_list.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for friend access
namespace deglib::builder {
class EvenRegularGraphBuilder;
}

namespace deglib::graph {

/**
 * Priority Queue based on std::vector using binary heap operations.
 */
template <class Compare, class ObjectType>
class PQV : public std::vector<ObjectType> {
    Compare comp;

  public:
    explicit PQV(Compare cmp = Compare()) : comp(cmp) {}

    /// Const-correct access to the top element of the heap.
    [[nodiscard]] const ObjectType& top() const { return this->front(); }

    /// Construct element in-place at the end and restore heap invariants.
    template <class... Args>
    void emplace(Args&&... args) {
        this->emplace_back(std::forward<Args>(args)...);
        std::push_heap(this->begin(), this->end(), comp);
    }

    /// Removes the top element from the heap.
    void pop() {
        std::pop_heap(this->begin(), this->end(), comp);
        this->pop_back();
    }

    /// Replaces the top element in-place, sifts it down in single-pass O(log k) time,
    /// and returns a const reference to the new top element.
    template <class... Args>
    const ObjectType& replace_top(Args&&... args) {
        ObjectType val(std::forward<Args>(args)...);
        size_t len = this->size();
        size_t parent = 0;
        size_t child = 1;

        while (child < len) {
            // Find the larger child (comp(a, b) means a < b for max-heap)
            if (child + 1 < len && comp((*this)[child], (*this)[child + 1])) {
                child++;
            }
            // If val is not less than the larger child, heap property holds
            if (!comp(val, (*this)[child])) {
                break;
            }
            (*this)[parent] = std::move((*this)[child]);
            parent = child;
            child = 2 * parent + 1;
        }
        (*this)[parent] = std::move(val);
        return this->front();
    }

    /// Re-establishes the heap order using the internal comparator.
    void heapify() { std::make_heap(this->begin(), this->end(), comp); }

    /// Sorts the heap elements in ascending order using the internal comparator,
    /// consuming the heap property. After calling this, top() is no longer valid
    /// until heapify() is called again.
    void sort() { std::sort_heap(this->begin(), this->end(), comp); }
};

/**
 * Search result set (Max-Heap: top() returns the entry with the LARGEST distance).
 *
 * Note: Elements are maintained in max-heap order within the underlying vector.
 * Iterating or indexing the container directly will yield elements in arbitrary (unsorted) heap order.
 * To access elements in ordered sequence (closest distance first), extract them using top() and pop(),
 * or call sort() explicitly to sort the underlying storage in ascending distance order.
 */
using ResultSet = PQV<std::less<ObjectDistance>, ObjectDistance>;

// Unchecked search candidates (Min-Heap: top() returns the entry with the SMALLEST distance)
using UncheckedSet = PQV<std::greater<ObjectDistance>, ObjectDistance>;

/**
 * Reusable scratch workspace and result container for all internal graph search and exploration kernels.
 *
 * Eliminates per-query heap allocations when passed by reference across repeated searches
 * (e.g. in Searcher, batch loops, or graph builder), and records exact graph distance evaluations
 * and vertex expansions (hops).
 */
struct alignas(64) SearchWorkspace {
    deglib::search::ResultList pool;    ///< Sorted beam buffer populated by ef and efQueue search (offset 0, 40 bytes).
    uint32_t graph_distance_count = 0;  ///< Number of graph distance evaluations performed.
    uint32_t graph_hop_count = 0;       ///< Number of vertex expansions (hops) performed.
    bool used_pool = false;             ///< True when results live in `pool` (ef), false when in `results` (eps).

    ResultSet results;           ///< Max-heap result container populated by eps search.
    UncheckedSet next_vertices;  ///< Min-heap exploration frontier used by eps and efQueue search.

    /// Resets buffers and telemetry counters for an eps search.
    void reset_for_eps(size_t k, size_t edges_per_vertex) noexcept {
        results.clear();
        results.reserve(k + 1);
        next_vertices.clear();
        next_vertices.reserve(k * edges_per_vertex);
        graph_distance_count = 0;
        graph_hop_count = 0;
        used_pool = false;
    }

    /// Resets buffers and telemetry counters for an ef search.
    void reset_for_ef(uint32_t k, uint32_t ef, size_t edges_per_vertex, bool need_queue) noexcept {
        const int32_t cap = static_cast<int32_t>(std::max(k, ef));
        pool.reset(static_cast<int32_t>(ef), cap);
        if (need_queue) {
            next_vertices.clear();
            next_vertices.reserve(static_cast<size_t>(ef) * edges_per_vertex);
        }
        graph_distance_count = 0;
        graph_hop_count = 0;
        used_pool = true;
    }

    /// Returns the number of retained results from the most recent search.
    [[nodiscard]] size_t size() const noexcept { return used_pool ? pool.size() : results.size(); }

    /// Returns the number of retained results capped at `k`.
    [[nodiscard]] size_t result_count(size_t k) const noexcept { return std::min(k, size()); }

    /// Returns whether the most recent search produced zero results.
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /// Extracts up to `k` internal vertex IDs without sorting or popping (used for post-graph reranking).
    uint32_t extract_internal_ids(uint32_t k, uint32_t* out_internal_ids) const noexcept {
        if (used_pool) {
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>(k, pool.size()));
            for (uint32_t i = 0; i < count; ++i) {
                out_internal_ids[i] = pool[i].getIdentifier();
            }
            return count;
        }
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(k, results.size()));
        for (uint32_t i = 0; i < count; ++i) {
            out_internal_ids[i] = results[i].getIdentifier();
        }
        return count;
    }

    /**
     * Drains up to `k` results into flat output buffers, mapping internal indices to external labels via `graph`.
     *
     * - For `ef` search (`used_pool == true`), `pool` is already sorted ascending by distance.
     * - For `eps` search (`used_pool == false`), pops the max-heap in reverse for ascending order when
     *   `unsorted == false`, or copies the heap buffer linearly in O(k) when `unsorted == true`.
     * - When `pad_missing == true` and fewer than `k` results exist, unfilled slots up to `k` are padded
     *   with `UINT32_MAX` in `out_indices` and `quiet_NaN()` in `out_distances`.
     */
    template <typename GraphT>
    uint32_t drain_into(const GraphT& graph, uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted, bool pad_missing = false) noexcept {
        uint32_t count = 0;
        if (used_pool) {
            count = static_cast<uint32_t>(std::min<size_t>(k, pool.size()));
            for (uint32_t i = 0; i < count; ++i) {
                out_indices[i] = graph.getExternalLabel(pool[i].getIdentifier());
            }
            if (out_distances) {
                for (uint32_t i = 0; i < count; ++i) {
                    out_distances[i] = pool[i].getDistance();
                }
            }
        } else {
            while (results.size() > k) {
                results.pop();
            }
            count = static_cast<uint32_t>(results.size());
            if (unsorted) {
                for (uint32_t i = 0; i < count; ++i) {
                    out_indices[i] = graph.getExternalLabel(results[i].getIdentifier());
                }
                if (out_distances) {
                    for (uint32_t i = 0; i < count; ++i) {
                        out_distances[i] = results[i].getDistance();
                    }
                }
            } else if (out_distances) {
                for (uint32_t i = count; i > 0; --i) {
                    const auto next = results.top();
                    out_indices[i - 1] = graph.getExternalLabel(next.getIdentifier());
                    out_distances[i - 1] = next.getDistance();
                    results.pop();
                }
            } else {
                for (uint32_t i = count; i > 0; --i) {
                    out_indices[i - 1] = graph.getExternalLabel(results.top().getIdentifier());
                    results.pop();
                }
            }
        }

        if (pad_missing) {
            for (uint32_t i = count; i < k; ++i) {
                out_indices[i] = std::numeric_limits<uint32_t>::max();
                if (out_distances) {
                    out_distances[i] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        return count;
    }
};

/**
 * Abstract base interface for all low-level internal graph representations.
 * All methods operate on internal_index (0..N-1) for maximum memory performance.
 */
class InternalGraph {
  public:
    virtual ~InternalGraph() = default;

    [[nodiscard]] virtual uint32_t size() const noexcept = 0;
    [[nodiscard]] virtual uint8_t getEdgesPerVertex() const noexcept = 0;
    [[nodiscard]] virtual const deglib::distances::FloatSpace& getFeatureSpace() const noexcept = 0;

    [[nodiscard]] virtual uint32_t getExternalLabel(uint32_t internal_index) const noexcept = 0;
    [[nodiscard]] virtual uint32_t getInternalIndex(uint32_t external_label) const noexcept = 0;

    /**
     * @brief Returns all external labels ordered by their internal index (0 .. size()-1).
     *
     * In static graphs (ReadOnlyGraph or mutable graphs without deletions), this matches the builder's
     * insertion order. In dynamic graphs, deletions swap vertices, so internal positions shift; this
     * method reflects the current internal layout.
     *
     * Use this vector to align external feature arrays (e.g. for custom_features or rerank_data)
     * into the contiguous internal order expected by C++ methods.
     */
    [[nodiscard]] virtual std::vector<uint32_t> getExternalLabels() const {
        const uint32_t n = size();
        std::vector<uint32_t> labels(n);
        for (uint32_t i = 0; i < n; ++i) {
            labels[i] = getExternalLabel(i);
        }
        return labels;
    }

    [[nodiscard]] virtual const uint32_t* getNeighborIndices(uint32_t internal_index) const noexcept = 0;
    [[nodiscard]] virtual const std::byte* getFeatureVector(uint32_t internal_index) const noexcept = 0;

    [[nodiscard]] virtual bool hasVertex(uint32_t external_label) const noexcept = 0;
    [[nodiscard]] virtual bool hasEdge(uint32_t internal_index, uint32_t neighbor_index) const noexcept = 0;

    const std::vector<uint32_t>& getEntryVertexIndices() const { return entry_vertex_indices_; }
    void setEntryVertexIndices(std::vector<uint32_t> indices) const {
        if (!indices.empty()) {
            entry_vertex_indices_ = std::move(indices);
        }
    }
    int32_t getPo() const noexcept { return po_; }
    int32_t getPl() const noexcept { return pl_; }
    int32_t getNl() const noexcept { return nl_; }

  protected:
    void setPo(int32_t po) const noexcept {
        if (po > 0) po_ = po;
    }
    void setPl(int32_t pl) const noexcept {
        if (pl > 0) pl_ = pl;
    }
    void setNl(int32_t nl) const noexcept {
        if (nl > 0) nl_ = nl;
    }
    void setPrefetch(int32_t po, int32_t pl, int32_t nl = 3) const noexcept {
        if (po > 0) po_ = po;
        if (pl > 0) pl_ = pl;
        if (nl > 0) nl_ = nl;
    }

  public:
    /**
     * @brief Auto-tunes traversal prefetch parameters (po, pl, nl) by empirically timing graph
     * traversal over sampled vertices.
     *
     * The graph self-samples its own stored features as queries and exercises `search_intern`
     * with a fixed operating point (`k = 100`, `eps = 0.1f`, `max_distance_computation_count = 10000`).
     * Optionally computes a 1-median entry vertex on `median_subset` sampled vertices.
     *
     * @param sample_count  Number of vertices sampled as queries (capped at graph size; default 300).
     * @param median_subset Subset size for computing the graph-median entry point (`<= 1` keeps native entries).
     */
    void optimize(size_t sample_count = 300, size_t median_subset = 4096) const {
        const uint32_t n_vertices = size();
        const size_t n_queries = std::min(sample_count, static_cast<size_t>(n_vertices));
        // Below one full neighborhood the traversal does no real work; the timing would be meaningless.
        if (n_queries == 0 || n_queries < getEdgesPerVertex()) return;
        constexpr uint32_t k = 100;
        constexpr float eps = 0.1f;
        constexpr uint32_t max_dist = 10000;

        // Graph-median entry point (1-median on a random subset of the graph's OWN stored features):
        //   median = argmin_m  sum_{i in sample} dist_func(vec[m], vec[i])
        // Uses only getFeatureVector + the graph's native dist_func, so it is quantizer- and data-type-free
        // (no synthetic mean, no per-type decode). Replaces the entry vertices with this central vertex.
        // median_subset <= 1 keeps the native entries (disabled). Cost: O(median_subset^2) dist calls, once.
        if (median_subset > 1) {
            const size_t S = std::min(median_subset, static_cast<size_t>(n_vertices));
            std::vector<uint32_t> midx(S);
            uint32_t mrng = 12345u;  // fixed: reproduces the validated median vertex (matches 8.1)
            for (auto& idx : midx) {
                mrng ^= mrng << 13;
                mrng ^= mrng >> 17;
                mrng ^= mrng << 5;
                idx = mrng % n_vertices;
            }
            const auto med_dist = getFeatureSpace().get_dist_func();
            const void* med_param = getFeatureSpace().get_dist_func_param();
            double best_sum = std::numeric_limits<double>::max();
            uint32_t median = midx[0];
            for (size_t a = 0; a < S; ++a) {
                const std::byte* va = getFeatureVector(midx[a]);
                double sum = 0.0;
                for (size_t b = 0; b < S; ++b) sum += static_cast<double>(med_dist(va, getFeatureVector(midx[b]), med_param));
                if (sum < best_sum) {
                    best_sum = sum;
                    median = midx[a];
                }
            }
            setEntryVertexIndices({median});
        }

        std::vector<uint32_t> sample_idx(n_queries);
        uint32_t rng = 7u * 2654435761u + 1u;  // fixed sampling seed for the traversal tuning (not caller-facing).
        for (auto& idx : sample_idx) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            idx = rng % n_vertices;
        }

        // Neighbor list size is fixed per graph, so nl_fixed covers all neighbor ID cache lines directly.
        auto cache_lines_for = [](size_t bytes) noexcept -> int32_t {
            return std::max<int32_t>(1, static_cast<int32_t>((bytes + deglib::memory::L1_CACHE_LINE_SIZE - 1) / deglib::memory::L1_CACHE_LINE_SIZE));
        };
        const int32_t nl_fixed = cache_lines_for(static_cast<size_t>(getEdgesPerVertex()) * sizeof(uint32_t));
        const int32_t max_po = std::min<int32_t>(20, static_cast<int32_t>(getEdgesPerVertex()));
        const int32_t max_pl = std::min<int32_t>(20, cache_lines_for(getFeatureSpace().get_data_size()));

        SearchWorkspace ws;
        const std::vector<uint32_t>& entries = getEntryVertexIndices();
        auto timeit = [&] {
            const auto t_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < n_queries; ++i) {
                search_intern(ws, entries, getFeatureVector(sample_idx[i]), k, eps, true, nullptr, max_dist);
            }
            return std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t_start).count();
        };

        setPrefetch(1, 1, nl_fixed);
        timeit();
        timeit();
        timeit();

        std::vector<int32_t> try_pos;
        for (const int32_t p : {1, 2, 3, 4, 6, 8, 12, 16, 20}) {
            if (p <= max_po) try_pos.push_back(p);
        }
        if (try_pos.empty() || try_pos.back() != max_po) try_pos.push_back(max_po);

        const int32_t min_pl = std::max<int32_t>(1, max_pl / 2);
        std::vector<int32_t> try_pls;
        for (const int32_t l : {1, 2, 3, 4, 6, 8, 12, 16, 20}) {
            if (l >= min_pl && l <= max_pl) try_pls.push_back(l);
        }
        if (try_pls.empty() || try_pls.back() != max_pl) try_pls.push_back(max_pl);

        auto timeit_median3 = [&] {
            double t0 = timeit();
            double t1 = timeit();
            double t2 = timeit();
            if (t0 > t1) std::swap(t0, t1);
            if (t1 > t2) std::swap(t1, t2);
            if (t0 > t1) std::swap(t0, t1);
            return t1;
        };

        // Two-stage coordinate search with median-of-3 timing:
        // Stage 1: sweep po at full feature-vector prefetch (max_pl)
        const size_t init_pl_idx = try_pls.size() - 1;
        size_t best_po_idx = 0;
        size_t best_pl_idx = init_pl_idx;
        double best_time = std::numeric_limits<double>::max();

        for (size_t pi = 0; pi < try_pos.size(); ++pi) {
            setPrefetch(try_pos[pi], try_pls[init_pl_idx], nl_fixed);
            const double dur = timeit_median3();
            if (dur < best_time) {
                best_time = dur;
                best_po_idx = pi;
            }
        }

        // Stage 2: sweep pl (>= max_pl / 2) at best_po
        if (try_pls.size() > 1) {
            for (size_t li = 0; li < try_pls.size(); ++li) {
                if (li == init_pl_idx) continue;
                setPrefetch(try_pos[best_po_idx], try_pls[li], nl_fixed);
                const double dur = timeit_median3();
                if (dur < best_time) {
                    best_time = dur;
                    best_pl_idx = li;
                }
            }
        }

        setPrefetch(try_pos[best_po_idx], try_pls[best_pl_idx], nl_fixed);
    }
    /**
     * @brief Traverses the graph from the given entry vertices towards to_vertex and backtracks the path.
     *
     * @param entry_vertex_indices Internal indices to start search from.
     * @param to_vertex Target internal index.
     * @param eps Exploration factor widening the search radius (0.0 = greedy descent).
     * @param k Size of result set driving exploration radius.
     * @return Path from target back to entry vertex with internal indices and distances; empty if unreachable.
     */
    virtual std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const = 0;

    /**
     * @brief Performs approximate nearest neighbor search using dynamic exploration radius (eps) into `ws`.
     *
     * Reuses `ws` buffers across queries without heap reallocations and populates `ws.results`,
     * `ws.graph_distance_count`, and `ws.graph_hop_count`. Operates on internal indices (0..N-1).
     */
    template <typename T>
    void search(
        SearchWorkspace& ws,
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        validate_query_bytes(query.size_bytes());
        search_intern(ws, getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count);
    }

    /**
     * @brief Performs approximate nearest neighbor search using dynamic exploration radius (eps).
     *
     * Convenience overload allocating and returning a fresh `SearchWorkspace`.
     */
    template <typename T>
    [[nodiscard]] SearchWorkspace search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search(ws, query, k, eps, filter, max_distance_computation_count);
        return ws;
    }

    /**
     * @brief Performs approximate nearest neighbor search from custom entry vertices using dynamic exploration radius (eps) into `ws`.
     */
    template <typename T>
    void search(
        SearchWorkspace& ws,
        std::span<const T> query,
        std::span<const uint32_t> entry_vertex_indices,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        validate_query_bytes(query.size_bytes());
        const std::span<const uint32_t> entries = entry_vertex_indices.empty() ? std::span<const uint32_t>(getEntryVertexIndices()) : entry_vertex_indices;
        search_intern(ws, entries, reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count);
    }

    /**
     * @brief Performs approximate nearest neighbor search from custom entry vertices using dynamic exploration radius (eps).
     */
    template <typename T>
    [[nodiscard]] SearchWorkspace search(
        std::span<const T> query,
        std::span<const uint32_t> entry_vertex_indices,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search(ws, query, entry_vertex_indices, k, eps, filter, max_distance_computation_count);
        return ws;
    }

    /**
     * @brief Performs approximate nearest neighbor search using fixed candidate pool size (ef) into `ws`.
     *
     * Populates `ws.pool` (sorted ascending by distance), `ws.graph_distance_count`, and `ws.graph_hop_count`.
     */
    template <typename T>
    void search_ef(
        SearchWorkspace& ws,
        std::span<const T> query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        validate_query_bytes(query.size_bytes());
        search_ef_intern(
            ws, getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, ef, include_entry, filter, max_distance_computation_count
        );
    }

    /**
     * @brief Performs approximate nearest neighbor search using fixed candidate pool size (ef).
     *
     * Convenience overload allocating and returning a fresh `SearchWorkspace`.
     */
    template <typename T>
    [[nodiscard]] SearchWorkspace search_ef(
        std::span<const T> query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search_ef(ws, query, k, ef, include_entry, filter, max_distance_computation_count);
        return ws;
    }

    /**
     * @brief Performs approximate nearest neighbor search from custom entry vertices using fixed candidate pool size (ef) into `ws`.
     */
    template <typename T>
    void search_ef(
        SearchWorkspace& ws,
        std::span<const T> query,
        std::span<const uint32_t> entry_vertex_indices,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        validate_query_bytes(query.size_bytes());
        const std::span<const uint32_t> entries = entry_vertex_indices.empty() ? std::span<const uint32_t>(getEntryVertexIndices()) : entry_vertex_indices;
        search_ef_intern(ws, entries, reinterpret_cast<const std::byte*>(query.data()), k, ef, include_entry, filter, max_distance_computation_count);
    }

    /**
     * @brief Performs approximate nearest neighbor search from custom entry vertices using fixed candidate pool size (ef).
     */
    template <typename T>
    [[nodiscard]] SearchWorkspace search_ef(
        std::span<const T> query,
        std::span<const uint32_t> entry_vertex_indices,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search_ef(ws, query, entry_vertex_indices, k, ef, include_entry, filter, max_distance_computation_count);
        return ws;
    }

    /**
     * @brief Explores the graph starting from an internal vertex index using dynamic exploration radius (eps) into `ws`.
     */
    void explore(
        SearchWorkspace& ws,
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        const auto query_ptr = getFeatureVector(entry_vertex_index);
        const uint32_t entries[1] = {entry_vertex_index};
        search_intern(ws, entries, query_ptr, k, eps, include_entry, filter, max_distance_computation_count);
    }

    /**
     * @brief Explores the graph starting from an internal vertex index using dynamic exploration radius (eps).
     */
    [[nodiscard]] SearchWorkspace explore(
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        SearchWorkspace ws;
        explore(ws, entry_vertex_index, k, max_distance_computation_count, eps, include_entry, filter);
        return ws;
    }

    /**
     * @brief Explores the graph starting from an internal vertex index using fixed beam width (ef) into `ws`.
     */
    void explore_ef(
        SearchWorkspace& ws,
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t ef,
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        const auto query_ptr = getFeatureVector(entry_vertex_index);
        const uint32_t entries[1] = {entry_vertex_index};
        search_ef_intern(ws, entries, query_ptr, k, ef, include_entry, filter, max_distance_computation_count);
    }

    /**
     * @brief Explores the graph starting from an internal vertex index using fixed beam width (ef).
     */
    [[nodiscard]] SearchWorkspace explore_ef(
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t ef,
        const uint32_t max_distance_computation_count = 0,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        SearchWorkspace ws;
        explore_ef(ws, entry_vertex_index, k, ef, max_distance_computation_count, include_entry, filter);
        return ws;
    }

  private:
    void validate_query_bytes(size_t query_bytes) const {
        if (query_bytes < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query_bytes) + " bytes"
            );
        }
    }

  protected:
    mutable std::vector<uint32_t> entry_vertex_indices_{0};
    // Prefetch tuning knobs: performance cache state, not logical identity, hence mutable so
    // optimize()/setPrefetch() can run through a const InternalGraph reference.
    mutable int32_t po_ = 8;
    mutable int32_t pl_ = 3;
    mutable int32_t nl_ = 3;

    /**
     * Virtual internal search entry points implemented by derived graph classes, writing results and
     * telemetry directly into `ws`.
     */
    virtual void search_intern(
        SearchWorkspace& ws,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const = 0;

    virtual void search_ef_intern(
        SearchWorkspace& ws,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const = 0;

    /**
     * Convenience wrapper allocating and returning a fresh `SearchWorkspace` for internal callers.
     */
    [[nodiscard]] SearchWorkspace search_intern(
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search_intern(ws, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count);
        return ws;
    }

    [[nodiscard]] SearchWorkspace search_ef_intern(
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        SearchWorkspace ws;
        search_ef_intern(ws, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count);
        return ws;
    }

    /**
     * Statically dispatched exploration and k-NN search implementation.
     * Inlines graph member accesses (features, neighbors, labels) via GraphType to avoid vtable calls.
     */
    template <typename GraphType, bool use_max_distance_count, bool use_filter>
    static void searchImpl(
        SearchWorkspace& ws,
        const GraphType& self,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t initial_k,
        const float eps,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        const size_t vertex_count = self.size();
        const size_t k = std::min(vertex_count, static_cast<size_t>(initial_k));
        ws.reset_for_eps(k, self.edges_per_vertex_);
        if (k == 0) return;

        const auto dist_func = self.feature_space_.get_dist_func();
        const auto batch_dist_func = self.feature_space_.get_batch_dist_func();
        const auto dist_func_param = self.feature_space_.get_dist_func_param();

        // set of checked vertex ids
        const auto vl = self.visited_list_pool_->getFreeVisitedList();
        auto* checked_ids = vl->get_visited();
        const auto checked_ids_tag = vl->get_tag();

        auto& next_vertices = ws.next_vertices;
        auto& results = ws.results;

        // if the filter only contains few valid ids brute force them all
        if constexpr (use_filter) {
            if (vertex_count < 1'000 || (filter->get_inclusion_rate() * vertex_count) < 10'000 || filter->get_inclusion_rate() < 0.10f) {
                uint32_t bf_dist_count = 0;
                auto radius = std::numeric_limits<float>::max();
                filter->for_each_valid_label([&](uint32_t valid_label) {
                    auto valid_index = self.getInternalIndex(valid_label);
                    const auto feature = self.feature_by_index(valid_index);
                    const auto distance = dist_func(query, feature, dist_func_param);
                    ++bf_dist_count;

                    // remember the vertex, if its better than the worst in the result list
                    if (distance < radius) {
                        results.emplace(valid_index, distance);

                        // update the search radius
                        if (results.size() > k) {
                            results.pop();
                            radius = results.top().getDistance();
                        }
                    }
                });
                ws.graph_distance_count = bf_dist_count;
                return;
            }
        }

        uint32_t dist_count = 0;
        uint32_t hop_count = 0;

        // copy the initial entry vertices and their distances to the query into the three containers
        for (const uint32_t index : entry_vertex_indices) {
            if (index >= vertex_count || checked_ids[index] == checked_ids_tag) continue;
            checked_ids[index] = checked_ids_tag;

            const auto feature = self.feature_by_index(index);
            const auto distance = dist_func(query, feature, dist_func_param);
            next_vertices.emplace(index, distance);
            if (include_entry) {
                if constexpr (use_filter) {
                    if (filter->is_valid(self.label_by_index(index))) {
                        results.emplace(index, distance);
                    }
                } else {
                    results.emplace(index, distance);
                }
                if (results.size() > k) results.pop();
            }

            // early stop after reaching distance budget
            ++dist_count;
            if constexpr (use_max_distance_count) {
                if (dist_count >= max_distance_computation_count) {
                    ws.graph_distance_count = dist_count;
                    ws.graph_hop_count = hop_count;
                    return;
                }
            }
        }

        // search radius
        auto radius = (results.size() < k) ? std::numeric_limits<float>::max() : results.top().getDistance();
        auto exploration_radius = (radius == std::numeric_limits<float>::max()) ? radius : radius * ((radius < 0) ? (1 - eps) : (1 + eps));

        const int32_t po = self.getPo();
        const size_t prefetch_bytes = static_cast<size_t>(self.getPl()) * deglib::memory::L1_CACHE_LINE_SIZE;
        const int32_t nl = self.getNl();
        const size_t edges_per_vertex = self.edges_per_vertex_;
        alignas(64) uint32_t good_neighbors[256];
        alignas(64) const void* good_features[256];
        alignas(64) float good_distances[256];

        // iterate as long as good elements are in the next_vertices queue
        while (next_vertices.empty() == false) {
            const auto next_vertex = next_vertices.top();
            next_vertices.pop();

            // max distance reached
            if (next_vertex.getDistance() > exploration_radius) break;

            ++hop_count;

            const auto neighbor_indices = self.neighbors_by_index(next_vertex.getIdentifier());
            int32_t good_neighbor_count = 0;
            for (size_t i = 0; i < edges_per_vertex; ++i) {
                const uint32_t neighbor_index = neighbor_indices[i];
                if (checked_ids[neighbor_index] == checked_ids_tag) continue;
                checked_ids[neighbor_index] = checked_ids_tag;
                good_neighbors[good_neighbor_count] = neighbor_index;
                good_features[good_neighbor_count] = self.feature_by_index(neighbor_index);
                ++good_neighbor_count;
            }

            if (good_neighbor_count == 0) continue;

            // Cap the neighbor count based on the remaining distance budget
            if constexpr (use_max_distance_count) {
                const uint32_t remaining = max_distance_computation_count - dist_count;
                if (static_cast<uint32_t>(good_neighbor_count) > remaining) {
                    good_neighbor_count = static_cast<int32_t>(remaining);
                }
                if (good_neighbor_count <= 0) break;
            }

            batch_dist_func(query, good_features, static_cast<size_t>(good_neighbor_count), dist_func_param, good_distances, po, prefetch_bytes);

            for (int32_t i = 0; i < good_neighbor_count; ++i) {
                const uint32_t neighbor_index = good_neighbors[i];
                const float neighbor_distance = good_distances[i];

                // check the neighborhood of this vertex later, if its good enough
                if (neighbor_distance <= exploration_radius) {
                    next_vertices.emplace(neighbor_index, neighbor_distance);
                    deglib::memory::prefetch(
                        reinterpret_cast<const char*>(self.neighbors_by_index(neighbor_index)), static_cast<size_t>(nl) * deglib::memory::L1_CACHE_LINE_SIZE
                    );

                    // remember the vertex, if its better than the worst in the result list
                    if (neighbor_distance < radius) {
                        if constexpr (use_filter) {
                            if (filter->is_valid(self.label_by_index(neighbor_index))) {
                                results.emplace(neighbor_index, neighbor_distance);
                            }
                        } else {
                            results.emplace(neighbor_index, neighbor_distance);
                        }

                        // update the search radius
                        if (results.size() > k) {
                            results.pop();
                            radius = results.top().getDistance();
                            exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
                        }
                    }
                }
            }

            dist_count += static_cast<uint32_t>(good_neighbor_count);
            if constexpr (use_max_distance_count) {
                if (dist_count >= max_distance_computation_count) {
                    break;
                }
            }
        }

        ws.graph_distance_count = dist_count;
        ws.graph_hop_count = hop_count;
    }

    /**
     * Dispatches runtime filter and distance budget settings to compile-time specialized searchImpl instantiations.
     */
    template <typename GraphType>
    static void searchInternImpl(
        SearchWorkspace& ws,
        const GraphType& self,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) {
        if (filter) {
            if (max_distance_computation_count == 0) {
                searchImpl<GraphType, false, true>(ws, self, entry_vertex_indices, query, k, eps, include_entry, filter, 0);
            } else {
                searchImpl<GraphType, true, true>(ws, self, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count);
            }
        } else {
            if (max_distance_computation_count == 0) {
                searchImpl<GraphType, false, false>(ws, self, entry_vertex_indices, query, k, eps, include_entry, nullptr, 0);
            } else {
                searchImpl<GraphType, true, false>(ws, self, entry_vertex_indices, query, k, eps, include_entry, nullptr, max_distance_computation_count);
            }
        }
    }

    /**
     * ef-bounded search with the exploration frontier separated from the retained result set.
     *
     * Needed whenever the traversal must visit vertices that must not be reported: filtered-out
     * candidates would otherwise occupy the fixed result capacity and collapse the search radius,
     * and entry vertices must stay expandable while being excluded from the results.
     * The frontier holds every candidate within the current ef-th best valid distance, so stopping
     * at the first candidate beyond it is exact for the ef best valid results.
     */
    template <typename GraphType, bool use_max_distance_count, bool use_filter>
    static void searchEfQueueImpl(
        SearchWorkspace& ws,
        const GraphType& self,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        const size_t vertex_count = self.size();
        ws.reset_for_ef(k, ef, self.edges_per_vertex_, /*need_queue=*/true);
        if (vertex_count == 0 || k == 0) return;

        const auto dist_func = self.feature_space_.get_dist_func();
        const auto batch_dist_func = self.feature_space_.get_batch_dist_func();
        const auto dist_func_param = self.feature_space_.get_dist_func_param();

        // set of checked vertex ids
        const auto vl = self.visited_list_pool_->getFreeVisitedList();
        auto* checked_ids = vl->get_visited();
        const auto checked_ids_tag = vl->get_tag();

        // invalid candidates are traversed through next_vertices, but never reach valid_results
        auto& next_vertices = ws.next_vertices;
        auto& valid_results = ws.pool;

        uint32_t dist_count = 0;
        uint32_t hop_count = 0;

        // seed the traversal with every entry vertex
        for (const uint32_t ep : entry_vertex_indices) {
            if (ep >= vertex_count || checked_ids[ep] == checked_ids_tag) continue;
            checked_ids[ep] = checked_ids_tag;

            const auto feature = self.feature_by_index(ep);
            const float dist = dist_func(query, feature, dist_func_param);
            next_vertices.emplace(ep, dist);
            if (include_entry) {
                if constexpr (use_filter) {
                    if (filter->is_valid(self.label_by_index(ep))) valid_results.insert(ep, dist);
                } else {
                    valid_results.insert(ep, dist);
                }
            }

            // early stop after reaching distance budget
            ++dist_count;
            if constexpr (use_max_distance_count) {
                if (dist_count >= max_distance_computation_count) {
                    ws.graph_distance_count = dist_count;
                    ws.graph_hop_count = hop_count;
                    valid_results.truncate(k);
                    return;
                }
            }
        }

        const int32_t po = self.getPo();
        const size_t prefetch_bytes = static_cast<size_t>(self.getPl()) * deglib::memory::L1_CACHE_LINE_SIZE;
        const int32_t nl = self.getNl();
        const size_t edges_per_vertex = self.edges_per_vertex_;
        alignas(64) uint32_t good_neighbors[256];
        alignas(64) const void* good_features[256];
        alignas(64) float good_distances[256];

        // iterate as long as candidates within the result radius are in the next_vertices queue
        while (next_vertices.empty() == false) {
            const auto next_vertex = next_vertices.top();
            next_vertices.pop();

            // the ef-th best valid distance is the exact routing radius of an unfiltered ef search
            const float radius = valid_results.size() < static_cast<size_t>(ef) ? std::numeric_limits<float>::max() : valid_results[ef - 1].getDistance();

            // no remaining candidate can improve the ef best valid results
            if (next_vertex.getDistance() > radius) break;

            ++hop_count;

            const auto neighbor_indices = self.neighbors_by_index(next_vertex.getIdentifier());
            int32_t good_neighbor_count = 0;
            for (size_t i = 0; i < edges_per_vertex; ++i) {
                const uint32_t neighbor_index = neighbor_indices[i];
                if (checked_ids[neighbor_index] == checked_ids_tag) continue;
                checked_ids[neighbor_index] = checked_ids_tag;
                good_neighbors[good_neighbor_count] = neighbor_index;
                good_features[good_neighbor_count] = self.feature_by_index(neighbor_index);
                ++good_neighbor_count;
            }

            if (good_neighbor_count == 0) continue;

            // Cap the neighbor count based on the remaining distance budget
            if constexpr (use_max_distance_count) {
                const uint32_t remaining = max_distance_computation_count - dist_count;
                if (static_cast<uint32_t>(good_neighbor_count) > remaining) {
                    good_neighbor_count = static_cast<int32_t>(remaining);
                }
                if (good_neighbor_count <= 0) break;
            }

            batch_dist_func(query, good_features, static_cast<size_t>(good_neighbor_count), dist_func_param, good_distances, po, prefetch_bytes);

            for (int32_t i = 0; i < good_neighbor_count; ++i) {
                const uint32_t neighbor_index = good_neighbors[i];
                const float dist = good_distances[i];

                // check the neighborhood of this vertex later, if its good enough
                if (dist <= radius) {
                    next_vertices.emplace(neighbor_index, dist);
                    deglib::memory::prefetch(
                        reinterpret_cast<const char*>(self.neighbors_by_index(neighbor_index)), static_cast<size_t>(nl) * deglib::memory::L1_CACHE_LINE_SIZE
                    );
                }

                if constexpr (use_filter) {
                    if (filter->is_valid(self.label_by_index(neighbor_index))) valid_results.insert(neighbor_index, dist);
                } else {
                    valid_results.insert(neighbor_index, dist);
                }
            }

            dist_count += static_cast<uint32_t>(good_neighbor_count);
            if constexpr (use_max_distance_count) {
                if (dist_count >= max_distance_computation_count) {
                    break;
                }
            }
        }

        ws.graph_distance_count = dist_count;
        ws.graph_hop_count = hop_count;
        valid_results.truncate(k);
    }

    /**
     * ef-bounded graph search.
     *
     * The plain case — no filter, entry vertices reported — runs the beam variant, where a single
     * ResultList serves as frontier and result set at once. Every other combination needs both
     * separated, see searchEfQueueImpl.
     */
    template <typename GraphType, bool use_max_distance_count, bool use_filter>
    static void searchEfImpl(
        SearchWorkspace& ws,
        const GraphType& self,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        if constexpr (use_filter) {
            searchEfQueueImpl<GraphType, use_max_distance_count, true>(
                ws, self, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count
            );
        } else {
            if (!include_entry) {
                searchEfQueueImpl<GraphType, use_max_distance_count, false>(
                    ws, self, entry_vertex_indices, query, k, ef, false, nullptr, max_distance_computation_count
                );
                return;
            }

            const size_t vertex_count = self.size();
            ws.reset_for_ef(k, ef, self.edges_per_vertex_, /*need_queue=*/false);
            if (vertex_count == 0 || k == 0) return;

            const auto dist_func = self.feature_space_.get_dist_func();
            const auto batch_dist_func = self.feature_space_.get_batch_dist_func();
            const auto dist_func_param = self.feature_space_.get_dist_func_param();

            // set of checked vertex ids
            const auto vl = self.visited_list_pool_->getFreeVisitedList();
            auto* checked_ids = vl->get_visited();
            const auto checked_ids_tag = vl->get_tag();

            auto& pool = ws.pool;
            uint32_t dist_count = 0;
            uint32_t hop_count = 0;

            // every entry vertex seeds the beam and is part of the result set
            for (const uint32_t ep : entry_vertex_indices) {
                if (ep >= vertex_count || checked_ids[ep] == checked_ids_tag) continue;
                checked_ids[ep] = checked_ids_tag;

                const auto feature = self.feature_by_index(ep);
                const float dist = dist_func(query, feature, dist_func_param);
                pool.insert(ep, dist);

                // early stop after reaching distance budget
                ++dist_count;
                if constexpr (use_max_distance_count) {
                    if (dist_count >= max_distance_computation_count) {
                        ws.graph_distance_count = dist_count;
                        ws.graph_hop_count = hop_count;
                        pool.truncate(k);
                        return;
                    }
                }
            }

            const int32_t po = self.getPo();
            const size_t prefetch_bytes = static_cast<size_t>(self.getPl()) * deglib::memory::L1_CACHE_LINE_SIZE;
            const int32_t nl = self.getNl();
            const size_t edges_per_vertex = self.edges_per_vertex_;
            alignas(64) uint32_t good_neighbors[256];
            alignas(64) const void* good_features[256];
            alignas(64) float good_distances[256];

            while (pool.has_next()) {
                const uint32_t u = pool.pop();
                ++hop_count;

                const auto neighbor_indices = self.neighbors_by_index(u);
                int32_t good_neighbor_count = 0;
                for (size_t i = 0; i < edges_per_vertex; ++i) {
                    const uint32_t neighbor_index = neighbor_indices[i];
                    if (checked_ids[neighbor_index] == checked_ids_tag) {
                        continue;
                    }
                    checked_ids[neighbor_index] = checked_ids_tag;
                    good_neighbors[good_neighbor_count] = neighbor_index;
                    good_features[good_neighbor_count] = self.feature_by_index(neighbor_index);
                    ++good_neighbor_count;
                }

                if (good_neighbor_count == 0) continue;

                // Cap the neighbor count based on the remaining distance budget
                if constexpr (use_max_distance_count) {
                    const uint32_t remaining = max_distance_computation_count - dist_count;
                    if (static_cast<uint32_t>(good_neighbor_count) > remaining) {
                        good_neighbor_count = static_cast<int32_t>(remaining);
                    }
                    if (good_neighbor_count <= 0) break;
                }

                batch_dist_func(query, good_features, static_cast<size_t>(good_neighbor_count), dist_func_param, good_distances, po, prefetch_bytes);

                for (int32_t i = 0; i < good_neighbor_count; ++i) {
                    const uint32_t neighbor_index = good_neighbors[i];
                    if (pool.insert(neighbor_index, good_distances[i])) {
                        deglib::memory::prefetch(
                            reinterpret_cast<const char*>(self.neighbors_by_index(neighbor_index)), static_cast<size_t>(nl) * deglib::memory::L1_CACHE_LINE_SIZE
                        );
                    }
                }

                dist_count += static_cast<uint32_t>(good_neighbor_count);
                if constexpr (use_max_distance_count) {
                    if (dist_count >= max_distance_computation_count) {
                        break;
                    }
                }
            }

            ws.graph_distance_count = dist_count;
            ws.graph_hop_count = hop_count;
            pool.truncate(k);
        }
    }

    /**
     * Dispatches runtime filter and distance budget settings to compile-time specialized searchEfImpl instantiations.
     */
    template <typename GraphType>
    static void searchEfInternImpl(
        SearchWorkspace& ws,
        const GraphType& self,
        std::span<const uint32_t> entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) {
        if (filter) {
            if (max_distance_computation_count == 0) {
                searchEfImpl<GraphType, false, true>(ws, self, entry_vertex_indices, query, k, ef, include_entry, filter, 0);
            } else {
                searchEfImpl<GraphType, true, true>(ws, self, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count);
            }
        } else {
            if (max_distance_computation_count == 0) {
                searchEfImpl<GraphType, false, false>(ws, self, entry_vertex_indices, query, k, ef, include_entry, nullptr, 0);
            } else {
                searchEfImpl<GraphType, true, false>(ws, self, entry_vertex_indices, query, k, ef, include_entry, nullptr, max_distance_computation_count);
            }
        }
    }

    /**
     * Greedy best-first reachability search.
     * Backtracks predecessors and returns an ordered path from to_vertex back to entry.
     */
    template <typename GraphType>
    static std::vector<deglib::graph::ObjectDistance>
    hasPathImpl(const GraphType& self, const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) {
        const auto query = self.feature_by_index(to_vertex);
        const auto dist_func = self.feature_space_.get_dist_func();
        const auto dist_func_param = self.feature_space_.get_dist_func_param();
        const auto feature_size = self.feature_space_.get_data_size();

        // set of checked vertex ids
        const auto vl = self.visited_list_pool_->getFreeVisitedList();
        auto* checked_ids = vl->get_visited();
        const auto checked_ids_tag = vl->get_tag();

        // items to traverse next
        auto next_vertices = deglib::graph::UncheckedSet();

        // trackable information
        auto trackback = std::unordered_map<uint32_t, deglib::graph::ObjectDistance>();

        // result set
        auto results = deglib::graph::ResultSet();

        // copy the initial entry vertices and their distances to the query into the three containers
        for (auto&& index : entry_vertex_indices) {
            if (checked_ids[index] != checked_ids_tag) {
                checked_ids[index] = checked_ids_tag;

                const auto feature = self.feature_by_index(index);
                const auto distance = dist_func(query, feature, dist_func_param);
                results.emplace(index, distance);
                if (results.size() > k) results.pop();
                next_vertices.emplace(index, distance);
                trackback.emplace(index, deglib::graph::ObjectDistance(index, distance));
            }
        }

        // search radius
        auto radius = (results.size() < k) ? std::numeric_limits<float>::max() : results.top().getDistance();
        auto exploration_radius = (radius == std::numeric_limits<float>::max()) ? radius : radius * ((radius < 0) ? (1 - eps) : (1 + eps));

        // iterate as long as good elements are in the next_vertices queue
        auto good_neighbors = std::array<uint32_t, 256>();
        while (next_vertices.empty() == false) {
            // next vertex to check
            const auto next_vertex = next_vertices.top();
            next_vertices.pop();

            // max distance reached
            if (next_vertex.getDistance() > exploration_radius) break;

            size_t good_neighbor_count = 0;
            const auto neighbor_indices = self.neighbors_by_index(next_vertex.getIdentifier());
            for (size_t i = 0; i < self.edges_per_vertex_; i++) {
                const auto neighbor_index = neighbor_indices[i];

                // found our target vertex, create a path back to the entry vertex
                if (neighbor_index == to_vertex) {
                    auto path = std::vector<deglib::graph::ObjectDistance>();
                    path.emplace_back(to_vertex, 0.f);
                    path.emplace_back(next_vertex.getIdentifier(), next_vertex.getDistance());

                    auto last_vertex = trackback.find(next_vertex.getIdentifier());
                    while (last_vertex != trackback.cend() && last_vertex->first != last_vertex->second.getIdentifier()) {
                        path.emplace_back(last_vertex->second.getIdentifier(), last_vertex->second.getDistance());
                        last_vertex = trackback.find(last_vertex->second.getIdentifier());
                    }

                    return path;
                }

                // collect
                if (checked_ids[neighbor_index] != checked_ids_tag) {
                    checked_ids[neighbor_index] = checked_ids_tag;
                    good_neighbors[good_neighbor_count++] = neighbor_index;
                }
            }

            if (good_neighbor_count == 0) continue;

            memory::prefetch(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[0])), feature_size);
            for (size_t i = 0; i < good_neighbor_count; i++) {
                memory::prefetch(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])), feature_size);

                const auto neighbor_index = good_neighbors[i];
                const auto neighbor_feature_vector = self.feature_by_index(neighbor_index);
                const auto neighbor_distance = dist_func(query, neighbor_feature_vector, dist_func_param);

                // check the neighborhood of this vertex later, if its good enough
                if (neighbor_distance <= exploration_radius) {
                    next_vertices.emplace(neighbor_index, neighbor_distance);
                    trackback.insert({neighbor_index, deglib::graph::ObjectDistance(next_vertex.getIdentifier(), next_vertex.getDistance())});

                    // remember the vertex, if its better than the worst in the result list
                    if (neighbor_distance < radius) {
                        results.emplace(neighbor_index, neighbor_distance);

                        // update the search radius
                        if (results.size() > k) {
                            results.pop();
                            radius = results.top().getDistance();
                            exploration_radius = radius * ((radius < 0) ? (1 - eps) : (1 + eps));
                        }
                    }
                }
            }
        }

        // there is no path
        return std::vector<deglib::graph::ObjectDistance>();
    }

    friend class deglib::builder::EvenRegularGraphBuilder;
};

}  // namespace deglib::graph
