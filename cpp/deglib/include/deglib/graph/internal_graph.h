#pragma once

#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/graph/visited_list_pool.h"
#include "deglib/utils/memory.h"
#include "deglib/search/result_list.h"

#include <algorithm>
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
 * Abstract base interface for all low-level internal graph representations.
 * All methods operate on internal_index (0..N-1) for maximum memory performance.
 */
class InternalGraph {
  public:
    virtual ~InternalGraph() = default;
    virtual const uint32_t size() const = 0;
    virtual const uint8_t getEdgesPerVertex() const = 0;
    virtual const deglib::distances::FloatSpace& getFeatureSpace() const = 0;

    virtual const uint32_t getExternalLabel(const uint32_t internal_index) const = 0;
    virtual const uint32_t getInternalIndex(const uint32_t external_label) const = 0;
    virtual const uint32_t* getNeighborIndices(const uint32_t internal_index) const = 0;
    virtual const std::byte* getFeatureVector(const uint32_t internal_index) const = 0;

    virtual const bool hasVertex(const uint32_t external_label) const = 0;
    virtual const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const = 0;

  protected:
    std::vector<uint32_t> entry_vertex_indices_{0};
    int32_t po_ = 8;
    int32_t pl_ = 3;
    int32_t nl_ = 3;

  public:
    const std::vector<uint32_t>& getEntryVertexIndices() const { return entry_vertex_indices_; }
    void setEntryVertexIndices(std::vector<uint32_t> indices) {
        if (!indices.empty()) {
            entry_vertex_indices_ = std::move(indices);
        }
    }
    int32_t getPo() const noexcept { return po_; }
    int32_t getPl() const noexcept { return pl_; }
    int32_t getNl() const noexcept { return nl_; }
    void setPo(int32_t po) noexcept {
        if (po > 0) po_ = po;
    }
    void setPl(int32_t pl) noexcept {
        if (pl > 0) pl_ = pl;
    }
    void setNl(int32_t nl) noexcept {
        if (nl > 0) nl_ = nl;
    }
    void setPrefetch(int32_t po, int32_t pl, int32_t nl = 3) noexcept {
        if (po > 0) po_ = po;
        if (pl > 0) pl_ = pl;
        if (nl > 0) nl_ = nl;
    }
    /**
     * Perform a search but stops when the to_vertex was found.
     */
    virtual std::vector<deglib::graph::ObjectDistance>
    hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const = 0;

    /**
     * Bounds-checked internal search for query vectors.
     */
    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if (query.size_bytes() < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query.size_bytes()) + " bytes"
            );
        }
        return search_intern(getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count);
    }

    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        const std::vector<uint32_t>& entry_vertex_indices,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if (query.size_bytes() < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query.size_bytes()) + " bytes"
            );
        }
        if (entry_vertex_indices.empty()) {
            return search_intern(
                getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count
            );
        }
        return search_intern(entry_vertex_indices, reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count);
    }

    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        std::span<const uint32_t> entry_vertex_indices,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if (entry_vertex_indices.empty()) {
            return search(query, k, eps, filter, max_distance_computation_count);
        }
        const std::vector<uint32_t> entries(entry_vertex_indices.begin(), entry_vertex_indices.end());
        return search(query, entries, k, eps, filter, max_distance_computation_count);
    }

    /**
     * Bounds-checked internal search for query vectors using ResultList with fixed ef budget.
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
        if (query.size_bytes() < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query.size_bytes()) + " bytes"
            );
        }
        return search_ef_intern(
            getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, ef, include_entry, filter, max_distance_computation_count
        );
    }

    template <typename T>
    std::vector<deglib::graph::ObjectDistance> search_ef(
        std::span<const T> query,
        const std::vector<uint32_t>& entry_vertex_indices,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const {
        if (query.size_bytes() < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query.size_bytes()) + " bytes"
            );
        }
        if (entry_vertex_indices.empty()) {
            return search_ef_intern(
                getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, ef, include_entry, filter, max_distance_computation_count
            );
        }
        return search_ef_intern(
            entry_vertex_indices, reinterpret_cast<const std::byte*>(query.data()), k, ef, include_entry, filter, max_distance_computation_count
        );
    }

    /**
     * Internal graph exploration starting at entry_vertex_index.
     */
    deglib::graph::ResultSet explore(
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr
    ) const {
        const auto query_ptr = getFeatureVector(entry_vertex_index);
        return search_intern({entry_vertex_index}, query_ptr, k, eps, include_entry, filter, max_distance_computation_count);
    }

  protected:
    /**
     * Virtual internal search entry point implemented by derived graph classes.
     */
    virtual deglib::graph::ResultSet search_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const = 0;

    virtual std::vector<deglib::graph::ObjectDistance> search_ef_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) const = 0;

    /**
     * Statically dispatched exploration and k-NN search implementation.
     * Inlines graph member accesses (features, neighbors, labels) via GraphType to avoid vtable calls.
     */
    template <typename GraphType, deglib::distances::DistanceFunction COMPARATOR, bool use_max_distance_count, bool use_filter>
    static deglib::graph::ResultSet searchImpl(
        const GraphType& self,
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t initial_k,
        const float eps,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        uint32_t distance_computation_count = 0;
        const auto dist_func_param = self.feature_space_.get_dist_func_param();
        const size_t vertex_count = self.size();
        size_t k = std::min(vertex_count, static_cast<size_t>(initial_k));

        // set of checked vertex ids
        const auto vl = self.visited_list_pool_->getFreeVisitedList();
        auto* checked_ids = vl->get_visited();
        const auto checked_ids_tag = vl->get_tag();

        // items to traverse next
        auto next_vertices = deglib::graph::UncheckedSet();
        next_vertices.reserve(k * self.edges_per_vertex_);

        // result set
        auto results = deglib::graph::ResultSet();
        results.reserve(k + 1);

        // if the filter only contains few valid ids brute force them all
        if constexpr (use_filter) {
            if (vertex_count < 1'000 || (filter->get_inclusion_rate() * vertex_count) < 10'000 || filter->get_inclusion_rate() < 0.10f) {
                auto radius = std::numeric_limits<float>::max();
                filter->for_each_valid_label([&](uint32_t valid_label) {
                    auto valid_index = self.getInternalIndex(valid_label);
                    const auto feature = reinterpret_cast<const float*>(self.feature_by_index(valid_index));
                    const auto distance = COMPARATOR::compare(query, feature, dist_func_param);

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
                return results;
            }
        }

        // copy the initial entry vertices and their distances to the query into the three containers
        for (auto&& index : entry_vertex_indices) {
            if (checked_ids[index] != checked_ids_tag) {
                checked_ids[index] = checked_ids_tag;

                const auto feature = self.feature_by_index(index);
                const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
                next_vertices.emplace(index, distance);
                if (include_entry) {
                    if constexpr (use_filter) {
                        if (filter->is_valid(self.label_by_index(index))) {
                            results.emplace(index, distance);
                        }
                    } else {
                        results.emplace(index, distance);
                    }
                }

                // early stop after to many computations
                if constexpr (use_max_distance_count) {
                    if (++distance_computation_count >= max_distance_computation_count) {
                        return results;
                    }
                }
            }
        }

        // search radius
        auto radius = std::numeric_limits<float>::max();
        auto exploration_radius = radius;

        const int32_t po = self.getPo();
        const int32_t pl = self.getPl();
        const int32_t nl = self.getNl();
        const size_t edges_per_vertex = self.edges_per_vertex_;
        alignas(64) uint32_t good_neighbors[256];

        auto prefetch_feature = [pl](const char* ptr) { deglib::memory::prefetch(ptr, static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE); };

        // iterate as long as good elements are in the next_vertices queue
        while (next_vertices.empty() == false) {
            const auto next_vertex = next_vertices.top();
            next_vertices.pop();

            // max distance reached
            if (next_vertex.getDistance() > exploration_radius) break;

            const auto neighbor_indices = self.neighbors_by_index(next_vertex.getIdentifier());
            int32_t good_neighbor_count = 0;
            for (size_t i = 0; i < edges_per_vertex; ++i) {
                const uint32_t neighbor_index = neighbor_indices[i];
                if (checked_ids[neighbor_index] == checked_ids_tag) continue;
                checked_ids[neighbor_index] = checked_ids_tag;
                good_neighbors[good_neighbor_count++] = neighbor_index;
            }

            if (good_neighbor_count == 0) continue;

            // Cap the neighbor count based on the remaining distance budget
            if constexpr (use_max_distance_count) {
                const uint32_t remaining = max_distance_computation_count - distance_computation_count;
                if (static_cast<uint32_t>(good_neighbor_count) > remaining) good_neighbor_count = static_cast<int32_t>(remaining);
            }

            for (int32_t i = 0; i < std::min<int32_t>(po, good_neighbor_count); ++i) {
                prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i])));
            }

            for (int32_t i = 0; i < good_neighbor_count; ++i) {
                if (i + po < good_neighbor_count) {
                    prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i + po])));
                }
                const uint32_t neighbor_index = good_neighbors[i];
                const auto feature = self.feature_by_index(neighbor_index);
                const float neighbor_distance = COMPARATOR::compare(query, feature, dist_func_param);

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

                // early stop after to many computations
                if constexpr (use_max_distance_count) {
                    if (++distance_computation_count >= max_distance_computation_count) {
                        return results;
                    }
                }
            }
        }

        return results;
    }

    /**
     * Dispatches runtime metric and filter settings to compile-time specialized searchImpl instantiations.
     */
    template <typename GraphType>
    static deglib::graph::ResultSet searchInternImpl(
        const GraphType& self,
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) {
        return self.feature_space_.compute([&]<deglib::distances::DistanceFunction Dist>(Dist) -> deglib::graph::ResultSet {
            if (filter) {
                if (max_distance_computation_count == 0) {
                    return searchImpl<GraphType, Dist, false, true>(self, entry_vertex_indices, query, k, eps, include_entry, filter, 0);
                } else {
                    return searchImpl<GraphType, Dist, true, true>(
                        self, entry_vertex_indices, query, k, eps, include_entry, filter, max_distance_computation_count
                    );
                }
            } else {
                if (max_distance_computation_count == 0) {
                    return searchImpl<GraphType, Dist, false, false>(self, entry_vertex_indices, query, k, eps, include_entry, nullptr, 0);
                } else {
                    return searchImpl<GraphType, Dist, true, false>(
                        self, entry_vertex_indices, query, k, eps, include_entry, nullptr, max_distance_computation_count
                    );
                }
            }
        });
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
    template <typename GraphType, deglib::distances::DistanceFunction COMPARATOR, bool use_max_distance_count, bool use_filter>
    static std::vector<deglib::graph::ObjectDistance> searchEfQueueImpl(
        const GraphType& self,
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        const auto dist_func_param = self.feature_space_.get_dist_func_param();
        const size_t vertex_count = self.size();
        const int32_t capacity = static_cast<int32_t>(std::max(k, ef));

        // set of checked vertex ids
        const auto vl = self.visited_list_pool_->getFreeVisitedList();
        auto* checked_ids = vl->get_visited();
        const auto checked_ids_tag = vl->get_tag();

        uint32_t distance_computation_count = 0;

        // invalid candidates are traversed through next_vertices, but never reach valid_results
        auto next_vertices = deglib::graph::UncheckedSet();
        next_vertices.reserve(static_cast<size_t>(ef) * self.edges_per_vertex_);
        deglib::search::ResultList valid_results(static_cast<int32_t>(ef), capacity);

        // seed the traversal with every entry vertex
        for (auto ep : entry_vertex_indices) {
            if (ep >= vertex_count || checked_ids[ep] == checked_ids_tag) continue;
            checked_ids[ep] = checked_ids_tag;

            const auto feature = self.feature_by_index(ep);
            const float dist = COMPARATOR::compare(query, feature, dist_func_param);
            next_vertices.emplace(ep, dist);
            if (include_entry) {
                if constexpr (use_filter) {
                    if (filter->is_valid(self.label_by_index(ep))) valid_results.insert(ep, dist);
                } else {
                    valid_results.insert(ep, dist);
                }
            }

            // early stop after to many computations
            if constexpr (use_max_distance_count) {
                if (++distance_computation_count >= max_distance_computation_count) {
                    return std::move(valid_results).to_vector(k);
                }
            }
        }

        const int32_t po = self.getPo();
        const int32_t pl = self.getPl();
        const int32_t nl = self.getNl();
        const size_t edges_per_vertex = self.edges_per_vertex_;
        alignas(64) uint32_t good_neighbors[256];

        auto prefetch_feature = [pl](const char* ptr) { deglib::memory::prefetch(ptr, static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE); };

        // iterate as long as candidates within the result radius are in the next_vertices queue
        while (next_vertices.empty() == false) {
            const auto next_vertex = next_vertices.top();
            next_vertices.pop();

            // the ef-th best valid distance is the exact routing radius of an unfiltered ef search
            const float radius = valid_results.size() < static_cast<size_t>(ef) ? std::numeric_limits<float>::max() : valid_results[ef - 1].getDistance();

            // no remaining candidate can improve the ef best valid results
            if (next_vertex.getDistance() > radius) break;

            const auto neighbor_indices = self.neighbors_by_index(next_vertex.getIdentifier());
            int32_t good_neighbor_count = 0;
            for (size_t i = 0; i < edges_per_vertex; ++i) {
                const uint32_t neighbor_index = neighbor_indices[i];
                if (checked_ids[neighbor_index] == checked_ids_tag) continue;
                checked_ids[neighbor_index] = checked_ids_tag;
                good_neighbors[good_neighbor_count++] = neighbor_index;
            }

            if (good_neighbor_count == 0) continue;

            // Cap the neighbor count based on the remaining distance budget
            if constexpr (use_max_distance_count) {
                const uint32_t remaining = max_distance_computation_count - distance_computation_count;
                if (static_cast<uint32_t>(good_neighbor_count) > remaining) good_neighbor_count = static_cast<int32_t>(remaining);
            }

            for (int32_t i = 0; i < std::min<int32_t>(po, good_neighbor_count); ++i) {
                prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i])));
            }

            for (int32_t i = 0; i < good_neighbor_count; ++i) {
                if (i + po < good_neighbor_count) {
                    prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i + po])));
                }
                const uint32_t neighbor_index = good_neighbors[i];
                const auto feature = self.feature_by_index(neighbor_index);
                const float dist = COMPARATOR::compare(query, feature, dist_func_param);

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

                // early stop after to many computations
                if constexpr (use_max_distance_count) {
                    if (++distance_computation_count >= max_distance_computation_count) {
                        return std::move(valid_results).to_vector(k);
                    }
                }
            }
        }

        return std::move(valid_results).to_vector(k);
    }

    /**
     * ef-bounded graph search.
     *
     * The plain case — no filter, entry vertices reported — runs the beam variant, where a single
     * ResultList serves as frontier and result set at once. Every other combination needs both
     * separated, see searchEfQueueImpl.
     */
    template <typename GraphType, deglib::distances::DistanceFunction COMPARATOR, bool use_max_distance_count, bool use_filter>
    static std::vector<deglib::graph::ObjectDistance> searchEfImpl(
        const GraphType& self,
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry,
        const deglib::search::Filter* filter,
        const uint32_t max_distance_computation_count
    ) {
        if constexpr (use_filter) {
            return searchEfQueueImpl<GraphType, COMPARATOR, use_max_distance_count, true>(
                self, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count
            );
        } else {
            if (!include_entry) {
                return searchEfQueueImpl<GraphType, COMPARATOR, use_max_distance_count, false>(
                    self, entry_vertex_indices, query, k, ef, false, nullptr, max_distance_computation_count
                );
            }

            const auto dist_func_param = self.feature_space_.get_dist_func_param();
            const size_t vertex_count = self.size();
            const int32_t capacity = static_cast<int32_t>(std::max(k, ef));

            // set of checked vertex ids
            const auto vl = self.visited_list_pool_->getFreeVisitedList();
            auto* checked_ids = vl->get_visited();
            const auto checked_ids_tag = vl->get_tag();

            uint32_t distance_computation_count = 0;

            deglib::search::ResultList pool(static_cast<int32_t>(ef), capacity);

            // every entry vertex seeds the beam and is part of the result set
            for (auto ep : entry_vertex_indices) {
                if (ep >= vertex_count || checked_ids[ep] == checked_ids_tag) continue;
                checked_ids[ep] = checked_ids_tag;

                const auto feature = self.feature_by_index(ep);
                const float dist = COMPARATOR::compare(query, feature, dist_func_param);
                pool.insert(ep, dist);

                // early stop after to many computations
                if constexpr (use_max_distance_count) {
                    if (++distance_computation_count >= max_distance_computation_count) {
                        return std::move(pool).to_vector(k);
                    }
                }
            }

            const int32_t po = self.getPo();
            const int32_t pl = self.getPl();
            const int32_t nl = self.getNl();
            const size_t edges_per_vertex = self.edges_per_vertex_;
            alignas(64) uint32_t good_neighbors[256];

            auto prefetch_feature = [pl](const char* ptr) { deglib::memory::prefetch(ptr, static_cast<size_t>(pl) * deglib::memory::L1_CACHE_LINE_SIZE); };

            while (pool.has_next()) {
                uint32_t u = pool.pop();
                const auto neighbor_indices = self.neighbors_by_index(u);

                int32_t good_neighbor_count = 0;
                for (size_t i = 0; i < edges_per_vertex; ++i) {
                    uint32_t neighbor_index = neighbor_indices[i];
                    if (checked_ids[neighbor_index] == checked_ids_tag) {
                        continue;
                    }
                    checked_ids[neighbor_index] = checked_ids_tag;
                    good_neighbors[good_neighbor_count++] = neighbor_index;
                }

                // Cap the neighbor count based on the remaining distance budget
                if constexpr (use_max_distance_count) {
                    const uint32_t remaining = max_distance_computation_count - distance_computation_count;
                    if (static_cast<uint32_t>(good_neighbor_count) > remaining) good_neighbor_count = static_cast<int32_t>(remaining);
                }

                for (int32_t i = 0; i < std::min<int32_t>(po, good_neighbor_count); ++i) {
                    prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i])));
                }

                for (int32_t i = 0; i < good_neighbor_count; ++i) {
                    if (i + po < good_neighbor_count) {
                        prefetch_feature(reinterpret_cast<const char*>(self.feature_by_index(good_neighbors[i + po])));
                    }
                    uint32_t neighbor_index = good_neighbors[i];
                    const auto feature = self.feature_by_index(neighbor_index);
                    const float dist = COMPARATOR::compare(query, feature, dist_func_param);
                    if (pool.insert(neighbor_index, dist)) {
                        deglib::memory::prefetch(
                            reinterpret_cast<const char*>(self.neighbors_by_index(neighbor_index)), static_cast<size_t>(nl) * deglib::memory::L1_CACHE_LINE_SIZE
                        );
                    }

                    // early stop after to many computations
                    if constexpr (use_max_distance_count) {
                        if (++distance_computation_count >= max_distance_computation_count) {
                            return std::move(pool).to_vector(k);
                        }
                    }
                }
            }

            return std::move(pool).to_vector(k);
        }
    }

    /**
     * Dispatches runtime metric, filter and distance budget settings to compile-time specialized searchEfImpl instantiations.
     */
    template <typename GraphType>
    static std::vector<deglib::graph::ObjectDistance> searchEfInternImpl(
        const GraphType& self,
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const uint32_t ef,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0
    ) {
        return self.feature_space_.compute([&]<deglib::distances::DistanceFunction Dist>(Dist) -> std::vector<deglib::graph::ObjectDistance> {
            if (filter) {
                if (max_distance_computation_count == 0) {
                    return searchEfImpl<GraphType, Dist, false, true>(self, entry_vertex_indices, query, k, ef, include_entry, filter, 0);
                } else {
                    return searchEfImpl<GraphType, Dist, true, true>(
                        self, entry_vertex_indices, query, k, ef, include_entry, filter, max_distance_computation_count
                    );
                }
            } else {
                if (max_distance_computation_count == 0) {
                    return searchEfImpl<GraphType, Dist, false, false>(self, entry_vertex_indices, query, k, ef, include_entry, nullptr, 0);
                } else {
                    return searchEfImpl<GraphType, Dist, true, false>(
                        self, entry_vertex_indices, query, k, ef, include_entry, nullptr, max_distance_computation_count
                    );
                }
            }
        });
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
                next_vertices.emplace(index, distance);
                trackback.emplace(index, deglib::graph::ObjectDistance(index, distance));
            }
        }

        // search radius
        auto radius = std::numeric_limits<float>::max();
        auto exploration_radius = radius;

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
