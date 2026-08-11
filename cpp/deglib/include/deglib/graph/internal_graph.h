#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "deglib/distances.h"
#include "deglib/filter.h"

// Forward declaration for friend access
namespace deglib::builder {
class EvenRegularGraphBuilder;
}

namespace deglib::graph {

/**
 * Represents a pair of a vertex identifier and its corresponding distance.
 * Designed as a trivially default-constructible type (POD) for zero-overhead allocations.
 */
class ObjectDistance
{
    uint32_t identifier_;
    float distance_;

public:
    // Default constructor generates zero instructions (uninitialized memory for maximum performance)
    ObjectDistance() = default;

    constexpr ObjectDistance(const uint32_t identifier, const float distance) noexcept
        : identifier_(identifier), distance_(distance) {}

    [[nodiscard]] constexpr uint32_t getIdentifier() const noexcept { 
        return identifier_; 
    }

    [[nodiscard]] constexpr float getDistance() const noexcept { 
        return distance_; 
    }

    constexpr bool operator==(const ObjectDistance& o) const noexcept { 
        return distance_ == o.distance_ && identifier_ == o.identifier_; 
    }

    constexpr bool operator<(const ObjectDistance& o) const noexcept {
        if (distance_ == o.distance_)
            return identifier_ < o.identifier_;
        return distance_ < o.distance_;
    }

    constexpr bool operator>(const ObjectDistance& o) const noexcept {
        if (distance_ == o.distance_)
            return identifier_ > o.identifier_;
        return distance_ > o.distance_;
    }
};

/**
 * Priority Queue based on std::vector using binary heap operations.
 */
template<class Compare, class ObjectType>
class PQV : public std::vector<ObjectType> {
    Compare comp;

public:
    explicit PQV(Compare cmp = Compare()) : comp(cmp) {}

    /// Const-correct access to the top element of the heap.
    [[nodiscard]] const ObjectType& top() const { 
        return this->front(); 
    }

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
    void heapify() {
        std::make_heap(this->begin(), this->end(), comp);
    }

    /// Sorts the heap elements in ascending order using the internal comparator,
    /// consuming the heap property. After calling this, top() is no longer valid
    /// until heapify() is called again.
    void sort() {
        std::sort_heap(this->begin(), this->end(), comp);
    }
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
class InternalGraph
{
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

    const std::vector<uint32_t> getEntryVertexIndices() const {
      return std::vector<uint32_t> { 0 };
    }

    /**
     * Perform a search but stops when the to_vertex was found.
     */
    virtual std::vector<deglib::graph::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const = 0;

    /**
     * Bounds-checked internal search for query vectors.
     */
    template <typename T>
    deglib::graph::ResultSet search(
        std::span<const T> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0) const 
    {
        if (query.size_bytes() < getFeatureSpace().get_data_size()) {
            throw std::invalid_argument(
                "Search query buffer mismatch: expected at least " + std::to_string(getFeatureSpace().get_data_size()) +
                " bytes (dim=" + std::to_string(getFeatureSpace().dim()) + "), got " + std::to_string(query.size_bytes()) + " bytes");
        }
        return search_intern(getEntryVertexIndices(), reinterpret_cast<const std::byte*>(query.data()), k, eps, true, filter, max_distance_computation_count);
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
        const deglib::search::Filter* filter = nullptr) const
    {
        const auto query_ptr = getFeatureVector(entry_vertex_index);
        return search_intern({ entry_vertex_index }, query_ptr, k, eps, include_entry, filter, max_distance_computation_count);
    }

    virtual deglib::graph::ResultSet search_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::search::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0) const = 0;

    friend class deglib::builder::EvenRegularGraphBuilder;
};

} // namespace deglib::graph



