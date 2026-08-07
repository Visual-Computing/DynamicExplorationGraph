#pragma once

#include <cstddef>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include "deglib/distances.h"

#include "deglib/filter.h"

// Forward declaration for friend access
namespace deglib::builder {
class EvenRegularGraphBuilder;
}

namespace deglib::search
{

class ObjectDistance
{
    uint32_t internal_index_;
    float distance_;

  public:
    ObjectDistance() {}

    ObjectDistance(const uint32_t internal_index, const float distance) : internal_index_(internal_index), distance_(distance) {}

    inline const uint32_t getInternalIndex() const { 
      return internal_index_; 
    }

    inline const float getDistance() const { 
      return distance_; 
    }

    inline bool operator==(const ObjectDistance& o) const { 
      return (distance_ == o.distance_) && (internal_index_ == o.internal_index_); 
    }

    inline bool operator<(const ObjectDistance& o) const {
      if (distance_ == o.distance_)
        return internal_index_ < o.internal_index_;
      else
        return distance_ < o.distance_;
    }

    inline bool operator>(const ObjectDistance& o) const {
      if (distance_ == o.distance_)
        return internal_index_ > o.internal_index_;
      else
        return distance_ > o.distance_;
    }
};

template<class Compare, class ObjectType>
class PQV : public std::vector<ObjectType> {
  Compare comp;
  public:
    PQV(Compare cmp = Compare()) : comp(cmp) {}

    const ObjectType& top() { return this->front(); }

    template <class... _Valty>
    void emplace(_Valty&&... _Val) {
      this->emplace_back(std::forward<_Valty>(_Val)...);
      std::push_heap(this->begin(), this->end(), comp);
    }

    void push(const ObjectType& x) {
      this->push_back(x);
      std::push_heap(this->begin(),this->end(), comp);
    }

    void pop() {
      std::pop_heap(this->begin(),this->end(), comp);
      this->pop_back();
    }
};

// search result set containing vertex ids and distances
typedef PQV<std::less<ObjectDistance>, ObjectDistance> ResultSet;

// set of unchecked vertex ids
typedef PQV<std::greater<ObjectDistance>, ObjectDistance> UncheckedSet;

class SearchGraph
{
  public:    
    virtual ~SearchGraph() = default;
    virtual const uint32_t size() const = 0;
    virtual const uint8_t getEdgesPerVertex() const = 0;
    virtual const deglib::FloatSpace& getFeatureSpace() const = 0;

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
    virtual std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const = 0;

    /**
     * Bounds-checked public search for query vectors (float span).
     */
    deglib::search::ResultSet search(
        std::span<const float> query,
        const uint32_t k,
        const float eps = 0.0f,
        const deglib::graph::Filter* filter = nullptr,
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
     * Public graph exploration starting at entry_vertex_index.
     */
    deglib::search::ResultSet explore(
        const uint32_t entry_vertex_index,
        const uint32_t k,
        const uint32_t max_distance_computation_count = 0,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::graph::Filter* filter = nullptr) const
    {
        const auto query_ptr = getFeatureVector(entry_vertex_index);
        return search_intern({ entry_vertex_index }, query_ptr, k, eps, include_entry, filter, max_distance_computation_count);
    }

  protected:
    /**
     * Internal raw-pointer search implementation.
     */
    virtual deglib::search::ResultSet search_intern(
        const std::vector<uint32_t>& entry_vertex_indices,
        const std::byte* query,
        const uint32_t k,
        const float eps = 0.0f,
        const bool include_entry = true,
        const deglib::graph::Filter* filter = nullptr,
        const uint32_t max_distance_computation_count = 0) const = 0;

    friend class deglib::builder::EvenRegularGraphBuilder;
};

} // end namespace deglib::search
