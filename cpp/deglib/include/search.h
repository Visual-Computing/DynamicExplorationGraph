#pragma once

#include <queue>
#include <distances.h>

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



/**
 * priority queue with access to the internal data.
 * therefore access to the unsorted data is possible.
 * 
 * https://stackoverflow.com/questions/4484767/how-to-iterate-over-a-priority-queue
 * https://www.linuxtopia.org/online_books/programming_books/c++_practical_programming/c++_practical_programming_189.html
 */
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
// typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::greater<ObjectDistance>> UncheckedSet;
typedef PQV<std::greater<ObjectDistance>, ObjectDistance> UncheckedSet;




class SearchGraph
{
  public:    
    virtual ~SearchGraph() = default;
    virtual const uint32_t size() const = 0;
    virtual const uint8_t getEdgesPerVertex() const = 0;
    virtual const deglib::SpaceInterface<float>& getFeatureSpace() const = 0;

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
     * Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.
     * 
     * Eps greater 0 extends the search range and takes additional graph vertices into account. 
     * 
     * It is possible to limit the amount of work by specifing a maximal number of distances to be calculated.
     * For lower numbers it is recommended to set eps to 0 since its very unlikly the method can make use of the extended the search range.
     * 
     * The starting point of the search is determined be the graph
     */
    deglib::search::ResultSet search(const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) const {
      return search(getEntryVertexIndices(), query, eps,  k, filter, max_distance_computation_count);
    };

    /**
     * Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.
     * 
     * Eps greater 0 extends the search range and takes additional graph vertices into account. 
     * 
     * It is possible to limit the amount of work by specifing a maximal number of distances to be calculated.
     * For lower numbers it is recommended to set eps to 0 since its very unlikly the method can make use of the extended the search range.
     */
    virtual deglib::search::ResultSet search(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const deglib::graph::Filter* filter = nullptr, const uint32_t max_distance_computation_count = 0) const = 0;

    /**
     * A exploration for similar element, limited by max_distance_computation_count
     */
    virtual deglib::search::ResultSet explore(const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count) const = 0;
};

} // end namespace deglib::search
