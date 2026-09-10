#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace deglib::search {

/**
 * A pair of a vertex identifier and its corresponding distance.
 *
 * Supports an arbitrary boolean flag (isFlagged / setFlagged) stored in bit 31
 * without increasing memory footprint. getIdentifier() always returns the 31-bit identifier.
 */
class ObjectDistance {
    uint32_t identifier_;
    float distance_;

  public:
    static constexpr uint32_t kFlagMask = 1u << 31;
    static constexpr uint32_t kIdentifierMask = ~kFlagMask;  // 0x7FFFFFFF

    constexpr ObjectDistance() noexcept = default;
    constexpr ObjectDistance(const uint32_t identifier, const float distance) noexcept : identifier_(identifier), distance_(distance) {}

    /// Returns the vertex identifier with the flag bit (bit 31) stripped off.
    [[nodiscard]] constexpr uint32_t getIdentifier() const noexcept { return identifier_ & kIdentifierMask; }

    /// Returns the distance value.
    [[nodiscard]] constexpr float getDistance() const noexcept { return distance_; }

    /// Checks whether the boolean flag (bit 31) is set.
    [[nodiscard]] constexpr bool isFlagged() const noexcept { return (identifier_ & kFlagMask) != 0; }

    /// Sets or clears the boolean flag (bit 31).
    constexpr void setFlagged(const bool flagged = true) noexcept {
        if (flagged) {
            identifier_ |= kFlagMask;
        } else {
            identifier_ &= kIdentifierMask;
        }
    }

    constexpr bool operator==(const ObjectDistance& o) const noexcept { return distance_ == o.distance_ && getIdentifier() == o.getIdentifier(); }

    constexpr bool operator<(const ObjectDistance& o) const noexcept {
        if (distance_ == o.distance_) return getIdentifier() < o.getIdentifier();
        return distance_ < o.distance_;
    }

    constexpr bool operator>(const ObjectDistance& o) const noexcept {
        if (distance_ == o.distance_) return getIdentifier() > o.getIdentifier();
        return distance_ > o.distance_;
    }
};

/**
 * Internal candidate buffer and beam-search frontier used during graph traversal.
 */
class ResultList {
  public:
    // --- Lifecycle ---

    /// Constructs an empty ResultList with zero capacity.
    ResultList() = default;

    /**
     * Constructs a ResultList with specified exploration budget and result capacity.
     *
     * @param ef       Exploration budget: maximum number of candidates permitted to be expanded via pop().
     * @param capacity Maximum number of nearest results retained (typically max(k, ef)).
     */
    ResultList(int32_t ef, int32_t capacity) : ef_(ef), capacity_(capacity), data_(capacity + 1) {}

    ResultList(const ResultList&) = delete;
    ResultList(ResultList&& rhs) noexcept = default;
    ResultList& operator=(const ResultList&) = delete;
    ResultList& operator=(ResultList&& rhs) noexcept = default;

    // --- Search Traversal API ---

    /**
     * Inserts a candidate vertex into the sorted list.
     * If the list is at capacity and dist >= worst retained distance, insertion is skipped.
     * Otherwise, locates the insertion index via binary search, shifts elements, and inserts in-place.
     *
     * @param u    Vertex identifier.
     * @param dist Distance to query vector.
     * @return     True if candidate was inserted into the retained capacity, false if rejected.
     */
    inline bool insert(uint32_t u, float dist) noexcept {
        const ObjectDistance cand(u, dist);
        if (size_ == capacity_ && !(cand < data_[size_ - 1])) {
            return false;
        }
        int32_t lo = find_bsearch(cand);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(ObjectDistance));
        data_[lo] = cand;
        if (size_ < capacity_) {
            size_++;
        }
        if (lo < cur_) {
            cur_ = lo;
        }
        return true;
    }

    /**
     * Returns whether there are further unexpanded candidates within the exploration budget (ef).
     */
    [[nodiscard]] inline bool has_next() const noexcept { return cur_ < size_ && cur_ < ef_; }

    /**
     * Marks the next unexpanded candidate as flagged and returns its vertex identifier.
     * Advances the internal cursor to the next unflagged candidate.
     *
     * @return Vertex identifier of the expanded candidate.
     */
    inline uint32_t pop() noexcept {
        data_[cur_].setFlagged(true);
        int32_t pre = cur_;
        while (cur_ < size_ && data_[cur_].isFlagged()) {
            cur_++;
        }
        return data_[pre].getIdentifier();
    }

    // --- Result & Container Access API ---

    /// Returns the number of valid nearest neighbor results currently stored in the list.
    [[nodiscard]] inline size_t size() const noexcept { return static_cast<size_t>(size_); }

    /// Checks if the result list contains zero results.
    [[nodiscard]] inline bool empty() const noexcept { return size_ == 0; }

    /// Returns the maximum capacity of this result list.
    [[nodiscard]] inline size_t capacity() const noexcept { return static_cast<size_t>(capacity_); }

    /// Returns the vertex identifier at 0-based rank index i (sorted ascending by distance).
    [[nodiscard]] inline uint32_t id(size_t i) const noexcept { return data_[i].getIdentifier(); }

    /// Returns the distance at 0-based rank index i (sorted ascending by distance).
    [[nodiscard]] inline float dist(size_t i) const noexcept { return data_[i].getDistance(); }

    /// Const random-access to the ObjectDistance entry at 0-based rank index i.
    [[nodiscard]] inline const ObjectDistance& operator[](size_t i) const noexcept { return data_[i]; }

    /// Mutable random-access to the ObjectDistance entry at 0-based rank index i.
    [[nodiscard]] inline ObjectDistance& operator[](size_t i) noexcept { return data_[i]; }

    /// Returns a reference to the closest result (first element).
    [[nodiscard]] inline const ObjectDistance& front() const noexcept { return data_.front(); }

    /// Returns a reference to the furthest valid result (last element).
    [[nodiscard]] inline const ObjectDistance& back() const noexcept { return data_[size_ - 1]; }

    /// Pointer to the underlying contiguous ObjectDistance array.
    [[nodiscard]] inline const ObjectDistance* data() const noexcept { return data_.data(); }

    /// Const iterator to the beginning of the valid results.
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }

    /// Const iterator to the end of the valid results.
    [[nodiscard]] auto end() const noexcept { return data_.begin() + size_; }

    /// Mutable iterator to the beginning of the valid results.
    [[nodiscard]] auto begin() noexcept { return data_.begin(); }

    /// Mutable iterator to the end of the valid results.
    [[nodiscard]] auto end() noexcept { return data_.begin() + size_; }

    /// Explicit const iterator to the beginning of the valid results.
    [[nodiscard]] auto cbegin() const noexcept { return data_.cbegin(); }

    /// Explicit const iterator to the end of the valid results.
    [[nodiscard]] auto cend() const noexcept { return data_.cbegin() + size_; }

    /// Const reverse iterator to the reverse beginning of the valid results.
    [[nodiscard]] auto rbegin() const noexcept { return std::make_reverse_iterator(end()); }

    /// Const reverse iterator to the reverse end of the valid results.
    [[nodiscard]] auto rend() const noexcept { return std::make_reverse_iterator(begin()); }

    /// Mutable reverse iterator to the reverse beginning of the valid results.
    [[nodiscard]] auto rbegin() noexcept { return std::make_reverse_iterator(end()); }

    /// Mutable reverse iterator to the reverse end of the valid results.
    [[nodiscard]] auto rend() noexcept { return std::make_reverse_iterator(begin()); }

    /// Explicit const reverse iterator to the reverse beginning of the valid results.
    [[nodiscard]] auto crbegin() const noexcept { return std::make_reverse_iterator(cend()); }

    /// Explicit const reverse iterator to the reverse end of the valid results.
    [[nodiscard]] auto crend() const noexcept { return std::make_reverse_iterator(cbegin()); }

    /// Returns a C++20 std::span viewing all valid ObjectDistance results.
    [[nodiscard]] inline std::span<const ObjectDistance> span() const noexcept { return {data_.data(), static_cast<size_t>(size_)}; }

    /**
     * Truncates internal results to at most k elements and moves them into a std::vector.
     */
    [[nodiscard]] std::vector<ObjectDistance> to_vector(size_t k) && noexcept {
        const size_t count = std::min<size_t>(k, static_cast<size_t>(size_));
        data_.resize(count);
        return std::move(data_);
    }

  private:
    int32_t size_ = 0;
    int32_t cur_ = 0;
    int32_t ef_ = 0;
    int32_t capacity_ = 0;

    std::vector<ObjectDistance> data_;

    /**
     * Binary search to locate the insertion index for a given distance to maintain ascending order.
     */
    [[nodiscard]] inline int32_t find_bsearch(const ObjectDistance& cand) const noexcept {
        int32_t lo = 0, hi = size_;
        while (lo < hi) {
            int32_t mid = (lo + hi) / 2;
            if (data_[mid] > cand) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
};

}  // namespace deglib::search

namespace deglib::graph {
using ObjectDistance = deglib::search::ObjectDistance;
}  // namespace deglib::graph
