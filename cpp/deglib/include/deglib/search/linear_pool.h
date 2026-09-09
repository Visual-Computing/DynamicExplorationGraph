#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace deglib::search {

template <typename DistT = float>
struct Neighbor {
    uint32_t id;
    DistT distance;

    Neighbor() = default;
    Neighbor(uint32_t id, DistT distance) : id(id), distance(distance) {}

    inline friend bool operator<(const Neighbor& lhs, const Neighbor& rhs) noexcept {
        return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.id < rhs.id);
    }

    inline friend bool operator>(const Neighbor& lhs, const Neighbor& rhs) noexcept { return !(lhs < rhs); }
};

template <typename DistT = float>
struct LinearPool {
    using dist_type = DistT;

    int32_t size_ = 0;
    int32_t cur_ = 0;
    int32_t ef_ = 0;
    int32_t capacity_ = 0;

    std::vector<Neighbor<DistT>> data_;

    LinearPool() = default;

    LinearPool(int32_t ef, int32_t capacity) : ef_(ef), capacity_(capacity), data_(capacity + 1) {}

    LinearPool(const LinearPool&) = delete;
    LinearPool(LinearPool&& rhs) noexcept = default;
    LinearPool& operator=(const LinearPool&) = delete;
    LinearPool& operator=(LinearPool&& rhs) noexcept = default;

    void reset(int32_t ef, int32_t cap) {
        size_ = 0;
        cur_ = 0;
        ef_ = ef;
        capacity_ = cap;
        if (static_cast<int32_t>(data_.size()) < cap + 1) {
            data_.resize(cap + 1);
        }
    }

    [[nodiscard]] inline int32_t find_bsearch(DistT dist) const noexcept {
        int32_t lo = 0, hi = size_;
        while (lo < hi) {
            int32_t mid = (lo + hi) / 2;
            if (data_[mid].distance > dist) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    inline bool insert(uint32_t u, DistT dist) noexcept {
        if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
            return false;
        }
        int32_t lo = find_bsearch(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<DistT>));
        data_[lo] = {u, dist};
        if (size_ < capacity_) {
            size_++;
        }
        if (lo < cur_) {
            cur_ = lo;
        }
        return true;
    }

    static constexpr uint32_t kMask = 0x7FFFFFFF;

    [[nodiscard]] inline uint32_t get_id(uint32_t raw_id) const noexcept { return raw_id & kMask; }

    inline void set_checked(uint32_t& raw_id) noexcept { raw_id |= (1u << 31); }

    [[nodiscard]] inline bool is_checked(uint32_t raw_id) const noexcept { return (raw_id >> 31) & 1; }

    inline uint32_t pop() noexcept {
        set_checked(data_[cur_].id);
        int32_t pre = cur_;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
            cur_++;
        }
        return get_id(data_[pre].id);
    }

    [[nodiscard]] inline bool has_next() const noexcept { return cur_ < size_ && cur_ < ef_; }

    [[nodiscard]] inline uint32_t id(int32_t i) const noexcept { return get_id(data_[i].id); }

    [[nodiscard]] inline DistT dist(int32_t i) const noexcept { return data_[i].distance; }

    [[nodiscard]] inline int32_t size() const noexcept { return size_; }

    [[nodiscard]] inline int32_t capacity() const noexcept { return capacity_; }

    void to_sorted(uint32_t* ids, float* scores, int32_t length) const noexcept {
        const int32_t count = std::min(length, size_);
        for (int32_t i = 0; i < count; ++i) {
            ids[i] = id(i);
            if (scores) {
                scores[i] = static_cast<float>(dist(i));
            }
        }
    }
};

}  // namespace deglib::search
