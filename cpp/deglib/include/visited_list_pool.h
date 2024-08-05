#pragma once

#include <algorithm>
#include <mutex>
#include <vector>
#include <deque>
#include <cstdint>

/**
 * Ref https://raw.githubusercontent.com/nmslib/hnswlib/master/hnswlib/visited_list_pool.h
 */
namespace deglib::graph {

class VisitedList {
private:
    uint16_t current_tag_{1};
    std::unique_ptr<uint16_t[]> slots_;
    unsigned int num_elements_;

public:
    explicit VisitedList(int numelements1) : slots_(std::make_unique<uint16_t[]>(numelements1)), num_elements_(numelements1) {}

    [[nodiscard]] auto* get_visited() const {
        return slots_.get();
    }

    [[nodiscard]] auto get_tag() const {
        return current_tag_;
    }

    void reset() {
        ++current_tag_;
        if (current_tag_ == 0) {
            std::fill_n(slots_.get(), num_elements_, 0);
            ++current_tag_;
        }
    }
};

class VisitedListPool {
private:
    using ListPtr = std::unique_ptr<VisitedList>;

    std::deque<ListPtr> pool_;
    std::mutex pool_guard_;
    int num_elements_;

 public:
    VisitedListPool(int initmaxpools, int numelements) : num_elements_(numelements) {
        for (int i = 0; i < initmaxpools; i++)
            pool_.push_front(std::make_unique<VisitedList>(numelements));
    }

    class FreeVisitedList {
        private:
            friend VisitedListPool;
            
            VisitedListPool& pool_;
            ListPtr list_;

            FreeVisitedList(VisitedListPool& pool, ListPtr list) : pool_(pool), list_(std::move(list)) {}

        public:        
            FreeVisitedList(const FreeVisitedList& other) = delete;
            FreeVisitedList(FreeVisitedList&& other) = delete;
            FreeVisitedList& operator=(const FreeVisitedList& other) = delete;
            FreeVisitedList& operator=(FreeVisitedList&& other) = delete;

            ~FreeVisitedList() noexcept {
                pool_.releaseVisitedList(std::move(list_));
            }

            auto operator->() const {
                return list_.get();
            }
    };

    FreeVisitedList getFreeVisitedList() {
        ListPtr rez = popVisitedList();
        if (rez) {
            rez->reset();
        } else {
            rez = std::make_unique<VisitedList>(num_elements_);
        }
        return {*this, std::move(rez)};
    }

private:
    void releaseVisitedList(ListPtr vl) {
        std::unique_lock <std::mutex> lock(pool_guard_);
        pool_.push_back(std::move(vl));
    }

    ListPtr popVisitedList() {
        ListPtr rez;
        std::unique_lock <std::mutex> lock(pool_guard_);
        if (!pool_.empty()) {
            rez = std::move(pool_.front());
            pool_.pop_front();
        }
        return rez;
    }
};

}  // namespace deglib::graph