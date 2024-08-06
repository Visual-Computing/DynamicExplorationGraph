#pragma once

#include "intrusive_list.h"

#include <algorithm>
#include <mutex>
#include <memory>
#include <cstdint>

/**
 * Ref https://raw.githubusercontent.com/nmslib/hnswlib/master/hnswlib/visited_list_pool.h
 */
namespace deglib::graph {

class VisitedListPool;

class VisitedList {
private:
    friend VisitedListPool;

    VisitedList* next_;
    VisitedList* prev_;
    std::unique_ptr<uint16_t[]> slots_;
    uint32_t num_elements_;
    uint16_t current_tag_{1};

public:
    explicit VisitedList(uint32_t numelements1) : slots_(std::make_unique<uint16_t[]>(numelements1)), num_elements_(numelements1) {}

    [[nodiscard]] auto* get_visited() const {
        return slots_.get();
    }

    [[nodiscard]] auto get_tag() const {
        return current_tag_;
    }

    void reset() {
        ++current_tag_;
        if (current_tag_ == 0) {
            std::fill_n(slots_.get(), num_elements_, uint16_t{});
            ++current_tag_;
        }
    }
};

class VisitedListPool {
private:
    IntrusiveList<VisitedList, &VisitedList::next_, &VisitedList::prev_> pool_;
    std::mutex pool_guard_;
    uint32_t num_elements_;

 public:
    VisitedListPool(uint32_t initmaxpools, uint32_t numelements) : num_elements_(numelements) {
        for (uint32_t i = 0; i < initmaxpools; i++)
            pool_.push_back(new VisitedList(numelements));
    }

    ~VisitedListPool() noexcept {
        while (!pool_.empty()) {
            delete pool_.pop_front();
        }
    }

    class FreeVisitedList {
        private:
            friend VisitedListPool;
            
            VisitedListPool& pool_;
            VisitedList& list_;

            FreeVisitedList(VisitedListPool& pool, VisitedList& list) : pool_(pool), list_(list) {}

        public:        
            FreeVisitedList(const FreeVisitedList& other) = delete;
            FreeVisitedList(FreeVisitedList&& other) = delete;
            FreeVisitedList& operator=(const FreeVisitedList& other) = delete;
            FreeVisitedList& operator=(FreeVisitedList&& other) = delete;

            ~FreeVisitedList() noexcept {
                pool_.releaseVisitedList(list_);
            }

            auto operator->() const {
                return &list_;
            }
    };

    [[nodiscard]] FreeVisitedList getFreeVisitedList() {
        auto rez = popVisitedList();
        if (rez) {
            rez->reset();
        } else {
            rez = new VisitedList(num_elements_);
        }
        return {*this, *rez};
    }

private:
    void releaseVisitedList(VisitedList& vl) {
        std::unique_lock <std::mutex> lock(pool_guard_);
        pool_.push_back(&vl);
    }

    VisitedList* popVisitedList() {
        VisitedList* rez{};
        std::unique_lock <std::mutex> lock(pool_guard_);
        if (!pool_.empty()) {
            rez = pool_.pop_front();
        }
        return rez;
    }
};

}  // namespace deglib::graph