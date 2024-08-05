#pragma once

#include <algorithm>
#include <mutex>
#include <vector>
#include <deque>
#include <cstdint>

/**
 * Copied from https://raw.githubusercontent.com/nmslib/hnswlib/master/hnswlib/visited_list_pool.h
 */
namespace deglib::graph {

class VisitedList {
private:
    uint16_t tag{1};
    std::unique_ptr<uint16_t[]> mass;
    unsigned int numelements;

public:
    explicit VisitedList(int numelements1) : mass(std::make_unique<uint16_t[]>(numelements1)), numelements(numelements1) {}

    [[nodiscard]] auto* get_visited() const {
        return mass.get();
    }

    [[nodiscard]] auto get_tag() const {
        return tag;
    }

    void reset() {
        ++tag;
        if (tag == 0) {
            std::fill_n(mass.get(), numelements, 0);
            ++tag;
        }
    }
};


///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
private:
    using ListPtr = std::unique_ptr<VisitedList>;

    std::deque<ListPtr> pool;
    std::mutex poolguard;
    int numelements;

 public:
    VisitedListPool(int initmaxpools, int numelements) : numelements(numelements) {
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(std::make_unique<VisitedList>(numelements));
    }

    class FreeVisitedList {
        private:
            friend VisitedListPool;
            
            VisitedListPool& self;
            ListPtr list;

            FreeVisitedList(VisitedListPool& self, ListPtr list) : self(self), list(std::move(list)) {}

        public:        
            FreeVisitedList(const FreeVisitedList& other) = delete;
            FreeVisitedList(FreeVisitedList&& other) = delete;
            FreeVisitedList& operator=(const FreeVisitedList& other) = delete;
            FreeVisitedList& operator=(FreeVisitedList&& other) = delete;

            ~FreeVisitedList() noexcept {
                self.releaseVisitedList(std::move(list));
            }

            auto operator->() const {
                return list.get();
            }
    };

    FreeVisitedList getFreeVisitedList() {
        ListPtr rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = std::move(pool.front());
                pool.pop_front();
            } else {
                rez = std::make_unique<VisitedList>(numelements);
            }
        }
        rez->reset();
        return {*this, std::move(rez)};
    }

private:
    void releaseVisitedList(ListPtr vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_back(std::move(vl));
    }
};

}  // namespace deglib::graph