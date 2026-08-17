#pragma once

#include <utility>

namespace deglib::graph {

/*
Usage:

``` 
struct Entry {
    Entry* next_; // list hooks
    Entry* prev_; // list hooks
    // ... other members
};

using List = IntrusiveList<Entry, &Entry::next_, &Entry::prev_>;
```

Make sure entries added to the list have a stable address.


Implementation adapted from
https://github.com/facebookexperimental/libunifex/blob/main/include/unifex/detail/intrusive_list.hpp
 */
template <class T, T* T::*Next, T* T::*Prev>
class IntrusiveList
{
  private:
    T* head_{};
    T* tail_{};

  public:
    IntrusiveList() = default;

    IntrusiveList(const IntrusiveList&) = delete;

    IntrusiveList(IntrusiveList&& other) noexcept
        : head_(std::exchange(other.head_, nullptr)), tail_(std::exchange(other.tail_, nullptr))
    {
    }

    ~IntrusiveList() = default;

    IntrusiveList& operator=(const IntrusiveList&) = delete;
    IntrusiveList& operator=(IntrusiveList&&) = delete;

    [[nodiscard]] bool empty() const noexcept { return head_ == nullptr; }

    void push_back(T* item) noexcept
    {
        item->*Prev = tail_;
        item->*Next = nullptr;
        if (tail_ == nullptr)
            head_ = item;
        else
            tail_->*Next = item;
        tail_ = item;
    }

    [[nodiscard]] T* pop_front() noexcept
    {
        T* item = head_;
        head_ = item->*Next;
        if (head_ != nullptr)
            head_->*Prev = nullptr;
        else
            tail_ = nullptr;
        return item;
    }
};

}  // namespace deglib::graph