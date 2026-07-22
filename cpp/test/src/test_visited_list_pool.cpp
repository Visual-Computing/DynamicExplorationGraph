// test_visited_list_pool.cpp — Unit tests for VisitedList and VisitedListPool

#include <cstdint>
#include <thread>
#include <vector>

#include "visited_list_pool.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  VisitedList
// ---------------------------------------------------------------------------

TEST(VisitedList, CtorInitializesToZero) {
    deglib::graph::VisitedList vl(100);
    auto* slots = vl.get_visited();
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(slots[i], 0u);
    }
}

TEST(VisitedList, TagStartsAtOne) {
    deglib::graph::VisitedList vl(10);
    EXPECT_EQ(vl.get_tag(), 1u);
}

TEST(VisitedList, MarkVisited) {
    deglib::graph::VisitedList vl(10);
    auto* slots = vl.get_visited();
    auto tag = vl.get_tag();

    slots[3] = tag;
    slots[7] = tag;

    EXPECT_EQ(slots[3], tag);
    EXPECT_EQ(slots[7], tag);
    EXPECT_NE(slots[0], tag);
}

TEST(VisitedList, ResetIncrementsTag) {
    deglib::graph::VisitedList vl(10);
    auto tag1 = vl.get_tag();
    vl.reset();
    auto tag2 = vl.get_tag();
    EXPECT_EQ(tag2, tag1 + 1);
}

TEST(VisitedList, ResetDoesNotClearSlots) {
    deglib::graph::VisitedList vl(10);
    auto* slots = vl.get_visited();
    auto tag = vl.get_tag();

    slots[5] = tag;
    vl.reset();

    // slots should still contain the old tag value
    EXPECT_EQ(slots[5], tag);
}

TEST(VisitedList, TagOverflowHandling) {
    deglib::graph::VisitedList vl(10);
    for (int i = 0; i < 100; ++i) {
        vl.reset();
    }
    auto* slots = vl.get_visited();
    auto tag = vl.get_tag();
    slots[5] = tag;
    EXPECT_EQ(slots[5], tag);
}

// ---------------------------------------------------------------------------
//  VisitedListPool
// ---------------------------------------------------------------------------

TEST(VisitedListPool, CtorCreatesPools) {
    deglib::graph::VisitedListPool pool(3, 100);
    auto fl1 = pool.getFreeVisitedList();
    auto fl2 = pool.getFreeVisitedList();
    auto fl3 = pool.getFreeVisitedList();

    EXPECT_NE(fl1.operator->(), nullptr);
    EXPECT_NE(fl2.operator->(), nullptr);
    EXPECT_NE(fl3.operator->(), nullptr);
}

TEST(VisitedListPool, GetAndRelease) {
    deglib::graph::VisitedListPool pool(1, 10);
    auto fl = pool.getFreeVisitedList();
    auto* slots = fl->get_visited();
    auto tag = fl->get_tag();

    slots[5] = tag;
    // fl goes out of scope -> released back to pool
    EXPECT_EQ(slots[5], tag);
}

TEST(VisitedListPool, ReleaseReusesList) {
    deglib::graph::VisitedListPool pool(1, 10);

    {
        auto fl = pool.getFreeVisitedList();
        auto* slots = fl->get_visited();
        auto tag = fl->get_tag();
        slots[3] = tag;
    } // released

    {
        auto fl = pool.getFreeVisitedList();
        // reset() only increments the tag, it does NOT clear slots.
        // The old value remains but the new tag won't match it.
        auto* slots = fl->get_visited();
        auto tag = fl->get_tag();
        EXPECT_NE(slots[3], tag); // old value, new tag -> mismatch
    }
}

TEST(VisitedListPool, MoreGetsThanPoolSize) {
    deglib::graph::VisitedListPool pool(2, 50);

    auto fl1 = pool.getFreeVisitedList();
    auto fl2 = pool.getFreeVisitedList();
    auto fl3 = pool.getFreeVisitedList();

    EXPECT_NE(fl1.operator->(), nullptr);
    EXPECT_NE(fl2.operator->(), nullptr);
    EXPECT_NE(fl3.operator->(), nullptr);
    // fl1, fl2, fl3 released automatically when scope ends
}

TEST(VisitedListPool, TagIsFreshAfterGet) {
    deglib::graph::VisitedListPool pool(1, 10);

    auto fl1 = pool.getFreeVisitedList();
    auto tag1 = fl1->get_tag();
    auto* slots1 = fl1->get_visited();
    slots1[5] = tag1;
    // fl1 released automatically

    auto fl2 = pool.getFreeVisitedList();
    auto tag2 = fl2->get_tag();
    // tag should be different (incremented by reset())
    EXPECT_NE(tag2, tag1);
    // slots should still have old value but tag doesn't match
    EXPECT_NE(fl2->get_visited()[5], tag2);
}

TEST(VisitedListPool, ThreadSafetyBasic) {
    deglib::graph::VisitedListPool pool(10, 100);

    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&pool]() {
            for (int j = 0; j < 100; ++j) {
                auto fl = pool.getFreeVisitedList();
                auto* slots = fl->get_visited();
                auto tag = fl->get_tag();
                slots[j % 100] = tag;
                // fl released automatically
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}
