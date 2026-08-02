// test_filter.cpp — Unit tests for deglib::graph::Filter

#include <vector>
#include <cstdint>
#include <cstdlib>

#include "filter.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  Filter — basic bitset operations
// ---------------------------------------------------------------------------

TEST(Filter, EmptyFilter) {
    std::vector<int> empty;
    deglib::graph::Filter f(empty.data(), 0, 10, 10);
    EXPECT_EQ(f.size(), 0u);
    EXPECT_FALSE(f.is_valid(0));
    EXPECT_FALSE(f.is_valid(5));
    EXPECT_FALSE(f.is_valid(10));
}

TEST(Filter, SingleLabel) {
    int labels[] = {5};
    deglib::graph::Filter f(labels, 1, 10, 11);
    EXPECT_EQ(f.size(), 1u);
    EXPECT_TRUE(f.is_valid(5));
    EXPECT_FALSE(f.is_valid(0));
    EXPECT_FALSE(f.is_valid(10));
}

TEST(Filter, MultipleLabels) {
    int labels[] = {0, 3, 7, 10};
    deglib::graph::Filter f(labels, 4, 10, 11);
    EXPECT_EQ(f.size(), 4u);
    EXPECT_TRUE(f.is_valid(0));
    EXPECT_TRUE(f.is_valid(3));
    EXPECT_TRUE(f.is_valid(7));
    EXPECT_TRUE(f.is_valid(10));
    EXPECT_FALSE(f.is_valid(1));
    EXPECT_FALSE(f.is_valid(5));
}

TEST(Filter, DuplicatesIgnored) {
    int labels[] = {5, 5, 5, 5};
    deglib::graph::Filter f(labels, 4, 10, 11);
    EXPECT_EQ(f.size(), 1u); // only one unique label
    EXPECT_TRUE(f.is_valid(5));
}

TEST(Filter, OutOfRangeLabelsIgnored) {
    int labels[] = {-1, 0, 15, 5};
    deglib::graph::Filter f(labels, 4, 10, 11);
    // -1 and 15 are out of range [0, 10], only 0 and 5 are valid
    EXPECT_EQ(f.size(), 2u);
    EXPECT_TRUE(f.is_valid(0));
    EXPECT_TRUE(f.is_valid(5));
    EXPECT_FALSE(f.is_valid(-1));
    EXPECT_FALSE(f.is_valid(15));
}

TEST(Filter, BoundaryLabels) {
    int labels[] = {0, 10};
    deglib::graph::Filter f(labels, 2, 10, 11);
    EXPECT_EQ(f.size(), 2u);
    EXPECT_TRUE(f.is_valid(0));
    EXPECT_TRUE(f.is_valid(10));
    EXPECT_FALSE(f.is_valid(11));
}

TEST(Filter, LargeMaxValue) {
    int labels[] = {0, 63, 64, 127, 128};
    deglib::graph::Filter f(labels, 5, 128, 129);
    EXPECT_EQ(f.size(), 5u);
    EXPECT_TRUE(f.is_valid(0));
    EXPECT_TRUE(f.is_valid(63));
    EXPECT_TRUE(f.is_valid(64));
    EXPECT_TRUE(f.is_valid(127));
    EXPECT_TRUE(f.is_valid(128));
    EXPECT_FALSE(f.is_valid(1));
    EXPECT_FALSE(f.is_valid(62));
}

TEST(Filter, InclusionRate) {
    int labels[] = {0, 1, 2};
    deglib::graph::Filter f(labels, 3, 9, 10);
    EXPECT_DOUBLE_EQ(f.get_inclusion_rate(), 0.3);
}

TEST(Filter, InclusionRateFull) {
    std::vector<int> labels(10);
    for (int i = 0; i < 10; ++i) labels[i] = i;
    deglib::graph::Filter f(labels.data(), 10, 9, 10);
    EXPECT_DOUBLE_EQ(f.get_inclusion_rate(), 1.0);
}

TEST(Filter, ForEachValidLabel) {
    int labels[] = {1, 3, 5};
    deglib::graph::Filter f(labels, 3, 5, 6);

    std::vector<int> collected;
    f.for_each_valid_label([&](int label) { collected.push_back(label); });

    EXPECT_EQ(collected.size(), 3u);
    // for_each iterates in ascending order (bit 0, 1, 2, ...)
    EXPECT_EQ(collected[0], 1);
    EXPECT_EQ(collected[1], 3);
    EXPECT_EQ(collected[2], 5);
}

TEST(Filter, ForEachValidLabelEmpty) {
    std::vector<int> empty;
    deglib::graph::Filter f(empty.data(), 0, 10, 11);
    int count = 0;
    f.for_each_valid_label([&](int) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST(Filter, BitsetCorrectness) {
    // Test that labels spanning multiple 64-bit words work correctly.
    std::vector<int> labels;
    labels.push_back(0);
    labels.push_back(63);
    labels.push_back(64);
    labels.push_back(127);

    deglib::graph::Filter f(labels.data(), labels.size(), 127, 128);
    EXPECT_EQ(f.size(), 4u);

    EXPECT_TRUE(f.is_valid(0));
    EXPECT_TRUE(f.is_valid(63));
    EXPECT_TRUE(f.is_valid(64));
    EXPECT_TRUE(f.is_valid(127));
    EXPECT_FALSE(f.is_valid(1));
    EXPECT_FALSE(f.is_valid(62));
    EXPECT_FALSE(f.is_valid(65));
    EXPECT_FALSE(f.is_valid(126));
}
