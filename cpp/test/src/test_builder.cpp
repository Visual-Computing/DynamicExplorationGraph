// test_builder.cpp — Unit tests for deglib::builder::UnionFind

#include <cstdint>
#include <limits>
#include <unordered_map>

#include "builder.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  UnionFind
// ---------------------------------------------------------------------------

TEST(UnionFind, DefaultValues) {
    deglib::builder::UnionFind uf(10);
    EXPECT_NE(uf.getDefaultValue(), 0u);
    EXPECT_EQ(uf.getDefaultValue(), std::numeric_limits<uint32_t>::max());
}

TEST(UnionFind, FindUnknownElement) {
    deglib::builder::UnionFind uf(10);
    EXPECT_EQ(uf.Find(5), uf.getDefaultValue());
}

TEST(UnionFind, SingleElement) {
    deglib::builder::UnionFind uf(10);
    uf.Update(5, 5); // element is its own parent
    EXPECT_EQ(uf.Find(5), 5u);
}

TEST(UnionFind, UnionTwoElements) {
    deglib::builder::UnionFind uf(10);
    uf.Update(5, 5);
    uf.Update(7, 7);
    uf.Union(5, 7);

    // After Union(5, 7), one becomes parent of the other
    uint32_t root5 = uf.Find(5);
    uint32_t root7 = uf.Find(7);
    EXPECT_EQ(root5, root7); // same root
}

TEST(UnionFind, ChainFind) {
    deglib::builder::UnionFind uf(10);
    uf.Update(1, 1);
    uf.Update(2, 2);
    uf.Update(3, 3);

    uf.Union(1, 2); // 1 → 2 or 2 → 1
    uf.Union(2, 3); // chain: 1-2-3

    uint32_t root1 = uf.Find(1);
    uint32_t root2 = uf.Find(2);
    uint32_t root3 = uf.Find(3);
    EXPECT_EQ(root1, root2);
    EXPECT_EQ(root2, root3);
}

TEST(UnionFind, UpdateDirectParent) {
    deglib::builder::UnionFind uf(10);
    uf.Update(1, 1);
    uf.Update(2, 2);
    uf.Update(1, 2); // directly set 1's parent to 2

    EXPECT_EQ(uf.Find(1), 2u);
    EXPECT_EQ(uf.Find(2), 2u);
}

TEST(UnionFind, DisjointSets) {
    deglib::builder::UnionFind uf(10);
    uf.Update(1, 1);
    uf.Update(2, 2);
    uf.Update(5, 5);
    uf.Update(6, 6);

    uf.Union(1, 2);
    uf.Union(5, 6);

    EXPECT_EQ(uf.Find(1), uf.Find(2));
    EXPECT_EQ(uf.Find(5), uf.Find(6));
    EXPECT_NE(uf.Find(1), uf.Find(5)); // different sets
}

TEST(UnionFind, LargeIds) {
    deglib::builder::UnionFind uf(1000);
    uf.Update(999, 999);
    uf.Update(500, 500);
    uf.Union(999, 500);

    EXPECT_EQ(uf.Find(999), uf.Find(500));
    EXPECT_NE(uf.Find(1), uf.Find(500)); // 1 is unknown
}

// ---------------------------------------------------------------------------
//  GraphEdge
// ---------------------------------------------------------------------------

TEST(GraphEdge, Ctor) {
    deglib::builder::GraphEdge e(1, 2, 3.5f);
    EXPECT_EQ(e.from_vertex, 1u);
    EXPECT_EQ(e.to_vertex, 2u);
    EXPECT_NEAR(e.weight, 3.5f, 1e-6f);
}

// ---------------------------------------------------------------------------
//  BuilderAddTask / BuilderRemoveTask / BuilderChange
// ---------------------------------------------------------------------------

TEST(BuilderAddTask, Ctor) {
    std::vector<std::byte> feat(4);
    deglib::builder::BuilderAddTask task(42, 100, std::move(feat));
    EXPECT_EQ(task.label, 42u);
    EXPECT_EQ(task.manipulation_index, 100u);
    EXPECT_EQ(task.feature.size(), 4u);
}

TEST(BuilderRemoveTask, Ctor) {
    deglib::builder::BuilderRemoveTask task(42, 100);
    EXPECT_EQ(task.label, 42u);
    EXPECT_EQ(task.manipulation_index, 100u);
}

TEST(BuilderChange, Ctor) {
    deglib::builder::BuilderChange change(5, 3, 1.5f, 7, 2.5f);
    EXPECT_EQ(change.internal_index, 5u);
    EXPECT_EQ(change.from_neighbor_index, 3u);
    EXPECT_NEAR(change.from_neighbor_weight, 1.5f, 1e-6f);
    EXPECT_EQ(change.to_neighbor_index, 7u);
    EXPECT_NEAR(change.to_neighbor_weight, 2.5f, 1e-6f);
}

// ---------------------------------------------------------------------------
//  BuilderStatus
// ---------------------------------------------------------------------------

TEST(BuilderStatus, DefaultValues) {
    deglib::builder::BuilderStatus status{};
    EXPECT_EQ(status.step, 0u);
    EXPECT_EQ(status.added, 0u);
    EXPECT_EQ(status.deleted, 0u);
    EXPECT_EQ(status.improved, 0u);
    EXPECT_EQ(status.tries, 0u);
}

// ---------------------------------------------------------------------------
//  ReachableGroup
// ---------------------------------------------------------------------------

TEST(ReachableGroup, Ctor) {
    deglib::builder::ReachableGroup rg(5, 10);
    EXPECT_EQ(rg.getVertexIndex(), 5u);
    EXPECT_EQ(rg.size(), 1u); // starts with self
    EXPECT_EQ(rg.getMissingEdgeSize(), 1u); // self-loop missing
}

TEST(ReachableGroup, HasEdgeRemovesFromMissing) {
    deglib::builder::ReachableGroup rg(5, 10);
    EXPECT_EQ(rg.getMissingEdgeSize(), 1u);

    rg.hasEdge(5); // self is now connected
    EXPECT_EQ(rg.getMissingEdgeSize(), 0u);
    EXPECT_EQ(rg.size(), 1u);
}

TEST(ReachableGroup, CopyFrom) {
    deglib::builder::ReachableGroup rg1(1, 10);
    deglib::builder::ReachableGroup rg2(2, 10);

    rg1.missing_edges_.insert(3);
    rg1.reachable_vertices_.insert(4);
    rg2.missing_edges_.insert(5);
    rg2.reachable_vertices_.insert(6);

    rg1.copyFrom(rg2);

    EXPECT_TRUE(rg1.missing_edges_.contains(3));
    EXPECT_TRUE(rg1.missing_edges_.contains(5));
    EXPECT_TRUE(rg1.reachable_vertices_.contains(4));
    EXPECT_TRUE(rg1.reachable_vertices_.contains(6));
}

TEST(ReachableGroup, CopyFromSelf) {
    deglib::builder::ReachableGroup rg(1, 10);
    rg.missing_edges_.insert(99);
    auto before = rg.missing_edges_.size();
    rg.copyFrom(rg); // should be a no-op
    EXPECT_EQ(rg.missing_edges_.size(), before);
}
