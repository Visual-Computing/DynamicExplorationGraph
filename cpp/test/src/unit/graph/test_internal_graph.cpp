// test_internal_graph.cpp — Unit tests for deglib::graph::InternalGraph and related types
//
// Tests ObjectDistance, ResultSet, UncheckedSet (defined in internal_graph.h),
// and the InternalGraph abstract interface contract via a mock implementation.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "deglib/graph/internal_graph.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  ObjectDistance
// ---------------------------------------------------------------------------

TEST(ObjectDistance, DefaultCtor) {
   deglib::graph::ObjectDistance od;
   // default ctor leaves members uninitialized — just check it compiles
}

TEST(ObjectDistance, Ctor) {
   deglib::graph::ObjectDistance od(42, 3.14f);
   EXPECT_EQ(od.getInternalIndex(), 42u);
   EXPECT_NEAR(od.getDistance(), 3.14f, 1e-6f);
}

TEST(ObjectDistance, Equality) {
   deglib::graph::ObjectDistance a(1, 2.0f);
   deglib::graph::ObjectDistance b(1, 2.0f);
   deglib::graph::ObjectDistance c(2, 2.0f);
   deglib::graph::ObjectDistance d(1, 3.0f);

   EXPECT_TRUE(a == b);
   EXPECT_FALSE(a == c);
   EXPECT_FALSE(a == d);
}

TEST(ObjectDistance, LessThanByDistance) {
   deglib::graph::ObjectDistance a(10, 1.0f);
   deglib::graph::ObjectDistance b(5, 2.0f);
   EXPECT_TRUE(a < b);
   EXPECT_FALSE(b < a);
}

TEST(ObjectDistance, LessThanByIndexWhenDistanceEqual) {
   deglib::graph::ObjectDistance a(5, 1.0f);
   deglib::graph::ObjectDistance b(3, 1.0f);
   // a < b should be false (same distance, a.index > b.index)
   EXPECT_FALSE(a < b);
   // b < a should be true (same distance, b.index < a.index)
   EXPECT_TRUE(b < a);
}

TEST(ObjectDistance, GreaterThan) {
   deglib::graph::ObjectDistance a(10, 5.0f);
   deglib::graph::ObjectDistance b(5, 2.0f);
   EXPECT_TRUE(a > b);
   EXPECT_FALSE(b > a);
}

// ---------------------------------------------------------------------------
//  ResultSet (PQV with std::less — max-heap by distance)
// ---------------------------------------------------------------------------

TEST(ResultSet, Empty) {
   deglib::graph::ResultSet rs;
   EXPECT_TRUE(rs.empty());
   EXPECT_EQ(rs.size(), 0u);
}

TEST(ResultSet, EmplaceAndTop) {
   deglib::graph::ResultSet rs;
   rs.emplace(5, 1.0f);
   rs.emplace(3, 2.0f);
   rs.emplace(7, 0.5f);

   // ResultSet uses std::less<ObjectDistance> → max-heap
   // Top should be the element with the largest distance
   auto top = rs.top();
   EXPECT_NEAR(top.getDistance(), 2.0f, 1e-6f);
   EXPECT_EQ(top.getInternalIndex(), 3u);
}

TEST(ResultSet, PopRemovesMax) {
   deglib::graph::ResultSet rs;
   rs.emplace(1, 1.0f);
   rs.emplace(2, 3.0f);
   rs.emplace(3, 2.0f);

   rs.pop(); // remove max (dist=3)
   EXPECT_NEAR(rs.top().getDistance(), 2.0f, 1e-6f);

   rs.pop(); // remove max (dist=2)
   EXPECT_NEAR(rs.top().getDistance(), 1.0f, 1e-6f);
   rs.pop(); // remove last (dist=1)
   EXPECT_TRUE(rs.empty());
}

TEST(ResultSet, Reserve) {
   deglib::graph::ResultSet rs;
   rs.reserve(100);
   for (int i = 0; i < 50; ++i) {
       rs.emplace(static_cast<uint32_t>(i), static_cast<float>(i));
   }
   EXPECT_EQ(rs.size(), 50u);
}

// ---------------------------------------------------------------------------
//  UncheckedSet (PQV with std::greater — min-heap by distance)
// ---------------------------------------------------------------------------

TEST(UncheckedSet, MinHeapOrder) {
   deglib::graph::UncheckedSet us;
   us.emplace(1, 5.0f);
   us.emplace(2, 1.0f);
   us.emplace(3, 3.0f);

   // UncheckedSet uses std::greater → min-heap
   auto top = us.top();
   EXPECT_NEAR(top.getDistance(), 1.0f, 1e-6f);
   EXPECT_EQ(top.getInternalIndex(), 2u);
}

TEST(UncheckedSet, PopRemovesMin) {
   deglib::graph::UncheckedSet us;
   us.emplace(1, 5.0f);
   us.emplace(2, 1.0f);
   us.emplace(3, 3.0f);

   us.pop(); // remove min (dist=1)
   EXPECT_NEAR(us.top().getDistance(), 3.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
//  Mock implementation of InternalGraph for testing the interface
// ---------------------------------------------------------------------------

class MockInternalGraph : public deglib::graph::InternalGraph {
   uint32_t size_;
   uint8_t edges_per_vertex_;
   deglib::FloatSpace feature_space_;
   std::vector<uint32_t> labels_;
   std::vector<std::vector<uint32_t>> neighbors_;
   std::vector<std::vector<float>> weights_;
   std::vector<std::vector<std::byte>> features_;

public:
   MockInternalGraph(uint32_t size, uint8_t epv, deglib::FloatSpace fs)
       : size_(size), edges_per_vertex_(epv), feature_space_(std::move(fs)),
         labels_(size), neighbors_(size, std::vector<uint32_t>(epv)),
         weights_(size, std::vector<float>(epv, 0.0f)),
         features_(size, std::vector<std::byte>(feature_space_.get_data_size())) {}

   const uint32_t size() const override { return size_; }
   const uint8_t getEdgesPerVertex() const override { return edges_per_vertex_; }
   const deglib::FloatSpace& getFeatureSpace() const override { return feature_space_; }

   const uint32_t getExternalLabel(uint32_t idx) const override { return labels_[idx]; }
   const uint32_t getInternalIndex(uint32_t label) const override {
       for (uint32_t i = 0; i < size_; ++i)
           if (labels_[i] == label) return i;
       return 0;
   }
   const uint32_t* getNeighborIndices(uint32_t idx) const override { return neighbors_[idx].data(); }
   const std::byte* getFeatureVector(uint32_t idx) const override { return features_[idx].data(); }

   const bool hasVertex(uint32_t label) const override {
       for (uint32_t i = 0; i < size_; ++i)
           if (labels_[i] == label) return true;
       return false;
   }
   const bool hasEdge(uint32_t idx, uint32_t neighbor) const override {
       return std::binary_search(neighbors_[idx].begin(), neighbors_[idx].end(), neighbor);
   }

   std::vector<deglib::graph::ObjectDistance> hasPath(const std::vector<uint32_t>&, uint32_t, float, uint32_t) const override {
       return {};
   }
   deglib::graph::ResultSet search_intern(const std::vector<uint32_t>&, const std::byte*,
                                          const uint32_t, const float = 0.0f, const bool = true,
                                          const deglib::graph::Filter* = nullptr, const uint32_t = 0) const override {
       return deglib::graph::ResultSet();
   }

   // helpers for test setup
   void setLabel(uint32_t idx, uint32_t label) { labels_[idx] = label; }
   void setNeighbors(uint32_t idx, const std::vector<uint32_t>& n) { neighbors_[idx] = n; }
   void setFeature(uint32_t idx, const std::vector<std::byte>& f) { features_[idx] = f; }
};

// ---------------------------------------------------------------------------
//  InternalGraph interface contract tests
// ---------------------------------------------------------------------------

TEST(InternalGraph, InterfaceContract) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   MockInternalGraph graph(3, 2, std::move(fs));

   graph.setLabel(0, 10);
   graph.setLabel(1, 20);
   graph.setLabel(2, 30);

   EXPECT_EQ(graph.size(), 3u);
   EXPECT_EQ(graph.getEdgesPerVertex(), 2u);
   EXPECT_EQ(graph.getFeatureSpace().dim(), 4u);
}

TEST(InternalGraph, HasVertex) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   MockInternalGraph graph(3, 2, std::move(fs));
   graph.setLabel(0, 10);
   graph.setLabel(1, 20);
   graph.setLabel(2, 30);

   EXPECT_TRUE(graph.hasVertex(10));
   EXPECT_TRUE(graph.hasVertex(30));
   EXPECT_FALSE(graph.hasVertex(99));
}

TEST(InternalGraph, GetInternalIndex) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   MockInternalGraph graph(3, 2, std::move(fs));
   graph.setLabel(0, 10);
   graph.setLabel(1, 20);
   graph.setLabel(2, 30);

   EXPECT_EQ(graph.getInternalIndex(10), 0u);
   EXPECT_EQ(graph.getInternalIndex(20), 1u);
   EXPECT_EQ(graph.getInternalIndex(30), 2u);
}

TEST(InternalGraph, HasEdge) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   MockInternalGraph graph(3, 2, std::move(fs));
   graph.setLabel(0, 10);
   graph.setLabel(1, 20);
   graph.setLabel(2, 30);
   graph.setNeighbors(0, {1, 2});
   graph.setNeighbors(1, {0, 2});
   graph.setNeighbors(2, {0, 1});

   EXPECT_TRUE(graph.hasEdge(0, 1));
   EXPECT_TRUE(graph.hasEdge(0, 2));
   EXPECT_FALSE(graph.hasEdge(0, 0));
}

TEST(InternalGraph, GetEntryVertexIndices) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   MockInternalGraph graph(3, 2, std::move(fs));
   auto entries = graph.getEntryVertexIndices();
   EXPECT_EQ(entries.size(), 1u);
   EXPECT_EQ(entries[0], 0u);
}
