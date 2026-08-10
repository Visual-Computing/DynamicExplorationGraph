// test_graph.cpp — Unit tests for DynamicExplorationGraph facade
//
// Verifies that DynamicExplorationGraph strictly uses external labels (User Object IDs)
// for all inputs and outputs, while the underlying InternalGraph implementations
// operate strictly on internal indices (0..N-1).
//
// Non-sequential external labels (1005, 9999, 42, 707, 12345) are used so that
// any accidental swap of external labels and internal indices will instantly
// fail test assertions.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "deglib/graph.h"
#include "deglib/graph/mutable_graph.h"
#include "deglib/graph/readonly_graph.h"
#include "gtest/gtest.h"

namespace {

// Non-sequential, arbitrary external labels — deliberately distinct from 0..N-1
constexpr std::array<uint32_t, 5> kExternalLabels = {1005, 9999, 42, 707, 12345};

// Internal indices are always 0..N-1
constexpr std::array<uint32_t, 5> kInternalIndices = {0, 1, 2, 3, 4};

inline std::vector<float> make_vec_4d(float x, float y, float z, float w) {
   return {x, y, z, w};
}

inline std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
   auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
   std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
   return bytes;
}

// Helper: set up edges for a MutableGraph using changeEdges (bypasses self-loop requirement)
void set_edges(deglib::graph::MutableGraph& graph, uint32_t vertex,
              const std::vector<uint32_t>& neighbors, const std::vector<float>& weights) {
   graph.changeEdges(vertex, neighbors.data(), weights.data());
}

// Build a fully-connected mutable graph with non-sequential external labels.
// Vertices are placed at distinct positions so search/explore return deterministic results.
// Internal indices: 0..4, External labels: 1005, 9999, 42, 707, 12345
deglib::graph::SizeBoundedGraph build_test_graph() {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   // Place vertices at distinct 4D positions
   graph.addVertex(kExternalLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExternalLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExternalLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExternalLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExternalLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   // Fully connect: each vertex connects to all others (neighbors must be sorted ascending)
   // Internal index 0 connects to 0(self), 1, 2, 3
   set_edges(graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
   // Internal index 1 connects to 0, 1(self), 2, 3
   set_edges(graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
   // Internal index 2 connects to 0, 1, 2(self), 3
   set_edges(graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
   // Internal index 3 connects to 0, 1, 2, 3(self)
   set_edges(graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});
   // Internal index 4 has self-loop only (not connected to others)
   set_edges(graph, 4, {4, 4, 4, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

   return graph;
}

} // anonymous namespace

// ===========================================================================
//  DynamicExplorationGraph: search() returns external labels
// ===========================================================================

TEST(DEGSearchReturnsExternalLabels, SearchReturnsExternalLabels) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Query near vertex 0 (external label 1005, position 0.0)
   std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
   auto results = deg.search(std::span<const float>(query), 3, 0.0f);

   ASSERT_GT(results.size(), 0u);

   // Every identifier in the result must be an external label, not an internal index
   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       // Must be a valid external label
       EXPECT_TRUE(valid_labels.contains(id))
           << "Result identifier " << id << " is not a valid external label";
       // Must NOT be an internal index (0..4)
       EXPECT_FALSE(internal_indices.contains(id))
           << "Result identifier " << id << " is an internal index, not an external label";
       results.pop();
   }
}

TEST(DEGSearchReturnsExternalLabels, SearchFindsCorrectExternalLabel) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Query exactly at vertex 0's position (external label 1005)
   std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
   auto results = deg.search(std::span<const float>(query), 1, 0.0f);

   ASSERT_GE(results.size(), 1u);
   // The nearest neighbor should be vertex 0 with external label 1005
   EXPECT_EQ(results.top().getIdentifier(), kExternalLabels[0]);
   EXPECT_EQ(results.top().getIdentifier(), 1005u);
}

// ===========================================================================
//  DynamicExplorationGraph: explore() accepts external label, returns external labels
// ===========================================================================

TEST(DEGExploreReturnsExternalLabels, ExploreAcceptsExternalLabelAndReturnsExternalLabels) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Explore starting from external label 1005 (internal index 0)
   auto results = deg.explore(kExternalLabels[0], 3, 0, 0.0f, /*include_entry=*/true, nullptr);

   ASSERT_GT(results.size(), 0u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(valid_labels.contains(id))
           << "Result identifier " << id << " is not a valid external label";
       EXPECT_FALSE(internal_indices.contains(id))
           << "Result identifier " << id << " is an internal index, not an external label";
       results.pop();
   }
}

TEST(DEGExploreReturnsExternalLabels, ExploreFromDifferentExternalLabel) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Explore from external label 9999 (internal index 1, position 1.0)
   auto results = deg.explore(kExternalLabels[1], 2, 0, 0.0f, /*include_entry=*/false, nullptr);

   ASSERT_GT(results.size(), 0u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(valid_labels.contains(id))
           << "Result identifier " << id << " is not a valid external label";
       EXPECT_FALSE(internal_indices.contains(id))
           << "Result identifier " << id << " is an internal index, not an external label";
       results.pop();
   }
}

TEST(DEGExploreReturnsExternalLabels, ExploreIncludeEntryReturnsExternalLabel) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // When include_entry=true, the entry vertex's external label must appear in results
   auto results = deg.explore(kExternalLabels[2], 4, 0, 0.0f, /*include_entry=*/true, nullptr);

   ASSERT_GT(results.size(), 0u);

   bool found_entry = false;
   while (!results.empty()) {
       if (results.top().getIdentifier() == kExternalLabels[2]) {
           found_entry = true;
       }
       results.pop();
   }
   EXPECT_TRUE(found_entry) << "Entry vertex external label " << kExternalLabels[2]
                            << " should be in results when include_entry=true";
}

// ===========================================================================
//  DynamicExplorationGraph: getNeighbors() accepts external label, returns external labels
// ===========================================================================

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsAcceptsExternalLabel) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Get neighbors of vertex with external label 1005 (internal index 0)
   // Internal index 0 has neighbors {0, 1, 2, 3} → external labels {1005, 9999, 42, 707}
   auto neighbors = deg.getNeighbors(kExternalLabels[0]);

   ASSERT_EQ(neighbors.size(), 4u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   for (uint32_t n : neighbors) {
       EXPECT_TRUE(valid_labels.contains(n))
           << "Neighbor " << n << " is not a valid external label";
       EXPECT_FALSE(internal_indices.contains(n))
           << "Neighbor " << n << " is an internal index, not an external label";
   }
}

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsReturnsCorrectExternalLabels) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Vertex 0 (external label 1005) has internal neighbors {0, 1, 2, 3}
   // which map to external labels {1005, 9999, 42, 707}
   auto neighbors = deg.getNeighbors(kExternalLabels[0]);

   ASSERT_EQ(neighbors.size(), 4u);

   std::unordered_set<uint32_t> expected = {1005, 9999, 42, 707};
   std::unordered_set<uint32_t> actual(neighbors.begin(), neighbors.end());

   EXPECT_EQ(actual, expected);
}

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsFromDifferentVertex) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Vertex 1 (external label 9999) has internal neighbors {0, 1, 2, 3}
   // which map to external labels {1005, 9999, 42, 707}
   auto neighbors = deg.getNeighbors(kExternalLabels[1]);

   ASSERT_EQ(neighbors.size(), 4u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   for (uint32_t n : neighbors) {
       EXPECT_TRUE(valid_labels.contains(n))
           << "Neighbor " << n << " is not a valid external label";
   }
}

// ===========================================================================
//  DynamicExplorationGraph: hasVertex() uses external labels
// ===========================================================================

TEST(DEGHasVertexUsesExternalLabels, HasVertexTrueForValidExternalLabels) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   for (uint32_t label : kExternalLabels) {
       EXPECT_TRUE(deg.hasVertex(label))
           << "hasVertex(" << label << ") should return true for valid external label";
   }
}

TEST(DEGHasVertexUsesExternalLabels, HasVertexFalseForInvalidExternalLabels) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Labels that are not in the graph
   EXPECT_FALSE(deg.hasVertex(0u));
   EXPECT_FALSE(deg.hasVertex(1u));
   EXPECT_FALSE(deg.hasVertex(999u));
   EXPECT_FALSE(deg.hasVertex(12346u));
}

TEST(DEGHasVertexUsesExternalLabels, HasVertexFalseForInternalIndices) {
   auto graph = build_test_graph();
   deglib::DynamicExplorationGraph deg(graph);

   // Internal indices 0..4 should NOT be valid external labels
   for (uint32_t idx : kInternalIndices) {
       EXPECT_FALSE(deg.hasVertex(idx))
           << "hasVertex(" << idx << ") should return false — " << idx
           << " is an internal index, not an external label";
   }
}

// ===========================================================================
//  DynamicExplorationGraph: search() with ReadOnlyGraph backend
// ===========================================================================

TEST(DEGSearchReturnsExternalLabels, SearchWithReadOnlyGraphBackend) {
   auto size_bounded = build_test_graph();
   auto readonly = deglib::graph::convert_to_readonly_graph(size_bounded);
   deglib::DynamicExplorationGraph deg(readonly);

   // Query near vertex 0 (external label 1005)
   std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
   auto results = deg.search(std::span<const float>(query), 3, 0.0f);

   ASSERT_GT(results.size(), 0u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(valid_labels.contains(id))
           << "Result identifier " << id << " is not a valid external label";
       EXPECT_FALSE(internal_indices.contains(id))
           << "Result identifier " << id << " is an internal index, not an external label";
       results.pop();
   }
}

TEST(DEGExploreReturnsExternalLabels, ExploreWithReadOnlyGraphBackend) {
   auto size_bounded = build_test_graph();
   auto readonly = deglib::graph::convert_to_readonly_graph(size_bounded);
   deglib::DynamicExplorationGraph deg(readonly);

   // Explore from external label 42 (internal index 2)
   auto results = deg.explore(kExternalLabels[2], 3, 0, 0.0f, /*include_entry=*/true, nullptr);

   ASSERT_GT(results.size(), 0u);

   std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
   std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(valid_labels.contains(id))
           << "Result identifier " << id << " is not a valid external label";
       EXPECT_FALSE(internal_indices.contains(id))
           << "Result identifier " << id << " is an internal index, not an external label";
       results.pop();
   }
}
