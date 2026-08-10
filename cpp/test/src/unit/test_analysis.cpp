// test_analysis.cpp — Unit tests for deglib::analysis reachability and graph analysis functions
//
// Tests calc_search_reachability, calc_exploration_reach, and analyze_graph
// using a MockInternalGraph with a small, deterministic graph topology.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "deglib/analysis.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

namespace {

// ---------------------------------------------------------------------------
//  Mock implementation of InternalGraph for testing analysis functions
// ---------------------------------------------------------------------------

class MockInternalGraph : public deglib::graph::InternalGraph {
  uint32_t size_;
  uint8_t edges_per_vertex_;
  deglib::distances::FloatSpace feature_space_;
  std::vector<uint32_t> labels_;
  std::vector<std::vector<uint32_t>> neighbors_;
  std::vector<std::vector<float>> weights_;
  std::vector<std::vector<std::byte>> features_;

public:
  MockInternalGraph(uint32_t size, uint8_t epv, deglib::distances::FloatSpace fs)
      : size_(size), edges_per_vertex_(epv), feature_space_(std::move(fs)),
        labels_(size), neighbors_(size, std::vector<uint32_t>(epv, (std::numeric_limits<uint32_t>::max)())),
        weights_(size, std::vector<float>(epv, 0.0f)),
        features_(size, std::vector<std::byte>(feature_space_.get_data_size())) {}

  const uint32_t size() const override { return size_; }
  const uint8_t getEdgesPerVertex() const override { return edges_per_vertex_; }
  const deglib::distances::FloatSpace& getFeatureSpace() const override { return feature_space_; }

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
                                          const deglib::search::Filter* = nullptr, const uint32_t = 0) const override {
      return deglib::graph::ResultSet();
  }

  // helpers for test setup
  void setLabel(uint32_t idx, uint32_t label) { labels_[idx] = label; }
  void setNeighbors(uint32_t idx, const std::vector<uint32_t>& n) { neighbors_[idx] = n; }
};

// Build a simple 4-vertex graph:
//  0 -> 1, 2, 3
//  1 -> 0, 2, 3
//  2 -> 0, 1, 3
//  3 -> 0, 1, 2
// This is a fully-connected graph where every vertex can reach every other vertex.
MockInternalGraph build_fully_connected_graph() {
  deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
  MockInternalGraph graph(4, 4, std::move(space));

  graph.setLabel(0, 0);
  graph.setLabel(1, 1);
  graph.setLabel(2, 2);
  graph.setLabel(3, 3);

  graph.setNeighbors(0, {1, 2, 3, (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(1, {0, 2, 3, (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(2, {0, 1, 3, (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(3, {0, 1, 2, (std::numeric_limits<uint32_t>::max)()});

  return graph;
}

// Build a 4-vertex graph with two disconnected components:
//  Component A: 0 -> 1, 1 -> 0
//  Component B: 2 -> 3, 3 -> 2
MockInternalGraph build_disconnected_graph() {
  deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
  MockInternalGraph graph(4, 4, std::move(space));

  graph.setLabel(0, 0);
  graph.setLabel(1, 1);
  graph.setLabel(2, 2);
  graph.setLabel(3, 3);

  graph.setNeighbors(0, {1, (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(1, {0, (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(2, {3, (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)()});
  graph.setNeighbors(3, {2, (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)(), (std::numeric_limits<uint32_t>::max)()});

  return graph;
}

} // anonymous namespace

// ===========================================================================
//  calc_search_reachability
// ===========================================================================

TEST(DegAnalysisSearchReachability, FullyConnectedGraph) {
  auto graph = build_fully_connected_graph();
  // Entry vertex is 0 (default from InternalGraph::getEntryVertexIndices)
  // From 0 we can reach 1, 2, 3 — all 4 vertices
  uint32_t reachable = deglib::analysis::calc_search_reachability(graph);
  EXPECT_EQ(reachable, 4u);
}

TEST(DegAnalysisSearchReachability, DisconnectedGraph) {
  auto graph = build_disconnected_graph();
  // Entry vertex is 0. From 0 we can only reach 0 and 1 (component A)
  uint32_t reachable = deglib::analysis::calc_search_reachability(graph);
  EXPECT_EQ(reachable, 2u);
}

// ===========================================================================
//  calc_exploration_reach
// ===========================================================================

TEST(DegAnalysisExplorationReach, FullyConnectedGraph) {
  auto graph = build_fully_connected_graph();
  // Every vertex can reach all 4 vertices, so average reach = 4
  float avg_reach = deglib::analysis::calc_exploration_reach(graph);
  EXPECT_FLOAT_EQ(avg_reach, 4.0f);
}

TEST(DegAnalysisExplorationReach, DisconnectedGraph) {
  auto graph = build_disconnected_graph();
  // Each vertex can reach 2 vertices (itself + one neighbor)
  float avg_reach = deglib::analysis::calc_exploration_reach(graph);
  EXPECT_FLOAT_EQ(avg_reach, 2.0f);
}

// ===========================================================================
//  analyze_graph
// ===========================================================================

TEST(DegAnalysisAnalyzeGraph, FullyConnectedGraph) {
  auto graph = build_fully_connected_graph();
  auto stats = deglib::analysis::analyze_graph(graph);

  EXPECT_EQ(stats.vertex_count, 4u);
  EXPECT_EQ(stats.feature_dims, 4u);
  EXPECT_EQ(stats.edges_per_vertex, 4u);
  // 3 valid edges per vertex * 4 vertices = 12 total edges
  EXPECT_EQ(stats.edge_count, 12u);
  // avg out-degree = 12 / 4 = 3.0
  EXPECT_FLOAT_EQ(stats.avg_out_degree, 3.0f);
  EXPECT_EQ(stats.min_out_degree, 3u);
  EXPECT_EQ(stats.max_out_degree, 3u);
  // In-degree: each vertex is reachable from 3 others
  EXPECT_FLOAT_EQ(stats.avg_in_degree, 3.0f);
  EXPECT_EQ(stats.min_in_degree, 3u);
  EXPECT_EQ(stats.max_in_degree, 3u);
  // No source vertices (all have in-degree > 0)
  EXPECT_EQ(stats.source_vertices, 0u);
  // Search reachability: all 4 vertices reachable from entry
  EXPECT_FLOAT_EQ(stats.search_reachability, 1.0f);
  // Exploration reachability: each vertex reaches 4, avg = 4, normalized = 4/4 = 1.0
  EXPECT_FLOAT_EQ(stats.exploration_reachability, 1.0f);
  // Memory: 4 * (4*4 + 4*4 + 4*4) = 4 * 48 = 192
  EXPECT_EQ(stats.memory_bytes, 192u);
}

TEST(DegAnalysisAnalyzeGraph, DisconnectedGraph) {
  auto graph = build_disconnected_graph();
  auto stats = deglib::analysis::analyze_graph(graph);

  EXPECT_EQ(stats.vertex_count, 4u);
  EXPECT_EQ(stats.edge_count, 4u);  // 1 valid edge per vertex * 4 vertices
  EXPECT_FLOAT_EQ(stats.avg_out_degree, 1.0f);
  EXPECT_EQ(stats.min_out_degree, 1u);
  EXPECT_EQ(stats.max_out_degree, 1u);
  // In-degree: each vertex has exactly 1 incoming edge
  EXPECT_FLOAT_EQ(stats.avg_in_degree, 1.0f);
  EXPECT_EQ(stats.min_in_degree, 1u);
  EXPECT_EQ(stats.max_in_degree, 1u);
  // No source vertices
  EXPECT_EQ(stats.source_vertices, 0u);
  // Search reachability: only 2 of 4 reachable from entry vertex 0
  EXPECT_FLOAT_EQ(stats.search_reachability, 0.5f);
  // Exploration reachability: each vertex reaches 2, avg = 2, normalized = 2/4 = 0.5
  EXPECT_FLOAT_EQ(stats.exploration_reachability, 0.5f);
}

TEST(DegAnalysisAnalyzeGraph, EmptyGraph) {
  deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
  MockInternalGraph graph(0, 4, std::move(space));
  auto stats = deglib::analysis::analyze_graph(graph);

  EXPECT_EQ(stats.vertex_count, 0u);
  EXPECT_EQ(stats.edge_count, 0u);
  EXPECT_FLOAT_EQ(stats.avg_out_degree, 0.0f);
  EXPECT_FLOAT_EQ(stats.avg_in_degree, 0.0f);
  EXPECT_FLOAT_EQ(stats.search_reachability, 0.0f);
  EXPECT_FLOAT_EQ(stats.exploration_reachability, 0.0f);
}
