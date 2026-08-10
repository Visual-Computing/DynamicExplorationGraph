// test_pruning.cpp — Unit tests for deglib::optimization::pruning methods
//
// Covers: prune_worst_edges, remove_non_mrng_edges, remove_non_mrng_edges_weight_sorted,
//         remove_non_mrng_edges_iterative

#include <cmath>
#include <cstdint>
#include <vector>

#include "deglib/optimization/pruning.h"
#include "deglib/graph/sizebounded_graph.h"
#include "deglib/analysis.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

static std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
    return bytes;
}

static std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<uint8_t>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(uint8_t));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(uint8_t));
    return bytes;
}

static std::unique_ptr<std::byte[]> make_vec_bytes(float x, float y, float z, float w) {
    std::vector<float> v = {x, y, z, w};
    return make_float_bytes(v);
}

// Create a simple 5-vertex graph with 4 edges per vertex
static deglib::graph::SizeBoundedGraph create_test_graph() {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    // Add 5 vertices at distinct positions
    graph.addVertex(0, make_vec_bytes(0.0f, 0.0f, 0.0f, 0.0f).get());
    graph.addVertex(1, make_vec_bytes(1.0f, 0.0f, 0.0f, 0.0f).get());
    graph.addVertex(2, make_vec_bytes(2.0f, 0.0f, 0.0f, 0.0f).get());
    graph.addVertex(3, make_vec_bytes(3.0f, 0.0f, 0.0f, 0.0f).get());
    graph.addVertex(4, make_vec_bytes(4.0f, 0.0f, 0.0f, 0.0f).get());

    // Set up edges: vertex 0 connects to 1,2,3,4 with weights 1,4,9,16
    graph.changeEdge(0, 0, 1, 1.0f);
    graph.changeEdge(0, 0, 2, 4.0f);
    graph.changeEdge(0, 0, 3, 9.0f);
    graph.changeEdge(0, 0, 4, 16.0f);

    // Vertex 1 connects to 0,2,3,4
    graph.changeEdge(1, 1, 0, 1.0f);
    graph.changeEdge(1, 1, 2, 1.0f);
    graph.changeEdge(1, 1, 3, 4.0f);
    graph.changeEdge(1, 1, 4, 9.0f);

    // Vertex 2 connects to 0,1,3,4
    graph.changeEdge(2, 2, 0, 4.0f);
    graph.changeEdge(2, 2, 1, 1.0f);
    graph.changeEdge(2, 2, 3, 1.0f);
    graph.changeEdge(2, 2, 4, 4.0f);

    // Vertex 3 connects to 0,1,2,4
    graph.changeEdge(3, 3, 0, 9.0f);
    graph.changeEdge(3, 3, 1, 4.0f);
    graph.changeEdge(3, 3, 2, 1.0f);
    graph.changeEdge(3, 3, 4, 1.0f);

    // Vertex 4 connects to 0,1,2,3
    graph.changeEdge(4, 4, 0, 16.0f);
    graph.changeEdge(4, 4, 1, 9.0f);
    graph.changeEdge(4, 4, 2, 4.0f);
    graph.changeEdge(4, 4, 3, 1.0f);

    return graph;
}

// ---------------------------------------------------------------------------
//  prune_worst_edges
// ---------------------------------------------------------------------------

TEST(PruningTest, PruneWorstEdgesRemovesHighestWeight) {
    auto graph = create_test_graph();
    const auto edge_per_vertex = graph.getEdgesPerVertex();

    // Vertex 0 has neighbors 1(1.0), 2(4.0), 3(9.0), 4(16.0)
    // After pruning worst 1, neighbor 4 (weight 16.0) should become a self-loop
    deglib::optimization::pruning::prune_worst_edges(graph, 1, 1);

    const auto* indices = graph.getNeighborIndices(0);
    const auto* weights = graph.getNeighborWeights(0);

    // Self-loops have weight 0.0 and index == vertex_index
    bool found_self_loop = false;
    for (uint32_t n = 0; n < edge_per_vertex; n++) {
        if (indices[n] == 0 && weights[n] == 0.0f) {
            found_self_loop = true;
        }
    }
    EXPECT_TRUE(found_self_loop);
}

TEST(PruningTest, PruneWorstEdgesZeroDoesNothing) {
    auto graph = create_test_graph();

    // Pruning 0 worst edges should not change anything
    deglib::optimization::pruning::prune_worst_edges(graph, 0, 1);

    // Vertex 0 should still have its original edges
    const auto* indices = graph.getNeighborIndices(0);
    const auto* weights = graph.getNeighborWeights(0);
    const auto edge_per_vertex = graph.getEdgesPerVertex();

    // Should not have any self-loops (all neighbors are distinct from vertex 0)
    for (uint32_t n = 0; n < edge_per_vertex; n++) {
        EXPECT_NE(indices[n], 0u);
    }
}

TEST(PruningTest, PruneWorstEdgesAllBecomesSelfLoops) {
    auto graph = create_test_graph();
    const auto edge_per_vertex = graph.getEdgesPerVertex();

    // Prune all edges -> all become self-loops
    deglib::optimization::pruning::prune_worst_edges(graph, edge_per_vertex, 1);

    const auto* indices = graph.getNeighborIndices(0);
    const auto* weights = graph.getNeighborWeights(0);

    for (uint32_t n = 0; n < edge_per_vertex; n++) {
        EXPECT_EQ(indices[n], 0u);
        EXPECT_EQ(weights[n], 0.0f);
    }
}

// ---------------------------------------------------------------------------
//  remove_non_mrng_edges
// ---------------------------------------------------------------------------

TEST(PruningTest, RemoveNonMrngEdgesReducesNonRngCount) {
    auto graph = create_test_graph();

    uint32_t before = deglib::analysis::calc_non_rng_edges(graph);
    uint32_t removed = deglib::optimization::pruning::remove_non_mrng_edges(graph, 1);
    uint32_t after = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_GT(before, 0u);
    EXPECT_GT(removed, 0u);
    EXPECT_LT(after, before);
}

TEST(PruningTest, RemoveNonMrngEdgesIdempotent) {
    auto graph = create_test_graph();

    // First removal
    deglib::optimization::pruning::remove_non_mrng_edges(graph, 1);
    uint32_t after_first = deglib::analysis::calc_non_rng_edges(graph);

    // Second removal should not remove anything (already MRNG)
    uint32_t removed_second = deglib::optimization::pruning::remove_non_mrng_edges(graph, 1);
    uint32_t after_second = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_EQ(removed_second, 0u);
    EXPECT_EQ(after_first, after_second);
}

// ---------------------------------------------------------------------------
//  remove_non_mrng_edges_weight_sorted
// ---------------------------------------------------------------------------

TEST(PruningTest, RemoveNonMrngEdgesWeightSortedReducesNonRngCount) {
    auto graph = create_test_graph();

    uint32_t before = deglib::analysis::calc_non_rng_edges(graph);
    uint32_t removed = deglib::optimization::pruning::remove_non_mrng_edges_weight_sorted(graph, 1);
    uint32_t after = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_GT(before, 0u);
    EXPECT_GT(removed, 0u);
    EXPECT_LT(after, before);
}

TEST(PruningTest, RemoveNonMrngEdgesWeightSortedIdempotent) {
    auto graph = create_test_graph();

    deglib::optimization::pruning::remove_non_mrng_edges_weight_sorted(graph, 1);
    uint32_t after_first = deglib::analysis::calc_non_rng_edges(graph);

    uint32_t removed_second = deglib::optimization::pruning::remove_non_mrng_edges_weight_sorted(graph, 1);
    uint32_t after_second = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_EQ(removed_second, 0u);
    EXPECT_EQ(after_first, after_second);
}

// ---------------------------------------------------------------------------
//  remove_non_mrng_edges_iterative
// ---------------------------------------------------------------------------

TEST(PruningTest, RemoveNonMrngEdgesIterativeReducesNonRngCount) {
    auto graph = create_test_graph();

    uint32_t before = deglib::analysis::calc_non_rng_edges(graph);
    uint32_t removed = deglib::optimization::pruning::remove_non_mrng_edges_iterative(graph, 1);
    uint32_t after = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_GT(before, 0u);
    EXPECT_GT(removed, 0u);
    EXPECT_LT(after, before);
}

TEST(PruningTest, RemoveNonMrngEdgesIterativeIdempotent) {
    auto graph = create_test_graph();

    deglib::optimization::pruning::remove_non_mrng_edges_iterative(graph, 1);
    uint32_t after_first = deglib::analysis::calc_non_rng_edges(graph);

    uint32_t removed_second = deglib::optimization::pruning::remove_non_mrng_edges_iterative(graph, 1);
    uint32_t after_second = deglib::analysis::calc_non_rng_edges(graph);

    EXPECT_EQ(removed_second, 0u);
    EXPECT_EQ(after_first, after_second);
}

// ---------------------------------------------------------------------------
//  Cross-method consistency
// ---------------------------------------------------------------------------

TEST(PruningTest, AllMethodsProduceMrngConformGraph) {
    auto graph1 = create_test_graph();
    auto graph2 = create_test_graph();
    auto graph3 = create_test_graph();

    uint32_t removed1 = deglib::optimization::pruning::remove_non_mrng_edges(graph1, 1);
    uint32_t removed2 = deglib::optimization::pruning::remove_non_mrng_edges_weight_sorted(graph2, 1);
    uint32_t removed3 = deglib::optimization::pruning::remove_non_mrng_edges_iterative(graph3, 1);

    // All methods should reduce non-RNG edges to zero (MRNG-conform graph)
    EXPECT_EQ(deglib::analysis::calc_non_rng_edges(graph1), 0u);
    EXPECT_EQ(deglib::analysis::calc_non_rng_edges(graph2), 0u);
    EXPECT_EQ(deglib::analysis::calc_non_rng_edges(graph3), 0u);

    // All methods should remove at least one edge (the test graph has non-RNG edges)
    EXPECT_GT(removed1, 0u);
    EXPECT_GT(removed2, 0u);
    EXPECT_GT(removed3, 0u);
}
