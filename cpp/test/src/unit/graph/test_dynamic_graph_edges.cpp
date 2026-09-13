#include "test_dynamic_graph_common.h"

// ---------------------------------------------------------------------------
//  3. Edge Management
// ---------------------------------------------------------------------------

TEST(DynamicGraph, ChangeEdgeAndChangeEdges) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::DynamicGraph graph(4, space, /*chunk_size=*/8);

    for (uint32_t i = 0; i < 5; ++i) {
        auto v = make_vec_4d(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
        graph.addVertex(i, make_float_bytes(v).get());
    }

    // Initial state: self-loops
    const uint32_t* n0 = graph.getNeighborIndices(0);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(n0[i], 0);
    }

    // Change edges directly using sorted list
    uint32_t new_neighbors[4] = {1, 2, 3, 4};
    float new_weights[4] = {1.0f, 4.0f, 9.0f, 16.0f};
    graph.changeEdges(0, new_neighbors, new_weights);

    EXPECT_TRUE(graph.hasEdge(0, 1));
    EXPECT_TRUE(graph.hasEdge(0, 2));
    EXPECT_TRUE(graph.hasEdge(0, 3));
    EXPECT_TRUE(graph.hasEdge(0, 4));
    EXPECT_FALSE(graph.hasEdge(0, 0));
    EXPECT_NEAR(graph.getEdgeWeight(0, 3), 9.0f, 1e-4f);

    // Swap an edge
    EXPECT_TRUE(graph.changeEdge(0, 3, 0, 5.0f));
    EXPECT_TRUE(graph.hasEdge(0, 0));
    EXPECT_FALSE(graph.hasEdge(0, 3));
    EXPECT_NEAR(graph.getEdgeWeight(0, 0), 5.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
//  4. Compact Swap-with-Last Vertex Removal & Chunk Deallocation
// ---------------------------------------------------------------------------

TEST(DynamicGraph, RemoveVertexSwapWithLast) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    // 4 vertices per chunk
    deglib::graph::DynamicGraph graph(2, space, 4);

    // Add 5 vertices: 0..3 in Chunk 0, 4 in Chunk 1
    for (uint32_t i = 0; i < 5; ++i) {
        auto v = make_vec_4d(static_cast<float>(i * 10), 0.0f, 0.0f, 0.0f);
        graph.addVertex(100 + i, make_float_bytes(v).get());
    }

    EXPECT_EQ(graph.size(), 5u);
    EXPECT_EQ(graph.chunk_count(), 2u);

    // Set some edges between vertex 1 (internal 1, label 101) and vertex 4 (internal 4, label 104)
    uint32_t n1[2] = {0, 4};
    float w1[2] = {1.0f, 2.0f};
    graph.changeEdges(1, n1, w1);

    uint32_t n4[2] = {1, 4};
    float w4[2] = {2.0f, 0.0f};
    graph.changeEdges(4, n4, w4);

    EXPECT_TRUE(graph.hasEdge(1, 4));
    EXPECT_TRUE(graph.hasEdge(4, 1));

    // Remove vertex with label 101 (internal index 1).
    // Internal index 4 (label 104, last element) should move to internal index 1.
    auto involved = graph.removeVertex(101);
    EXPECT_EQ(graph.size(), 4u);
    EXPECT_FALSE(graph.hasVertex(101));
    EXPECT_TRUE(graph.hasVertex(104));

    // Chunk 1 had 1 element (index 4). After removal, size is 4, so Chunk 1 was freed!
    EXPECT_EQ(graph.chunk_count(), 1u);
    EXPECT_EQ(graph.capacity(), 4u);

    // Label 104 is now at internal index 1
    EXPECT_EQ(graph.getInternalIndex(104), 1u);
    EXPECT_EQ(graph.getExternalLabel(1), 104u);
    const float* f1 = reinterpret_cast<const float*>(graph.getFeatureVector(1));
    EXPECT_NEAR(f1[0], 40.0f, 1e-4f);  // feature of old vertex 4

    // Remove remaining elements down to 0
    graph.removeVertex(100);
    graph.removeVertex(102);
    graph.removeVertex(103);
    EXPECT_EQ(graph.size(), 1u);
    EXPECT_EQ(graph.chunk_count(), 1u);

    graph.removeVertex(104);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.chunk_count(), 0u);
    EXPECT_EQ(graph.capacity(), 0u);

    // Re-add a vertex after empty
    auto v_new = make_vec_4d(99.0f, 0.0f, 0.0f, 0.0f);
    uint32_t idx_new = graph.addVertex(999, make_float_bytes(v_new).get());
    EXPECT_EQ(idx_new, 0u);
    EXPECT_EQ(graph.size(), 1u);
    EXPECT_EQ(graph.chunk_count(), 1u);
    EXPECT_EQ(graph.capacity(), 4u);
}
