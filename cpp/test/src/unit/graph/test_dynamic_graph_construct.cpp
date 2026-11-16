#include "test_dynamic_graph_common.h"

#include <stdexcept>

// ---------------------------------------------------------------------------
//  1. Construction & Chunk Sizing
// ---------------------------------------------------------------------------

TEST(DynamicGraph, ConstructionEmpty) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::DynamicGraph graph(4, space, /*chunk_size=*/8);  // 8 vertices per chunk

    EXPECT_EQ(graph.size(), 0);
    EXPECT_EQ(graph.capacity(), 0);
    EXPECT_EQ(graph.chunk_count(), 0);
    EXPECT_EQ(graph.chunk_capacity(), 8);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 4);
}

TEST(DynamicGraph, AutoRoundsChunkSizeToPowerOfTwo) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::DynamicGraph graph(4, space, /*chunk_size=*/10);  // rounds up to 16

    EXPECT_EQ(graph.chunk_capacity(), 16);
}

TEST(DynamicGraph, RejectsOddEdgesPerVertex) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    EXPECT_THROW(deglib::graph::DynamicGraph(3, space, 8), std::invalid_argument);
}

TEST(DynamicGraph, DenseAndSparseLabelLookup) {
    constexpr uint32_t n = 8;
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::DynamicGraph graph(4, space, /*chunk_size=*/8);

    std::array<float, 4> vec{0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t i = 0; i < n; ++i) {
        vec[0] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(vec.data()));
    }

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_TRUE(graph.hasVertex(i));
        EXPECT_EQ(graph.getInternalIndex(i), i);
    }
    EXPECT_FALSE(graph.hasVertex(n));

    // Removing vertex 2 swaps the last vertex (7) into internal slot 2 -> tests fallback
    graph.removeVertex(2);
    EXPECT_FALSE(graph.hasVertex(2));
    EXPECT_TRUE(graph.hasVertex(7));
    EXPECT_EQ(graph.getInternalIndex(7), 2u);
    EXPECT_FALSE(graph.hasVertex(n));
}
