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
