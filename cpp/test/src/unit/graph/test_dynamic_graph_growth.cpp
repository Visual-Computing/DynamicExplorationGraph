#include "test_dynamic_graph_common.h"

// ---------------------------------------------------------------------------
//  2. Dynamic Chunk Growth on AddVertex
// ---------------------------------------------------------------------------

TEST(DynamicGraph, ChunkGrowthAcrossBoundaries) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    // 8 vertices per chunk
    deglib::graph::DynamicGraph graph(4, space, /*chunk_size=*/8);

    EXPECT_EQ(graph.chunk_count(), 0);
    EXPECT_EQ(graph.capacity(), 0);

    // Add 1st vertex -> 1st chunk allocated
    auto v0 = make_vec_4d(1.0f, 2.0f, 3.0f, 4.0f);
    uint32_t idx0 = graph.addVertex(100, make_float_bytes(v0).get());
    EXPECT_EQ(idx0, 0);
    EXPECT_EQ(graph.size(), 1);
    EXPECT_EQ(graph.chunk_count(), 1);
    EXPECT_EQ(graph.capacity(), 8);
    EXPECT_TRUE(graph.hasVertex(100));
    EXPECT_EQ(graph.getInternalIndex(100), 0);
    EXPECT_EQ(graph.getExternalLabel(0), 100);

    // Add up to 8 vertices -> still 1 chunk
    for (uint32_t i = 1; i < 8; ++i) {
        auto vi = make_vec_4d(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
        uint32_t idx = graph.addVertex(100 + i, make_float_bytes(vi).get());
        EXPECT_EQ(idx, i);
        EXPECT_EQ(graph.chunk_count(), 1);
        EXPECT_EQ(graph.capacity(), 8);
    }
    EXPECT_EQ(graph.size(), 8);

    // Add 9th vertex -> triggers 2nd chunk
    auto v8 = make_vec_4d(8.0f, 0.0f, 0.0f, 0.0f);
    uint32_t idx8 = graph.addVertex(108, make_float_bytes(v8).get());
    EXPECT_EQ(idx8, 8);
    EXPECT_EQ(graph.size(), 9);
    EXPECT_EQ(graph.chunk_count(), 2);
    EXPECT_EQ(graph.capacity(), 16);

    // Add up to 25 vertices -> 4 chunks total (capacity 32)
    for (uint32_t i = 9; i < 25; ++i) {
        auto vi = make_vec_4d(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
        graph.addVertex(100 + i, make_float_bytes(vi).get());
    }
    EXPECT_EQ(graph.size(), 25);
    EXPECT_EQ(graph.chunk_count(), 4);
    EXPECT_EQ(graph.capacity(), 32);

    // Check all labels and feature vectors
    for (uint32_t i = 0; i < 25; ++i) {
        EXPECT_TRUE(graph.hasVertex(100 + i));
        EXPECT_EQ(graph.getInternalIndex(100 + i), i);
        EXPECT_EQ(graph.getExternalLabel(i), 100 + i);
        const float* f = reinterpret_cast<const float*>(graph.getFeatureVector(i));
        if (i == 0) {
            EXPECT_FLOAT_EQ(f[0], 1.0f);
            EXPECT_FLOAT_EQ(f[1], 2.0f);
        } else {
            EXPECT_FLOAT_EQ(f[0], static_cast<float>(i));
        }
    }
}
