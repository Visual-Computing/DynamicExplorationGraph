#include "deglib/analysis.h"
#include "deglib/filter.h"
#include "deglib/graph/dynamic_graph.h"
#include "deglib/graph/readonly_graph.h"
#include "deglib/graph/sizebounded_graph.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<float> make_vec_4d(float x, float y, float z, float w) { return {x, y, z, w}; }

std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
    return bytes;
}

}  // anonymous namespace

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

// ---------------------------------------------------------------------------
//  5. Search, Exploration & Random Graph
// ---------------------------------------------------------------------------

TEST(DynamicGraph, RandomGraphAndSearch) {
    const uint32_t count = 60;
    const uint8_t edges = 4;
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);

    std::vector<float> data(count * 4);
    for (size_t i = 0; i < count * 4; ++i) {
        data[i] = static_cast<float>(i % 17) * 0.1f;
    }

    auto graph = deglib::graph::DynamicGraph::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, edges, space, 42, /*chunk_size=*/8);

    EXPECT_EQ(graph.size(), count);
    EXPECT_EQ(graph.getEdgesPerVertex(), edges);
    EXPECT_GE(graph.capacity(), count);

    // Search query
    std::vector<float> query = {0.1f, 0.2f, 0.3f, 0.4f};
    auto results = graph.search(std::span<const float>(query), 5);
    EXPECT_EQ(results.size(), 5);

    // Exploration from vertex 0
    auto explore_results = graph.explore(0, 5);
    EXPECT_EQ(explore_results.size(), 5);

    // Pathfinding
    auto path = graph.hasPath(graph.getEntryVertexIndices(), 10, 0.1f, 10);
    // Path should either be empty or reach target 10
    if (!path.empty()) {
        EXPECT_EQ(path.front().getIdentifier(), 10);
    }
}

// ---------------------------------------------------------------------------
//  6. Save and Load
// ---------------------------------------------------------------------------

TEST(DynamicGraph, SaveAndLoad) {
    const uint32_t count = 30;
    const uint8_t edges = 4;
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);

    std::vector<float> data(count * 4);
    for (size_t i = 0; i < count * 4; ++i) {
        data[i] = static_cast<float>(i) * 0.05f;
    }

    auto orig_graph =
        deglib::graph::DynamicGraph::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, edges, space, 123, /*chunk_size=*/8);

    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_dynamic_graph.deg";
    EXPECT_TRUE(orig_graph.saveGraph(temp_path.string().c_str()));

    auto loaded_graph = deglib::graph::load_dynamic_graph(temp_path.string().c_str(), /*chunk_size=*/8);
    EXPECT_EQ(loaded_graph.size(), orig_graph.size());
    EXPECT_EQ(loaded_graph.getEdgesPerVertex(), orig_graph.getEdgesPerVertex());
    EXPECT_EQ(loaded_graph.getFeatureSpace().dim(), orig_graph.getFeatureSpace().dim());

    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(loaded_graph.getExternalLabel(i), orig_graph.getExternalLabel(i));
        const uint32_t* orig_n = orig_graph.getNeighborIndices(i);
        const uint32_t* loaded_n = loaded_graph.getNeighborIndices(i);
        for (uint8_t e = 0; e < edges; ++e) {
            EXPECT_EQ(orig_n[e], loaded_n[e]);
        }
    }

    std::filesystem::remove(temp_path);
}

// ---------------------------------------------------------------------------
//  7. Interoperability & Facade Conversion
// ---------------------------------------------------------------------------

TEST(DynamicGraph, FacadeConversion) {
    const uint32_t count = 20;
    const uint8_t edges = 4;
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);

    std::vector<float> data(count * 4, 1.0f);
    auto deg = deglib::DynamicExplorationGraph::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, edges, space, 7);

    // Convert from SizeBoundedGraph to DynamicGraph
    auto dynamic_deg = deg.to_dynamic(/*chunk_size=*/8);
    EXPECT_EQ(dynamic_deg.size(), count);
    EXPECT_TRUE(dynamic_deg.isMutable());

    // Convert DynamicGraph to ReadOnlyGraph
    auto readonly_deg = dynamic_deg.to_readonly();
    EXPECT_EQ(readonly_deg.size(), count);
    EXPECT_FALSE(readonly_deg.isMutable());

    // Convert ReadOnlyGraph back to mutable SizeBoundedGraph
    auto sizebounded_deg = readonly_deg.to_mutable();
    EXPECT_EQ(sizebounded_deg.size(), count);
    EXPECT_TRUE(sizebounded_deg.isMutable());
}
