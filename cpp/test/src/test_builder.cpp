// test_builder.cpp — Unit tests for deglib::builder
//
// Covers: UnionFind, GraphEdge, BuilderAddTask/RemoveTask/Change,
// BuilderStatus, ReachableGroup, and comprehensive EvenRegularGraphBuilder
// tests with real feature vectors and multiple distance functions (L2,
// InnerProduct, L2_Uint8, EvpBits).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include "builder.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
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

// ---------------------------------------------------------------------------
//  EvenRegularGraphBuilder Tests with Various Distance Functions
// ---------------------------------------------------------------------------

// Helper function: Create float feature vectors
std::vector<std::byte> createFloatFeature(const std::vector<float>& values) {
    std::vector<std::byte> feature(values.size() * sizeof(float));
    std::memcpy(feature.data(), values.data(), feature.size());
    return feature;
}

// Helper function: Create uint8 feature vectors
std::vector<std::byte> createUint8Feature(const std::vector<uint8_t>& values) {
    std::vector<std::byte> feature(values.size() * sizeof(uint8_t));
    std::memcpy(feature.data(), values.data(), feature.size());
    return feature;
}

// Helper function: Create EvpBits feature vectors
// EvpBits format: 2*dim/8 bytes (ones and negative_ones packed)
std::vector<std::byte> createEvpBitsFeature(const std::vector<uint8_t>& bit_values) {
    size_t byte_count = bit_values.size() / 8;
    std::vector<std::byte> feature(2 * byte_count);
    
    // Pack ones and negative_ones separately
    uint8_t ones_byte = 0, negatives_byte = 0;
    for (size_t i = 0; i < bit_values.size(); ++i) {
        if (bit_values[i] > 0) {
            ones_byte |= (1 << (i % 8));
        } else if (bit_values[i] < 0) {
            negatives_byte |= (1 << (i % 8));
        }
        
        if (i % 8 == 7) {
            feature[i / 8] = std::byte(ones_byte);
            feature[byte_count + i / 8] = std::byte(negatives_byte);
            ones_byte = 0;
            negatives_byte = 0;
        }
    }
    return feature;
}

// Test: Build graph with L2 distance on float vectors
TEST(EvenRegularGraphBuilder, BuildGraphWithL2Float) {
    auto size = 12;
    auto edges_per_vertex = 4;  // Must be even
    size_t feature_dim = 4;

    // Create graph with L2 metric
    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));

    // Create builder
    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Add float feature vectors
    std::vector<std::vector<float>> features = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.1f, 2.1f, 3.1f, 4.1f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {5.1f, 6.1f, 7.1f, 8.1f},
        {10.0f, 10.0f, 10.0f, 10.0f},
        {10.1f, 10.1f, 10.1f, 10.1f},
        {15.0f, 15.0f, 15.0f, 15.0f},
        {15.1f, 15.1f, 15.1f, 15.1f},
        {20.0f, 20.0f, 20.0f, 20.0f},
        {20.1f, 20.1f, 20.1f, 20.1f},
        {25.0f, 25.0f, 25.0f, 25.0f},
        {25.1f, 25.1f, 25.1f, 25.1f}
    };

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createFloatFeature(features[i]));
    }

    // Execute build with callback
    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    // Verify graph is connected
    EXPECT_EQ(graph.size(), 12u);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        // First neighbor should be valid (0 <= neighbor < size)
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}

// Test: Build graph with InnerProduct distance on float vectors
TEST(EvenRegularGraphBuilder, BuildGraphWithInnerProductFloat) {
    auto size = 10;
    auto edges_per_vertex = 4;  // Must be even
    size_t feature_dim = 4;

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::InnerProduct);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Normalized float vectors for inner product
    std::vector<std::vector<float>> features = {
        {0.5f, 0.5f, 0.5f, 0.5f},
        {0.51f, 0.51f, 0.51f, 0.51f},
        {0.707f, 0.0f, 0.707f, 0.0f},
        {0.7f, 0.0f, 0.71f, 0.0f},
        {0.0f, 0.707f, 0.0f, 0.707f},
        {0.0f, 0.71f, 0.0f, 0.71f},
        {0.5f, 0.0f, 0.5f, 0.707f},
        {0.51f, 0.0f, 0.51f, 0.7f},
        {0.707f, 0.5f, 0.0f, 0.5f},
        {0.71f, 0.51f, 0.0f, 0.51f}
    };

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createFloatFeature(features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    EXPECT_EQ(graph.size(), 10u);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}

// Test: Build graph with L2Uint8 distance on uint8 vectors
TEST(EvenRegularGraphBuilder, BuildGraphWithL2Uint8) {
    auto size = 10;
    auto edges_per_vertex = 4;  // Must be even
    size_t feature_dim = 4;

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2_Uint8);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Uint8 feature vectors (quantized values 0-255)
    std::vector<std::vector<uint8_t>> features = {
        {100, 120, 130, 140},
        {101, 121, 131, 141},
        {150, 160, 170, 180},
        {151, 161, 171, 181},
        {200, 210, 220, 230},
        {201, 211, 221, 231},
        {220, 230, 240, 250},
        {221, 231, 241, 251},
        {240, 245, 250, 255},
        {241, 246, 251, 255}
    };

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createUint8Feature(features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    EXPECT_EQ(graph.size(), 10u);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}

// Test: Build larger graph with multiple distance functions to verify consistency
TEST(EvenRegularGraphBuilder, BuildGraphLargerDataset) {
    auto size = 30;
    auto edges_per_vertex = 6;  // Must be even
    size_t feature_dim = 4;

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(123);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Create 20 float vectors in 4D space with clustered structure
    std::vector<std::vector<float>> features;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Cluster 1: around (1, 1, 1, 1)
    for (int i = 0; i < 10; ++i) {
        float x = 1.0f + dist(rnd) * 0.1f;
        float y = 1.0f + dist(rnd) * 0.1f;
        float z = 1.0f + dist(rnd) * 0.1f;
        float w = 1.0f + dist(rnd) * 0.1f;
        features.push_back({x, y, z, w});
    }

    // Cluster 2: around (5, 5, 5, 5)
    for (int i = 0; i < 10; ++i) {
        float x = 5.0f + dist(rnd) * 0.1f;
        float y = 5.0f + dist(rnd) * 0.1f;
        float z = 5.0f + dist(rnd) * 0.1f;
        float w = 5.0f + dist(rnd) * 0.1f;
        features.push_back({x, y, z, w});
    }

    // Cluster 3: around (10, 10, 10, 10)
    for (int i = 0; i < 10; ++i) {
        float x = 10.0f + dist(rnd) * 0.1f;
        float y = 10.0f + dist(rnd) * 0.1f;
        float z = 10.0f + dist(rnd) * 0.1f;
        float w = 10.0f + dist(rnd) * 0.1f;
        features.push_back({x, y, z, w});
    }

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createFloatFeature(features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    // Verify graph structure
    EXPECT_EQ(graph.size(), size);
    
    // Verify each vertex has at least some neighbors
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }

    // Verify that graph is well-connected (sample vertices should have neighbors)
    uint32_t idx0 = graph.getInternalIndex(0);
    uint32_t idx1 = graph.getInternalIndex(1);
    auto neighbors0 = graph.getNeighborIndices(idx0);
    auto neighbors1 = graph.getNeighborIndices(idx1);
    EXPECT_GE(neighbors0[0], 0u);
    EXPECT_LT(neighbors0[0], graph.size());
    EXPECT_GE(neighbors1[0], 0u);
    EXPECT_LT(neighbors1[0], graph.size());
}

// Test: Build graph with custom optimization parameters
TEST(EvenRegularGraphBuilder, BuildGraphWithCustomParameters) {
    auto size = 12;
    auto edges_per_vertex = 6;  // Must be even
    size_t feature_dim = 5;

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(456);

    // Create builder with custom parameters
    deglib::builder::EvenRegularGraphBuilder builder(
        graph,
        rnd,
        deglib::builder::OptimizationTarget::HighLID,
        6,      // extend_k
        0.2f,   // extend_eps
        4,      // improve_k
        0.15f,  // improve_eps
        5,      // max_path_length
        2,      // swap_tries
        1       // additional_swap_tries
    );

    // Create feature vectors
    std::vector<std::vector<float>> features;
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    
    for (int i = 0; i < 12; ++i) {
        features.push_back({
            dist(rnd),
            dist(rnd),
            dist(rnd),
            dist(rnd),
            dist(rnd)
        });
    }

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createFloatFeature(features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    EXPECT_EQ(graph.size(), size);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}

// Test: Build graph and verify edge weights are reasonable
TEST(EvenRegularGraphBuilder, VerifyEdgeWeights) {
    auto size = 10;
    auto edges_per_vertex = 4;  // Must be even
    size_t feature_dim = 3;

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(789);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Create well-separated vectors so distances are predictable
    std::vector<std::vector<float>> features = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},    // distance = 1.0
        {0.0f, 1.0f, 0.0f, 0.0f},    // distance = 1.0
        {0.0f, 0.0f, 1.0f, 0.0f},    // distance = 1.0
        {10.0f, 0.0f, 0.0f, 0.0f},   // distance = 10.0
        {0.0f, 10.0f, 0.0f, 0.0f},   // distance = 10.0
        {0.0f, 0.0f, 10.0f, 0.0f},   // distance = 10.0
        {20.0f, 0.0f, 0.0f, 0.0f},   // distance = 20.0
        {0.0f, 20.0f, 0.0f, 0.0f},   // distance = 20.0
        {0.0f, 0.0f, 20.0f, 0.0f}    // distance = 20.0
    };

    for (uint32_t i = 0; i < features.size(); ++i) {
        builder.addEntry(i, createFloatFeature(features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    // Verify that edge weights are positive
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto weights = graph.getNeighborWeights(internal_idx);
        for (uint32_t j = 0; j < edges_per_vertex; ++j) {
            EXPECT_GE(weights[j], 0.0f);
        }
    }
}

// Test: Build graph with different optimization targets
TEST(EvenRegularGraphBuilder, BuildGraphWithDifferentOptimizationTargets) {
    std::vector<deglib::builder::OptimizationTarget> targets = {
        deglib::builder::OptimizationTarget::StreamingData_SchemeA,
        deglib::builder::OptimizationTarget::StreamingData_SchemeB,
        deglib::builder::OptimizationTarget::StreamingData_SchemeC,
        deglib::builder::OptimizationTarget::StreamingData_SchemeD,
        deglib::builder::OptimizationTarget::HighLID,
        deglib::builder::OptimizationTarget::LowLID
    };

    size_t feature_dim = 4;

    for (auto target : targets) {
        auto size = 12;
        auto edges_per_vertex = 4;  // Must be even

        deglib::FloatSpace feature_space(feature_dim, deglib::Metric::L2);
        deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
        std::mt19937 rnd(999);

        deglib::builder::EvenRegularGraphBuilder builder(
            graph, rnd, target, 4, 0.1f, 2, 0.05f);

        std::vector<std::vector<float>> features = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.1f, 2.1f, 3.1f, 4.1f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {5.1f, 6.1f, 7.1f, 8.1f},
            {10.0f, 11.0f, 12.0f, 13.0f},
            {10.1f, 11.1f, 12.1f, 13.1f},
            {15.0f, 16.0f, 17.0f, 18.0f},
            {15.1f, 16.1f, 17.1f, 18.1f},
            {20.0f, 21.0f, 22.0f, 23.0f},
            {20.1f, 21.1f, 22.1f, 23.1f},
            {25.0f, 26.0f, 27.0f, 28.0f},
            {25.1f, 26.1f, 27.1f, 28.1f}
        };

        for (uint32_t i = 0; i < features.size(); ++i) {
            builder.addEntry(i, createFloatFeature(features[i]));
        }

        auto callback = [](deglib::builder::BuilderStatus& status) {};
        builder.build(callback);

        EXPECT_EQ(graph.size(), size);
        for (uint32_t i = 0; i < graph.size(); ++i) {
            uint32_t internal_idx = graph.getInternalIndex(i);
            auto neighbors = graph.getNeighborIndices(internal_idx);
            EXPECT_GE(neighbors[0], 0u);
            EXPECT_LT(neighbors[0], graph.size());
        }
    }
}

// Test: Build graph with EvpBits distance function
TEST(EvenRegularGraphBuilder, BuildGraphWithEvpBits) {
    auto size = 12;
    auto edges_per_vertex = 4;  // Must be even
    size_t feature_dim = 32;    // EvpBits works with bit dimensions

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::EvpBits);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(555);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Create EvpBits feature vectors (bit arrays)
    // For EvpBits with dimension 32, we need 32 bits (4 bytes) per feature
    std::vector<std::vector<uint8_t>> bit_features = {
        {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1},
        {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},
        {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0},
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0},
        {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1},
        {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0},
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0},
        {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1}
    };

    for (uint32_t i = 0; i < bit_features.size(); ++i) {
        builder.addEntry(i, createEvpBitsFeature(bit_features[i]));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    EXPECT_EQ(graph.size(), 12u);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}

// Test: Build graph with EvpBits on larger dataset
TEST(EvenRegularGraphBuilder, BuildGraphWithEvpBitsLarger) {
    auto size = 20;
    auto edges_per_vertex = 6;  // Must be even
    size_t feature_dim = 64;    // Larger bit dimension

    deglib::FloatSpace feature_space(feature_dim, deglib::Metric::EvpBits);
    deglib::graph::SizeBoundedGraph graph(size, edges_per_vertex, std::move(feature_space));
    std::mt19937 rnd(666);
    std::uniform_int_distribution<int> bit_dist(0, 1);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rnd);

    // Create random EvpBits feature vectors
    for (uint32_t i = 0; i < 20; ++i) {
        std::vector<uint8_t> bits(feature_dim);
        for (size_t j = 0; j < feature_dim; ++j) {
            bits[j] = static_cast<uint8_t>(bit_dist(rnd));
        }
        builder.addEntry(i, createEvpBitsFeature(bits));
    }

    auto callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(callback);

    EXPECT_EQ(graph.size(), 20u);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        uint32_t internal_idx = graph.getInternalIndex(i);
        auto neighbors = graph.getNeighborIndices(internal_idx);
        EXPECT_GE(neighbors[0], 0u);
        EXPECT_LT(neighbors[0], graph.size());
    }
}
