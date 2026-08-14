// test_sizebounded_graph.cpp — Google Test suite for SizeBoundedGraph
//
// Covers: construction, vertex management, edge management, label lookup,
// feature storage, capacity, search, save/load, multi-operation cycles.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "deglib/graph/sizebounded_graph.h"
#include "deglib/graph/readonly_graph.h"
#include "deglib/analysis.h"
#include "deglib/filter.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

static std::vector<float> make_vec_4d(float x, float y, float z, float w) {
    return {x, y, z, w};
}

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

// ---------------------------------------------------------------------------
//  1. Construction
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, ConstructionEmpty) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(100, 4, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.capacity(), 100u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
    EXPECT_EQ(space.dim(), 4u);
    EXPECT_EQ(space.metric(), deglib::distances::Metric::FP32_L2);
}

TEST(SizeBoundedGraph, RejectsOddEdgesPerVertex) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    EXPECT_THROW(deglib::graph::SizeBoundedGraph graph(50, 3, space), std::invalid_argument);
}

TEST(SizeBoundedGraph, InnerProductMetric) {
    deglib::distances::FloatSpace space(8, deglib::distances::Metric::FP32_InnerProduct);
    deglib::graph::SizeBoundedGraph graph(200, 8, space);

    EXPECT_EQ(graph.getEdgesPerVertex(), 8u);
    EXPECT_EQ(graph.capacity(), 200u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_InnerProduct);
}

TEST(SizeBoundedGraph, L2Uint8Metric) {
    deglib::distances::FloatSpace space(128, deglib::distances::Metric::Uint8_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 128u);
}

// ---------------------------------------------------------------------------
//  2. Vertex Management
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, AddVertex) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    auto v0 = make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f);
    auto v1 = make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f);
    auto v2 = make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f);

    auto idx0 = graph.addVertex(0, make_float_bytes(v0).get());
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(graph.size(), 1u);
    EXPECT_TRUE(graph.hasVertex(0));

    auto idx1 = graph.addVertex(1, make_float_bytes(v1).get());
    EXPECT_EQ(idx1, 1u);
    EXPECT_EQ(graph.size(), 2u);
    EXPECT_TRUE(graph.hasVertex(1));

    auto idx2 = graph.addVertex(2, make_float_bytes(v2).get());
    EXPECT_EQ(idx2, 2u);
}

TEST(SizeBoundedGraph, ExternalLabels) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    std::vector<float> v(4, 0.0f);
    graph.addVertex(0, make_float_bytes(v).get());
    graph.addVertex(1, make_float_bytes(v).get());
    graph.addVertex(2, make_float_bytes(v).get());

    EXPECT_EQ(graph.getExternalLabel(0), 0u);
    EXPECT_EQ(graph.getExternalLabel(1), 1u);
    EXPECT_EQ(graph.getExternalLabel(2), 2u);
}

TEST(SizeBoundedGraph, RemoveSingleVertex) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(4, 2, space);

    std::vector<float> v(4, 0.0f);
    graph.addVertex(0, make_float_bytes(v).get());

    auto neighbors = graph.removeVertex(0);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_FALSE(graph.hasVertex(0));
}

TEST(SizeBoundedGraph, RemoveMiddleVertex) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 2, space);

    for (int i = 0; i < 4; ++i) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, make_float_bytes(v).get());
    }

    EXPECT_EQ(graph.size(), 4u);
    graph.removeVertex(1);

    EXPECT_EQ(graph.size(), 3u);
    EXPECT_FALSE(graph.hasVertex(1));
    EXPECT_TRUE(graph.hasVertex(0));
    EXPECT_TRUE(graph.hasVertex(2));
    EXPECT_TRUE(graph.hasVertex(3));
}

TEST(SizeBoundedGraph, SelfLoopInitialization) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    std::vector<float> v = {1.0f, 0.0f, 0.0f, 0.0f};
    graph.addVertex(5, make_float_bytes(v).get());

    EXPECT_TRUE(graph.hasEdge(0, 0));
    EXPECT_FALSE(graph.hasEdge(0, 1));
}

// ---------------------------------------------------------------------------
//  3. Edge Management
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, SetAndQueryEdge) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {3.0f, 4.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));

    graph.changeEdge(0, 0, 1, 5.0f);
    graph.changeEdge(1, 1, 0, 5.0f);

    EXPECT_TRUE(graph.hasEdge(0, 1));
    EXPECT_TRUE(graph.hasEdge(1, 0));
    EXPECT_NEAR(graph.getEdgeWeight(0, 1), 5.0f, 1e-4f);
    EXPECT_NEAR(graph.getEdgeWeight(1, 0), 5.0f, 1e-4f);
    EXPECT_EQ(graph.getEdgeWeight(0, 2), -1.0f);
}

TEST(SizeBoundedGraph, ChangeEdgeSwap) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 4; ++i) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(i);
        graph.addVertex(i, make_float_bytes(v).get());
    }

    graph.changeEdge(0, 0, 1, 1.0f);
    graph.changeEdge(1, 1, 0, 1.0f);
    graph.changeEdge(2, 2, 0, 4.0f);
    graph.changeEdge(0, 0, 2, 4.0f);
    graph.changeEdge(3, 3, 0, 9.0f);
    graph.changeEdge(0, 0, 3, 9.0f);

    EXPECT_TRUE(graph.hasEdge(0, 1));
    EXPECT_TRUE(graph.hasEdge(0, 2));
    EXPECT_TRUE(graph.hasEdge(0, 3));
    EXPECT_TRUE(graph.hasEdge(1, 0));
    EXPECT_TRUE(graph.hasEdge(2, 0));
    EXPECT_TRUE(graph.hasEdge(3, 0));
    EXPECT_NEAR(graph.getEdgeWeight(0, 1), 1.0f, 1e-4f);
    EXPECT_NEAR(graph.getEdgeWeight(0, 2), 4.0f, 1e-4f);
    EXPECT_NEAR(graph.getEdgeWeight(0, 3), 9.0f, 1e-4f);
}

TEST(SizeBoundedGraph, ChangeEdgesSorted) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(i);
        graph.addVertex(i, make_float_bytes(v).get());
    }

    uint32_t sorted_neighbors[] = {0, 1, 2, 3};
    float weights[] = {0.0f, 1.0f, 4.0f, 9.0f};
    graph.changeEdges(0, sorted_neighbors, weights);

    const auto* neighbors = graph.getNeighborIndices(0);
    EXPECT_EQ(neighbors[0], 0u);
    EXPECT_EQ(neighbors[1], 1u);
    EXPECT_EQ(neighbors[2], 2u);
    EXPECT_EQ(neighbors[3], 3u);
}

// ---------------------------------------------------------------------------
//  4. Label Lookup & Feature Storage
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, LabelLookup) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 2, space);

    for (auto lbl : {100u, 200u, 300u}) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(lbl / 100);
        graph.addVertex(lbl, make_float_bytes(v).get());
    }

    EXPECT_EQ(graph.getInternalIndex(100), 0u);
    EXPECT_EQ(graph.getInternalIndex(200), 1u);
    EXPECT_EQ(graph.getInternalIndex(300), 2u);
    EXPECT_EQ(graph.getExternalLabel(0), 100u);
    EXPECT_EQ(graph.getExternalLabel(1), 200u);
    EXPECT_EQ(graph.getExternalLabel(2), 300u);
}

TEST(SizeBoundedGraph, FeatureVectorStorage) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(2, 2, space);

    std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v2 = {5.0f, 6.0f, 7.0f, 8.0f};

    graph.addVertex(1, make_float_bytes(v1).get());
    graph.addVertex(2, make_float_bytes(v2).get());

    const float* fp1 = reinterpret_cast<const float*>(graph.getFeatureVector(0));
    const float* fp2 = reinterpret_cast<const float*>(graph.getFeatureVector(1));

    EXPECT_NEAR(fp1[0], 1.0f, 1e-6f);
    EXPECT_NEAR(fp1[1], 2.0f, 1e-6f);
    EXPECT_NEAR(fp2[0], 5.0f, 1e-6f);
    EXPECT_NEAR(fp2[3], 8.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
//  5. Capacity & Edge Cases
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, LargeGraph) {
    deglib::distances::FloatSpace space(128, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(1000, 16, space);

    EXPECT_EQ(graph.getEdgesPerVertex(), 16u);
    EXPECT_EQ(graph.capacity(), 1000u);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 128u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_L2);
}

TEST(SizeBoundedGraph, SmallGraph) {
    deglib::distances::FloatSpace space(2, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(3, 2, space);

    std::vector<float> v(2, 0.0f);
    graph.addVertex(0, make_float_bytes(v).get());

    EXPECT_EQ(graph.size(), 1u);
    EXPECT_TRUE(graph.hasEdge(0, 0));
}

TEST(SizeBoundedGraph, VariousDimensions) {
    std::vector<size_t> dims = {1, 4, 8, 64, 128, 192};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
        deglib::graph::SizeBoundedGraph graph(3, 2, space);

        std::vector<float> v(dim, 0.0f);
        v[0] = 1.0f;
        graph.addVertex(0, make_float_bytes(v).get());

        EXPECT_EQ(graph.size(), 1u);
        EXPECT_EQ(graph.getFeatureSpace().dim(), dim);
    }

    std::vector<size_t> u8_dims = {64, 128, 192};
    for (size_t dim : u8_dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);
        deglib::graph::SizeBoundedGraph graph(3, 2, space);

        std::vector<uint8_t> v(dim, 0);
        v[0] = 1;
        graph.addVertex(0, make_float_bytes(v).get());

        EXPECT_EQ(graph.getFeatureSpace().dim(), dim);
    }
}

// ---------------------------------------------------------------------------
//  6. Search
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, SearchBasic) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    float query[] = {0.5f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search(std::span<const float>(query, 4), 5, 0.0f);

    EXPECT_GT(results.size(), 0u);
    if (results.size() > 0) {
        EXPECT_GE(results.top().getIdentifier(), 0u);
    }
}

TEST(SizeBoundedGraph, SearchWithFilter) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    int valid[] = {0};
    deglib::search::Filter filter(valid, 1, 10, 10);

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search(std::span<const float>(query, 4), 3, 0.0f, &filter);

    EXPECT_GE(results.size(), 0u);
}

// ---------------------------------------------------------------------------
//  7. Save / Load
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, SaveGraph) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    std::string path = "test/graph_save_test.deg";
    bool saved = graph.saveGraph(path.c_str());
    EXPECT_TRUE(saved);

    auto size = std::filesystem::file_size(path);
    EXPECT_GT(size, 0u);
    std::filesystem::remove(path);
}

TEST(SizeBoundedGraph, SaveLoadHeader) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {7.0f, 10.0f, 0.0f, 0.0f};
    float v1[] = {0.5f, -1.0f, 2.0f, 3.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.changeEdge(0, 0, 1, 5.0f);

    std::string path = "test/graph_load_test.deg";
    graph.saveGraph(path.c_str());

    std::ifstream ifs(path, std::ios::binary);
    EXPECT_TRUE(ifs.is_open());

    if (ifs.is_open()) {
        uint8_t metric_type = 0;
        uint16_t dim = 0;
        uint32_t graph_size = 0;
        uint8_t edges = 0;
        ifs.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        ifs.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
        ifs.read(reinterpret_cast<char*>(&edges), sizeof(edges));

        EXPECT_EQ(static_cast<int>(metric_type), 1); // L2
        EXPECT_EQ(dim, 4u);
        EXPECT_EQ(graph_size, 2u);
        EXPECT_EQ(edges, 4u);

        ifs.close();
    }

    std::filesystem::remove(path);
}

TEST(SizeBoundedGraph, SaveUint8Graph) {
    deglib::distances::FloatSpace space(128, deglib::distances::Metric::Uint8_L2);
    deglib::graph::SizeBoundedGraph graph(3, 2, space);

    std::vector<uint8_t> v0(128, 0);
    std::vector<uint8_t> v1(128, 1);
    std::vector<uint8_t> v2(128, 255);

    graph.addVertex(0, make_float_bytes(v0).get());
    graph.addVertex(1, make_float_bytes(v1).get());
    graph.addVertex(2, make_float_bytes(v2).get());

    std::string path = "test/graph_uint8_test.deg";
    bool saved_ok = graph.saveGraph(path.c_str());
    EXPECT_TRUE(saved_ok);

    auto size = std::filesystem::file_size(path);
    EXPECT_GT(size, 0u);
    std::filesystem::remove(path);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);
}

// ---------------------------------------------------------------------------
//  8. Multiple Operations
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, MultipleAddRemoveCycles) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 2, space);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, make_float_bytes(v).get());
    }
    EXPECT_EQ(graph.size(), 5u);

    graph.removeVertex(0);
    graph.removeVertex(2);
    graph.removeVertex(4);
    EXPECT_EQ(graph.size(), 2u);
    EXPECT_TRUE(graph.hasVertex(1));
    EXPECT_TRUE(graph.hasVertex(3));

    for (int i = 5; i < 8; ++i) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, make_float_bytes(v).get());
    }
    for (int i = 1; i <= 7; ++i) {
        if (graph.hasVertex(static_cast<uint32_t>(i)))
            graph.removeVertex(static_cast<uint32_t>(i));
    }
    EXPECT_EQ(graph.size(), 0u);
}

// ---------------------------------------------------------------------------
//  8. VisitedListPool
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, VisitedListPool) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search(std::span<const float>(query, 4), 1);
    EXPECT_GT(results.size(), 0u);
}

// ---------------------------------------------------------------------------
//  9. Search & Explore Dedicated Verification
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, SearchWithMaxDistanceCount) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    for (int i = 0; i < 10; ++i) {
        std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
        graph.addVertex(i, make_float_bytes(v).get());
    }

    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search(std::span<const float>(query), 5, 0.1f, nullptr, 2);

    EXPECT_GT(results.size(), 0u);
    EXPECT_LE(results.size(), 5u);
}

TEST(SizeBoundedGraph, ExploreBasicIncludeEntry) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
        graph.addVertex(i, make_float_bytes(v).get());
    }

    // Connect node 0 to 1 and 2
    uint32_t sorted_neighbors[] = {0, 1, 2, 3};
    float weights[] = {0.0f, 1.0f, 4.0f, 9.0f};
    graph.changeEdges(0, sorted_neighbors, weights);

    auto results = graph.explore(0, 3, 0, 0.0f, /*include_entry=*/true, nullptr);

    EXPECT_GT(results.size(), 0u);
    bool found_entry = false;
    while (!results.empty()) {
        if (results.top().getIdentifier() == 0u) {
            found_entry = true;
        }
        results.pop();
    }
    EXPECT_TRUE(found_entry);
}

TEST(SizeBoundedGraph, ExploreExcludeEntry) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
        graph.addVertex(i, make_float_bytes(v).get());
    }

    uint32_t sorted_neighbors[] = {0, 1, 2, 3};
    float weights[] = {0.0f, 1.0f, 4.0f, 9.0f};
    graph.changeEdges(0, sorted_neighbors, weights);

    auto results = graph.explore(0, 2, 0, 0.0f, /*include_entry=*/false, nullptr);

    EXPECT_GT(results.size(), 0u);
}

TEST(SizeBoundedGraph, ExploreWithMaxDistanceCount) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    for (int i = 0; i < 10; ++i) {
        std::vector<float> v = {static_cast<float>(i), 0.0f, 0.0f, 0.0f};
        graph.addVertex(i, make_float_bytes(v).get());
    }

    auto results = graph.explore(0, 5, 2, 0.0f, /*include_entry=*/true, nullptr);

    EXPECT_GT(results.size(), 0u);
    EXPECT_LE(results.size(), 5u);
}
// ---------------------------------------------------------------------------
//  10. Internal vs External ID Verification
// ---------------------------------------------------------------------------
//
// These tests verify that SizeBoundedGraph (as an InternalGraph implementation)
// strictly operates on internal indices (0..N-1) for search(), explore(),
// hasPath(), and getNeighborIndices(), while hasVertex() and getExternalLabel()
// / getInternalIndex() handle the external label ↔ internal index mapping.
//
// Non-sequential external labels (1005, 9999, 42, 707, 12345) are used so that
// any accidental swap of external labels and internal indices will instantly
// fail test assertions.

namespace {

// Non-sequential, arbitrary external labels — deliberately distinct from 0..N-1
constexpr std::array<uint32_t, 5> kExtLabels = {1005, 9999, 42, 707, 12345};

} // anonymous namespace

TEST(SizeBoundedGraphInternalIndicesInSearch, SearchReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   // Add vertices with non-sequential external labels at distinct positions
   graph.addVertex(kExtLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   // Fully connect vertices 0..3 (neighbors must be sorted ascending)
   uint32_t sorted_nbrs[] = {0, 1, 2, 3};
   float weights[] = {0.0f, 1.0f, 2.0f, 3.0f};
   graph.changeEdges(0, sorted_nbrs, weights);
   graph.changeEdges(1, sorted_nbrs, weights);
   graph.changeEdges(2, sorted_nbrs, weights);
   graph.changeEdges(3, sorted_nbrs, weights);
   // Vertex 4 has self-loop only

   // search() on InternalGraph returns internal indices in the ResultSet
   std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
   auto results = graph.search(std::span<const float>(query), 3, 0.0f);

   ASSERT_GT(results.size(), 0u);

   // Every identifier must be an internal index (0..4), NOT an external label
   std::unordered_set<uint32_t> internal_indices = {0, 1, 2, 3, 4};
   std::unordered_set<uint32_t> ext_labels(kExtLabels.begin(), kExtLabels.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(internal_indices.contains(id))
           << "search() result identifier " << id << " is not a valid internal index (0..N-1)";
       EXPECT_FALSE(ext_labels.contains(id))
           << "search() result identifier " << id << " is an external label, not an internal index";
       results.pop();
   }
}

TEST(SizeBoundedGraphInternalIndicesInSearch, ExploreReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   graph.addVertex(kExtLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   uint32_t sorted_nbrs[] = {0, 1, 2, 3};
   float weights[] = {0.0f, 1.0f, 2.0f, 3.0f};
   graph.changeEdges(0, sorted_nbrs, weights);
   graph.changeEdges(1, sorted_nbrs, weights);
   graph.changeEdges(2, sorted_nbrs, weights);
   graph.changeEdges(3, sorted_nbrs, weights);

   // explore() on InternalGraph takes an internal index and returns internal indices
   auto results = graph.explore(0, 3, 0, 0.0f, /*include_entry=*/true, nullptr);

   ASSERT_GT(results.size(), 0u);

   std::unordered_set<uint32_t> internal_indices = {0, 1, 2, 3, 4};
   std::unordered_set<uint32_t> ext_labels(kExtLabels.begin(), kExtLabels.end());

   while (!results.empty()) {
       uint32_t id = results.top().getIdentifier();
       EXPECT_TRUE(internal_indices.contains(id))
           << "explore() result identifier " << id << " is not a valid internal index";
       EXPECT_FALSE(ext_labels.contains(id))
           << "explore() result identifier " << id << " is an external label, not an internal index";
       results.pop();
   }
}

TEST(SizeBoundedGraphInternalIndicesInSearch, HasPathReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   graph.addVertex(kExtLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   uint32_t sorted_nbrs[] = {0, 1, 2, 3};
   float weights[] = {0.0f, 1.0f, 2.0f, 3.0f};
   graph.changeEdges(0, sorted_nbrs, weights);
   graph.changeEdges(1, sorted_nbrs, weights);
   graph.changeEdges(2, sorted_nbrs, weights);
   graph.changeEdges(3, sorted_nbrs, weights);

   // hasPath() takes internal entry indices and a internal to_vertex index
   // Returns a vector of ObjectDistance with internal indices
   auto path = graph.hasPath({0}, 1, 0.0f, 5);

   ASSERT_GT(path.size(), 0u);

   std::unordered_set<uint32_t> internal_indices = {0, 1, 2, 3, 4};
   std::unordered_set<uint32_t> ext_labels(kExtLabels.begin(), kExtLabels.end());

   for (const auto& od : path) {
       uint32_t id = od.getIdentifier();
       EXPECT_TRUE(internal_indices.contains(id))
           << "hasPath() result identifier " << id << " is not a valid internal index";
       EXPECT_FALSE(ext_labels.contains(id))
           << "hasPath() result identifier " << id << " is an external label, not an internal index";
   }
}

TEST(SizeBoundedGraphInternalIndicesInSearch, GetNeighborIndicesReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   graph.addVertex(kExtLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   uint32_t sorted_nbrs[] = {0, 1, 2, 3};
   float weights[] = {0.0f, 1.0f, 2.0f, 3.0f};
   graph.changeEdges(0, sorted_nbrs, weights);

   // getNeighborIndices() takes an internal index and returns internal indices
   const auto* neighbors = graph.getNeighborIndices(0);
   uint8_t epv = graph.getEdgesPerVertex();

   std::unordered_set<uint32_t> internal_indices = {0, 1, 2, 3, 4};
   std::unordered_set<uint32_t> ext_labels(kExtLabels.begin(), kExtLabels.end());

   for (uint8_t i = 0; i < epv; ++i) {
       uint32_t n = neighbors[i];
       EXPECT_TRUE(internal_indices.contains(n))
           << "getNeighborIndices()[ " << i << "] = " << n << " is not a valid internal index";
       EXPECT_FALSE(ext_labels.contains(n))
           << "getNeighborIndices()[ " << i << "] = " << n << " is an external label, not an internal index";
   }
}

TEST(SizeBoundedGraphLabelMapping, BidirectionalTranslationWithNonSequentialLabels) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
   deglib::graph::SizeBoundedGraph graph(5, 4, space);

   // Add vertices with non-sequential external labels
   graph.addVertex(kExtLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
   graph.addVertex(kExtLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

   // getExternalLabel(internal_index) → external_label
   EXPECT_EQ(graph.getExternalLabel(0), kExtLabels[0]); // 1005
   EXPECT_EQ(graph.getExternalLabel(1), kExtLabels[1]); // 9999
   EXPECT_EQ(graph.getExternalLabel(2), kExtLabels[2]); // 42
   EXPECT_EQ(graph.getExternalLabel(3), kExtLabels[3]); // 707
   EXPECT_EQ(graph.getExternalLabel(4), kExtLabels[4]); // 12345

   // getInternalIndex(external_label) → internal_index
   EXPECT_EQ(graph.getInternalIndex(kExtLabels[0]), 0u);
   EXPECT_EQ(graph.getInternalIndex(kExtLabels[1]), 1u);
   EXPECT_EQ(graph.getInternalIndex(kExtLabels[2]), 2u);
   EXPECT_EQ(graph.getInternalIndex(kExtLabels[3]), 3u);
   EXPECT_EQ(graph.getInternalIndex(kExtLabels[4]), 4u);

   // Round-trip: internal → external → internal
   for (uint32_t i = 0; i < 5; ++i) {
       uint32_t ext = graph.getExternalLabel(i);
       uint32_t back = graph.getInternalIndex(ext);
       EXPECT_EQ(back, i) << "Round-trip failed for internal index " << i;
   }

    // hasVertex() takes external labels
    EXPECT_TRUE(graph.hasVertex(kExtLabels[0]));
    EXPECT_TRUE(graph.hasVertex(kExtLabels[2]));
    EXPECT_FALSE(graph.hasVertex(kExtLabels[0] + 1)); // 1006 — not in graph
    EXPECT_FALSE(graph.hasVertex(0u)); // 0 is an internal index, not an external label
    EXPECT_FALSE(graph.hasVertex(4u)); // 4 is an internal index, not an external label
}

// ---------------------------------------------------------------------------
//  Factory Methods: create_empty and create_random_graph
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, CreateEmpty) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::graph::SizeBoundedGraph::create_empty(100, 4, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.capacity(), 100u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 4u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_L2);
}

TEST(SizeBoundedGraph, CreateEmptyUint8) {
    deglib::distances::FloatSpace space(128, deglib::distances::Metric::Uint8_L2);
    auto graph = deglib::graph::SizeBoundedGraph::create_empty(50, 6, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.capacity(), 50u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 6u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);
}

TEST(SizeBoundedGraph, CreateRandomGraphFP32) {
    const uint32_t vertex_count = 100;
    const uint8_t edges_per_vertex = 8;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Create feature data: vertex_count vectors of dim floats
    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Validate graph structure
    EXPECT_EQ(graph.size(), vertex_count);
    EXPECT_EQ(graph.getEdgesPerVertex(), edges_per_vertex);

    // Check regularity (no self-loops, sorted, unique neighbors)
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph, vertex_count, true));

    // Check connectivity
    EXPECT_TRUE(deglib::analysis::check_graph_connectivity(graph));

    // Check edge weights match distances
    EXPECT_TRUE(deglib::analysis::check_graph_weights(graph));
}

TEST(SizeBoundedGraph, CreateRandomGraphUInt8) {
    const uint32_t vertex_count = 50;
    const uint8_t edges_per_vertex = 6;
    const uint32_t dim = 8;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim);
    uint8_t* feature_uint8 = reinterpret_cast<uint8_t*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_uint8[i * dim + d] = static_cast<uint8_t>((i + d) % 256);
        }
    }

    auto graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 42);

    EXPECT_EQ(graph.size(), vertex_count);
    EXPECT_EQ(graph.getEdgesPerVertex(), edges_per_vertex);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);

    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph, vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_connectivity(graph));
    EXPECT_TRUE(deglib::analysis::check_graph_weights(graph));
}

// ---------------------------------------------------------------------------
//  FromGraph: Copy from existing graph with weight recalculation
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, FromGraphSameFeatures) {
    const uint32_t vertex_count = 50;
    const uint8_t edges_per_vertex = 8;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Build a source graph using create_random_graph
    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Copy using from_graph with the same feature space
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.getEdgesPerVertex(), edges_per_vertex);
    EXPECT_EQ(copied_graph.capacity(), vertex_count);
    EXPECT_NE(dynamic_cast<const deglib::graph::MutableGraph*>(&copied_graph), nullptr);

    // Verify topology (neighbor indices) matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_neighbors = copied_graph.getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e])
                << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify edge weights are recalculated correctly
    const auto dist_func = space.get_dist_func();
    const auto dist_func_param = space.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                source_graph.getFeatureVector(i),
                source_graph.getFeatureVector(neighbor_idx),
                dist_func_param);
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-5f)
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify graph is mutable (addVertex/removeVertex work)
    EXPECT_TRUE(copied_graph.hasVertex(0));
    auto removed = copied_graph.removeVertex(0);
    EXPECT_EQ(removed.size(), edges_per_vertex);
    EXPECT_FALSE(copied_graph.hasVertex(0));
    EXPECT_EQ(copied_graph.size(), vertex_count - 1);

    // Add vertex back
    copied_graph.addVertex(0, feature_bytes.get());
    EXPECT_TRUE(copied_graph.hasVertex(0));
    EXPECT_EQ(copied_graph.size(), vertex_count);
}

TEST(SizeBoundedGraph, FromGraphDefaultFeatureSpace) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Copy using from_graph without specifying feature space (fast-path copy)
    const uint32_t new_capacity = 50;
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(source_graph, new_capacity);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.capacity(), new_capacity);
    EXPECT_EQ(copied_graph.getEdgesPerVertex(), edges_per_vertex);

    // Verify weights are copied directly
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_weights = source_graph.getNeighborWeights(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_weights[e], copy_weights[e])
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, FromGraphFromReadOnly) {
    const uint32_t vertex_count = 30;
    const uint8_t edges_per_vertex = 6;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Build a source SizeBoundedGraph
    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i * 2 + d);
        }
    }

    auto source_sbg = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Convert to ReadOnlyGraph
    auto readonly = deglib::graph::convert_to_readonly_graph(source_sbg);

    // Copy from ReadOnlyGraph using from_graph
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        readonly, space);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.getEdgesPerVertex(), edges_per_vertex);
    EXPECT_NE(dynamic_cast<const deglib::graph::MutableGraph*>(&copied_graph), nullptr);

    // Verify topology matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = readonly.getNeighborIndices(i);
        const auto* copy_neighbors = copied_graph.getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e])
                << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify weights are correct (ReadOnlyGraph has no weights, so they must be recalculated)
    const auto dist_func = space.get_dist_func();
    const auto dist_func_param = space.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = readonly.getNeighborIndices(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                readonly.getFeatureVector(i),
                readonly.getFeatureVector(neighbor_idx),
                dist_func_param);
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-5f)
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, FromGraphCustomFeatures) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Build a source graph
    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Create custom features with a different scale (multiply by 10)
    auto custom_feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* custom_floats = reinterpret_cast<float*>(custom_feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            custom_floats[i * dim + d] = static_cast<float>(i + d) * 10.0f;
        }
    }

    // Copy using from_graph with custom features
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space, custom_feature_bytes.get());

    EXPECT_EQ(copied_graph.size(), vertex_count);

    // Verify topology (neighbor indices) matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_neighbors = copied_graph.getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e])
                << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify edge weights are recalculated with the new (scaled) features
    const auto dist_func = space.get_dist_func();
    const auto dist_func_param = space.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                custom_feature_bytes.get() + size_t(i) * dim * sizeof(float),
                custom_feature_bytes.get() + size_t(neighbor_idx) * dim * sizeof(float),
                dist_func_param);
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-3f)
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, FromGraphNewMaxSize) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Copy with a larger capacity
    const uint32_t new_capacity = 100;
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space, nullptr, new_capacity);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.capacity(), new_capacity);
    EXPECT_NE(dynamic_cast<const deglib::graph::MutableGraph*>(&copied_graph), nullptr);

    // Should be able to add more vertices
    auto new_feature = make_float_bytes(make_vec_4d(100.0f, 0.0f, 0.0f, 0.0f));
    copied_graph.addVertex(9999, new_feature.get());
    EXPECT_EQ(copied_graph.size(), vertex_count + 1);
    EXPECT_TRUE(copied_graph.hasVertex(9999));
}

TEST(SizeBoundedGraph, FromGraphInnerProduct) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 8;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_InnerProduct);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d + 1);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Copy with same feature space (InnerProduct)
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_InnerProduct);

    // Verify weights are recalculated correctly using InnerProduct metric
    const auto dist_func = space.get_dist_func();
    const auto dist_func_param = space.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                source_graph.getFeatureVector(i),
                source_graph.getFeatureVector(neighbor_idx),
                dist_func_param);
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-5f)
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, FromGraphSearchable) {
    const uint32_t vertex_count = 50;
    const uint8_t edges_per_vertex = 8;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Copy using from_graph
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space);

    // Search should work and return the same results
    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto source_results = source_graph.search(std::span<const float>(query), 5, 0.0f);
    auto copied_results = copied_graph.search(std::span<const float>(query), 5, 0.0f);

    ASSERT_EQ(source_results.size(), copied_results.size());

    // Compare results (both should return the same internal indices)
    auto src_res = source_results;
    auto copy_res = copied_results;
    while (!src_res.empty() && !copy_res.empty()) {
        EXPECT_EQ(src_res.top().getIdentifier(), copy_res.top().getIdentifier());
        EXPECT_NEAR(src_res.top().getDistance(), copy_res.top().getDistance(), 1e-5f);
        src_res.pop();
        copy_res.pop();
    }
}

TEST(SizeBoundedGraph, FromGraphDifferentFeatureSpace) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space_l2(dim, deglib::distances::Metric::FP32_L2);
    deglib::distances::FloatSpace space_ip(dim, deglib::distances::Metric::FP32_InnerProduct);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto source_graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space_l2, 7);

    // Copy with a different feature space (InnerProduct)
    auto copied_graph = deglib::graph::SizeBoundedGraph::from_graph(
        source_graph, space_ip);

    EXPECT_EQ(copied_graph.size(), vertex_count);
    EXPECT_EQ(copied_graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_InnerProduct);

    // Verify topology matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_neighbors = copied_graph.getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e])
                << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify weights are recalculated with InnerProduct metric
    const auto dist_func = space_ip.get_dist_func();
    const auto dist_func_param = space_ip.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.getNeighborIndices(i);
        const auto* copy_weights = copied_graph.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                source_graph.getFeatureVector(i),
                source_graph.getFeatureVector(neighbor_idx),
                dist_func_param);
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-5f)
                << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, CreateRandomGraphSearchable) {
    const uint32_t vertex_count = 30;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i * 10 + d);
        }
    }

    auto graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Use explore from vertex 0 (internal index 0) — should find vertex 0 as nearest
    auto results = graph.explore(0, 5, 0, 0.0f, /*include_entry=*/true, nullptr);
    EXPECT_GT(results.size(), 0u);

    // Vertex 0 should be in the results (it's the entry point)
    bool found_vertex_0 = false;
    while (!results.empty()) {
        if (results.top().getIdentifier() == 0u) {
            found_vertex_0 = true;
        }
        results.pop();
    }
    EXPECT_TRUE(found_vertex_0) << "Vertex 0 should be in explore results";
}

TEST(SizeBoundedGraph, CreateRandomGraphDeterministic) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto graph1 = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 123);
    auto graph2 = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 123);

    // Same seed should produce same graph
    EXPECT_EQ(graph1.size(), graph2.size());
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* n1 = graph1.getNeighborIndices(i);
        const auto* n2 = graph2.getNeighborIndices(i);
        const auto* w1 = graph1.getNeighborWeights(i);
        const auto* w2 = graph2.getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(n1[e], n2[e]) << "Neighbor mismatch at vertex " << i << " edge " << e;
            EXPECT_NEAR(w1[e], w2[e], 1e-6f) << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(SizeBoundedGraph, CreateRandomGraphDifferentSeeds) {
    const uint32_t vertex_count = 20;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 4;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d);
        }
    }

    auto graph1 = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 1);
    auto graph2 = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 999);

    // Different seeds may produce different graphs (not guaranteed, but likely)
    // At minimum, both should be valid
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph1, vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph2, vertex_count, true));
}

TEST(SizeBoundedGraph, CreateRandomGraphInnerProduct) {
    const uint32_t vertex_count = 30;
    const uint8_t edges_per_vertex = 4;
    const uint32_t dim = 8;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_InnerProduct);

    auto feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* feature_floats = reinterpret_cast<float*>(feature_bytes.get());
    // Normalize for inner product
    for (uint32_t i = 0; i < vertex_count; i++) {
        float norm = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] = static_cast<float>(i + d + 1);
            norm += feature_floats[i * dim + d] * feature_floats[i * dim + d];
        }
        norm = std::sqrt(norm);
        for (uint32_t d = 0; d < dim; d++) {
            feature_floats[i * dim + d] /= norm;
        }
    }

    auto graph = deglib::graph::SizeBoundedGraph::create_random_graph(
        feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    EXPECT_EQ(graph.size(), vertex_count);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph, vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_connectivity(graph));
    EXPECT_TRUE(deglib::analysis::check_graph_weights(graph));
}
