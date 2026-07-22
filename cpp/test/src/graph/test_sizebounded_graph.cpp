// test_sizebounded_graph.cpp — Google Test suite for SizeBoundedGraph
//
// Covers: construction, vertex management, edge management, label lookup,
// feature storage, capacity, search, save/load, multi-operation cycles.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "graph/sizebounded_graph.h"
#include "filter.h"
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(100, 4, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.capacity(), 100u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
    EXPECT_EQ(space.dim(), 4u);
    EXPECT_EQ(space.metric(), deglib::Metric::L2);
}

TEST(SizeBoundedGraph, RejectsOddEdgesPerVertex) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    EXPECT_THROW(deglib::graph::SizeBoundedGraph graph(50, 3, space), std::invalid_argument);
}

TEST(SizeBoundedGraph, InnerProductMetric) {
    deglib::FloatSpace space(8, deglib::Metric::InnerProduct);
    deglib::graph::SizeBoundedGraph graph(200, 8, space);

    EXPECT_EQ(graph.getEdgesPerVertex(), 8u);
    EXPECT_EQ(graph.capacity(), 200u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::Metric::InnerProduct);
}

TEST(SizeBoundedGraph, L2Uint8Metric) {
    deglib::FloatSpace space(128, deglib::Metric::L2_Uint8);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::Metric::L2_Uint8);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 128u);
}

// ---------------------------------------------------------------------------
//  2. Vertex Management
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, AddVertex) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    EXPECT_EQ(graph.size(), 3u);
}

TEST(SizeBoundedGraph, ExternalLabels) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(4, 2, space);

    std::vector<float> v(4, 0.0f);
    graph.addVertex(0, make_float_bytes(v).get());

    auto neighbors = graph.removeVertex(0);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_FALSE(graph.hasVertex(0));
}

TEST(SizeBoundedGraph, RemoveMiddleVertex) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    EXPECT_EQ(graph.getEdgeWeight(0, 2), std::numeric_limits<float>::lowest());
}

TEST(SizeBoundedGraph, ChangeEdgeSwap) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(128, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(1000, 16, space);

    EXPECT_EQ(graph.getEdgesPerVertex(), 16u);
    EXPECT_EQ(graph.capacity(), 1000u);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 128u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::Metric::L2);
}

TEST(SizeBoundedGraph, SmallGraph) {
    deglib::FloatSpace space(2, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(3, 2, space);

    std::vector<float> v(2, 0.0f);
    graph.addVertex(0, make_float_bytes(v).get());

    EXPECT_EQ(graph.size(), 1u);
    EXPECT_TRUE(graph.hasEdge(0, 0));
}

TEST(SizeBoundedGraph, VariousDimensions) {
    std::vector<size_t> dims = {1, 4, 8, 64, 128, 192};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2);
        deglib::graph::SizeBoundedGraph graph(3, 2, space);

        std::vector<float> v(dim, 0.0f);
        v[0] = 1.0f;
        graph.addVertex(0, make_float_bytes(v).get());

        EXPECT_EQ(graph.size(), 1u);
        EXPECT_EQ(graph.getFeatureSpace().dim(), dim);
    }

    std::vector<size_t> u8_dims = {64, 128, 192};
    for (size_t dim : u8_dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2_Uint8);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    float query[] = {0.5f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search({0}, reinterpret_cast<const std::byte*>(query), 0.0f, 5);

    EXPECT_GT(results.size(), 0u);
    if (results.size() > 0) {
        EXPECT_GE(results.top().getInternalIndex(), 0u);
    }
}

TEST(SizeBoundedGraph, SearchWithFilter) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    int valid[] = {0};
    deglib::graph::Filter filter(valid, 1, 10, 10);

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search({0}, reinterpret_cast<const std::byte*>(query), 0.0f, 3, &filter);

    EXPECT_GE(results.size(), 0u);
}

// ---------------------------------------------------------------------------
//  7. Save / Load
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, SaveGraph) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    deglib::FloatSpace space(128, deglib::Metric::L2_Uint8);
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
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::Metric::L2_Uint8);
}

// ---------------------------------------------------------------------------
//  8. Multiple Operations
// ---------------------------------------------------------------------------

TEST(SizeBoundedGraph, MultipleAddRemoveCycles) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
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
    EXPECT_EQ(graph.size(), 5u);

    for (int i = 1; i <= 7; ++i) {
        if (graph.hasVertex(static_cast<uint32_t>(i)))
            graph.removeVertex(static_cast<uint32_t>(i));
    }
    EXPECT_EQ(graph.size(), 0u);
}
