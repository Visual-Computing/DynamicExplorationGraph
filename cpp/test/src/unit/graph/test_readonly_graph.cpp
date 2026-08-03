// test_readonly_graph.cpp — Unit tests for ReadOnlyGraph

#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include <algorithm>

#include "gtest/gtest.h"
#include "graph/readonly_graph.h"
#include "graph/sizebounded_graph.h"

namespace {

inline std::vector<float> make_float_vec(size_t n, int seed = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>((seed + static_cast<int>(i)) % 100);
    }
    return v;
}

inline std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
    return bytes;
}

// Helper: set up edges for a SizeBoundedGraph using changeEdges (bypasses self-loop requirement)
void set_edges(deglib::graph::SizeBoundedGraph& graph, uint32_t vertex,
               const std::vector<uint32_t>& neighbors, const std::vector<float>& weights) {
    graph.changeEdges(vertex, neighbors.data(), weights.data());
}

} // anonymous namespace

namespace {

TEST(ReadOnlyGraph, Construction) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraph graph(10, 4, space);
    EXPECT_EQ(graph.capacity(), 10u);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
}

TEST(ReadOnlyGraph, ConstructionOddEdges) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    EXPECT_THROW((deglib::graph::ReadOnlyGraph(10, 3, space)), std::invalid_argument);
}

TEST(ReadOnlyGraph, CopyFromSizeBoundedGraph) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 4; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // add edges using changeEdges (avoids self-loop requirement of changeEdge)
    // neighbors must be sorted ascending
    set_edges(mutable_graph, 0, {0, 0, 1, 2}, {0.0f, 0.0f, 1.0f, 2.0f});
    set_edges(mutable_graph, 1, {0, 1, 1, 1}, {1.0f, 0.0f, 0.0f, 0.0f});
    set_edges(mutable_graph, 2, {0, 2, 2, 2}, {2.0f, 0.0f, 0.0f, 0.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    EXPECT_EQ(graph.size(), 4u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);

    // check labels
    EXPECT_TRUE(graph.hasVertex(0));
    EXPECT_TRUE(graph.hasVertex(1));
    EXPECT_TRUE(graph.hasVertex(2));
    EXPECT_TRUE(graph.hasVertex(3));
    EXPECT_FALSE(graph.hasVertex(99));

    // check edges
    EXPECT_TRUE(graph.hasEdge(graph.getInternalIndex(0), graph.getInternalIndex(1)));
    EXPECT_TRUE(graph.hasEdge(graph.getInternalIndex(1), graph.getInternalIndex(0)));
    EXPECT_TRUE(graph.hasEdge(graph.getInternalIndex(0), graph.getInternalIndex(2)));
    EXPECT_TRUE(graph.hasEdge(graph.getInternalIndex(2), graph.getInternalIndex(0)));
}

TEST(ReadOnlyGraph, InternalExternalIndex) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 4; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i * 10, make_float_bytes(v).get());
    }

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    for (int i = 0; i < 4; ++i) {
        uint32_t ext = i * 10;
        uint32_t internal = graph.getInternalIndex(ext);
        EXPECT_EQ(graph.getExternalLabel(internal), ext);
    }
}

TEST(ReadOnlyGraph, FeatureVector) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    std::vector<float> expected(4, 42.0f);
    mutable_graph.addVertex(0, make_float_bytes(expected).get());

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    const auto* feature = reinterpret_cast<const float*>(graph.getFeatureVector(0));
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(feature[i], expected[i]);
    }
}

TEST(ReadOnlyGraph, NeighborIndices) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    for (int i = 0; i < 4; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // fully connected graph (neighbors must be sorted ascending)
    set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    const auto* neighbors = graph.getNeighborIndices(0);
    // neighbors are sorted, self-loop at index 0
    EXPECT_EQ(neighbors[0], 0u);
    EXPECT_EQ(neighbors[1], 1u);
    EXPECT_EQ(neighbors[2], 2u);
    EXPECT_EQ(neighbors[3], 3u);
}

TEST(ReadOnlyGraph, HasEdge) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 3; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // add edge 0-1 (neighbors must be sorted ascending)
    set_edges(mutable_graph, 0, {0, 0, 0, 1}, {0.0f, 0.0f, 0.0f, 1.0f});
    set_edges(mutable_graph, 1, {0, 1, 1, 1}, {1.0f, 0.0f, 0.0f, 0.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    EXPECT_TRUE(graph.hasEdge(0, 1));
    EXPECT_TRUE(graph.hasEdge(1, 0));
    EXPECT_FALSE(graph.hasEdge(0, 2));
}

TEST(ReadOnlyGraph, Explore) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    // create 4 vertices with distinct features
    auto v0 = make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    auto v1 = make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    auto v2 = make_float_bytes(std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    auto v3 = make_float_bytes(std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f});
    
    mutable_graph.addVertex(0, v0.get());
    mutable_graph.addVertex(1, v1.get());
    mutable_graph.addVertex(2, v2.get());
    mutable_graph.addVertex(3, v3.get());

    // fully connected graph (neighbors must be sorted ascending)
    set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 1.0f, 1.0f});
    set_edges(mutable_graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.414f, 1.414f});
    set_edges(mutable_graph, 2, {0, 1, 2, 3}, {1.0f, 1.414f, 0.0f, 1.414f});
    set_edges(mutable_graph, 3, {0, 1, 2, 3}, {1.0f, 1.414f, 1.414f, 0.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    // verify graph has 4 vertices
    EXPECT_EQ(graph.size(), 4u);

    // debug: check neighbors of vertex 0
    const auto* neighbors = graph.getNeighborIndices(0);
    // neighbors should be {0, 1, 2, 3}
    EXPECT_EQ(neighbors[0], 0u);
    EXPECT_EQ(neighbors[1], 1u);
    EXPECT_EQ(neighbors[2], 2u);
    EXPECT_EQ(neighbors[3], 3u);

    // debug: check that hasEdge works
    EXPECT_TRUE(graph.hasEdge(0, 1));
    EXPECT_TRUE(graph.hasEdge(0, 2));
    EXPECT_TRUE(graph.hasEdge(0, 3));

    // explore from vertex 0, find 3 nearest neighbors (excluding entry)
    auto results = graph.explore(0, 3, false);
    
    // all 3 other vertices should be found (distance 1.0 each)
    EXPECT_EQ(results.size(), 3u);

    // all other vertices should be found
    std::unordered_set<uint32_t> found;
    while (!results.empty()) {
        found.insert(results.top().getInternalIndex());
        results.pop();
    }
    EXPECT_TRUE(found.contains(1));
    EXPECT_TRUE(found.contains(2));
    EXPECT_TRUE(found.contains(3));
}

TEST(ReadOnlyGraph, Search) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    // create vertices close together in a cluster
    for (int i = 0; i < 8; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);  // small distances between vertices
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // connect each vertex to its neighbors so the graph is connected
    // (neighbors must be sorted ascending)
    for (int i = 0; i < 8; ++i) {
        std::vector<uint32_t> neighbors;
        std::vector<float> weights;
        // connect to previous and next vertices to ensure connectivity
        if (i > 0) {
            neighbors.push_back(static_cast<uint32_t>(i - 1));
            weights.push_back(1.0f);
        }
        if (i > 1) {
            neighbors.push_back(static_cast<uint32_t>(i - 2));
            weights.push_back(1.0f);
        }
        if (i < 7) {
            neighbors.push_back(static_cast<uint32_t>(i + 1));
            weights.push_back(1.0f);
        }
        if (i < 6) {
            neighbors.push_back(static_cast<uint32_t>(i + 2));
            weights.push_back(1.0f);
        }
        // fill remaining with self
        while (neighbors.size() < 4) {
            neighbors.push_back(static_cast<uint32_t>(i));
            weights.push_back(0.0f);
        }
        // sort neighbors to satisfy binary_search requirement
        std::vector<uint32_t> sorted_neighbors = neighbors;
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
        std::vector<float> sorted_weights(weights.size());
        for (size_t k = 0; k < neighbors.size(); ++k) {
            auto it = std::find(sorted_neighbors.begin(), sorted_neighbors.end(), neighbors[k]);
            size_t idx = std::distance(sorted_neighbors.begin(), it);
            sorted_weights[idx] = weights[k];
        }
        set_edges(mutable_graph, static_cast<uint32_t>(i), sorted_neighbors, sorted_weights);
    }

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    // search for something close to vertex 4
    std::vector<float> query = {4.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search({0}, make_float_bytes(query).get(), 0.1f, 3);

    EXPECT_GT(results.size(), 0u);
    // ResultSet is a max-heap (std::less), so top() returns the worst of the top-k results.
    // Find the best (minimum distance) result by iterating through all results.
    auto best = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.getDistance() < b.getDistance(); });
    EXPECT_EQ(best->getInternalIndex(), graph.getInternalIndex(4));
}

TEST(ReadOnlyGraph, HasPath) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    // create a line: 0 -- 1 -- 2 -- 3
    for (int i = 0; i < 4; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i) * 10.0f;
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // set up edges: 0-1, 1-2, 2-3 (neighbors must be sorted ascending)
    set_edges(mutable_graph, 0, {0, 0, 0, 1}, {0.0f, 0.0f, 0.0f, 10.0f});
    set_edges(mutable_graph, 1, {0, 1, 1, 2}, {10.0f, 0.0f, 0.0f, 10.0f});
    set_edges(mutable_graph, 2, {1, 2, 2, 3}, {10.0f, 0.0f, 0.0f, 10.0f});
    set_edges(mutable_graph, 3, {2, 3, 3, 3}, {10.0f, 0.0f, 0.0f, 0.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto path = graph.hasPath({0}, graph.getInternalIndex(3), 0.1f, 1);
    EXPECT_FALSE(path.empty());
}

TEST(ReadOnlyGraph, HasPathNoConnection) {
    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    // create two disconnected components
    for (int i = 0; i < 4; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i) * 10.0f;
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    // only connect 0-1
    set_edges(mutable_graph, 0, {0, 0, 0, 1}, {0.0f, 0.0f, 0.0f, 10.0f});
    set_edges(mutable_graph, 1, {0, 1, 1, 1}, {10.0f, 0.0f, 0.0f, 0.0f});

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto path = graph.hasPath({0}, graph.getInternalIndex(3), 0.1f, 1);
    EXPECT_TRUE(path.empty());
}

} // anonymous namespace
