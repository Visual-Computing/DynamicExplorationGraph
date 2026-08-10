// test_readonly_graph.cpp — Unit tests for ReadOnlyGraph

#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include <algorithm>

#include "gtest/gtest.h"
#include "deglib/graph/readonly_graph.h"
#include "deglib/graph/sizebounded_graph.h"

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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::ReadOnlyGraph graph(10, 4, space);
    EXPECT_EQ(graph.capacity(), 10u);
    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
}

TEST(ReadOnlyGraph, ConstructionOddEdges) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    EXPECT_THROW((deglib::graph::ReadOnlyGraph(10, 3, space)), std::invalid_argument);
}

TEST(ReadOnlyGraph, CopyFromSizeBoundedGraph) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    auto results = graph.explore(0, 3, 0, 0.0f, /*include_entry=*/false);
    
    // all 3 other vertices should be found (distance 1.0 each)
    EXPECT_EQ(results.size(), 3u);

    // all other vertices should be found
    std::unordered_set<uint32_t> found;
    while (!results.empty()) {
        found.insert(results.top().getIdentifier());
        results.pop();
    }
    EXPECT_TRUE(found.contains(1));
    EXPECT_TRUE(found.contains(2));
    EXPECT_TRUE(found.contains(3));
}

TEST(ReadOnlyGraph, Search) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
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
    auto results = graph.search(std::span<const float>(query), 3, 0.1f);

    EXPECT_GT(results.size(), 0u);
    // ResultSet is a max-heap (std::less), so top() returns the worst of the top-k results.
    // Find the best (minimum distance) result by iterating through all results.
    auto best = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.getDistance() < b.getDistance(); });
    EXPECT_EQ(best->getIdentifier(), graph.getInternalIndex(4));
}

TEST(ReadOnlyGraph, HasPath) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 4.0f, 9.0f});
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto path = graph.hasPath({0}, 1, 0.0f, 5);
    EXPECT_GT(path.size(), 0u);
}

TEST(ReadOnlyGraph, HasPathNoConnection) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto path = graph.hasPath({0}, 4, 0.0f, 5);
    EXPECT_EQ(path.size(), static_cast<size_t>(0));
}

TEST(ReadOnlyGraph, ExploreBasicIncludeEntry) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 4.0f, 9.0f});
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

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

TEST(ReadOnlyGraph, ExploreExcludeEntry) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

    for (int i = 0; i < 5; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 4.0f, 9.0f});
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto results = graph.explore(0, 2, 0, 0.0f, /*include_entry=*/false, nullptr);
    EXPECT_GT(results.size(), 0u);
}

TEST(ReadOnlyGraph, ExploreWithMaxDistanceCount) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
    deglib::graph::SizeBoundedGraph mutable_graph(10, 4, space);

    for (int i = 0; i < 10; ++i) {
        auto v = make_float_vec(4);
        v[0] = static_cast<float>(i);
        mutable_graph.addVertex(i, make_float_bytes(v).get());
    }

    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

    auto results = graph.explore(0, 5, 2, 0.0f, /*include_entry=*/true, nullptr);
    EXPECT_GT(results.size(), 0u);
    EXPECT_LE(results.size(), 5u);
}
// ---------------------------------------------------------------------------
//  Internal vs External ID Verification
// ---------------------------------------------------------------------------
//
// These tests verify that ReadOnlyGraph (as an InternalGraph implementation)
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

TEST(ReadOnlyGraphInternalIndicesInSearch, SearchReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
   deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

   // Add vertices with non-sequential external labels at distinct positions
   mutable_graph.addVertex(kExtLabels[0], make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[1], make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[2], make_float_bytes(std::vector<float>{2.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[3], make_float_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[4], make_float_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 0.0f}).get());

   // Fully connect vertices 0..3 (neighbors must be sorted ascending)
   set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
   set_edges(mutable_graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
   set_edges(mutable_graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
   set_edges(mutable_graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});
   // Vertex 4 has self-loop only

   deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

   // search() on InternalGraph returns internal indices in the ResultSet
   std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
   auto results = graph.search(std::span<const float>(query), 3, 0.0f);

   ASSERT_GT(results.size(), 0u);

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

TEST(ReadOnlyGraphInternalIndicesInSearch, ExploreReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
   deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

   mutable_graph.addVertex(kExtLabels[0], make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[1], make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[2], make_float_bytes(std::vector<float>{2.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[3], make_float_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[4], make_float_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 0.0f}).get());

   set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
   set_edges(mutable_graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
   set_edges(mutable_graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
   set_edges(mutable_graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});

   deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

   // explore() takes an internal index and returns internal indices
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

TEST(ReadOnlyGraphInternalIndicesInSearch, HasPathReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
   deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

   mutable_graph.addVertex(kExtLabels[0], make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[1], make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[2], make_float_bytes(std::vector<float>{2.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[3], make_float_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[4], make_float_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 0.0f}).get());

   set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
   set_edges(mutable_graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
   set_edges(mutable_graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
   set_edges(mutable_graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});

   deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

   // hasPath() takes internal entry indices and a internal to_vertex index
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

TEST(ReadOnlyGraphInternalIndicesInSearch, GetNeighborIndicesReturnsInternalIndices) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
   deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

   mutable_graph.addVertex(kExtLabels[0], make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[1], make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[2], make_float_bytes(std::vector<float>{2.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[3], make_float_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[4], make_float_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 0.0f}).get());

   set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});

   deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), 4, space, mutable_graph);

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

TEST(ReadOnlyGraphSaveLoadLabelPreservation, SaveLoadPreservesExternalLabels) {
   deglib::distances::FloatSpace space(4, deglib::distances::Metric::L2);
   deglib::graph::SizeBoundedGraph mutable_graph(5, 4, space);

   // Add vertices with non-sequential external labels
   mutable_graph.addVertex(kExtLabels[0], make_float_bytes(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[1], make_float_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[2], make_float_bytes(std::vector<float>{2.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[3], make_float_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 0.0f}).get());
   mutable_graph.addVertex(kExtLabels[4], make_float_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 0.0f}).get());

   set_edges(mutable_graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
   set_edges(mutable_graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
   set_edges(mutable_graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
   set_edges(mutable_graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});

   // Save the SizeBoundedGraph to disk
   std::string path = "test/readonly_label_preservation_test.deg";
   bool saved = mutable_graph.saveGraph(path.c_str());
   EXPECT_TRUE(saved);

   // Load as ReadOnlyGraph
   auto loaded = deglib::graph::load_readonly_graph(path.c_str());

   // Verify external labels are preserved after save/load
   EXPECT_EQ(loaded.size(), 5u);
   EXPECT_EQ(loaded.getExternalLabel(0), kExtLabels[0]);
   EXPECT_EQ(loaded.getExternalLabel(1), kExtLabels[1]);
   EXPECT_EQ(loaded.getExternalLabel(2), kExtLabels[2]);
   EXPECT_EQ(loaded.getExternalLabel(3), kExtLabels[3]);
   EXPECT_EQ(loaded.getExternalLabel(4), kExtLabels[4]);

   // Verify bidirectional mapping is preserved
   EXPECT_EQ(loaded.getInternalIndex(kExtLabels[0]), 0u);
   EXPECT_EQ(loaded.getInternalIndex(kExtLabels[1]), 1u);
   EXPECT_EQ(loaded.getInternalIndex(kExtLabels[2]), 2u);
   EXPECT_EQ(loaded.getInternalIndex(kExtLabels[3]), 3u);
   EXPECT_EQ(loaded.getInternalIndex(kExtLabels[4]), 4u);

   // Verify hasVertex works with external labels
   for (uint32_t label : kExtLabels) {
       EXPECT_TRUE(loaded.hasVertex(label));
   }

   // Verify getNeighborIndices returns internal indices (not external labels)
   const auto* neighbors = loaded.getNeighborIndices(0);
   uint8_t epv = loaded.getEdgesPerVertex();
   std::unordered_set<uint32_t> internal_indices = {0, 1, 2, 3, 4};
   for (uint8_t i = 0; i < epv; ++i) {
       EXPECT_TRUE(internal_indices.contains(neighbors[i]))
           << "getNeighborIndices()[ " << i << "] = " << neighbors[i]
           << " is not a valid internal index after save/load";
   }

   std::filesystem::remove(path);
}

} // anonymous namespace
