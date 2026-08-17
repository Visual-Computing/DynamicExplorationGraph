// test_graph.cpp — Unit tests for DynamicExplorationGraph facade
//
// Verifies that DynamicExplorationGraph strictly uses external labels (User Object IDs)
// for all inputs and outputs, while the underlying InternalGraph implementations
// operate strictly on internal indices (0..N-1).
//
// Non-sequential external labels (1005, 9999, 42, 707, 12345) are used so that
// any accidental swap of external labels and internal indices will instantly
// fail test assertions.

#include "deglib/analysis.h"
#include "deglib/graph.h"
#include "deglib/graph/mutable_graph.h"
#include "deglib/graph/readonly_graph.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

// Non-sequential, arbitrary external labels — deliberately distinct from 0..N-1
constexpr std::array<uint32_t, 5> kExternalLabels = {1005, 9999, 42, 707, 12345};

// Internal indices are always 0..N-1
constexpr std::array<uint32_t, 5> kInternalIndices = {0, 1, 2, 3, 4};

inline std::vector<float> make_vec_4d(float x, float y, float z, float w) { return {x, y, z, w}; }

inline std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
    return bytes;
}

// Helper: set up edges for a MutableGraph using changeEdges (bypasses self-loop requirement)
void set_edges(deglib::graph::MutableGraph& graph, uint32_t vertex, const std::vector<uint32_t>& neighbors, const std::vector<float>& weights) {
    graph.changeEdges(vertex, neighbors.data(), weights.data());
}

// Build a fully-connected mutable graph with non-sequential external labels.
// Vertices are placed at distinct positions so search/explore return deterministic results.
// Internal indices: 0..4, External labels: 1005, 9999, 42, 707, 12345
deglib::graph::SizeBoundedGraph build_test_graph() {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    // Place vertices at distinct 4D positions
    graph.addVertex(kExternalLabels[0], make_float_bytes(make_vec_4d(0.0f, 0.0f, 0.0f, 0.0f)).get());
    graph.addVertex(kExternalLabels[1], make_float_bytes(make_vec_4d(1.0f, 0.0f, 0.0f, 0.0f)).get());
    graph.addVertex(kExternalLabels[2], make_float_bytes(make_vec_4d(2.0f, 0.0f, 0.0f, 0.0f)).get());
    graph.addVertex(kExternalLabels[3], make_float_bytes(make_vec_4d(3.0f, 0.0f, 0.0f, 0.0f)).get());
    graph.addVertex(kExternalLabels[4], make_float_bytes(make_vec_4d(4.0f, 0.0f, 0.0f, 0.0f)).get());

    // Fully connect: each vertex connects to all others (neighbors must be sorted ascending)
    // Internal index 0 connects to 0(self), 1, 2, 3
    set_edges(graph, 0, {0, 1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f});
    // Internal index 1 connects to 0, 1(self), 2, 3
    set_edges(graph, 1, {0, 1, 2, 3}, {1.0f, 0.0f, 1.0f, 2.0f});
    // Internal index 2 connects to 0, 1, 2(self), 3
    set_edges(graph, 2, {0, 1, 2, 3}, {2.0f, 1.0f, 0.0f, 1.0f});
    // Internal index 3 connects to 0, 1, 2, 3(self)
    set_edges(graph, 3, {0, 1, 2, 3}, {3.0f, 2.0f, 1.0f, 0.0f});
    // Internal index 4 has self-loop only (not connected to others)
    set_edges(graph, 4, {4, 4, 4, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    return graph;
}

}  // anonymous namespace

// ===========================================================================
//  DynamicExplorationGraph: search() returns external labels
// ===========================================================================

TEST(DEGSearchReturnsExternalLabels, SearchReturnsExternalLabels) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Query near vertex 0 (external label 1005, position 0.0)
    std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
    auto results = deg.search(std::span<const float>(query), 3, 0.0f);

    ASSERT_GT(results.size(), 0u);

    // Every identifier in the result must be an external label, not an internal index
    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        // Must be a valid external label
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        // Must NOT be an internal index (0..4)
        EXPECT_FALSE(internal_indices.contains(id)) << "Result identifier " << id << " is an internal index, not an external label";
        results.pop();
    }
}

TEST(DEGSearchReturnsExternalLabels, SearchFindsCorrectExternalLabel) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Query exactly at vertex 0's position (external label 1005)
    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = deg.search(std::span<const float>(query), 1, 0.0f);

    ASSERT_GE(results.size(), 1u);
    // The nearest neighbor should be vertex 0 with external label 1005
    EXPECT_EQ(results.top().getIdentifier(), kExternalLabels[0]);
    EXPECT_EQ(results.top().getIdentifier(), 1005u);
}

// ===========================================================================
//  DynamicExplorationGraph: explore() accepts external label, returns external labels
// ===========================================================================

TEST(DEGExploreReturnsExternalLabels, ExploreAcceptsExternalLabelAndReturnsExternalLabels) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Explore starting from external label 1005 (internal index 0)
    auto results = deg.explore(kExternalLabels[0], 3, 0, 0.0f, /*include_entry=*/true, nullptr);

    ASSERT_GT(results.size(), 0u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        EXPECT_FALSE(internal_indices.contains(id)) << "Result identifier " << id << " is an internal index, not an external label";
        results.pop();
    }
}

TEST(DEGExploreReturnsExternalLabels, ExploreFromDifferentExternalLabel) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Explore from external label 9999 (internal index 1, position 1.0)
    auto results = deg.explore(kExternalLabels[1], 2, 0, 0.0f, /*include_entry=*/false, nullptr);

    ASSERT_GT(results.size(), 0u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        EXPECT_FALSE(internal_indices.contains(id)) << "Result identifier " << id << " is an internal index, not an external label";
        results.pop();
    }
}

TEST(DEGExploreReturnsExternalLabels, ExploreIncludeEntryReturnsExternalLabel) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // When include_entry=true, the entry vertex's external label must appear in results
    auto results = deg.explore(kExternalLabels[2], 4, 0, 0.0f, /*include_entry=*/true, nullptr);

    ASSERT_GT(results.size(), 0u);

    bool found_entry = false;
    while (!results.empty()) {
        if (results.top().getIdentifier() == kExternalLabels[2]) {
            found_entry = true;
        }
        results.pop();
    }
    EXPECT_TRUE(found_entry) << "Entry vertex external label " << kExternalLabels[2] << " should be in results when include_entry=true";
}

// ===========================================================================
//  DynamicExplorationGraph: getNeighbors() accepts external label, returns external labels
// ===========================================================================

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsAcceptsExternalLabel) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Get neighbors of vertex with external label 1005 (internal index 0)
    // Internal index 0 has neighbors {0, 1, 2, 3} → external labels {1005, 9999, 42, 707}
    auto neighbors = deg.getNeighbors(kExternalLabels[0]);

    ASSERT_EQ(neighbors.size(), 4u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    for (uint32_t n : neighbors) {
        EXPECT_TRUE(valid_labels.contains(n)) << "Neighbor " << n << " is not a valid external label";
        EXPECT_FALSE(internal_indices.contains(n)) << "Neighbor " << n << " is an internal index, not an external label";
    }
}

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsReturnsCorrectExternalLabels) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Vertex 0 (external label 1005) has internal neighbors {0, 1, 2, 3}
    // which map to external labels {1005, 9999, 42, 707}
    auto neighbors = deg.getNeighbors(kExternalLabels[0]);

    ASSERT_EQ(neighbors.size(), 4u);

    std::unordered_set<uint32_t> expected = {1005, 9999, 42, 707};
    std::unordered_set<uint32_t> actual(neighbors.begin(), neighbors.end());

    EXPECT_EQ(actual, expected);
}

TEST(DEGGetNeighborsReturnsExternalLabels, GetNeighborsFromDifferentVertex) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Vertex 1 (external label 9999) has internal neighbors {0, 1, 2, 3}
    // which map to external labels {1005, 9999, 42, 707}
    auto neighbors = deg.getNeighbors(kExternalLabels[1]);

    ASSERT_EQ(neighbors.size(), 4u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    for (uint32_t n : neighbors) {
        EXPECT_TRUE(valid_labels.contains(n)) << "Neighbor " << n << " is not a valid external label";
    }
}

// ===========================================================================
//  DynamicExplorationGraph: hasVertex() uses external labels
// ===========================================================================

TEST(DEGHasVertexUsesExternalLabels, HasVertexTrueForValidExternalLabels) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    for (uint32_t label : kExternalLabels) {
        EXPECT_TRUE(deg.hasVertex(label)) << "hasVertex(" << label << ") should return true for valid external label";
    }
}

TEST(DEGHasVertexUsesExternalLabels, HasVertexFalseForInvalidExternalLabels) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Labels that are not in the graph
    EXPECT_FALSE(deg.hasVertex(0u));
    EXPECT_FALSE(deg.hasVertex(1u));
    EXPECT_FALSE(deg.hasVertex(999u));
    EXPECT_FALSE(deg.hasVertex(12346u));
}

TEST(DEGHasVertexUsesExternalLabels, HasVertexFalseForInternalIndices) {
    auto graph = build_test_graph();
    deglib::DynamicExplorationGraph deg(graph);

    // Internal indices 0..4 should NOT be valid external labels
    for (uint32_t idx : kInternalIndices) {
        EXPECT_FALSE(deg.hasVertex(idx)) << "hasVertex(" << idx << ") should return false — " << idx << " is an internal index, not an external label";
    }
}

// ===========================================================================
//  DynamicExplorationGraph: search() with ReadOnlyGraph backend
// ===========================================================================

TEST(DEGSearchReturnsExternalLabels, SearchWithReadOnlyGraphBackend) {
    auto size_bounded = build_test_graph();
    auto readonly = deglib::graph::convert_to_readonly_graph(size_bounded);
    deglib::DynamicExplorationGraph deg(readonly);

    // Query near vertex 0 (external label 1005)
    std::vector<float> query = {0.1f, 0.0f, 0.0f, 0.0f};
    auto results = deg.search(std::span<const float>(query), 3, 0.0f);

    ASSERT_GT(results.size(), 0u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        EXPECT_FALSE(internal_indices.contains(id)) << "Result identifier " << id << " is an internal index, not an external label";
        results.pop();
    }
}

TEST(DEGExploreReturnsExternalLabels, ExploreWithReadOnlyGraphBackend) {
    auto size_bounded = build_test_graph();
    auto readonly = deglib::graph::convert_to_readonly_graph(size_bounded);
    deglib::DynamicExplorationGraph deg(readonly);

    // Explore from external label 42 (internal index 2)
    auto results = deg.explore(kExternalLabels[2], 3, 0, 0.0f, /*include_entry=*/true, nullptr);

    ASSERT_GT(results.size(), 0u);

    std::unordered_set<uint32_t> valid_labels(kExternalLabels.begin(), kExternalLabels.end());
    std::unordered_set<uint32_t> internal_indices(kInternalIndices.begin(), kInternalIndices.end());

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        EXPECT_FALSE(internal_indices.contains(id)) << "Result identifier " << id << " is an internal index, not an external label";
        results.pop();
    }
}

// ===========================================================================
//  DynamicExplorationGraph: create_empty and create_random_graph
// ===========================================================================

TEST(DynamicExplorationGraphCreateEmpty, CreatesEmptyGraph) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::DynamicExplorationGraph::create_empty(100, 4, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 4u);
    EXPECT_EQ(graph.getFeatureSpace().dim(), 4u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::FP32_L2);
    EXPECT_TRUE(graph.isMutable());
}

TEST(DynamicExplorationGraphCreateEmpty, CreatesEmptyGraphUint8) {
    deglib::distances::FloatSpace space(128, deglib::distances::Metric::Uint8_L2);
    auto graph = deglib::DynamicExplorationGraph::create_empty(50, 6, space);

    EXPECT_EQ(graph.size(), 0u);
    EXPECT_EQ(graph.getEdgesPerVertex(), 6u);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);
    EXPECT_TRUE(graph.isMutable());
}

TEST(DynamicExplorationGraphCreateRandomGraph, CreatesValidGraph) {
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

    auto graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    EXPECT_EQ(graph.size(), vertex_count);
    EXPECT_EQ(graph.getEdgesPerVertex(), edges_per_vertex);
    EXPECT_TRUE(graph.isMutable());

    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph.internal(), vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_connectivity(graph.internal()));
    EXPECT_TRUE(deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph&>(graph.internal())));
}

TEST(DynamicExplorationGraphCreateRandomGraph, SearchReturnsExternalLabels) {
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

    auto graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Search returns external labels
    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search(std::span<const float>(query), 5, 0.0f);

    ASSERT_GT(results.size(), 0u);

    // External labels are 0..vertex_count-1 (same as internal indices in this test)
    std::unordered_set<uint32_t> valid_labels;
    for (uint32_t i = 0; i < vertex_count; i++) {
        valid_labels.insert(i);
    }

    while (!results.empty()) {
        uint32_t id = results.top().getIdentifier();
        EXPECT_TRUE(valid_labels.contains(id)) << "Result identifier " << id << " is not a valid external label";
        results.pop();
    }
}

TEST(DynamicExplorationGraphCreateRandomGraph, DeterministicWithSameSeed) {
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

    auto graph1 = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 123);
    auto graph2 = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 123);

    EXPECT_EQ(graph1.size(), graph2.size());

    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* n1 = graph1.internal().getNeighborIndices(i);
        const auto* n2 = graph2.internal().getNeighborIndices(i);
        const auto* w1 = static_cast<const deglib::graph::MutableGraph&>(graph1.internal()).getNeighborWeights(i);
        const auto* w2 = static_cast<const deglib::graph::MutableGraph&>(graph2.internal()).getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(n1[e], n2[e]) << "Neighbor mismatch at vertex " << i << " edge " << e;
            EXPECT_NEAR(w1[e], w2[e], 1e-6f) << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(DynamicExplorationGraphCreateRandomGraph, UInt8Metric) {
    const uint32_t vertex_count = 30;
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

    auto graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    EXPECT_EQ(graph.size(), vertex_count);
    EXPECT_EQ(graph.getFeatureSpace().metric(), deglib::distances::Metric::Uint8_L2);
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(graph.internal(), vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_connectivity(graph.internal()));
    EXPECT_TRUE(deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph&>(graph.internal())));
}

// ===========================================================================
//  DynamicExplorationGraph: from_graph and to_mutable
// ===========================================================================

TEST(DynamicExplorationGraphFromGraph, FromGraphCreatesMutableGraph) {
    const uint32_t vertex_count = 30;
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

    auto source_graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Convert to ReadOnlyGraph first
    auto readonly = source_graph.to_readonly();
    EXPECT_FALSE(readonly.isMutable());

    // Now create a mutable graph from the readonly graph
    auto mutable_graph = readonly.to_mutable();

    EXPECT_TRUE(mutable_graph.isMutable());
    EXPECT_EQ(mutable_graph.size(), vertex_count);
    EXPECT_EQ(mutable_graph.getEdgesPerVertex(), edges_per_vertex);

    // Verify graph regularity and weights
    EXPECT_TRUE(deglib::analysis::check_graph_regularity(mutable_graph.internal(), vertex_count, true));
    EXPECT_TRUE(deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph&>(mutable_graph.internal())));
}

TEST(DynamicExplorationGraphFromGraph, FromGraphStaticFactory) {
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

    auto source_graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Use to_mutable to create a mutable copy from the source graph
    auto mutable_graph = source_graph.to_mutable();

    EXPECT_TRUE(mutable_graph.isMutable());
    EXPECT_EQ(mutable_graph.size(), vertex_count);
    EXPECT_EQ(mutable_graph.getEdgesPerVertex(), edges_per_vertex);

    // Verify topology matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.internal().getNeighborIndices(i);
        const auto* copy_neighbors = mutable_graph.internal().getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e]) << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(DynamicExplorationGraphFromGraph, ToMutableSearchWorks) {
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

    auto source_graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Convert to readonly, then back to mutable
    auto readonly = source_graph.to_readonly();
    auto mutable_graph = readonly.to_mutable();

    // Search on both should return the same results
    std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto readonly_results = readonly.search(std::span<const float>(query), 5, 0.0f);
    auto mutable_results = mutable_graph.search(std::span<const float>(query), 5, 0.0f);

    ASSERT_EQ(readonly_results.size(), mutable_results.size());

    // Compare results
    auto rd_res = readonly_results;
    auto mu_res = mutable_results;
    while (!rd_res.empty() && !mu_res.empty()) {
        EXPECT_EQ(rd_res.top().getIdentifier(), mu_res.top().getIdentifier());
        EXPECT_NEAR(rd_res.top().getDistance(), mu_res.top().getDistance(), 1e-5f);
        rd_res.pop();
        mu_res.pop();
    }
}

TEST(DynamicExplorationGraphFromGraph, ToMutableWithCustomFeatures) {
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

    auto source_graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Create scaled custom features
    auto custom_feature_bytes = std::make_unique<std::byte[]>(size_t(vertex_count) * dim * sizeof(float));
    float* custom_floats = reinterpret_cast<float*>(custom_feature_bytes.get());
    for (uint32_t i = 0; i < vertex_count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            custom_floats[i * dim + d] = static_cast<float>(i + d) * 2.0f;
        }
    }

    // Convert to mutable with custom features
    auto mutable_graph = source_graph.to_mutable(space, custom_feature_bytes.get());

    EXPECT_TRUE(mutable_graph.isMutable());
    EXPECT_EQ(mutable_graph.size(), vertex_count);

    // Verify topology matches
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.internal().getNeighborIndices(i);
        const auto* copy_neighbors = mutable_graph.internal().getNeighborIndices(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            EXPECT_EQ(src_neighbors[e], copy_neighbors[e]) << "Neighbor index mismatch at vertex " << i << " edge " << e;
        }
    }

    // Verify weights are recalculated with custom features
    const auto dist_func = space.get_dist_func();
    const auto dist_func_param = space.get_dist_func_param();
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto* src_neighbors = source_graph.internal().getNeighborIndices(i);
        const auto* copy_weights = static_cast<const deglib::graph::MutableGraph&>(mutable_graph.internal()).getNeighborWeights(i);
        for (uint8_t e = 0; e < edges_per_vertex; e++) {
            const auto neighbor_idx = src_neighbors[e];
            const auto expected_weight = dist_func(
                custom_feature_bytes.get() + size_t(i) * dim * sizeof(float), custom_feature_bytes.get() + size_t(neighbor_idx) * dim * sizeof(float),
                dist_func_param
            );
            EXPECT_NEAR(copy_weights[e], expected_weight, 1e-4f) << "Weight mismatch at vertex " << i << " edge " << e;
        }
    }
}

TEST(DynamicExplorationGraphFromGraph, ToMutableWithNewMaxSize) {
    const uint32_t vertex_count = 10;
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

    auto source_graph = deglib::DynamicExplorationGraph::create_random_graph(feature_bytes.get(), vertex_count, edges_per_vertex, space, 7);

    // Convert to mutable with larger capacity using the direct to_mutable(new_max_size) overload
    const uint32_t new_capacity = 50;
    auto mutable_graph = source_graph.to_mutable(new_capacity);

    EXPECT_TRUE(mutable_graph.isMutable());
    EXPECT_EQ(mutable_graph.size(), vertex_count);

    // Should be able to add vertices
    auto new_feature = make_float_bytes(make_vec_4d(100.0f, 0.0f, 0.0f, 0.0f));
    static_cast<deglib::graph::MutableGraph&>(mutable_graph.internal()).addVertex(9999, new_feature.get());
    EXPECT_EQ(mutable_graph.size(), vertex_count + 1);
    EXPECT_TRUE(mutable_graph.hasVertex(9999));
}
