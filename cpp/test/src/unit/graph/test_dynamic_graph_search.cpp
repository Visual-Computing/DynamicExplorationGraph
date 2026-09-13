#include "test_dynamic_graph_common.h"

#include "deglib/deglib.h"
#include "deglib/graph/readonly_graph.h"
#include "deglib/graph/sizebounded_graph.h"

#include <filesystem>
#include <span>

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

    auto graph = deglib::graph::DynamicGraph(edges, space, /*chunk_size=*/8);
    deglib::builder::populate_random_graph(graph, reinterpret_cast<const std::byte*>(data.data()), count, 42);

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

    auto orig_graph = deglib::graph::DynamicGraph(edges, space, /*chunk_size=*/8);
    deglib::builder::populate_random_graph(orig_graph, reinterpret_cast<const std::byte*>(data.data()), count, 123);

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
    auto deg = deglib::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, edges, space, 7);

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
