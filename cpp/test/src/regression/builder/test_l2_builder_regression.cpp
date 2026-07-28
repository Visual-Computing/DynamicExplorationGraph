#include "test_regression.h"

// Regression tests for the graph builder (EvenRegularGraphBuilder) using the L2 metric.
// Tests all three deglib::builder::OptimizationTarget modes:
//   - LowLID, HighLID, StreamingData

TEST(DeglibBuilderRegressionL2, Benchmark_LowLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("LowLID", deglib::Metric::L2, 30000.0, 9.3, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::LowLID);
}

TEST(DeglibBuilderRegressionL2, Benchmark_HighLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("HighLID", deglib::Metric::L2, 16000.0, 10.0, 0.918,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::HighLID);
}

TEST(DeglibBuilderRegressionL2, Benchmark_StreamingData)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("StreamingData", deglib::Metric::L2, 23000.0, 23.8, 0.95,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::StreamingData);
}

static std::vector<uint32_t> build_graph_for_determinism(
    deglib::builder::OptimizationTarget optimization_target,
    size_t dim, size_t base_count, uint32_t edges_per_vertex,
    const std::vector<float>& base_data)
{
    const deglib::FloatSpace feature_space(dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex,
                                          std::move(feature_space));

    std::mt19937 rng(1337);
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target,
                                                     extend_k, extend_eps, improve_k, improve_eps,
                                                     max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(1);

    const size_t feature_bytes = dim * sizeof(float);
    const std::byte* base_bytes = reinterpret_cast<const std::byte*>(base_data.data());
    for (size_t i = 0; i < base_count; ++i)
    {
        const std::byte* ptr = base_bytes + i * feature_bytes;
        std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    std::vector<uint32_t> neighbors;
    neighbors.reserve(base_count * edges_per_vertex);
    for (uint32_t v = 0; v < base_count; ++v)
    {
        const uint32_t* nb = graph.getNeighborIndices(v);
        for (uint32_t e = 0; e < edges_per_vertex; ++e)
        {
            neighbors.push_back(nb[e]);
        }
    }
    return neighbors;
}

TEST(DeglibBuilderRegressionL2, StreamingDataDeterminism)
{
    const size_t dim = 64;
    const size_t base_count = 10000;
    const uint32_t edges_per_vertex = 32;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, 10, 50);

    auto graph1_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::StreamingData, dim, base_count, edges_per_vertex, base_data);
    auto graph2_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::StreamingData, dim, base_count, edges_per_vertex, base_data);

    ASSERT_EQ(graph1_neighbors.size(), graph2_neighbors.size())
        << "Graph neighbor count mismatch between two builds";

    size_t mismatch_count = 0;
    for (size_t i = 0; i < graph1_neighbors.size(); ++i)
    {
        if (graph1_neighbors[i] != graph2_neighbors[i])
            mismatch_count++;
    }

    double mismatch_pct = 100.0 * mismatch_count / graph1_neighbors.size();
    std::cout << "[StreamingDataDeterminism] mismatches: " << mismatch_count
              << " / " << graph1_neighbors.size()
              << " (" << mismatch_pct << "%)" << std::endl;

    EXPECT_EQ(0u, mismatch_count) << "Graph was not deterministic within the same process";
}

TEST(DeglibBuilderRegressionL2, LowLIDDeterminism)
{
    const size_t dim = 64;
    const size_t base_count = 10000;
    const uint32_t edges_per_vertex = 32;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, 10, 50);

    auto graph1_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::LowLID, dim, base_count, edges_per_vertex, base_data);
    auto graph2_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::LowLID, dim, base_count, edges_per_vertex, base_data);

    ASSERT_EQ(graph1_neighbors.size(), graph2_neighbors.size())
        << "Graph neighbor count mismatch between two builds";

    size_t mismatch_count = 0;
    for (size_t i = 0; i < graph1_neighbors.size(); ++i)
    {
        if (graph1_neighbors[i] != graph2_neighbors[i])
            mismatch_count++;
    }

    double mismatch_pct = 100.0 * mismatch_count / graph1_neighbors.size();
    std::cout << "[LowLIDDeterminism] mismatches: " << mismatch_count
              << " / " << graph1_neighbors.size()
              << " (" << mismatch_pct << "%)" << std::endl;

    EXPECT_EQ(0u, mismatch_count) << "Graph was not deterministic within the same process";
}

TEST(DeglibBuilderRegressionL2, HighLIDDeterminism)
{
    const size_t dim = 64;
    const size_t base_count = 10000;
    const uint32_t edges_per_vertex = 32;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, 10, 50);

    auto graph1_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::HighLID, dim, base_count, edges_per_vertex, base_data);
    auto graph2_neighbors = build_graph_for_determinism(
        deglib::builder::OptimizationTarget::HighLID, dim, base_count, edges_per_vertex, base_data);

    ASSERT_EQ(graph1_neighbors.size(), graph2_neighbors.size())
        << "Graph neighbor count mismatch between two builds";

    size_t mismatch_count = 0;
    for (size_t i = 0; i < graph1_neighbors.size(); ++i)
    {
        if (graph1_neighbors[i] != graph2_neighbors[i])
            mismatch_count++;
    }

    double mismatch_pct = 100.0 * mismatch_count / graph1_neighbors.size();
    std::cout << "[HighLIDDeterminism] mismatches: " << mismatch_count
              << " / " << graph1_neighbors.size()
              << " (" << mismatch_pct << "%)" << std::endl;

    EXPECT_EQ(0u, mismatch_count) << "Graph was not deterministic within the same process";
}
