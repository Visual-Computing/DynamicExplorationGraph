#include "common/test_helpers.h"

// ============================================================================
// Builder Performance Regression Benchmarks
// ============================================================================
// Consolidated 100k performance benchmark tests covering:
//   - DynamicGraph vs SizeBoundedGraph build & search comparison
//   - L2 Float metric (AVX512, AVX2, Scalar variants)
//   - InnerProduct Float metric (AVX512, AVX2, Scalar variants)
//   - L2 Uint8 metric (AVX512, AVX2, Scalar variants)
//   - InnerProduct FP16 metric (AVX512, AVX2, Scalar variants)
//   - OptimizationTarget modes (LowLID, HighLID, StreamingData)
//
// Each benchmark measures QPS, build time, and recall on 100,000 base vectors.
// QPS and build-time assertions are skipped when SKIP_PERFORMANCE_TESTS is set.
// ============================================================================

// ---------------------------------------------------------------------------
// DynamicGraph vs SizeBoundedGraph Comparative Benchmark (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, DynamicGraph_vs_SizeBoundedGraph_100k_Benchmark)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;
    const uint32_t edges_per_vertex = 32;
    const uint8_t extend_k = 32;
    const float extend_eps = 0.01f;
    const uint32_t search_k = 10;
    const float search_eps = 0.05f;
    const size_t num_runs = 20;
    const uint32_t thread_count = 1;
    const int num_rounds = 2;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, search_k);

    const size_t feature_bytes = dim * sizeof(float);
    const std::byte* base_bytes = reinterpret_cast<const std::byte*>(base_data.data());
    const std::byte* query_bytes = reinterpret_cast<const std::byte*>(query_data.data());
    const deglib::distances::FloatSpace feature_space(dim, deglib::distances::Metric::FP32_L2);

    auto run_single_benchmark = [&](deglib::graph::MutableGraph& graph) -> std::pair<double, double> {
        std::mt19937 rng(1337);
        deglib::builder::EvenRegularGraphBuilder builder(
            graph, rng, deglib::builder::OptimizationTarget::LowLID,
            extend_k, extend_eps, /*improve_k=*/0, /*improve_eps=*/0.0f,
            /*max_path_length=*/5, /*swap_tries=*/0, /*additional_swap_tries=*/0
        );
        builder.setThreadCount(thread_count);

        auto t_build_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < base_count; ++i)
        {
            const std::byte* ptr = base_bytes + i * feature_bytes;
            std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes);
            builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
        }

        auto build_callback = [](deglib::builder::BuilderStatus&) {};
        builder.build(build_callback);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double build_secs = std::chrono::duration<double>(t_build_end - t_build_start).count();

        auto run_search = [&]() -> std::pair<double, double> {
            size_t total_correct = 0;
            auto t_search_start = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < query_count; ++q)
            {
                const std::byte* q_ptr = query_bytes + q * feature_bytes;
                std::span<const float> q_span(reinterpret_cast<const float*>(q_ptr), dim);
                auto result = graph.search(q_span, search_k, search_eps, nullptr, 0);

                std::unordered_set<uint32_t> gt_set;
                if (!gt_data.empty() && q < gt_data.size())
                {
                    size_t eval_k = std::min(static_cast<size_t>(search_k), gt_data[q].size());
                    for (size_t i = 0; i < eval_k; ++i) gt_set.insert(gt_data[q][i]);
                }

                while (!result.empty())
                {
                    auto top_item = result.top();
                    result.pop();
                    uint32_t ext_label = graph.getExternalLabel(top_item.getIdentifier());
                    if (gt_set.count(ext_label)) total_correct++;
                }
            }
            auto t_search_end = std::chrono::high_resolution_clock::now();
            double search_secs = std::chrono::duration<double>(t_search_end - t_search_start).count();
            double qps = static_cast<double>(query_count) / search_secs;
            double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * search_k);
            return {qps, recall};
        };

        // Warmup search queries
        run_search();

        double total_qps = 0.0;
        double total_recall = 0.0;
        for (size_t r = 0; r < num_runs; ++r)
        {
            auto [qps, recall] = run_search();
            total_qps += qps;
            total_recall += recall;
        }
        double avg_qps = total_qps / num_runs;
        double avg_recall = total_recall / num_runs;

        EXPECT_GE(avg_recall + 1e-5, 0.88);
        return {build_secs, avg_qps};
    };

    double total_dyn_build = 0.0, total_dyn_qps = 0.0;
    double total_sb_build = 0.0, total_sb_qps = 0.0;

    for (int round = 0; round < num_rounds; ++round)
    {
        // DynamicGraph
        deglib::graph::DynamicGraph dyn_graph(edges_per_vertex, feature_space, /*chunk_size=*/1024);
        auto [dyn_build, dyn_qps] = run_single_benchmark(dyn_graph);
        total_dyn_build += dyn_build;
        total_dyn_qps += dyn_qps;

        // SizeBoundedGraph
        deglib::graph::SizeBoundedGraph sb_graph(static_cast<uint32_t>(base_count), edges_per_vertex, feature_space);
        auto [sb_build, sb_qps] = run_single_benchmark(sb_graph);
        total_sb_build += sb_build;
        total_sb_qps += sb_qps;
    }

    double avg_dyn_build = total_dyn_build / num_rounds;
    double avg_dyn_qps = total_dyn_qps / num_rounds;
    double avg_sb_build = total_sb_build / num_rounds;
    double avg_sb_qps = total_sb_qps / num_rounds;

    std::cout << "[COMPARE " << num_rounds << "x Avg (1 Thread)] DynamicGraph:     build_secs=" << avg_dyn_build
              << "  qps=" << avg_dyn_qps << std::endl;
    std::cout << "[COMPARE " << num_rounds << "x Avg (1 Thread)] SizeBoundedGraph: build_secs=" << avg_sb_build
              << "  qps=" << avg_sb_qps << std::endl;

    std::cout << "[COMPARE SUMMARY] Build overhead: "
              << ((avg_dyn_build - avg_sb_build) / avg_sb_build * 100.0) << "% ("
              << avg_dyn_build << "s vs " << avg_sb_build << "s), QPS ratio: "
              << (avg_dyn_qps / avg_sb_qps * 100.0) << "%\n";
}

// ---------------------------------------------------------------------------
// Multi-Threaded Builder Regression Benchmarks (1, 2, 4 Threads)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, MultiThreaded_1_Thread_Benchmark)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("MultiThreaded_1Thread", deglib::distances::Metric::FP32_L2, 45000.0, 4.0, 0.882,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        std::nullopt, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                        /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f,
                        /*thread_count=*/1);
}

TEST(DeglibBuilderRegression, MultiThreaded_2_Threads_Benchmark)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("MultiThreaded_2Threads", deglib::distances::Metric::FP32_L2, 45000.0, 2.4, 0.89201,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        std::nullopt, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                        /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f,
                        /*thread_count=*/2);
}

TEST(DeglibBuilderRegression, MultiThreaded_4_Threads_Benchmark)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("MultiThreaded_4Threads", deglib::distances::Metric::FP32_L2, 45000.0, 1.5, 0.89001,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        std::nullopt, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                        /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f,
                        /*thread_count=*/4);
}

// ---------------------------------------------------------------------------
// EVP Inner Product Metric Benchmark (100k)
// ---------------------------------------------------------------------------


#if defined(DEGLIB_X86)
TEST(DeglibBuilderRegression, EVPInnerProduct_Benchmark_AVX512)
{
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX-512 not available on this CPU";
    }
    const size_t dim = 512;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<std::byte> base_data;
    std::vector<std::byte> query_data;
    generate_synthetic_clustered_dataset_evp(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_evp(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("EVPInnerProduct_AVX512", deglib::distances::Metric::EVP_InnerProduct, 35000.0, 6.0, 0.898,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::evp_ip::EvpInnerProduct_AVX512<deglib::distances::ResidualMode::SimdOnly>{}, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                        /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f);
}

TEST(DeglibBuilderRegression, EVPInnerProduct_Benchmark_AVX2)
{
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 256;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<std::byte> base_data;
    std::vector<std::byte> query_data;
    generate_synthetic_clustered_dataset_evp(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_evp(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("EVPInnerProduct_AVX2", deglib::distances::Metric::EVP_InnerProduct, 41000.0, 4.7, 0.85301,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::evp_ip::EvpInnerProduct_AVX2<deglib::distances::ResidualMode::SimdOnly>{}, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                         /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f);
}
#endif

TEST(DeglibBuilderRegression, EVPInnerProduct_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<std::byte> base_data;
    std::vector<std::byte> query_data;
    generate_synthetic_clustered_dataset_evp(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_evp(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("EVPInnerProduct_Scalar", deglib::distances::Metric::EVP_InnerProduct, 47000.0, 4.0, 0.741,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::evp_ip::EvpInnerProduct{}, 50,
                        deglib::builder::OptimizationTarget::LowLID,
                        /*edges_per_vertex=*/32, /*extend_k=*/32, /*extend_eps=*/0.01f);
}



// ---------------------------------------------------------------------------
// L2 Float Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, L2Float_Benchmark_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_AVX512", deglib::distances::Metric::FP32_L2, 53000.0, 6.0, 0.961,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float_AVX512<deglib::distances::ResidualMode::DualOnly>{}, 100);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Float_Benchmark_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_AVX2", deglib::distances::Metric::FP32_L2, 53000.0, 6.0, 0.961,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float_AVX2<deglib::distances::ResidualMode::DualOnly>{}, 100);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Float_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_Scalar", deglib::distances::Metric::FP32_L2, 22000.0, 14.0, 0.971,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100);
}

// ---------------------------------------------------------------------------
// InnerProduct Float Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_AVX512", deglib::distances::Metric::FP32_InnerProduct, 42000.0, 4.5, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat_AVX512<deglib::distances::ResidualMode::DualOnly>{}, 50);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_AVX2", deglib::distances::Metric::FP32_InnerProduct, 42000.0, 4.5, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat_AVX2<deglib::distances::ResidualMode::DualOnly>{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_Scalar", deglib::distances::Metric::FP32_InnerProduct, 20000.0, 9.8, 0.867,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat{}, 50);
}

// ---------------------------------------------------------------------------
// L2 Uint8 Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, L2Uint8_Benchmark_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_AVX512", deglib::distances::Metric::Uint8_L2, 66000.0, 4.8, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8_AVX512<deglib::distances::ResidualMode::DualOnly>{}, 500);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Uint8_Benchmark_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_AVX2", deglib::distances::Metric::Uint8_L2, 66000.0, 4.9, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8_AVX2<deglib::distances::ResidualMode::DualOnly>{}, 500);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Uint8_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_Scalar", deglib::distances::Metric::Uint8_L2, 55000.0, 5.8, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8{}, 500);
}


// ---------------------------------------------------------------------------
// InnerProduct FP16 Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_AVX512", deglib::distances::Metric::FP16_InnerProduct, 52000.0, 4.2, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16_AVX512<deglib::distances::ResidualMode::DualOnly>{}, 50);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_AVX2", deglib::distances::Metric::FP16_InnerProduct, 52000.0, 4.2, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16_AVX2<deglib::distances::ResidualMode::DualOnly>{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_Scalar", deglib::distances::Metric::FP16_InnerProduct, 8000.0, 25.0, 0.867,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16{}, 50);
}

// ---------------------------------------------------------------------------
// OptimizationTarget Benchmarks (100k, L2 Float)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, Benchmark_LowLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("LowLID", deglib::distances::Metric::FP32_L2, 20000.0, 14, 0.971,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::LowLID);
}

TEST(DeglibBuilderRegression, Benchmark_HighLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("HighLID", deglib::distances::Metric::FP32_L2, 12000.0, 14.4, 0.96,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::HighLID);
}

TEST(DeglibBuilderRegression, Benchmark_StreamingData)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("StreamingData", deglib::distances::Metric::FP32_L2, 16000.0, 30, 0.91,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::StreamingData);
}



