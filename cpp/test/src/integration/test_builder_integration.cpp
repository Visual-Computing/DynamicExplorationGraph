#include "common/test_helpers.h"

// ============================================================================
// Builder & Search Recall Integration Tests
// ============================================================================
// Fast integration tests operating on 10,000 base vectors and 100 queries.
// Only recall correctness is verified — no QPS or build-time assertions.
// ============================================================================

// ---------------------------------------------------------------------------
// Dataset & Graph Determinism
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, DatasetBitExactness)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    const uint64_t base_hash = float_vector_checksum(base_data);
    const uint64_t query_hash = float_vector_checksum(query_data);

    auto gt_l2 = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_l2_hash = groundtruth_checksum(gt_l2);

    auto gt_ip = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_ip_hash = groundtruth_checksum(gt_ip);

    std::cout << "[DatasetBitExactness] base_hash = 0x" << std::hex << base_hash << std::dec << std::endl;
    std::cout << "[DatasetBitExactness] query_hash = 0x" << std::hex << query_hash << std::dec << std::endl;
    std::cout << "[DatasetBitExactness] gt_l2_hash = 0x" << std::hex << gt_l2_hash << std::dec << std::endl;
    std::cout << "[DatasetBitExactness] gt_ip_hash = 0x" << std::hex << gt_ip_hash << std::dec << std::endl;

    // Hardcoded expected hashes computed on Windows (10k dataset)
    // Checksum verification ensures 100% bit-exact dataset generation and groundtruth across OS/compilers.
    EXPECT_EQ(base_hash, 0x6dbf4d90119f156cULL) << "base_data checksum mismatch across platforms!";
    EXPECT_EQ(query_hash, 0xb2c6fdf3cd01e5b4ULL) << "query_data checksum mismatch across platforms!";
    EXPECT_EQ(gt_l2_hash, 0x37ac15b6b602a955ULL) << "gt_l2 checksum mismatch across platforms!";
    EXPECT_EQ(gt_ip_hash, 0x69f09ac67e0afed5ULL) << "gt_ip checksum mismatch across platforms!";
}

TEST(DeglibBuilderIntegration, Uint8DatasetBitExactness)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    const uint64_t base_hash = fnv1a_64(base_data.data(), base_data.size());
    const uint64_t query_hash = fnv1a_64(query_data.data(), query_data.size());

    auto gt_u8 = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_u8_hash = groundtruth_checksum(gt_u8);

    std::cout << "[Uint8DatasetBitExactness] uint8_base_hash = 0x" << std::hex << base_hash << std::dec << std::endl;
    std::cout << "[Uint8DatasetBitExactness] uint8_query_hash = 0x" << std::hex << query_hash << std::dec << std::endl;
    std::cout << "[Uint8DatasetBitExactness] gt_u8_hash = 0x" << std::hex << gt_u8_hash << std::dec << std::endl;

    EXPECT_EQ(base_hash, 0xbef9335301e291bfULL) << "uint8_base_data checksum mismatch across platforms!";
    EXPECT_EQ(query_hash, 0x499b735ff0f910e7ULL) << "uint8_query_data checksum mismatch across platforms!";
    EXPECT_EQ(gt_u8_hash, 0x36b390a7cae8797dULL) << "gt_u8 checksum mismatch across platforms!";
}

TEST(DeglibBuilderIntegration, LowLIDDeterminism)
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

TEST(DeglibBuilderIntegration, HighLIDDeterminism)
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

TEST(DeglibBuilderIntegration, StreamingDataDeterminism)
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

// ---------------------------------------------------------------------------
// L2 Float Metric Builder Tests (10k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, L2_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    run_builder_integration_test("L2_AVX512", deglib::Metric::L2, 0.96,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_l2::L2Float_AVX512<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderIntegration, L2_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    run_builder_integration_test("L2_AVX2", deglib::Metric::L2, 0.96,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_l2::L2Float_AVX2<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderIntegration, L2_Scalar)
{
    run_builder_integration_test("L2_Scalar", deglib::Metric::L2, 0.96,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_l2::L2Float{},
                                 deglib::builder::OptimizationTarget::LowLID);
}

// ---------------------------------------------------------------------------
// InnerProduct Float Metric Builder Tests (10k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, IP_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    run_builder_integration_test("IP_AVX512", deglib::Metric::InnerProduct, 0.86,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_ip::InnerProductFloat_AVX512<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderIntegration, IP_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    run_builder_integration_test("IP_AVX2", deglib::Metric::InnerProduct, 0.86,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_ip::InnerProductFloat_AVX2<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderIntegration, IP_Scalar)
{
    run_builder_integration_test("IP_Scalar", deglib::Metric::InnerProduct, 0.86,
                                 128, 10000, 100, 1000,
                                 deglib::distances::fp32_ip::InnerProductFloat{},
                                 deglib::builder::OptimizationTarget::LowLID);
}

// ---------------------------------------------------------------------------
// L2 Uint8 Metric Builder Tests (10k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, L2_Uint8_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    run_builder_integration_test("L2_Uint8_AVX512", deglib::Metric::L2_Uint8, 0.99,
                                 128, 10000, 100, 1000,
                                 deglib::distances::uint8_l2::L2Uint8_AVX512<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderIntegration, L2_Uint8_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    run_builder_integration_test("L2_Uint8_AVX2", deglib::Metric::L2_Uint8, 0.99,
                                 128, 10000, 100, 1000,
                                 deglib::distances::uint8_l2::L2Uint8_AVX2<>{},
                                 deglib::builder::OptimizationTarget::LowLID);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderIntegration, L2_Uint8_Scalar)
{
    run_builder_integration_test("L2_Uint8_Scalar", deglib::Metric::L2_Uint8, 0.99,
                                 128, 10000, 100, 1000,
                                 deglib::distances::uint8_l2::L2Uint8{},
                                 deglib::builder::OptimizationTarget::LowLID);
}

// ---------------------------------------------------------------------------
// OptimizationTarget Variants (10k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, Builder_L2_LowLID)
{
    run_builder_integration_test("Builder_L2_LowLID", deglib::Metric::L2, 0.96,
                                 128, 10000, 100, 1000,
                                 std::nullopt,
                                 deglib::builder::OptimizationTarget::LowLID);
}

TEST(DeglibBuilderIntegration, Builder_L2_HighLID)
{
    run_builder_integration_test("Builder_L2_HighLID", deglib::Metric::L2, 0.95,
                                 128, 10000, 100, 1000,
                                 std::nullopt,
                                 deglib::builder::OptimizationTarget::HighLID);
}

TEST(DeglibBuilderIntegration, Builder_L2_StreamingData)
{
    run_builder_integration_test("Builder_L2_StreamingData", deglib::Metric::L2, 0.90,
                                 128, 10000, 100, 1000,
                                 std::nullopt,
                                 deglib::builder::OptimizationTarget::StreamingData);
}

// ---------------------------------------------------------------------------
// EVP Inner Product Metric Builder Tests
// ---------------------------------------------------------------------------

TEST(DeglibBuilderIntegration, EvpDatasetBitExactness)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;
    const uint32_t non_zeros = 32;

    std::vector<std::byte> base_evp;
    std::vector<std::byte> query_evp;
    generate_synthetic_clustered_dataset_evp(base_count, dim, base_evp, query_evp, query_count, num_clusters, non_zeros);

    const uint64_t base_hash = byte_vector_checksum(base_evp);
    const uint64_t query_hash = byte_vector_checksum(query_evp);

    auto gt_evp = compute_groundtruth_evp(base_evp, base_count, query_evp, query_count, dim, 10);
    const uint64_t gt_evp_hash = groundtruth_checksum(gt_evp);

    std::cout << "[EvpDatasetBitExactness] base_hash = 0x" << std::hex << base_hash << std::dec << std::endl;
    std::cout << "[EvpDatasetBitExactness] query_hash = 0x" << std::hex << query_hash << std::dec << std::endl;
    std::cout << "[EvpDatasetBitExactness] gt_evp_hash = 0x" << std::hex << gt_evp_hash << std::dec << std::endl;

    // Hardcoded expected hashes computed on Windows (10k dataset)
    // Checksum verification ensures 100% bit-exact dataset generation and groundtruth across OS/compilers.
    EXPECT_EQ(base_hash, 0xe8de72f535d6e07dULL) << "base_evp checksum mismatch across platforms!";
    EXPECT_EQ(query_hash, 0xccb8e62c3224256fULL) << "query_evp checksum mismatch across platforms!";
    EXPECT_EQ(gt_evp_hash, 0xc01f476e432ca611ULL) << "gt_evp checksum mismatch across platforms!";
}

TEST(DeglibBuilderIntegration, EVPInnerProduct_Recall)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;
    const uint32_t non_zeros = 32;

    std::vector<std::byte> base_evp;
    std::vector<std::byte> query_evp;
    generate_synthetic_clustered_dataset_evp(base_count, dim, base_evp, query_evp, query_count, num_clusters, non_zeros);

    auto gt_evp = compute_groundtruth_evp(base_evp, base_count, query_evp, query_count, dim, 10);

    run_integration_test("EVPInnerProduct", deglib::Metric::EVPInnerProduct, 0.95,
                         base_evp.data(), query_evp.data(), base_count, query_count, dim, gt_evp);
}
