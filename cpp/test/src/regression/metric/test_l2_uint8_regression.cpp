#include "test_regression.h"

TEST(DeglibRegressionL2Uint8, MultiInstructionSetBenchmark)
{
#if defined(USE_AVX512)
    std::cout << "use AVX512  ...\n";
#elif defined(USE_AVX2)
    std::cout << "use AVX2  ...\n";
#elif defined(USE_SSE42)
    std::cout << "use SSE  ...\n";
#else
    std::cout << "use arch  ...\n";
#endif

    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

#if defined(USE_AVX512)
    run_regression_test("AVX512", deglib::Metric::L2_Uint8, 72000.0, 4.9, 0.99, base_data.data(), query_data.data(), base_count, query_count, dim, gt_data);
#elif defined(USE_AVX2)
    run_regression_test("AVX2", deglib::Metric::L2_Uint8, 69000.0, 4.8, 0.99, base_data.data(), query_data.data(), base_count, query_count, dim, gt_data);
#elif defined(USE_SSE42)
    run_regression_test("SSE", deglib::Metric::L2_Uint8, 1000.0, 60.0, 0.5, base_data.data(), query_data.data(), base_count, query_count, dim, gt_data);
#else
    run_regression_test("Scalar", deglib::Metric::L2_Uint8, 1000.0, 60.0, 0.5, base_data.data(), query_data.data(), base_count, query_count, dim, gt_data);
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecallAllVariantsSameDataset)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    // Scalar ground truth
    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    // Verify SIMD variants match scalar distance recall
    auto check_variant = [&](const char* name, auto dist_func)
    {
        std::vector<std::vector<uint32_t>> gt(query_count);
        for (int q = 0; q < static_cast<int>(query_count); ++q)
        {
            std::vector<std::pair<float, uint32_t>> dists(base_count);
            const uint8_t* q_vec = &query_data[q * dim];
            for (size_t i = 0; i < base_count; ++i)
            {
                const uint8_t* b_vec = &base_data[i * dim];
                size_t qty = dim;
                float d = dist_func(q_vec, b_vec, &qty);
                dists[i] = {d, static_cast<uint32_t>(i)};
            }
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
            gt[q].reserve(k);
            for (uint32_t i = 0; i < k; ++i) gt[q].push_back(dists[i].second);
        }

        size_t correct = 0;
        for (size_t q = 0; q < query_count; ++q)
        {
            std::unordered_set<uint32_t> gt_set(gt_scalar[q].begin(), gt_scalar[q].end());
            for (uint32_t idx : gt[q])
            {
                if (gt_set.count(idx)) ++correct;
            }
        }
        double recall = static_cast<double>(correct) / static_cast<double>(query_count * k);
        std::cout << "[DistanceRecall " << name << "] recall=" << recall << "  correct=" << correct << "/"
                  << (query_count * k) << std::endl;
        EXPECT_EQ(recall, 1.0) << "Distance recall between scalar and " << name << " L2Uint8 must be exactly 1.0";
    };

#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
    check_variant("L2Uint8Ext32", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::L2Uint8Ext32::compare(a, b, qty); });
    check_variant("L2Uint8Ext16", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::L2Uint8Ext16::compare(a, b, qty); });
#endif
    check_variant("L2Uint8", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::L2Uint8::compare(a, b, qty); });
}
