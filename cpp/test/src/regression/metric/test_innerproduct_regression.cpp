#include "test_regression.h"

TEST(DeglibRegressionIP, MultiInstructionSetBenchmark)
{
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    // Compute ground-truth using the scalar InnerProductFloat::compare() implementation.
    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    // Test each distance variant with its own performance thresholds
    // num_runs=50 extends the search measurement window to ~280ms per run,
    // reducing QPS noise from OS jitter and CPU power-state transitions.
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        run_regression_test("AVX512_16Ext", deglib::Metric::InnerProduct, 18000.0, 10.3, 0.774,
                            base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                            deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX512{}, 50);
    }
    if (deglib::cpu::has_avx2()) {
        run_regression_test("AVX2_16Ext", deglib::Metric::InnerProduct, 18000.0, 10.2, 0.774,
                            base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                            deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX2{}, 50);
    }
    if (deglib::cpu::has_sse42()) {
        run_regression_test("SSE_16Ext", deglib::Metric::InnerProduct, 15000.0, 11.8, 0.774,
                            base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                            deglib::distances::fp32_ip::InnerProductFloat16Ext_SSE{}, 50);
    }
#endif
    run_regression_test("Scalar", deglib::Metric::InnerProduct, 12000.0, 15.4, 0.78,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat{}, 50);
}

TEST(DeglibRegressionIP, DistanceRecallAllVariantsSameDataset)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    // Scalar ground truth
    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    // Verify SIMD variants match scalar distance recall
    auto check_variant = [&](const char* name, auto dist_func)
    {
        std::vector<std::vector<uint32_t>> gt(query_count);
        for (int q = 0; q < static_cast<int>(query_count); ++q)
        {
            std::vector<std::pair<float, uint32_t>> dists(base_count);
            const float* q_vec = &query_data[q * dim];
            for (size_t i = 0; i < base_count; ++i)
            {
                const float* b_vec = &base_data[i * dim];
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
        EXPECT_EQ(recall, 1.0) << "Distance recall between scalar and " << name << " InnerProduct must be exactly 1.0";
    };

#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        check_variant("InnerProductFloat16Ext_AVX512", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX512::compare(a, b, qty); });
        check_variant("InnerProductFloat16ExtResiduals_AVX512", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_AVX512::compare(a, b, qty); });
    }
    if (deglib::cpu::has_avx2()) {
        check_variant("InnerProductFloat16Ext_AVX2", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX2::compare(a, b, qty); });
        check_variant("InnerProductFloat8Ext_AVX2", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat8Ext_AVX2::compare(a, b, qty); });
        check_variant("InnerProductFloat16ExtResiduals_AVX2", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_AVX2::compare(a, b, qty); });
    }
    if (deglib::cpu::has_sse42()) {
        check_variant("InnerProductFloat16Ext_SSE", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16Ext_SSE::compare(a, b, qty); });
        check_variant("InnerProductFloat8Ext_SSE", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat8Ext_SSE::compare(a, b, qty); });
        check_variant("InnerProductFloat4Ext_SSE", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat4Ext_SSE::compare(a, b, qty); });
        check_variant("InnerProductFloat16ExtResiduals_SSE", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_SSE::compare(a, b, qty); });
        check_variant("InnerProductFloat4ExtResiduals_SSE", [](const void* a, const void* b, const void* qty)
                      { return deglib::distances::fp32_ip::InnerProductFloat4ExtResiduals_SSE::compare(a, b, qty); });
    }
#endif
    check_variant("InnerProductFloat", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::fp32_ip::InnerProductFloat::compare(a, b, qty); });
}