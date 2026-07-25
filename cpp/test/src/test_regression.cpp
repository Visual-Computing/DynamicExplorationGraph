#include <builder.h>
#include <deglib.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

// Compute exact brute-force L2 groundtruth for top-K neighbors
static std::vector<std::vector<uint32_t>> compute_groundtruth_l2(const std::vector<float>& base, size_t base_count,
                                                                 const std::vector<float>& query, size_t query_count,
                                                                 size_t dim, uint32_t k)
{
    std::vector<std::vector<uint32_t>> gt(query_count);

    for (int q = 0; q < static_cast<int>(query_count); ++q)
    {
        std::vector<std::pair<float, uint32_t>> dists(base_count);
        const float* q_vec = &query[q * dim];

        for (size_t i = 0; i < base_count; ++i)
        {
            const float* b_vec = &base[i * dim];
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d)
            {
                float diff = q_vec[d] - b_vec[d];
                sum += diff * diff;
            }
            dists[i] = {sum, static_cast<uint32_t>(i)};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].reserve(k);
        for (uint32_t i = 0; i < k; ++i)
        {
            gt[q].push_back(dists[i].second);
        }
    }

    return gt;
}

// Generate cross-platform deterministic clustered dataset (Gaussian Mixture with fixed seed)
static void generate_synthetic_clustered_dataset(size_t count, size_t dim, std::vector<float>& base,
                                                 std::vector<float>& query, size_t query_count,
                                                 size_t num_clusters = 20)
{
    base.resize(count * dim);
    query.resize(query_count * dim);

    std::mt19937 rng(42);
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim));
    std::normal_distribution<float> cent_dist(-1000.0f, 1000.0f);
    std::normal_distribution<float> noise_dist(-100.0f, 100.0f);

    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[c][d] = cent_dist(rng);
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            base[i * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            query[q * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }
}

static void run_reg(const char* name, double min_qps, double max_build_secs, double min_recall,
                    const std::vector<float>& base_data, const std::vector<float>& query_data, size_t base_count,
                    size_t query_count, size_t dim)
{
    std::cout << "--- Testing Instruction Variant: " << name << " ---" << std::endl;

    const uint32_t search_k = 10;
    const float search_eps = 0.05f;

    const uint32_t edges_per_vertex = 32;
    const deglib::builder::OptimizationTarget optimization_target = deglib::builder::OptimizationTarget::LowLID;
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t thread_count = 1;

    // Compute groundtruth using scalar groundtruth calculation
    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, search_k);

    // Build DEG Graph using the default distance function for the metric
    const deglib::FloatSpace feature_space(dim, deglib::Metric::L2);

    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex,
                                          std::move(feature_space));

    std::mt19937 rng(1337);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target, extend_k, extend_eps, improve_k,
                                                     improve_eps, max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);
    auto t_build_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < base_count; ++i)
    {
        const std::byte* ptr = reinterpret_cast<const std::byte*>(&base_data[i * dim]);
        std::vector<std::byte> feat_vec(ptr, ptr + dim * sizeof(float));
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_secs = std::chrono::duration<double>(t_build_end - t_build_start).count();

    auto entry_vertex_indices = graph.getEntryVertexIndices();

    auto run_search = [&]() -> std::pair<double, double>
    {
        size_t total_correct = 0;
        auto t_search_start = std::chrono::high_resolution_clock::now();

        for (size_t q = 0; q < query_count; ++q)
        {
            const std::byte* q_ptr = reinterpret_cast<const std::byte*>(&query_data[q * dim]);
            auto result = graph.search(entry_vertex_indices, q_ptr, search_eps, search_k, nullptr, 0);

            std::unordered_set<uint32_t> gt_set;
            if (!gt_data.empty() && q < gt_data.size())
            {
                size_t eval_k = std::min(static_cast<size_t>(search_k), gt_data[q].size());
                for (size_t i = 0; i < eval_k; ++i)
                {
                    gt_set.insert(gt_data[q][i]);
                }
            }

            while (!result.empty())
            {
                auto top_item = result.top();
                result.pop();
                uint32_t ext_label = graph.getExternalLabel(top_item.getInternalIndex());
                if (gt_set.count(ext_label))
                {
                    total_correct++;
                }
            }
        }

        auto t_search_end = std::chrono::high_resolution_clock::now();
        double search_secs = std::chrono::duration<double>(t_search_end - t_search_start).count();
        double qps = static_cast<double>(query_count) / search_secs;
        double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * search_k);
        return {qps, recall};
    };

    // Warm-up run
    run_search();

    // Measured runs: average of 3
    double total_qps = 0.0;
    double total_recall = 0.0;
    constexpr int num_runs = 3;
    for (int r = 0; r < num_runs; ++r)
    {
        auto [qps, recall] = run_search();
        total_qps += qps;
        total_recall += recall;
    }
    double qps = total_qps / num_runs;
    double recall = total_recall / num_runs;

    std::cout << "[" << name << "] build_secs=" << build_secs << "  qps=" << qps << "  recall=" << recall << std::endl;

    EXPECT_GE(recall, min_recall) << name << ": Recall@10 dropped below threshold";
    EXPECT_GT(qps, min_qps) << name << ": Search QPS dropped below threshold";
    EXPECT_LE(build_secs, max_build_secs) << name << ": Build time exceeded threshold";
}

TEST(DeglibRegression, MultiInstructionSetBenchmark)
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

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

#if defined(USE_AVX512)
    run_reg("AVX512", 41000.0, 6.0, 0.98, base_data, query_data, base_count, query_count, dim);
#elif defined(USE_AVX2)
    run_reg("AVX2", 38000.0, 6.2, 0.96, base_data, query_data, base_count, query_count, dim);
#elif defined(USE_SSE42)
    run_reg("SSE", 33000.0, 7.2, 0.96, base_data, query_data, base_count, query_count, dim);
#else
    run_reg("Scalar", 26000.0, 9.2, 0.989, base_data, query_data, base_count, query_count, dim);
#endif
}

// Compare the groundtruth computed with different distance functions on the SAME
// synthetic clustered dataset. All SIMD variants should produce identical top-K
// results when distances are exact integers (no floating-point rounding differences).
TEST(DeglibRegression, DistanceRecallAllVariantsSameDataset)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    // Scalar ground truth
    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    // Verify every SIMD variant matches scalar exactly
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
        EXPECT_EQ(recall, 1.0) << "Distance recall between scalar and " << name << " L2 must be exactly 1.0";
    };

    // In the current branch, L2Float16Ext is the combined SIMD variant that handles
    // AVX512, AVX2, and SSE42 internally. We test it alongside the scalar L2Float.
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
    check_variant("L2Float16Ext", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::L2Float16Ext::compare(a, b, qty); });
#endif
    check_variant("L2Float", [](const void* a, const void* b, const void* qty)
                  { return deglib::distances::L2Float::compare(a, b, qty); });
}
