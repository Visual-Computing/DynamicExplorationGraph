#include <gtest/gtest.h>
#include "flas/fast_linear_assignment_sorter.h"
#include "common/test_helpers.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static flas::FlasSettings local_default_flas_settings() {
    return flas::FlasSettings{};
}

// ============================================================================
//  MapField utility tests
// ============================================================================

TEST(MapFieldTest, InitMapField) {
    flas::MapField mf;
    float feature[3] = {1.0f, 2.0f, 3.0f};
    flas::init_map_field(mf, 42, feature);

    EXPECT_EQ(mf.id, 42);
    EXPECT_EQ(mf.feature, feature);
}

TEST(MapFieldTest, GetNumSwappable) {
    flas::MapField fields[5];
    flas::init_map_field(fields[0], 0, nullptr);
    flas::init_map_field(fields[1], 1, nullptr);
    flas::init_map_field(fields[2], 2, nullptr);
    flas::init_map_field(fields[3], 3, nullptr);
    flas::init_map_field(fields[4], 4, nullptr);

    EXPECT_EQ(5, 5);
}

// ============================================================================
//  FLAS sorting — 1D sort quality (columns=1, rows=N)
// ============================================================================

static std::vector<int> run_flas_1d(const float* features, int N, int dim, unsigned int seed) {
    auto mf = flas::make_map_fields(features, N, dim);

    flas::FlasSettings s = local_default_flas_settings();
    flas::RandomEngine r(seed);
    flas::do_sorting_1d(mf, dim, s, r, [](float) { return false; });

    std::vector<int> result(N);
    for (int i = 0; i < N; ++i) result[i] = mf[i].id;
    return result;
}

TEST(FlasSort, ValidPermutation1D) {
    const int N = 10;
    const int D = 1;
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (int i = 0; i < N; ++i) features[i] = static_cast<float>(i * 100);

    auto perm = run_flas_1d(features.data(), N, D, 42);

    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        ASSERT_GE(perm[i], 0);
        ASSERT_LT(perm[i], N);
        EXPECT_FALSE(seen[perm[i]]);
        seen[perm[i]] = true;
    }
}

TEST(FlasSort, PreservesAllElements1D) {
    const int N = 20;
    const int D = 4;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    auto perm = run_flas_1d(features.data(), N, D, 123);

    EXPECT_EQ(perm.size(), static_cast<size_t>(N));

    std::vector<int> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(sorted_perm[i], i);
    }
}

// ============================================================================
//  FLAS sorting — InnerProduct metric
// ============================================================================

TEST(FlasSort, InnerProductMetric1D) {
    const int N = 50;
    const int D = 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    auto map_fields = flas::make_map_fields(features.data(), N, D);

    flas::FlasSettings settings = local_default_flas_settings();
    settings.metric = flas::FlasMetric::InnerProduct;
    flas::RandomEngine flas_rng(42);
    flas::do_sorting_1d(map_fields, D, settings, flas_rng, [](float) { return false; });

    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        int id = map_fields[i].id;
        ASSERT_GE(id, 0);
        ASSERT_LT(id, N);
        EXPECT_FALSE(seen[id]);
        seen[id] = true;
    }
}

// ============================================================================
//  FLAS sorting — stress test
// ============================================================================

TEST(FlasSort, Stress_N1000_D8) {
    const int N = 1000;
    const int D = 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    auto map_fields = flas::make_map_fields(features.data(), N, D);

    flas::FlasSettings settings = local_default_flas_settings();
    flas::RandomEngine flas_rng(42);
    flas::do_sorting_1d(map_fields, D, settings, flas_rng, [](float) { return false; });

    // Verify permutation validity
    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        int id = map_fields[i].id;
        ASSERT_GE(id, 0);
        ASSERT_LT(id, N);
        EXPECT_FALSE(seen[id]);
        seen[id] = true;
    }
}

// ============================================================================
//  FLAS sorting — 2D Grid stress test (cols x rows)
// ============================================================================

TEST(FlasSort, 2DGrid_Cols20_Rows25_D8) {
    const int cols = 20;
    const int rows = 25;
    const int N = cols * rows;
    const int D = 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    auto map_fields = flas::make_map_fields(features.data(), N, D);

    flas::FlasSettings settings = local_default_flas_settings();
    flas::RandomEngine flas_rng(42);
    flas::do_sorting_1d(map_fields, D, settings, flas_rng, [](float) { return false; });

    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        int id = map_fields[i].id;
        ASSERT_GE(id, 0);
        ASSERT_LT(id, N);
        EXPECT_FALSE(seen[id]);
        seen[id] = true;
    }
}

// ============================================================================
//  FLAS settings tests
// ============================================================================

TEST(FlasSettingsTest, DefaultSettings) {
    flas::FlasSettings settings = flas::FlasSettings{};

    EXPECT_FLOAT_EQ(settings.initial_radius_factor, 0.5f);
    EXPECT_FLOAT_EQ(settings.radius_decay, 0.9f);
    EXPECT_FLOAT_EQ(settings.radius_end, 1.0f);
    EXPECT_EQ(settings.num_filters, 1);
    EXPECT_EQ(settings.max_swap_positions, 9);
    EXPECT_FLOAT_EQ(settings.sample_factor, 1.0f);
    EXPECT_EQ(settings.metric, flas::FlasMetric::L2);
}

// ============================================================================
//  FLAS sorting — callback early termination
// ============================================================================

TEST(FlasSort, CallbackEarlyTermination) {
    const int N = 100;
    const int D = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    auto map_fields = flas::make_map_fields(features.data(), N, D);

    flas::FlasSettings settings = local_default_flas_settings();
    flas::RandomEngine flas_rng(42);

    bool callback_called = false;
    flas::do_sorting_1d(map_fields, D, settings, flas_rng,
                    [&callback_called](float progress) {
                        callback_called = true;
                        return true; // terminate immediately
                    });

    EXPECT_TRUE(callback_called);
}

// ============================================================================
//  FLAS sorting — 1D sort quality comparison
//  Compares the 1D specialized FLAS against raw ordering.
// ============================================================================

#include <chrono>

static void run_1d_comparison_benchmark(int N, int D, unsigned int seed, int num_clusters) {
    std::vector<float> features;
    std::vector<float> dummy_queries;
    generate_synthetic_clustered_dataset(N, D, features, dummy_queries, 10, num_clusters);

    // FLAS sorting (fast_linear_assignment_sorter.h)
    auto mf = flas::make_map_fields(features.data(), N, D);
    flas::FlasSettings s;
    s.max_swap_positions = 50;
    s.radius_decay = 0.9f;
    flas::RandomEngine r(seed);
    auto t0 = std::chrono::high_resolution_clock::now();
    flas::do_sorting_1d(mf, D, s, r, [](float) { return false; });
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = mf[i].id;

    auto compute_locality = [&](const std::vector<int>& perm) {
        float sum = 0.0f;
        for (int i = 0; i < N - 1; ++i) {
            const float* a = &features[static_cast<size_t>(perm[i]) * D];
            const float* b = &features[static_cast<size_t>(perm[i + 1]) * D];
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = a[d] - b[d];
                dist += diff * diff;
            }
            sum += std::sqrt(dist);
        }
        return sum / static_cast<float>(N - 1);
    };

    std::vector<int> perm_raw(N);
    for (int i = 0; i < N; ++i) perm_raw[i] = i;

    float loc_raw = compute_locality(perm_raw);
    float loc = compute_locality(perm);

    // Check validity: Ensure every element 0..N-1 is present exactly once
    auto check_valid_permutation = [](const std::vector<int>& perm, int N) {
        std::vector<bool> seen(N, false);
        for (int id : perm) {
            if (id < 0 || id >= N || seen[id]) return false;
            seen[id] = true;
        }
        return true;
    };

    EXPECT_TRUE(check_valid_permutation(perm, N)) << "perm is NOT a valid permutation!";
    EXPECT_LT(loc, loc_raw);

    std::cout << "[FLAS 1D] N=" << N << " D=" << D << std::endl;
    std::cout << "  - Raw Locality:                   " << loc_raw << std::endl;
    std::cout << "  - FLAS Sorted:                    Locality=" << loc << " | Time=" << time_ms << " ms" << std::endl;
}

TEST(Flas1DComparison, Benchmark_N1000_D128) {
    run_1d_comparison_benchmark(1000, 128, 42, 20);
}

TEST(Flas1DTest, SortShuffled0To100) {
    const int N = 101;
    const int D = 1;
    const unsigned int seed = 42;

    std::vector<float> features(N);
    for (int i = 0; i < N; ++i) {
        features[i] = static_cast<float>(i);
    }
    std::mt19937 rng(seed);
    std::shuffle(features.begin(), features.end(), rng);

    auto calc_avg_neighbor_dist = [](const std::vector<float>& sorted_vals) {
        if (sorted_vals.size() <= 1) return 0.0f;
        float total_dist = 0.0f;
        int count = 0;
        for (size_t i = 0; i < sorted_vals.size() - 1; ++i) {
            total_dist += std::abs(sorted_vals[i + 1] - sorted_vals[i]);
            count++;
        }
        return total_dist / static_cast<float>(count);
    };

    // FLAS sorting (fast_linear_assignment_sorter.h)
    auto mf = flas::make_map_fields(features.data(), N, D);
    flas::FlasSettings s;
    s.max_swap_positions = 10;
    flas::RandomEngine r(seed);
    flas::do_sorting_1d(mf, D, s, r, [](float) { return false; });

    std::vector<float> res(N);
    for (int i = 0; i < N; ++i) res[i] = *mf[i].feature;
    float avg_dist = calc_avg_neighbor_dist(res);

    float avg_dist_raw = calc_avg_neighbor_dist(features);

    std::cout << "\n========================================================\n";
    std::cout << "  1D Test: Shuffled 0..100 (N=101, D=1)\n";
    std::cout << "========================================================\n";
    std::cout << "Raw shuffled values: ";
    for (float v : features) std::cout << v << " ";
    std::cout << "\n  -> Raw Avg Neighbor Distance: " << avg_dist_raw << "\n\n";

    std::cout << "1) FLAS Sorted (fast_linear_assignment_sorter.h):\n   Sorted: ";
    for (float v : res) std::cout << v << " ";
    std::cout << "\n   -> Avg Neighbor Distance: " << avg_dist << "\n";
    std::cout << "========================================================\n\n";
}
