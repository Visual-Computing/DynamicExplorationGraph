#include <gtest/gtest.h>
#include "flas/fast_linear_assignment_sorter.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static FlasSettings default_flas_settings() {
    FlasSettings settings;
    init_flas_settings(&settings);
    return settings;
}

// ============================================================================
//  Jonker-Volgenant assignment solver tests
// ============================================================================

TEST(JunkerVolgenantSolver, IdentityMatrix) {
    const int dim = 3;
    int matrix[9] = {
        0, 10, 10,
        10, 0, 10,
        10, 10, 0
    };

    int* assignment = compute_assignment(matrix, dim);
    ASSERT_NE(assignment, nullptr);

    EXPECT_EQ(assignment[0], 0);
    EXPECT_EQ(assignment[1], 1);
    EXPECT_EQ(assignment[2], 2);

    free(assignment);
}

TEST(JunkerVolgenantSolver, PermutedOptimalAssignment) {
    const int dim = 3;
    int matrix[9] = {
        10,  0, 10,
        10, 10,  0,
         0, 10, 10
    };

    int* assignment = compute_assignment(matrix, dim);
    ASSERT_NE(assignment, nullptr);

    EXPECT_EQ(assignment[0], 1);
    EXPECT_EQ(assignment[1], 2);
    EXPECT_EQ(assignment[2], 0);

    free(assignment);
}

TEST(JunkerVolgenantSolver, Dimension1) {
    const int dim = 1;
    int matrix[1] = { 42 };

    int* assignment = compute_assignment(matrix, dim);
    ASSERT_NE(assignment, nullptr);

    EXPECT_EQ(assignment[0], 0);

    free(assignment);
}

// ============================================================================
//  InternalData lifecycle
// ============================================================================

TEST(FlasInternalData, CreateAndFree) {
    const int N = 10;
    const int D = 3;

    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::vector<MapField> map_fields(N);
    for (int i = 0; i < N; ++i) {
        init_map_field(&map_fields[i], i, &features[static_cast<size_t>(i) * static_cast<size_t>(D)], true);
    }

    RandomEngine rng(42);
    InternalData data = create_internal_data(map_fields.data(), 1, N, D, 5, &rng);

    EXPECT_EQ(data.columns, 1);
    EXPECT_EQ(data.rows, N);
    EXPECT_EQ(data.grid_size, N);
    EXPECT_EQ(data.dim, D);
    EXPECT_EQ(data.num_swap_positions, 5);
    EXPECT_EQ(data.map_fields, map_fields.data());
    EXPECT_NE(data.som, nullptr);

    free_internal_data(&data);
}

// ============================================================================
//  FLAS sorting — 1D sort quality (columns=1, rows=N)
// ============================================================================

static std::vector<int> run_flas_1d(const float* features, int N, unsigned int seed) {
    std::vector<MapField> mf(N);
    for (int i = 0; i < N; ++i)
        init_map_field(&mf[i], i, &features[i], true);

    FlasSettings s = default_flas_settings();
    RandomEngine r(seed);
    do_sorting_full(mf.data(), 1, 1, N, &s, &r, [](float) { return false; });

    std::vector<int> result(N);
    for (int i = 0; i < N; ++i) result[i] = mf[i].id;
    return result;
}

TEST(FlasSort, ValidPermutation1D) {
    const int N = 10;
    std::vector<float> features(N);
    for (int i = 0; i < N; ++i) features[i] = static_cast<float>(i * 100);

    auto perm = run_flas_1d(features.data(), N, 42);

    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        ASSERT_GE(perm[i], 0);
        ASSERT_LT(perm[i], N);
        EXPECT_FALSE(seen[perm[i]]);
        seen[perm[i]] = true;
    }
}


TEST(FlasSort, Stress_N1000_D8) {
    const int N = 1000;
    const int D = 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    std::vector<MapField> map_fields(N);
    for (int i = 0; i < N; ++i) {
        init_map_field(&map_fields[i], i, &features[static_cast<size_t>(i) * static_cast<size_t>(D)], true);
    }

    FlasSettings settings = default_flas_settings();
    RandomEngine flas_rng(42);
    do_sorting_full(map_fields.data(), D, 1, N, &settings, &flas_rng, [](float) { return false; });

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
