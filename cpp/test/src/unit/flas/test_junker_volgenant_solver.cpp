#include <gtest/gtest.h>
#include "flas/junker_volgenant_solver.h"
#include "common/test_helpers.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// ============================================================================
//  Junker-Volgenant assignment solver tests
//
//  The test executable is named `test_junker_volgenant_solver` and this file
//  is its sole unit test source, so the Google Test suite is also named
//  `JunkerVolgenantSolver` to keep the mapping between executable and suite
//  unambiguous:
//
//    test_junker_volgenant_solver  ->  TEST(JunkerVolgenantSolver, ...)
// ============================================================================

// ---------------------------------------------------------------------------
// JVScratch — construction, init, reset, dim, accessors
// ---------------------------------------------------------------------------

TEST(JunkerVolgenantSolver, JVScratch_ZeroInitOnConstruction) {
    JVScratch scratch(4);
    EXPECT_EQ(scratch.dim(), 4);

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(scratch.perm()[i], 0);
        EXPECT_EQ(scratch.in_col()[i], 0);
        EXPECT_EQ(scratch.v()[i], 0);
        EXPECT_EQ(scratch.free()[i], 0);
        EXPECT_EQ(scratch.collist()[i], 0);
        EXPECT_EQ(scratch.matches()[i], 0);
        EXPECT_EQ(scratch.pred()[i], 0);
        EXPECT_EQ(scratch.d()[i], 0);
    }
}

TEST(JunkerVolgenantSolver, JVScratch_DefaultConstructorAndInit) {
    JVScratch scratch;
    EXPECT_EQ(scratch.dim(), 0);

    scratch.init(5);
    EXPECT_EQ(scratch.dim(), 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(scratch.perm()[i], 0);
        EXPECT_EQ(scratch.in_col()[i], 0);
        EXPECT_EQ(scratch.v()[i], 0);
        EXPECT_EQ(scratch.free()[i], 0);
        EXPECT_EQ(scratch.collist()[i], 0);
        EXPECT_EQ(scratch.matches()[i], 0);
        EXPECT_EQ(scratch.pred()[i], 0);
        EXPECT_EQ(scratch.d()[i], 0);
    }
}

TEST(JunkerVolgenantSolver, JVScratch_InitGrowsBuffer) {
    JVScratch scratch(2);
    EXPECT_EQ(scratch.dim(), 2);

    scratch.init(8);
    EXPECT_EQ(scratch.dim(), 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(scratch.perm()[i], 0);
        EXPECT_EQ(scratch.in_col()[i], 0);
        EXPECT_EQ(scratch.v()[i], 0);
        EXPECT_EQ(scratch.free()[i], 0);
        EXPECT_EQ(scratch.collist()[i], 0);
        EXPECT_EQ(scratch.matches()[i], 0);
        EXPECT_EQ(scratch.pred()[i], 0);
        EXPECT_EQ(scratch.d()[i], 0);
    }
}

TEST(JunkerVolgenantSolver, JVScratch_ResetZeroesInPlace) {
    JVScratch scratch(3);

    for (int i = 0; i < 3; i++) {
        scratch.perm()[i] = 99;
        scratch.in_col()[i] = 99;
        scratch.v()[i] = 99;
        scratch.free()[i] = 99;
        scratch.collist()[i] = 99;
        scratch.matches()[i] = 99;
        scratch.pred()[i] = 99;
        scratch.d()[i] = 99;
    }

    scratch.reset();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(scratch.perm()[i], 0);
        EXPECT_EQ(scratch.in_col()[i], 0);
        EXPECT_EQ(scratch.v()[i], 0);
        EXPECT_EQ(scratch.free()[i], 0);
        EXPECT_EQ(scratch.collist()[i], 0);
        EXPECT_EQ(scratch.matches()[i], 0);
        EXPECT_EQ(scratch.pred()[i], 0);
        EXPECT_EQ(scratch.d()[i], 0);
    }
}

// ---------------------------------------------------------------------------
// compute_assignment — convenience overload (returns std::vector<int>)
// ---------------------------------------------------------------------------

TEST(JunkerVolgenantSolver, IdentityMatrix) {
    const int dim = 3;
    int matrix[9] = {
        0, 10, 10,
        10, 0, 10,
        10, 10, 0
    };

    std::vector<int> assignment = compute_assignment(matrix, dim);
    ASSERT_FALSE(assignment.empty());

    EXPECT_EQ(assignment[0], 0);
    EXPECT_EQ(assignment[1], 1);
    EXPECT_EQ(assignment[2], 2);
}

TEST(JunkerVolgenantSolver, PermutedOptimalAssignment) {
    const int dim = 3;
    int matrix[9] = {
        10,  0, 10,
        10, 10,  0,
         0, 10, 10
    };

    std::vector<int> assignment = compute_assignment(matrix, dim);
    ASSERT_FALSE(assignment.empty());

    EXPECT_EQ(assignment[0], 1);
    EXPECT_EQ(assignment[1], 2);
    EXPECT_EQ(assignment[2], 0);
}

TEST(JunkerVolgenantSolver, Dimension1) {
    const int dim = 1;
    int matrix[1] = { 42 };

    std::vector<int> assignment = compute_assignment(matrix, dim);
    ASSERT_FALSE(assignment.empty());

    EXPECT_EQ(assignment[0], 0);
}

TEST(JunkerVolgenantSolver, LargerMatrixOptimalAssignment) {
    const int dim = 4;
    int matrix[16] = {
        10, 10, 10,  0,
        10, 10,  0, 10,
        10,  0, 10, 10,
         0, 10, 10, 10
    };

    std::vector<int> assignment = compute_assignment(matrix, dim);
    ASSERT_EQ(assignment.size(), static_cast<size_t>(dim));

    EXPECT_EQ(assignment[0], 3);
    EXPECT_EQ(assignment[1], 2);
    EXPECT_EQ(assignment[2], 1);
    EXPECT_EQ(assignment[3], 0);
}

TEST(JunkerVolgenantSolver, AssignmentIsAValidPermutation) {
    const int dim = 5;
    int matrix[25] = {
        5, 3, 8, 1, 9,
        2, 7, 4, 6, 3,
        8, 1, 5, 9, 2,
        3, 8, 2, 7, 4,
        9, 4, 6, 3, 1
    };

    std::vector<int> assignment = compute_assignment(matrix, dim);
    ASSERT_EQ(assignment.size(), static_cast<size_t>(dim));

    std::vector<int> sorted(assignment);
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < dim; i++) {
        EXPECT_EQ(sorted[i], i);
    }
}

// ---------------------------------------------------------------------------
// compute_assignment — scratch overload (zero-allocation path)
// ---------------------------------------------------------------------------

TEST(JunkerVolgenantSolver, ScratchOverloadProducesSameResult) {
    const int dim = 3;
    int matrix[9] = {
        10,  0, 10,
        10, 10,  0,
         0, 10, 10
    };

    std::vector<int> expected = compute_assignment(matrix, dim);

    JVScratch scratch(dim);
    compute_assignment(matrix, dim, scratch);
    std::vector<int> actual(scratch.perm(), scratch.perm() + dim);

    ASSERT_EQ(actual.size(), expected.size());
    for (int i = 0; i < dim; i++) {
        EXPECT_EQ(actual[i], expected[i]);
    }
}

TEST(JunkerVolgenantSolver, ScratchReuseAcrossMultipleCalls) {
    const int dim = 4;
    int matrix_a[16] = {
        0, 10, 10, 10,
        10, 0, 10, 10,
        10, 10, 0, 10,
        10, 10, 10, 0
    };
    int matrix_b[16] = {
        10, 10, 10, 0,
        10, 10, 0, 10,
        10, 0, 10, 10,
        0, 10, 10, 10
    };

    JVScratch scratch(dim);

    compute_assignment(matrix_a, dim, scratch);
    std::vector<int> result_a(scratch.perm(), scratch.perm() + dim);
    EXPECT_EQ(result_a[0], 0);
    EXPECT_EQ(result_a[1], 1);
    EXPECT_EQ(result_a[2], 2);
    EXPECT_EQ(result_a[3], 3);

    compute_assignment(matrix_b, dim, scratch);
    std::vector<int> result_b(scratch.perm(), scratch.perm() + dim);
    EXPECT_EQ(result_b[0], 3);
    EXPECT_EQ(result_b[1], 2);
    EXPECT_EQ(result_b[2], 1);
    EXPECT_EQ(result_b[3], 0);
}

TEST(JunkerVolgenantSolver, ScratchInitOnDimMismatch) {
    const int dim = 3;
    int matrix[9] = {
        0, 10, 10,
        10, 0, 10,
        10, 10, 0
    };

    JVScratch scratch(1);
    compute_assignment(matrix, dim, scratch);

    EXPECT_EQ(scratch.dim(), dim);
    std::vector<int> result(scratch.perm(), scratch.perm() + dim);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
}
