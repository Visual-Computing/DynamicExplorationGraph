#include <vector>

#include "gtest/gtest.h"
#include "test_helpers.h"

static void test_l2_matches_naive() {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = deglib::distances::L2Float::compare(v.data(), v.data(), &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Float, KnownValue) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    size_t dim = 3;
    float d = deglib::distances::L2Float::compare(a, b, &dim);
    EXPECT_NEAR(d, 27.0f, 1e-4f);
}

TEST(L2Float, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    size_t dim = a.size();
    float ab = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::L2Float::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Float, MatchesNaive_4) {
    auto a = make_float_vec(DIM_4);
    auto b = make_float_vec(DIM_4, 7);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Float, MatchesNaive_64) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 13);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-3f);
}

TEST(L2Float, MatchesNaive_128) {
    auto a = make_float_vec(DIM_128);
    auto b = make_float_vec(DIM_128, 21);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-3f);
}

TEST(L2Float, Dim1) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    size_t dim = 1;
    float d = deglib::distances::L2Float::compare(a, b, &dim);
    EXPECT_NEAR(d, 16.0f, 1e-4f);
}

TEST(L2Float, Dim0) {
    float a = 1.0f;
    float b = 2.0f;
    size_t dim = 0;
    float d = deglib::distances::L2Float::compare(&a, &b, &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Float4Ext, MatchesNaive) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float4Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float8Ext, MatchesNaive) {
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float8Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float16Ext, MatchesNaive) {
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float16Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}


TEST(FP32L2_Batch, CascadingBatch4) {
    // dim%4==0 but dim%8!=0 -> should use cascading AVX batch (AVX main + SSE tail)
    std::vector<size_t> dims = {12, 20, 28, 36, 44, 52, 60, 100, 124};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        auto db0 = make_float_vec(dim, 1);
        auto db1 = make_float_vec(dim, 2);
        auto db2 = make_float_vec(dim, 3);
        auto db3 = make_float_vec(dim, 4);

        const void* db_arr[4] = {db0.data(), db1.data(), db2.data(), db3.data()};
        std::vector<float> dists(4);
        deglib::distances::L2Float4Ext::compare_batch(query.data(), db_arr, 4, &dim, dists.data());

        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_arr[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
}

TEST(FP32L2_Batch, CascadingBatch8) {
    // dim%4==0 but dim%8!=0 -> should use cascading AVX batch (AVX main + SSE tail)
    std::vector<size_t> dims = {12, 20, 28, 36, 100};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(8);
        const void* db_ptrs[8];
        for (int j = 0; j < 8; ++j) {
            dbs[j] = make_float_vec(dim, j + 10);
            db_ptrs[j] = dbs[j].data();
        }

        std::vector<float> dists(8);
        deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 8, &dim, dists.data());

        for (int j = 0; j < 8; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
}

TEST(FP32L2_Batch, BatchRemainingWithCascade) {
    // count=13 -> 8 (batch8) + 4 (batch4) + 1 (compare) with cascading
    size_t dim = 100;
    auto query = make_float_vec(dim);
    std::vector<std::vector<float>> dbs(13);
    const void* db_ptrs[13];
    for (int j = 0; j < 13; ++j) {
        dbs[j] = make_float_vec(dim, j * 7);
        db_ptrs[j] = dbs[j].data();
    }

    std::vector<float> dists(13);
    deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());

    for (int j = 0; j < 13; ++j) {
        EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
            << "db=" << j;
    }
}

TEST(FP32L2_Batch, Batch4Aligned8) {
    // dim%8==0 via L2Float8Ext -> no tail, batch4 covers 4 vectors
    std::vector<size_t> dims = {8, 16, 32, 64, 128};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        auto db0 = make_float_vec(dim, 10);
        auto db1 = make_float_vec(dim, 20);
        auto db2 = make_float_vec(dim, 30);
        auto db3 = make_float_vec(dim, 40);

        const void* db_arr[4] = {db0.data(), db1.data(), db2.data(), db3.data()};
        std::vector<float> dists(4);
        deglib::distances::L2Float8Ext::compare_batch(query.data(), db_arr, 4, &dim, dists.data());

        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_arr[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
}

TEST(FP32L2_Batch, Batch8Aligned16) {
    // dim%16==0 via L2Float16Ext -> no tail, batch8 covers 8 vectors
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(8);
        const void* db_ptrs[8];
        for (int j = 0; j < 8; ++j) {
            dbs[j] = make_float_vec(dim, (j + 1) * 5);
            db_ptrs[j] = dbs[j].data();
        }

        std::vector<float> dists(8);
        deglib::distances::L2Float16Ext::compare_batch(query.data(), db_ptrs, 8, &dim, dists.data());

        for (int j = 0; j < 8; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
}

TEST(FP32L2_Batch, BatchCountZeroToOne) {
    // count=0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    std::vector<size_t> counts = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<size_t> dims = {4, 8, 12, 16, 20, 100, 128};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(12);
        const void* db_ptrs[12];
        for (int j = 0; j < 12; ++j) {
            dbs[j] = make_float_vec(dim, (j + 1) * 3);
            db_ptrs[j] = dbs[j].data();
        }

        for (size_t cnt : counts) {
            std::vector<float> dists(cnt);
            deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, cnt, &dim, dists.data());

            for (size_t j = 0; j < cnt; ++j) {
                EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
                    << "dim=" << dim << " count=" << cnt << " db=" << j;
            }
        }
    }
}

TEST(FP32L2_Batch, BatchDim4) {
    // dim=4 -> dim/8=0, batch processes nothing, tail handles everything
    size_t dim = 4;
    auto query = make_float_vec(dim);
    std::vector<std::vector<float>> dbs(13);
    const void* db_ptrs[13];
    for (int j = 0; j < 13; ++j) {
        dbs[j] = make_float_vec(dim, (j + 1) * 7);
        db_ptrs[j] = dbs[j].data();
    }

    std::vector<float> dists(13);
    deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());

    for (int j = 0; j < 13; ++j) {
        EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-4f)
            << "db=" << j;
    }
}

TEST(FP32L2_Batch, BatchDim1) {
    // dim=1 -> MIN_ALIGN=1, falls through to individual compare() calls
    size_t dim = 1;
    auto query = make_float_vec(dim);
    std::vector<std::vector<float>> dbs(10);
    const void* db_ptrs[10];
    for (int j = 0; j < 10; ++j) {
        dbs[j] = make_float_vec(dim, (j + 1) * 3);
        db_ptrs[j] = dbs[j].data();
    }

    std::vector<float> dists(10);
    deglib::distances::L2Float::compare_batch(query.data(), db_ptrs, 10, &dim, dists.data());

    for (int j = 0; j < 10; ++j) {
        EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-4f)
            << "db=" << j;
    }
}

TEST(FP32L2_Batch, BatchDim3) {
    // dim=3 -> MIN_ALIGN=1, individual compare() calls
    size_t dim = 3;
    auto query = make_float_vec(dim);
    std::vector<std::vector<float>> dbs(11);
    const void* db_ptrs[11];
    for (int j = 0; j < 11; ++j) {
        dbs[j] = make_float_vec(dim, (j + 1) * 5);
        db_ptrs[j] = dbs[j].data();
    }

    std::vector<float> dists(11);
    deglib::distances::L2Float::compare_batch(query.data(), db_ptrs, 11, &dim, dists.data());

    for (int j = 0; j < 11; ++j) {
        EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-4f)
            << "db=" << j;
    }
}

TEST(FP32L2_Batch, BatchLargeCount) {
    // count=100 to exercise multiple batch8 + batch4 + remainder
    size_t dim = 128;
    auto query = make_float_vec(dim);
    std::vector<std::vector<float>> dbs(100);
    const void* db_ptrs[100];
    for (int j = 0; j < 100; ++j) {
        dbs[j] = make_float_vec(dim, (j + 1) * 2);
        db_ptrs[j] = dbs[j].data();
    }

    std::vector<float> dists(100);
    deglib::distances::L2Float16Ext::compare_batch(query.data(), db_ptrs, 100, &dim, dists.data());

    for (int j = 0; j < 100; ++j) {
        EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
            << "db=" << j;
    }
}

TEST(FP32L2_Batch, BatchConsistencyCompareVsBatch) {
    // compare_batch results must match individual compare() calls for all MIN_ALIGN variants
    std::vector<size_t> dims = {1, 3, 4, 7, 8, 12, 16, 20, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(13);
        const void* db_ptrs[13];
        for (int j = 0; j < 13; ++j) {
            dbs[j] = make_float_vec(dim, (j + 1) * 11);
            db_ptrs[j] = dbs[j].data();
        }

        // L2Float (MIN_ALIGN=1)
        {
            std::vector<float> dists(13);
            deglib::distances::L2Float::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());
            for (int j = 0; j < 13; ++j) {
                EXPECT_NEAR(dists[j], deglib::distances::L2Float::compare(query.data(), db_ptrs[j], &dim), 1e-4f)
                    << "L2Float dim=" << dim << " db=" << j;
            }
        }
        if (dim % 4 == 0) {
            std::vector<float> dists(13);
            deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());
            for (int j = 0; j < 13; ++j) {
                EXPECT_NEAR(dists[j], deglib::distances::L2Float4Ext::compare(query.data(), db_ptrs[j], &dim), 1e-4f)
                    << "L2Float4Ext dim=" << dim << " db=" << j;
            }
        }
        if (dim % 8 == 0) {
            std::vector<float> dists(13);
            deglib::distances::L2Float8Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());
            for (int j = 0; j < 13; ++j) {
                EXPECT_NEAR(dists[j], deglib::distances::L2Float8Ext::compare(query.data(), db_ptrs[j], &dim), 1e-4f)
                    << "L2Float8Ext dim=" << dim << " db=" << j;
            }
        }
        if (dim % 16 == 0) {
            std::vector<float> dists(13);
            deglib::distances::L2Float16Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());
            for (int j = 0; j < 13; ++j) {
                EXPECT_NEAR(dists[j], deglib::distances::L2Float16Ext::compare(query.data(), db_ptrs[j], &dim), 1e-4f)
                    << "L2Float16Ext dim=" << dim << " db=" << j;
            }
        }
    }
}
