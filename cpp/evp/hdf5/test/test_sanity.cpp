// test_sanity.cpp — Sanity checks on dataset values WITHOUT ground truth files
//
// Checks non-negative distances, valid indices, and normalized query vectors.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

#include "hdf5_types.h"
#include "hdf5_io.h"
#include "hdf5_superblock.h"
#include "hdf5_heap.h"
#include "hdf5_btree.h"
#include "hdf5_snod.h"
#include "hdf5_ohdr.h"
#include "hdf5_scan.h"
#include "hdf5_readers.h"

#include "gtest/gtest.h"

using namespace hdf5_reader;
using namespace hdf5_reader::detail;

// ---------------------------------------------------------------------------
//  Paths (match the DATA_PATH preset)
// ---------------------------------------------------------------------------

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\small\\benchmark-dev-wikipedia-bge-m3-small.h5";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// fp16 (IEEE 754 binary16) to float32 conversion
static float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 0x1;
    int exp = (h >> 10) & 0x1f;
    int frac = h & 0x3ff;

    float f;
    if (exp == 0) {
        if (frac == 0) {
            f = 0.0f;
        } else {
            f = static_cast<float>(frac) * std::scalbnf(1.0f, -34.0f);
        }
    } else if (exp == 31) {
        f = (frac == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        f = std::scalbnf(1.0f + static_cast<float>(frac) * (1.0f / 1024.0f), exp - 15);
    }
    return sign ? -f : f;
}

// ---------------------------------------------------------------------------
//  allknn/dists: distances should be non-negative and reasonable
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, AllknnDists_NonNegative) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("allknn/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'allknn/dists' not found";

    size_t dims, count;
    auto data = read_dataset_as_floats(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 32u);
    EXPECT_EQ(count, 200000u);

    int idx = 0;
    for (size_t i = 0; i < 1000 && i < count; i++) {
        for (size_t j = 0; j < dims && idx < 1000; j++, idx++) {
            EXPECT_GE(data[i][j], -0.01f) << "Negative distance at row " << i << " col " << j;
            EXPECT_LT(data[i][j], 10000.0f) << "Distance too large (" << data[i][j] << ") at row " << i << " col " << j;
        }
    }
}

// ---------------------------------------------------------------------------
//  itest/knns: all indices must be valid (1 to train_size)
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, ItestKnns_IndicesValid) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/knns");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/knns' not found";

    size_t dims, count;
    auto data = read_dataset_as_ints(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1000u);
    EXPECT_EQ(count, 10000u);

    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < dims; j++) {
            int32_t idx = data[i][j];
            EXPECT_GE(idx, 1) << "Negative index at row " << i << " col " << j;
            EXPECT_LE(idx, 200000) << "Index > train size at row " << i << " col " << j << " (value=" << idx << ")";
        }
    }
}

// ---------------------------------------------------------------------------
//  itest/dists: distances should be non-negative and reasonable
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, ItestDists_NonNegative) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/dists' not found";

    size_t dims, count;
    auto data = read_dataset_as_floats(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1000u);
    EXPECT_EQ(count, 10000u);

    for (int i = 0; i < 1000; i++) {
        EXPECT_GE(data[0][i], 0.0f) << "Negative distance at index " << i;
        EXPECT_LT(data[0][i], 10000.0f) << "Distance too large (" << data[0][i] << ") at index " << i;
    }
}

// ---------------------------------------------------------------------------
//  itest/queries: vectors should be L2-normalized (norm ≈ 1.0)
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, ItestQueries_Normalized) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/queries");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/queries' not found";

    size_t dims, count;
    auto data = read_dataset_as_floats(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1024u);
    EXPECT_EQ(count, 10000u);

    for (size_t i = 0; i < 100; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < dims; j++) {
            float v = data[i][j];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        EXPECT_NEAR(norm, 1.0f, 0.1f) << "Query vector at row " << i << " not normalized (norm=" << norm << ")";
    }
}

// ---------------------------------------------------------------------------
//  otest/knns: all indices must be valid (0 to train_size-1)
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, OtestKnns_IndicesValid) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/knns");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/knns' not found";

    size_t dims, count;
    auto data = read_dataset_as_ints(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1000u);
    EXPECT_EQ(count, 10000u);

    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < dims; j++) {
            int32_t idx = data[i][j];
            EXPECT_GE(idx, 0) << "Negative index at row " << i << " col " << j;
            EXPECT_LT(idx, 200000) << "Index >= train size at row " << i << " col " << j << " (value=" << idx << ")";
        }
    }
}

// ---------------------------------------------------------------------------
//  otest/dists: distances should be non-negative and reasonable
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, OtestDists_NonNegative) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/dists' not found";

    size_t dims, count;
    auto data = read_dataset_as_floats(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1000u);
    EXPECT_EQ(count, 10000u);

    for (int i = 0; i < 1000; i++) {
        EXPECT_GE(data[0][i], 0.0f) << "Negative distance at index " << i;
        EXPECT_LT(data[0][i], 10000.0f) << "Distance too large (" << data[0][i] << ") at index " << i;
    }
}

// ---------------------------------------------------------------------------
//  otest/queries: fp16 vectors should be L2-normalized (norm ≈ 1.0)
// ---------------------------------------------------------------------------

TEST(Hdf5Sanity, OtestQueries_Normalized) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/queries");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/queries' not found";
    EXPECT_EQ(it->second.element_size, 2u);

    auto byte_vecs = read_dataset_as_vecs(SISAP_H5, it->second);
    size_t dims = it->second.num_cols;

    for (size_t i = 0; i < 100 && i < byte_vecs.size(); i++) {
        const std::vector<std::byte>& row = byte_vecs[i];
        const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(row.data());

        float norm = 0.0f;
        for (size_t j = 0; j < dims; j++) {
            float v = fp16_to_float(fp16_data[j]);
            norm += v * v;
        }
        norm = std::sqrt(norm);
        EXPECT_NEAR(norm, 1.0f, 0.1f) << "Query vector at row " << i << " not normalized (norm=" << norm << ")";
    }
}
