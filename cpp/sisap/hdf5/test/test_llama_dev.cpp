// test_llama_dev.cpp — Detailed tests verifying C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5
//
// Verifies structures, offsets, shapes, and exact contents for llama-dev dataset.

#include <cstdint>
#include <fstream>
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

static const char* LLAMA_H5 = "C:\\Data\\ANN\\sisap2026\\llama-dev\\llama-dev.h5";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  Dataset Structure and Offsets
// ---------------------------------------------------------------------------

TEST(Hdf5LlamaDev, MetadataAndOffsets) {
    if (!file_exists(LLAMA_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(LLAMA_H5);
    EXPECT_EQ(datasets.size(), 4u) << "Expected exactly 4 datasets in llama-dev.h5";

    // 1. train
    {
        auto it = datasets.find("train");
        ASSERT_NE(it, datasets.end()) << "Dataset 'train' not found";
        EXPECT_EQ(it->second.num_rows, 256921u);
        EXPECT_EQ(it->second.num_cols, 128u);
        EXPECT_EQ(it->second.element_size, 4u); // FP32
        EXPECT_EQ(it->second.file_offset, 2048u);
        EXPECT_EQ(it->second.total_bytes, 131543552u);
    }

    // 2. test/queries
    {
        auto it = datasets.find("test/queries");
        ASSERT_NE(it, datasets.end()) << "Dataset 'test/queries' not found";
        EXPECT_EQ(it->second.num_rows, 1000u);
        EXPECT_EQ(it->second.num_cols, 128u);
        EXPECT_EQ(it->second.element_size, 4u); // FP32
        EXPECT_EQ(it->second.file_offset, 131547648u);
        EXPECT_EQ(it->second.total_bytes, 512000u);
    }

    // 3. test/dists
    {
        auto it = datasets.find("test/dists");
        ASSERT_NE(it, datasets.end()) << "Dataset 'test/dists' not found";
        EXPECT_EQ(it->second.num_rows, 1000u);
        EXPECT_EQ(it->second.num_cols, 100u);
        EXPECT_EQ(it->second.element_size, 8u); // FP64 / Double
        EXPECT_EQ(it->second.file_offset, 132059648u);
        EXPECT_EQ(it->second.total_bytes, 800000u);
    }

    // 4. test/knns
    {
        auto it = datasets.find("test/knns");
        ASSERT_NE(it, datasets.end()) << "Dataset 'test/knns' not found";
        EXPECT_EQ(it->second.num_rows, 1000u);
        EXPECT_EQ(it->second.num_cols, 100u);
        EXPECT_EQ(it->second.element_size, 8u); // INT64
        EXPECT_EQ(it->second.file_offset, 132861696u);
        EXPECT_EQ(it->second.total_bytes, 800000u);
    }
}

// ---------------------------------------------------------------------------
//  Data Values Verification
// ---------------------------------------------------------------------------

TEST(Hdf5LlamaDev, VerifyTrainDataValues) {
    if (!file_exists(LLAMA_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(LLAMA_H5);
    auto it = datasets.find("train");
    ASSERT_NE(it, datasets.end());

    // Read the first chunk (1 row) to verify values
    auto data = read_matrix_fp32(LLAMA_H5, it->second, 0, 1);
    ASSERT_EQ(data.size(), 1u);
    ASSERT_EQ(data[0].size(), 128u);

    // Expected preview: [ 2.879948, 0.6283767, 1.5702908, -2.4001567, -0.6117648, 0.2660305, -2.2627392, -0.34680706, 0.4558941, 0.8036741 ]
    std::vector<float> expected = { 2.879948f, 0.6283767f, 1.5702908f, -2.4001567f, -0.6117648f, 0.2660305f, -2.2627392f, -0.34680706f, 0.4558941f, 0.8036741f };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(data[0][i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(Hdf5LlamaDev, VerifyTestQueriesValues) {
    if (!file_exists(LLAMA_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(LLAMA_H5);
    auto it = datasets.find("test/queries");
    ASSERT_NE(it, datasets.end());

    auto data = read_matrix_fp32(LLAMA_H5, it->second, 0, 1);
    ASSERT_EQ(data.size(), 1u);
    ASSERT_EQ(data[0].size(), 128u);

    // Expected preview: [-0.04855045, 0.09911887, 0.5128767, -0.13893484, 0.34128433, 0.14251602, -0.7772025, 0.79634196, -0.762619, 0.09785306]
    std::vector<float> expected = { -0.04855045f, 0.09911887f, 0.5128767f, -0.13893484f, 0.34128433f, 0.14251602f, -0.7772025f, 0.79634196f, -0.762619f, 0.09785306f };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(data[0][i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(Hdf5LlamaDev, VerifyTestDistsValues) {
    if (!file_exists(LLAMA_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(LLAMA_H5);
    auto it = datasets.find("test/dists");
    ASSERT_NE(it, datasets.end());

    auto data = read_matrix_fp64(LLAMA_H5, it->second, 0, 1);
    ASSERT_EQ(data.size(), 1u);
    ASSERT_EQ(data[0].size(), 100u);

    // Expected preview: [-35.93835068, -35.37674713, -34.76631927, -34.5580101, -34.49694824, -34.43302536, -34.36135101, -34.3237114, -34.27835083, -34.25453949]
    std::vector<double> expected = { -35.93835068, -35.37674713, -34.76631927, -34.5580101, -34.49694824, -34.43302536, -34.36135101, -34.3237114, -34.27835083, -34.25453949 };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(data[0][i], expected[i], 1e-5) << "Mismatch at index " << i;
    }
}

TEST(Hdf5LlamaDev, VerifyTestKnnsValues) {
    if (!file_exists(LLAMA_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(LLAMA_H5);
    auto it = datasets.find("test/knns");
    ASSERT_NE(it, datasets.end());

    auto data = read_matrix_int64(LLAMA_H5, it->second, 0, 1);
    ASSERT_EQ(data.size(), 1u);
    ASSERT_EQ(data[0].size(), 100u);

    // Expected preview: [174516, 253500, 201592, 201589, 253501, 219066, 215069, 161656, 245049, 232377]
    std::vector<int64_t> expected = { 174516, 253500, 201592, 201589, 253501, 219066, 215069, 161656, 245049, 232377 };
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(data[0][i], expected[i]) << "Mismatch at index " << i;
    }
}
