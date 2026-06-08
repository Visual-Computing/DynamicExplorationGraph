// test_dataset_dims_large.cpp — Verify dimensions of EACH dataset in the large SISAP HDF5 file
//
// Each dataset gets its own test case for clear PASS/FAIL per dataset.

#include <cstdint>
#include <fstream>

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

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\large\\benchmark-dev-wikipedia-bge-m3.h5";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  One test per dataset
// ---------------------------------------------------------------------------

TEST(Hdf5DatasetDimsLarge, train) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("train");
    ASSERT_NE(it, datasets.end()) << "Dataset 'train' not found";
    EXPECT_EQ(it->second.num_rows, 6350000u);
    EXPECT_EQ(it->second.num_cols, 1024u);
    EXPECT_EQ(it->second.element_size, 2u);
    EXPECT_EQ(it->second.total_bytes, 13004800000u);
}

TEST(Hdf5DatasetDimsLarge, allknn_dists) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("allknn/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'allknn/dists' not found";
    EXPECT_EQ(it->second.num_rows, 6350000u);
    EXPECT_EQ(it->second.num_cols, 32u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 812800000u);
}

TEST(Hdf5DatasetDimsLarge, allknn_knns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("allknn/knns");
    ASSERT_NE(it, datasets.end()) << "Dataset 'allknn/knns' not found";
    EXPECT_EQ(it->second.num_rows, 6350000u);
    EXPECT_EQ(it->second.num_cols, 32u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 812800000u);
}

TEST(Hdf5DatasetDimsLarge, itest_dists) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/dists' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1000u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 40000000u);
}

TEST(Hdf5DatasetDimsLarge, itest_knns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/knns");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/knns' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1000u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 40000000u);
}

TEST(Hdf5DatasetDimsLarge, itest_queries) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/queries");
    ASSERT_NE(it, datasets.end()) << "Dataset 'itest/queries' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1024u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 40960000u);
}

TEST(Hdf5DatasetDimsLarge, otest_dists) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/dists");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/dists' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1000u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 40000000u);
}

TEST(Hdf5DatasetDimsLarge, otest_knns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/knns");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/knns' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1000u);
    EXPECT_EQ(it->second.element_size, 4u);
    EXPECT_EQ(it->second.total_bytes, 40000000u);
}

TEST(Hdf5DatasetDimsLarge, otest_queries) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("otest/queries");
    ASSERT_NE(it, datasets.end()) << "Dataset 'otest/queries' not found";
    EXPECT_EQ(it->second.num_rows, 10000u);
    EXPECT_EQ(it->second.num_cols, 1024u);
    EXPECT_EQ(it->second.element_size, 2u);
    EXPECT_EQ(it->second.total_bytes, 20480000u);
}

// ---------------------------------------------------------------------------
//  scan_datasets returns exactly 9 entries
// ---------------------------------------------------------------------------

TEST(Hdf5DatasetDimsLarge, ScanReturnsAll9Datasets) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    EXPECT_EQ(datasets.size(), 9u) << "Expected 9 datasets, got " << datasets.size();
}
