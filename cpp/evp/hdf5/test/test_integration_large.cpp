// test_integration_large.cpp — End-to-end tests using the large SISAP HDF5 file
//
// Tests superblock, heap, btree, snod, ohdr, scan, and readers
// against benchmark-dev-wikipedia-bge-m3.h5 and its ground truth files.

#include <cstdint>
#include <cstring>
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

// ---------------------------------------------------------------------------
//  Paths (match the DATA_PATH preset)
// ---------------------------------------------------------------------------

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\large\\benchmark-dev-wikipedia-bge-m3.h5";
static const char* SISAP_ALLKNN_IVECS = "C:\\Data\\ANN\\sisap2026\\large\\allknn.ivecs";
static const char* SISAP_TRAIN_HVECS = "C:\\Data\\ANN\\sisap2026\\large\\train.hvecs";
static const char* SISAP_TRAIN_FVECS = "C:\\Data\\ANN\\sisap2026\\large\\train.fvecs";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  Superblock
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, Superblock) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    EXPECT_GT(sb.base, 0u);
    EXPECT_GT(sb.root_ohdr, 0u);
    EXPECT_NE(sb.root_btree, UNDEF64);
    EXPECT_NE(sb.root_heap, UNDEF64);
}

// ---------------------------------------------------------------------------
//  Heap
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, RootHeap) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap = parse_heap(f, sb.root_heap, sb.base);
    EXPECT_GT(heap.size(), 0u);

    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    ASSERT_GT(snod_addrs.size(), 0u);
    auto entries = parse_snod(f, snod_addrs[0], sb.base);
    ASSERT_GT(entries.size(), 0u);
    std::string first_name = heap_str(heap, entries[0].name_off);
    EXPECT_FALSE(first_name.empty());
}

// ---------------------------------------------------------------------------
//  B-Tree
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, RootBTree_HasLeaves) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    EXPECT_GE(snod_addrs.size(), 1u);
}

// ---------------------------------------------------------------------------
//  SNOD
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, RootSNOD_Entries) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    ASSERT_GT(snod_addrs.size(), 0u);

    auto entries = parse_snod(f, snod_addrs[0], sb.base);
    EXPECT_GT(entries.size(), 0u);
}

// ---------------------------------------------------------------------------
//  OHdr v1 (itest/knns uses v1)
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, OhdrV1_ItestKnns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint8_t> heap = parse_heap(f, sb.root_heap, sb.base);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    for (uint64_t snod_abs : snod_addrs) {
        auto entries = parse_snod(f, snod_abs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap, ste.name_off);
            if (name == "itest") {
                OhdrInfo grp = parse_ohdr(f, ste.ohdr_abs, sb.base);
                EXPECT_TRUE(grp.is_group);
                EXPECT_NE(grp.group_btree_abs, UNDEF64);

                std::vector<uint64_t> sub_snods;
                traverse_btree(f, grp.group_btree_abs, sb.base, sub_snods);
                ASSERT_GT(sub_snods.size(), 0u);

                std::vector<uint8_t> sub_heap = parse_heap(f, grp.group_heap_abs, sb.base);
                auto sub_entries = parse_snod(f, sub_snods[0], sb.base);
                for (const auto& sub_ste : sub_entries) {
                    std::string sub_name = heap_str(sub_heap, sub_ste.name_off);
                    if (sub_name == "knns") {
                        OhdrInfo info = parse_ohdr(f, sub_ste.ohdr_abs, sb.base);
                        EXPECT_EQ(info.dim0, 10000u);
                        EXPECT_EQ(info.dim1, 1000u);
                        EXPECT_EQ(info.elem_size, 4u);
                        EXPECT_GT(info.data_abs, 0u);
                        EXPECT_FALSE(info.is_group);
                        return;
                    }
                }
            }
        }
    }
    FAIL() << "itest/knns not found in file";
}

// ---------------------------------------------------------------------------
//  scan_datasets
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ScanDatasets_Basic) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    EXPECT_GT(datasets.size(), 0u);

    {
        auto it = datasets.find("itest/knns");
        ASSERT_NE(it, datasets.end());
        EXPECT_EQ(it->second.num_rows, 10000u);
        EXPECT_EQ(it->second.num_cols, 1000u);
        EXPECT_EQ(it->second.element_size, 4u);
        EXPECT_GT(it->second.file_offset, 0u);
        EXPECT_GT(it->second.total_bytes, 0u);
    }

    {
        auto it = datasets.find("itest/dists");
        ASSERT_NE(it, datasets.end());
        EXPECT_EQ(it->second.num_rows, 10000u);
        EXPECT_EQ(it->second.num_cols, 1000u);
        EXPECT_EQ(it->second.element_size, 4u);
    }

    {
        auto it = datasets.find("itest/queries");
        ASSERT_NE(it, datasets.end());
        EXPECT_EQ(it->second.num_rows, 10000u);
        EXPECT_EQ(it->second.num_cols, 1024u);
        EXPECT_EQ(it->second.element_size, 4u);
    }
}

// ---------------------------------------------------------------------------
//  print_datasets (should not throw)
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, PrintDatasets) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    print_datasets(datasets);
}

// ---------------------------------------------------------------------------
//  read_int32_flat + ground truth comparison
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ReadAllknnKnns_MatchesIvecs) {
    if (!file_exists(SISAP_H5) || !file_exists(SISAP_ALLKNN_IVECS)) {
        GTEST_SKIP();
    }

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("allknn/knns");
    ASSERT_NE(it, datasets.end());

    size_t dims, count;
    auto data = read_dataset_as_ints(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 32u);
    EXPECT_EQ(count, 6350000u);

    std::ifstream f(SISAP_ALLKNN_IVECS, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    int match = 0, total = 0;
    for (int i = 0; i < 100 && i < static_cast<int>(count); i++) {
        uint32_t row_dim;
        f.read(reinterpret_cast<char*>(&row_dim), 4);
        EXPECT_EQ(row_dim, 32u);

        std::vector<int32_t> ivecs_row(32);
        f.read(reinterpret_cast<char*>(ivecs_row.data()), 32 * 4);
        for (int j = 0; j < 32; j++) {
            total++;
            if (data[i][j] == ivecs_row[j]) {
                match++;
            }
        }
    }
    EXPECT_EQ(match, total) << "Matched " << match << "/" << total;
}

// ---------------------------------------------------------------------------
//  read_fp16_vectors + ground truth comparison (train.hvecs)
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ReadTrain_MatchesHvecs) {
    if (!file_exists(SISAP_H5) || !file_exists(SISAP_TRAIN_HVECS)) {
        GTEST_SKIP();
    }

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("train");
    ASSERT_NE(it, datasets.end());

    EXPECT_EQ(it->second.element_size, 2u);
    EXPECT_EQ(it->second.num_rows, 6350000u);
    EXPECT_EQ(it->second.num_cols, 1024u);

    std::ifstream f(SISAP_TRAIN_HVECS, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    auto vecs = read_dataset_as_vecs(SISAP_H5, it->second);
    const size_t bytes_per_row = it->second.num_cols * 2u;

    int match = 0, total = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t row_dim;
        f.read(reinterpret_cast<char*>(&row_dim), 4);
        EXPECT_EQ(row_dim, 1024u);

        std::vector<uint8_t> hvecs_row(1024 * 2);
        f.read(reinterpret_cast<char*>(hvecs_row.data()), 1024 * 2);

        const std::vector<std::byte>& hdf5_row = vecs[i];
        for (size_t j = 0; j < bytes_per_row; j++) {
            total++;
            if (static_cast<uint8_t>(hdf5_row[j]) == hvecs_row[j]) {
                match++;
            }
        }
    }
    EXPECT_EQ(match, total) << "Matched " << match << "/" << total;
}

// ---------------------------------------------------------------------------
//  read_int32_flat on itest/dists
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ReadItestDists_Positive) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/dists");
    ASSERT_NE(it, datasets.end());

    size_t dims, count;
    auto data = read_dataset_as_ints(SISAP_H5, it->second, dims, count);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(dims, 1000u);
    EXPECT_EQ(count, 10000u);

    const float* data_f = reinterpret_cast<const float*>(data[0].data());
    for (int i = 0; i < 20; i++) {
        EXPECT_GE(data_f[i], 0.0f) << "Row 0, col " << i;
    }
}

// ---------------------------------------------------------------------------
//  read_fp16_vectors
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ReadFP16Vectors) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);

    DatasetInfo* fp16 = nullptr;
    for (auto& [name, info] : datasets) {
        if (info.element_size == 2 && info.num_cols > 0) {
            fp16 = &info;
            break;
        }
    }

    if (!fp16) {
        GTEST_SKIP() << "No FP16 dataset found";
    }

    auto vecs = read_dataset_as_vecs(SISAP_H5, *fp16);
    EXPECT_GT(vecs.size(), 0u);
    EXPECT_GT(vecs[0].size(), 0u);
    EXPECT_EQ(vecs[0].size(), fp16->num_cols * 2u);
}

// ---------------------------------------------------------------------------
//  Error handling
// ---------------------------------------------------------------------------

TEST(Hdf5IntegrationLarge, ScanNonexistentFile) {
    EXPECT_THROW(scan_datasets("nonexistent.h5"), std::runtime_error);
}

TEST(Hdf5IntegrationLarge, ReadVecs_1DDataset) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    DatasetInfo info;
    info.element_size = 2;
    info.num_cols = 0;
    info.num_rows = 10;
    info.file_offset = 0;

    EXPECT_THROW(read_dataset_as_vecs(SISAP_H5, info), std::runtime_error);
}
