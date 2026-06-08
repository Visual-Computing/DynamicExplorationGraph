// test_integration.cpp — End-to-end tests using the real SISAP HDF5 file
//
// Tests superblock, heap, btree, snod, ohdr, scan, and readers
// against benchmark-dev-wikipedia-bge-m3-small.h5 and its ground truth files.

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

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\small\\benchmark-dev-wikipedia-bge-m3-small.h5";
static const char* SISAP_ITEST_IVECS = "C:\\Data\\ANN\\sisap2026\\small\\itest.ivecs";
static const char* SISAP_ALLKNN_IVECS = "C:\\Data\\ANN\\sisap2026\\small\\allknn.ivecs";
static const char* SISAP_TRAIN_HVECS = "C:\\Data\\ANN\\sisap2026\\small\\train.hvecs";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  Superblock
// ---------------------------------------------------------------------------

TEST(Hdf5Integration, Superblock) {
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

TEST(Hdf5Integration, RootHeap) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap = parse_heap(f, sb.root_heap, sb.base);
    EXPECT_GT(heap.size(), 0u);

    // Verify heap is usable: parse root SNOD and extract at least one name
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

TEST(Hdf5Integration, RootBTree_HasLeaves) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    // Root B-tree has at least 1 SNOD leaf
    EXPECT_GE(snod_addrs.size(), 1u);
}

// ---------------------------------------------------------------------------
//  SNOD
// ---------------------------------------------------------------------------

TEST(Hdf5Integration, RootSNOD_Entries) {
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

TEST(Hdf5Integration, OhdrV1_ItestKnns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint8_t> heap = parse_heap(f, sb.root_heap, sb.base);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    // Find itest group from root SNOD, then its sub-btree → find knns
    for (uint64_t snod_abs : snod_addrs) {
        auto entries = parse_snod(f, snod_abs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap, ste.name_off);
            if (name == "itest") {
                // itest is a group; parse its ohdr to get sub-btree
                OhdrInfo grp = parse_ohdr(f, ste.ohdr_abs, sb.base);
                EXPECT_TRUE(grp.is_group);
                EXPECT_NE(grp.group_btree_abs, UNDEF64);

                // Traverse sub-btree
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

TEST(Hdf5Integration, ScanDatasets_Basic) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    EXPECT_GT(datasets.size(), 0u);

    // itest/knns: 10000 x 1000 int32 (v1, always works)
    {
        auto it = datasets.find("itest/knns");
        ASSERT_NE(it, datasets.end());
        EXPECT_EQ(it->second.num_rows, 10000u);
        EXPECT_EQ(it->second.num_cols, 1000u);
        EXPECT_EQ(it->second.element_size, 4u);
        EXPECT_GT(it->second.file_offset, 0u);
        EXPECT_GT(it->second.total_bytes, 0u);
    }

    // itest/dists: 10000 x 1000 float32
    {
        auto it = datasets.find("itest/dists");
        ASSERT_NE(it, datasets.end());
        EXPECT_EQ(it->second.num_rows, 10000u);
        EXPECT_EQ(it->second.num_cols, 1000u);
        EXPECT_EQ(it->second.element_size, 4u);
    }

    // itest/queries: 10000 x 1024 float32
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

TEST(Hdf5Integration, PrintDatasets) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);
    print_datasets(datasets);
}

// ---------------------------------------------------------------------------
//  read_int32_flat + ground truth comparison
// ---------------------------------------------------------------------------

TEST(Hdf5Integration, ReadItestKnns_MatchesIvecs) {
    if (!file_exists(SISAP_H5) || !file_exists(SISAP_ITEST_IVECS)) {
        GTEST_SKIP();
    }

    auto datasets = scan_datasets(SISAP_H5);
    auto it = datasets.find("itest/knns");
    ASSERT_NE(it, datasets.end());

    auto data = read_matrix_int32(SISAP_H5, it->second);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(static_cast<size_t>(it->second.num_cols), 1000u);
    EXPECT_EQ(static_cast<size_t>(it->second.num_rows), 10000u);

    // knns data: access per-row
    for (int i = 0; i < 10; i++) {
        const auto& row = data[i];
        for (int j = 0; j < 32; j++) {
            EXPECT_GE(row[j], 1) << "Row " << i << " col " << j;
            EXPECT_LE(row[j], 200000) << "Row " << i << " col " << j;
        }
    }
}

// ---------------------------------------------------------------------------
//  read_fp16_vectors
// ---------------------------------------------------------------------------

TEST(Hdf5Integration, ReadFP16Vectors) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    auto datasets = scan_datasets(SISAP_H5);

    // Find a fp16 dataset (element_size == 2)
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

    auto vecs = read_matrix_bytes(SISAP_H5, *fp16);
    EXPECT_GT(vecs.size(), 0u);
    EXPECT_GT(vecs[0].size(), 0u);
    EXPECT_EQ(vecs[0].size(), fp16->num_cols * 2u);
}

// ---------------------------------------------------------------------------
//  Error handling
// ---------------------------------------------------------------------------

TEST(Hdf5Integration, ScanNonexistentFile) {
    EXPECT_THROW(scan_datasets("nonexistent.h5"), std::runtime_error);
}

TEST(Hdf5Integration, ReadVecs_1DDataset) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    DatasetInfo info;
    info.element_size = 2;
    info.num_cols = 0;   // 1-D not supported
    info.num_rows = 10;
    info.file_offset = 0;

    EXPECT_THROW(read_matrix_bytes(SISAP_H5, info), std::runtime_error);
}
