// test_structural.cpp — Structural parser tests (heap, btree, snod) against real file

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "hdf5_types.h"
#include "hdf5_io.h"
#include "hdf5_superblock.h"
#include "hdf5_heap.h"
#include "hdf5_btree.h"
#include "hdf5_snod.h"
#include "hdf5_ohdr.h"

#include "gtest/gtest.h"

using namespace hdf5_reader;
using namespace hdf5_reader::detail;

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\small\\benchmark-dev-wikipedia-bge-m3-small.h5";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  Superblock
// ---------------------------------------------------------------------------

TEST(Hdf5Structural, Superblock_Valid) {
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

TEST(Hdf5Structural, RootHeap_NonEmpty) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap = parse_heap(f, sb.root_heap, sb.base);
    EXPECT_GT(heap.size(), 0u);

    // Verify that heap_str can extract at least one name from the root SNOD
    auto heap_root = heap;
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    ASSERT_GT(snod_addrs.size(), 0u);

    auto entries = parse_snod(f, snod_addrs[0], sb.base);
    ASSERT_GT(entries.size(), 0u);
    std::string first_name = heap_str(heap_root, entries[0].name_off);
    EXPECT_FALSE(first_name.empty());
}

TEST(Hdf5Structural, GroupHeap_Itest) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);

    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    uint64_t itest_btree = UNDEF64, itest_heap = UNDEF64;
    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_root, ste.name_off);
            if (name == "itest") {
                OhdrInfo gi = parse_ohdr(f, ste.ohdr_abs, sb.base);
                itest_btree = gi.group_btree_abs;
                itest_heap = gi.group_heap_abs;
                break;
            }
        }
        if (itest_btree != UNDEF64) break;
    }
    ASSERT_NE(itest_btree, UNDEF64);

    // Parse itest sub-BTree and verify we can find dataset names from the group heap
    auto heap_itest = parse_heap(f, itest_heap, sb.base);
    std::vector<uint64_t> sub_snods;
    traverse_btree(f, itest_btree, sb.base, sub_snods);

    std::vector<std::string> names;
    for (uint64_t sabs : sub_snods) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_itest, ste.name_off);
            if (!name.empty()) names.push_back(name);
        }
    }
    // itest has 3 datasets: knns, dists, queries
    EXPECT_GE(names.size(), 3u);
    bool has_knns = std::find(names.begin(), names.end(), std::string("knns")) != names.end();
    bool has_dists = std::find(names.begin(), names.end(), std::string("dists")) != names.end();
    bool has_queries = std::find(names.begin(), names.end(), std::string("queries")) != names.end();
    EXPECT_TRUE(has_knns);
    EXPECT_TRUE(has_dists);
    EXPECT_TRUE(has_queries);
}

// ---------------------------------------------------------------------------
//  B-Tree
// ---------------------------------------------------------------------------

TEST(Hdf5Structural, RootBTree_HasLeaves) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);
    // Root B-tree has at least 1 SNOD leaf (actual count depends on HDF5 version/group layout)
    EXPECT_GE(snod_addrs.size(), 1u);
}

TEST(Hdf5Structural, SubBTree_Itest) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);

    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_root, ste.name_off);
            if (name == "itest") {
                OhdrInfo gi = parse_ohdr(f, ste.ohdr_abs, sb.base);
                std::vector<uint64_t> sub_snods;
                traverse_btree(f, gi.group_btree_abs, sb.base, sub_snods);
                // itest has 3 datasets (knns, dists, queries)
                EXPECT_GE(sub_snods.size(), 1u);
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  SNOD
// ---------------------------------------------------------------------------

TEST(Hdf5Structural, RootSNOD_HasEntries) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);

    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    size_t total_entries = 0;
    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        total_entries += entries.size();
    }
    // Root SNOD(s) should contain at least "itest" (the group we can parse reliably)
    EXPECT_GE(total_entries, 1u);
}

TEST(Hdf5Structural, SNOD_EntryFields) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);

    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& e : entries) {
            std::string name = heap_str(heap_root, e.name_off);
            EXPECT_FALSE(name.empty());
            EXPECT_GT(e.ohdr_abs, 0u);
            // cache_type: 0 for datasets, 1 for groups
            EXPECT_LE(e.cache_type, 1u);
        }
        return;
    }
}
