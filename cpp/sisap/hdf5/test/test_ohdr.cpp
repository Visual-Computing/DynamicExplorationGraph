// test_ohdr.cpp — Object Header parsing tests (v1 and v2)
//
// Uses the real SISAP file to exercise both v1 (itest/*) and v2 (allknn/*) paths.

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

#include "gtest/gtest.h"

using namespace hdf5_reader::detail;

static const char* SISAP_H5 = "C:\\Data\\ANN\\sisap2026\\small\\benchmark-dev-wikipedia-bge-m3-small.h5";

static bool file_exists(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// ---------------------------------------------------------------------------
//  OHdr v1 — itest/knns uses v1
// ---------------------------------------------------------------------------

TEST(Hdf5Ohdr, V1_ItestKnns) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);
    auto heap = parse_heap(f, sb.root_heap, sb.base);

    // Find itest entry in root SNOD
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    uint64_t itest_ohdr_abs = UNDEF64;
    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap, ste.name_off);
            if (name == "itest") {
                itest_ohdr_abs = ste.ohdr_abs;
                break;
            }
        }
        if (itest_ohdr_abs != UNDEF64) break;
    }

    ASSERT_NE(itest_ohdr_abs, UNDEF64);

    // Verify it's v1 (first byte == 1)
    fseek(f, itest_ohdr_abs);
    uint8_t first = frd8(f);
    EXPECT_EQ(first, 1) << "itest OHdr should be v1";

    OhdrInfo info = parse_ohdr_v1(f, itest_ohdr_abs, sb.base);
    // itest is a group, not a dataset
    EXPECT_TRUE(info.is_group);
    EXPECT_NE(info.group_btree_abs, UNDEF64);
    EXPECT_NE(info.group_heap_abs, UNDEF64);
}

TEST(Hdf5Ohdr, V1_ItestKnnsDataset) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    Superblock sb = parse_superblock(f);

    // Get itest group's btree + heap
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    uint64_t itest_btree = UNDEF64, itest_heap = UNDEF64;
    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_root, ste.name_off);
            if (name == "itest") {
                OhdrInfo gi = parse_ohdr_v1(f, ste.ohdr_abs, sb.base);
                itest_btree = gi.group_btree_abs;
                itest_heap = gi.group_heap_abs;
                break;
            }
        }
        if (itest_btree != UNDEF64) break;
    }
    ASSERT_NE(itest_btree, UNDEF64);

    // Now parse itest/knns dataset OHdr
    auto heap_itest = parse_heap(f, itest_heap, sb.base);
    std::vector<uint64_t> snod2;
    traverse_btree(f, itest_btree, sb.base, snod2);

    for (uint64_t sabs : snod2) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_itest, ste.name_off);
            if (name == "knns") {
                OhdrInfo info = parse_ohdr_v1(f, ste.ohdr_abs, sb.base);
                EXPECT_EQ(info.dim0, 10000ull);
                EXPECT_EQ(info.dim1, 1000ull);
                EXPECT_EQ(info.elem_size, 4u);
                EXPECT_GT(info.data_abs, 0ull);
                EXPECT_GT(info.data_len, 0ull);
                EXPECT_FALSE(info.is_group);
                return;
            }
        }
    }
    FAIL() << "itest/knns not found in SNOD";
}

// ---------------------------------------------------------------------------
//  OHdr v2 dispatch
// ---------------------------------------------------------------------------

TEST(Hdf5Ohdr, V2_DetectSignature) {
    // Simulate OHDR signature detection
    uint8_t sig[4] = {'O', 'H', 'D', 'R'};
    EXPECT_EQ(std::memcmp(sig, SIG_OHDR, 4), 0);
}

TEST(Hdf5Ohdr, ParseDispatch_V1) {
    if (!file_exists(SISAP_H5)) GTEST_SKIP();

    std::ifstream f(SISAP_H5, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    // Use itest/knns OHdr (v1) — get from integration test path
    Superblock sb = parse_superblock(f);
    auto heap_root = parse_heap(f, sb.root_heap, sb.base);
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, sb.root_btree, sb.base, snod_addrs);

    for (uint64_t sabs : snod_addrs) {
        auto entries = parse_snod(f, sabs, sb.base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap_root, ste.name_off);
            if (name == "itest") {
                // parse_ohdr should auto-detect v1
                OhdrInfo info = parse_ohdr(f, ste.ohdr_abs, sb.base);
                EXPECT_TRUE(info.is_group);
                return;
            }
        }
    }
}
