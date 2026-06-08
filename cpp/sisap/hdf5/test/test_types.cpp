// test_types.cpp — Type struct defaults and constants

#include <cstdint>
#include <cstring>

#include "hdf5_types.h"

#include "gtest/gtest.h"

using namespace hdf5_reader;
using namespace hdf5_reader::detail;

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------

TEST(Hdf5Types, UNDEF64) {
    EXPECT_EQ(UNDEF64, 0xFFFFFFFFFFFFFFFFull);
}

TEST(Hdf5Types, SIG_HDF5) {
    const uint8_t expected[8] = {0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a};
    EXPECT_EQ(std::memcmp(SIG_HDF5, expected, 8), 0);
}

TEST(Hdf5Types, SIG_TREE) {
    EXPECT_EQ(SIG_TREE[0], 'T');
    EXPECT_EQ(SIG_TREE[1], 'R');
    EXPECT_EQ(SIG_TREE[2], 'E');
    EXPECT_EQ(SIG_TREE[3], 'E');
}

TEST(Hdf5Types, SIG_SNOD) {
    EXPECT_EQ(SIG_SNOD[0], 'S');
    EXPECT_EQ(SIG_SNOD[1], 'N');
    EXPECT_EQ(SIG_SNOD[2], 'O');
    EXPECT_EQ(SIG_SNOD[3], 'D');
}

TEST(Hdf5Types, SIG_HEAP) {
    EXPECT_EQ(SIG_HEAP[0], 'H');
    EXPECT_EQ(SIG_HEAP[1], 'E');
    EXPECT_EQ(SIG_HEAP[2], 'A');
    EXPECT_EQ(SIG_HEAP[3], 'P');
}

TEST(Hdf5Types, SIG_OHDR) {
    EXPECT_EQ(SIG_OHDR[0], 'O');
    EXPECT_EQ(SIG_OHDR[1], 'H');
    EXPECT_EQ(SIG_OHDR[2], 'D');
    EXPECT_EQ(SIG_OHDR[3], 'R');
}

// ---------------------------------------------------------------------------
//  Struct default initialization
// ---------------------------------------------------------------------------

TEST(Hdf5Types, DatasetInfo_Default) {
    DatasetInfo info;
    EXPECT_EQ(info.name, "");
    EXPECT_EQ(info.file_offset, 0ull);
    EXPECT_EQ(info.total_bytes, 0ull);
    EXPECT_EQ(info.element_size, 0u);
    EXPECT_EQ(info.num_rows, 0ull);
    EXPECT_EQ(info.num_cols, 0ull);
}

TEST(Hdf5Types, OhdrInfo_Default) {
    OhdrInfo info;
    EXPECT_EQ(info.data_abs, 0ull);
    EXPECT_EQ(info.data_len, 0ull);
    EXPECT_EQ(info.elem_size, 0u);
    EXPECT_EQ(info.dim0, 0ull);
    EXPECT_EQ(info.dim1, 0ull);
    EXPECT_FALSE(info.is_group);
    EXPECT_EQ(info.group_btree_abs, UNDEF64);
    EXPECT_EQ(info.group_heap_abs, UNDEF64);
}

TEST(Hdf5Types, Superblock_Default) {
    Superblock sb;
    EXPECT_EQ(sb.base, 0ull);
    EXPECT_EQ(sb.root_ohdr, 0ull);
    EXPECT_EQ(sb.root_btree, 0ull);
    EXPECT_EQ(sb.root_heap, 0ull);
}

TEST(Hdf5Types, SteEntry_Default) {
    SteEntry e{};
    EXPECT_EQ(e.name_off, 0ull);
    EXPECT_EQ(e.ohdr_abs, 0ull);
    EXPECT_EQ(e.cache_type, 0u);
    EXPECT_EQ(e.scratch_btree, 0ull);
    EXPECT_EQ(e.scratch_heap, 0ull);
}
