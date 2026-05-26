// test_io.cpp — Low-level IO helpers

#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include "hdf5_io.h"

#include "gtest/gtest.h"

using namespace hdf5_reader::detail;

// ---------------------------------------------------------------------------
//  In-memory readers (r16/r32/r64)
// ---------------------------------------------------------------------------

TEST(Hdf5IO, r16_LittleEndian) {
    uint8_t buf[2] = {0x34, 0x12};
    EXPECT_EQ(r16(buf), 0x1234u);
}

TEST(Hdf5IO, r32_LittleEndian) {
    uint8_t buf[4] = {0x78, 0x56, 0x34, 0x12};
    EXPECT_EQ(r32(buf), 0x12345678u);
}

TEST(Hdf5IO, r64_LittleEndian) {
    uint8_t buf[8] = {0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90, 0x80};
    EXPECT_EQ(r64(buf), 0x8090A0B0C0D0E0F0ull);
}

TEST(Hdf5IO, r64_Zero) {
    uint8_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(r64(buf), 0ull);
}

TEST(Hdf5IO, r64_MaxUint64) {
    uint8_t buf[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    EXPECT_EQ(r64(buf), 0xFFFFFFFFFFFFFFFFull);
}

// ---------------------------------------------------------------------------
//  heap_str
// ---------------------------------------------------------------------------

TEST(Hdf5IO, heap_str_Basic) {
    std::vector<uint8_t> heap = {'a', 'b', 'c', 0, 'd', 'e', 0};
    EXPECT_EQ(heap_str(heap, 0), "abc");
    EXPECT_EQ(heap_str(heap, 4), "de");
    EXPECT_EQ(heap_str(heap, 5), "e");
}

TEST(Hdf5IO, heap_str_Empty) {
    std::vector<uint8_t> heap = {0, 'a', 'b'};
    EXPECT_EQ(heap_str(heap, 0), "");
}

TEST(Hdf5IO, heap_str_OutOfRange) {
    std::vector<uint8_t> heap = {'a', 'b', 'c'};
    EXPECT_THROW(heap_str(heap, 100), std::runtime_error);
}

TEST(Hdf5IO, heap_str_AtBoundary) {
    std::vector<uint8_t> heap = {'a', 'b', 0};
    EXPECT_EQ(heap_str(heap, 2), "");
}

TEST(Hdf5IO, heap_str_LongString) {
    std::vector<uint8_t> heap(100);
    for (int i = 0; i < 90; i++) heap[i] = 'A' + (i % 26);
    heap[90] = 0;
    EXPECT_EQ(heap_str(heap, 0).size(), 90u);
}
