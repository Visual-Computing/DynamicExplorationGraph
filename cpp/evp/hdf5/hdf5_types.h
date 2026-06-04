#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hdf5_reader {

// ============================================================================
// Public API types
// ============================================================================

struct DatasetInfo {
    std::string name;           ///< Full path, e.g. "train" or "allknn/indices"
    uint64_t    file_offset = 0; ///< Absolute byte offset of raw data in file
    uint64_t    total_bytes = 0; ///< Total size of the data block in bytes
    uint32_t    element_size = 0; ///< Bytes per scalar element (2=fp16, 4=int32/float32)
    uint64_t    num_rows    = 0; ///< First dimension (rows)
    uint64_t    num_cols    = 0; ///< Second dimension (cols), 0 if 1-D
};

// ============================================================================
// Internal types
// ============================================================================

namespace detail {

static constexpr uint64_t UNDEF64 = 0xFFFFFFFFFFFFFFFFULL;

static constexpr uint8_t SIG_HDF5[8] = { 0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a };
static constexpr uint8_t SIG_TREE[4] = { 'T', 'R', 'E', 'E' };
static constexpr uint8_t SIG_SNOD[4] = { 'S', 'N', 'O', 'D' };
static constexpr uint8_t SIG_HEAP[4] = { 'H', 'E', 'A', 'P' };
static constexpr uint8_t SIG_OHDR[4] = { 'O', 'H', 'D', 'R' };

// ============================================================================
// Superblock
// ============================================================================
struct Superblock {
    uint64_t base      = 0;
    uint64_t root_ohdr = 0; // absolute
    uint64_t root_btree = 0; // absolute (from root STE cache)
    uint64_t root_heap  = 0; // absolute (from root STE cache)
};

// ============================================================================
// Symbol Table Entry
// ============================================================================
struct SteEntry {
    uint64_t name_off;      // offset into local heap
    uint64_t ohdr_abs;      // absolute object header address
    uint32_t cache_type;
    uint64_t scratch_btree; // absolute (from scratch, for cache_type=1)
    uint64_t scratch_heap;  // absolute (from scratch, for cache_type=1)
};

// ============================================================================
// Dataset Object Header info
// ============================================================================
struct LinkInfo {
    std::string name;
    uint64_t    obj_abs;
};

struct OhdrInfo {
    uint64_t data_abs  = 0;  // absolute file offset of contiguous data
    uint64_t data_len  = 0;  // byte length
    uint32_t elem_size = 0;  // bytes per element
    uint64_t dim0      = 0;
    uint64_t dim1      = 0;
    bool     is_group  = false;  // has Symbol Table message (is a group)
    uint64_t group_btree_abs = UNDEF64; // for groups: absolute btree addr
    uint64_t group_heap_abs  = UNDEF64; // for groups: absolute heap addr
    std::vector<LinkInfo> links;       // for groups: list of child links
};

} // namespace detail
} // namespace hdf5_reader
