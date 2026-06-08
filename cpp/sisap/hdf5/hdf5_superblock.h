#pragma once

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {
namespace detail {

// ============================================================================
// Superblock (version 0)
// ============================================================================
inline Superblock parse_superblock(std::ifstream& f) {
    static const uint64_t CANDIDATES[] = { 0, 512, 1024, 2048, 4096 };
    uint8_t sig[8];
    uint64_t sb_abs = UNDEF64;
    for (uint64_t c : CANDIDATES) {
        f.seekg((std::streamoff)c);
        if (!f.good()) continue;
        f.read((char*)sig, 8);
        if (f.gcount() == 8 && memcmp(sig, SIG_HDF5, 8) == 0) { sb_abs = c; break; }
    }
    if (sb_abs == UNDEF64) throw std::runtime_error(
        "hdf5_reader: HDF5 superblock not found");

    fseek(f, sb_abs);
    fskip(f, 8);             // signature
    uint8_t ver = frd8(f);
    if (ver != 0) throw std::runtime_error(
        "hdf5_reader: only Superblock v0 supported, got v" + std::to_string(ver));

    // v0 after signature:
    // ver(1) freesp_ver(1) root_g_ver(1) res(1) shdr_ver(1) sz_off(1) sz_len(1) res(1)
    // leaf_k(2) intern_k(2) flags(4)
    // base(8) freesp(8) eof(8) drv(8)
    // root STE: link_name(8) ohdr(8) cache_type(4) res(4) scratch(16)
    fskip(f, 4);        // [9..12]: freesp_ver, rootg_ver, res, shdr_ver
    uint8_t sz_off = frd8(f); // [13]
    uint8_t sz_len = frd8(f); // [14]
    fskip(f, 1);              // [15]: reserved
    if (sz_off != 8 || sz_len != 8) throw std::runtime_error(
        "hdf5_reader: only 8-byte offsets/lengths supported (got sz_off="
        + std::to_string(sz_off) + " sz_len=" + std::to_string(sz_len) + ")");
    fskip(f, 2 + 2 + 4);    // [16..23]: leaf_k(2), intern_k(2), flags(4)

    Superblock sb;
    sb.base = frd64(f);    // base_address (absolute)
    fskip(f, 8 + 8 + 8);            // freesp, eof, drv_info

    // Root group STE
    fskip(f, 8);                // link_name_offset (always 0)
    uint64_t root_ohdr_rel = frd64(f);
    sb.root_ohdr = sb.base + root_ohdr_rel;
    uint32_t cache_type = frd32(f);
    fskip(f, 4);                // reserved
    uint64_t btree_rel = frd64(f);
    uint64_t heap_rel = frd64(f);
    sb.root_btree = (btree_rel != UNDEF64) ? sb.base + btree_rel : UNDEF64;
    sb.root_heap = (heap_rel != UNDEF64) ? sb.base + heap_rel : UNDEF64;
    (void)cache_type;
    return sb;
}

} // namespace detail
} // namespace hdf5_reader
