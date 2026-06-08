#pragma once

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {
namespace detail {

// ============================================================================
// B-Tree v1 traversal → collects all leaf SNOD addresses
// ============================================================================
inline void traverse_btree(std::ifstream& f, uint64_t btree_abs, uint64_t base,
                            std::vector<uint64_t>& snod_addrs)
{
    fseek(f, btree_abs);
    uint8_t sig[4]; f.read((char*)sig, 4);
    if (memcmp(sig, SIG_TREE, 4) != 0) throw std::runtime_error(
        "hdf5_reader: TREE sig missing at " + std::to_string(btree_abs));
    // HDF5 B-Tree v1 header (no version byte!):
    // Signature(4) | Node Type(1) | Node Level(1) | Entries Used(2)
    // | Left Sibling(8) | Right Sibling(8) → total 24 bytes
    uint8_t  node_type    = frd8(f);  // byte 4
    uint8_t  node_level   = frd8(f);  // byte 5
    uint16_t entries_used = frd16(f); // bytes 6-7
    frd64(f);                         // left_sibling  bytes 8-15
    frd64(f);                         // right_sibling bytes 16-23
    if (node_type != 0) throw std::runtime_error(
        "hdf5_reader: B-Tree node_type!=0 not supported at " + std::to_string(btree_abs));

    // Layout: key[0], child[0], key[1], child[1], ..., key[N]
    // entries_used = N children
    frd64(f);                   // key[0]
    std::vector<uint64_t> children;
    children.reserve(entries_used);
    for (uint16_t i = 0; i < entries_used; ++i) {
        uint64_t child_rel = frd64(f);
        uint64_t child_abs = (child_rel != UNDEF64) ? base + child_rel : UNDEF64;
        children.push_back(child_abs);
        frd64(f);               // key[i+1]
    }

    for (uint64_t child_abs : children) {
        if (child_abs == UNDEF64) continue;
        if (node_level == 0) {
            snod_addrs.push_back(child_abs);
        } else {
            traverse_btree(f, child_abs, base, snod_addrs);
        }
    }
}

} // namespace detail
} // namespace hdf5_reader
