#pragma once

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {
namespace detail {

// ============================================================================
// Symbol Table Node (SNOD)
// Returns: vector of STE entries (name_off → ohdr_abs, cache_type, scratch)
// ============================================================================
inline std::vector<SteEntry> parse_snod(std::ifstream& f, uint64_t snod_abs, uint64_t base) {
    fseek(f, snod_abs);
    uint8_t sig[4]; f.read((char*)sig, 4);
    if (memcmp(sig, SIG_SNOD, 4) != 0) throw std::runtime_error(
        "hdf5_reader: SNOD sig missing at " + std::to_string(snod_abs));
    frd8(f);            // version
    fskip(f, 1);        // reserved
    uint16_t n = frd16(f);

    std::vector<SteEntry> entries;
    entries.reserve(n);
    // Each STE = 40 bytes: name_off(8) ohdr_rel(8) cache_type(4) res(4) scratch(16)
    for (uint16_t i = 0; i < n; ++i) {
        SteEntry e{};
        e.name_off = frd64(f);
        uint64_t ohdr_rel = frd64(f);
        e.ohdr_abs = base + ohdr_rel;
        e.cache_type = frd32(f);
        fskip(f, 4);    // reserved
        // scratch (16 bytes): for cache_type=1 contains btree+heap relative addresses
        uint64_t sc_bt = frd64(f);
        uint64_t sc_hp = frd64(f);
        e.scratch_btree = (sc_bt != UNDEF64) ? base + sc_bt : UNDEF64;
        e.scratch_heap = (sc_hp != UNDEF64) ? base + sc_hp : UNDEF64;
        entries.push_back(e);
    }
    return entries;
}

} // namespace detail
} // namespace hdf5_reader
