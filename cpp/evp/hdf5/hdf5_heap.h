#pragma once

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {
namespace detail {

// ============================================================================
// Local Heap
// ============================================================================
inline std::vector<uint8_t> parse_heap(std::ifstream& f, uint64_t heap_abs, uint64_t base) {
    fseek(f, heap_abs);
    uint8_t sig[4]; f.read((char*)sig, 4);
    if (memcmp(sig, SIG_HEAP, 4) != 0) throw std::runtime_error(
        "hdf5_reader: HEAP sig missing at " + std::to_string(heap_abs));
    frd8(f);        // version
    fskip(f, 3);    // reserved
    uint64_t data_size = frd64(f);
    frd64(f);       // free_list_offset
    uint64_t data_rel = frd64(f);
    uint64_t data_abs = base + data_rel;
    std::vector<uint8_t> data((size_t)data_size);
    fseek(f, data_abs);
    f.read((char*)data.data(), (std::streamsize)data_size);
    return data;
}

} // namespace detail
} // namespace hdf5_reader
