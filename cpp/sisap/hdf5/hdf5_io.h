#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hdf5_types.h"

namespace hdf5_reader {
namespace detail {

// ---- Little-endian helpers ----
inline uint16_t r16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline uint32_t r32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint64_t r64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

inline uint8_t  frd8 (std::ifstream& f) { uint8_t  v = 0; f.read((char*)&v, 1); return v; }
inline uint16_t frd16(std::ifstream& f) { uint16_t v = 0; f.read((char*)&v, 2); return v; }
inline uint32_t frd32(std::ifstream& f) { uint32_t v = 0; f.read((char*)&v, 4); return v; }
inline uint64_t frd64(std::ifstream& f) { uint64_t v = 0; f.read((char*)&v, 8); return v; }
inline void     fskip(std::ifstream& f, std::streamoff n) { f.seekg(n, std::ios::cur); }
inline void     fseek(std::ifstream& f, uint64_t pos) {
    f.seekg((std::streamoff)pos, std::ios::beg);
    if (!f.good()) throw std::runtime_error(
        "hdf5_reader: seek failed at " + std::to_string(pos));
}

// ---- Read null-terminated string from heap ----
inline std::string heap_str(const std::vector<uint8_t>& heap, uint64_t off) {
    if (off >= heap.size()) throw std::runtime_error(
        "hdf5_reader: heap offset " + std::to_string(off) + " out of range");
    return std::string(reinterpret_cast<const char*>(heap.data() + off));
}

} // namespace detail
} // namespace hdf5_reader
