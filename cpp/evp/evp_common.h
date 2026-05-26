#pragma once

#include <chrono>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cassert>
#include <iostream>

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

inline std::vector<std::vector<std::byte>> hvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        perror("");
        abort();
    }

    // open as binary
    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    // read dimension header
    uint32_t dims = 0;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    assert((dims > 0 && dims < 1'000'000) && "unreasonable dimension");

    // compute number of rows
    size_t row_bytes = sizeof(uint32_t) + dims * sizeof(uint16_t);
    assert(file_size % row_bytes == 0 || !"weird file size");
    size_t n = (size_t)file_size / row_bytes;
    d_out = dims;
    n_out = n;

    std::vector<std::vector<std::byte>> vectors(n);
    for (size_t i = 0; i < n; ++i) {
        vectors[i].resize(dims * sizeof(uint16_t));
        // Seek to the start of this row's features (skip the 4-byte dimension header)
        ifstream.seekg(i * row_bytes + sizeof(uint32_t));
        ifstream.read(reinterpret_cast<char*>(vectors[i].data()), dims * sizeof(uint16_t));
    }

    ifstream.close();
    return vectors;
}
