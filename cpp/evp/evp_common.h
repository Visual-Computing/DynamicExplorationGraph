#pragma once

#include <chrono>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cassert>
#include <iostream>

namespace evp_common {

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

/**
 * @brief Writes a result list to a file in the ivecs binary format.
 *
 * Each row is serialised as:
 *   - 4 bytes : uint32_t  – number of elements in this row (= d)
 *   - d × 4 bytes : uint32_t[] – the element values
 *
 * Each row is written in full (row.size() elements). Callers are expected
 * to have already trimmed the rows to the desired length before calling.
 * If the output path is empty or the file cannot be opened, an error is
 * printed to stderr and the function returns without writing.
 *
 * @param output_path  Destination file path (binary).
 * @param results      2-D result list, one inner vector per query.
 */
inline void ivecs_write(
    const std::string& output_path,
    const std::vector<std::vector<uint32_t>>& results)
{
    if (output_path.empty()) {
        return;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "Error: Could not open output file '%s' for writing.\n",
                      output_path.c_str());
        return;
    }

    for (const auto& row : results) {
        const uint32_t d = static_cast<uint32_t>(row.size());
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(row.data()), d * sizeof(uint32_t));
    }

    std::printf("Successfully wrote %zu result rows (ivecs) to '%s'\n",
                results.size(), output_path.c_str());
}

inline float compute_recall(
    const std::vector<std::vector<uint32_t>>& results,
    const std::vector<std::vector<int32_t>>& gt_data,
    size_t count,
    uint32_t k_top)
{
    int total_hits = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& gt_row = gt_data[i];
        for (uint32_t k = 1; k <= k_top && k < (uint32_t)gt_row.size(); ++k) {
            uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1);
            const auto& row = results[i];
            const uint32_t row_len = static_cast<uint32_t>(std::min(row.size(), static_cast<size_t>(k_top)));
            for (uint32_t j = 0; j < row_len; ++j) {
                if (row[j] == gt_idx) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(total_hits) / (count * k_top);
}

} // namespace evp_common
