#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {

// ============================================================================
// Internal helpers
// ============================================================================

namespace detail {

inline void open_file(std::ifstream& f, const std::string& filepath) {
    f.open(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("hdf5_reader: cannot open: " + filepath);
    }
}

template <typename T>
inline void validate_dims(const DatasetInfo& info, const char* fn_name) {
    if (info.num_cols == 0) {
        throw std::runtime_error(std::string(fn_name) + ": 1-D dataset not supported");
    }
}

template <typename T>
inline void read_flat(std::ifstream& f, const DatasetInfo& info, std::vector<T>& out, size_t count, size_t dims) {
    detail::fseek(f, info.file_offset);
    out.resize(count * dims);
    const size_t bytes_per_row = dims * sizeof(T);
    for (size_t i = 0; i < count; ++i) {
        f.read(reinterpret_cast<char*>(out.data() + i * dims),
               static_cast<std::streamsize>(bytes_per_row));
        if (!f.good()) {
            throw std::runtime_error("hdf5_reader: read error at row " + std::to_string(i));
        }
    }
}

template <typename T>
inline void read_matrix(std::ifstream& f, const DatasetInfo& info, std::vector<std::vector<T>>& out, size_t count, size_t dims) {
    detail::fseek(f, info.file_offset);
    out.resize(count);
    const size_t bytes_per_row = dims * sizeof(T);
    for (size_t i = 0; i < count; ++i) {
        out[i].resize(dims);
        f.read(reinterpret_cast<char*>(out[i].data()),
               static_cast<std::streamsize>(bytes_per_row));
        if (!f.good()) {
            throw std::runtime_error("hdf5_reader: read error at row " + std::to_string(i));
        }
    }
}

} // namespace detail

// ============================================================================
// Flat readers — return std::vector<T> (row-major, count × dims)
// ============================================================================

inline std::vector<std::byte> read_flat_bytes(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<std::byte>(info, "read_flat_bytes");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::byte> out;
    detail::read_flat<std::byte>(f, info, out, count, dims);
    return out;
}

inline std::vector<uint8_t> read_flat_uint8(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint8_t>(info, "read_flat_uint8");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<uint8_t> out;
    detail::read_flat<uint8_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<int8_t> read_flat_int8(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int8_t>(info, "read_flat_int8");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<int8_t> out;
    detail::read_flat<int8_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<uint16_t> read_flat_uint16(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint16_t>(info, "read_flat_uint16");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<uint16_t> out;
    detail::read_flat<uint16_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<int16_t> read_flat_int16(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int16_t>(info, "read_flat_int16");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<int16_t> out;
    detail::read_flat<int16_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<uint32_t> read_flat_uint32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint32_t>(info, "read_flat_uint32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<uint32_t> out;
    detail::read_flat<uint32_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<int32_t> read_flat_int32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int32_t>(info, "read_flat_int32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<int32_t> out;
    detail::read_flat<int32_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<float> read_flat_fp32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<float>(info, "read_flat_fp32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<float> out;
    detail::read_flat<float>(f, info, out, count, dims);
    return out;
}

// ============================================================================
// Matrix readers — return std::vector<std::vector<T>> (one row per inner vector)
// ============================================================================

inline std::vector<std::vector<std::byte>> read_matrix_bytes(const std::string& filepath, const DatasetInfo& info) {
    if (info.num_cols == 0) {
        throw std::runtime_error("read_matrix_bytes: 1-D dataset not supported");
    }
    std::ifstream f;
    detail::open_file(f, filepath);
    detail::fseek(f, info.file_offset);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    const size_t bytes_per_row = dims * info.element_size;
    std::vector<std::vector<std::byte>> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i].resize(bytes_per_row);
        f.read(reinterpret_cast<char*>(out[i].data()), static_cast<std::streamsize>(bytes_per_row));
        if (!f.good()) {
            throw std::runtime_error("hdf5_reader: read error at row " + std::to_string(i));
        }
    }
    return out;
}

inline std::vector<std::vector<uint8_t>> read_matrix_uint8(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint8_t>(info, "read_matrix_uint8");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<uint8_t>> out;
    detail::read_matrix<uint8_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<int8_t>> read_matrix_int8(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int8_t>(info, "read_matrix_int8");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<int8_t>> out;
    detail::read_matrix<int8_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<uint16_t>> read_matrix_uint16(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint16_t>(info, "read_matrix_uint16");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<uint16_t>> out;
    detail::read_matrix<uint16_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<int16_t>> read_matrix_int16(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int16_t>(info, "read_matrix_int16");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<int16_t>> out;
    detail::read_matrix<int16_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<uint32_t>> read_matrix_uint32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<uint32_t>(info, "read_matrix_uint32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<uint32_t>> out;
    detail::read_matrix<uint32_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<int32_t>> read_matrix_int32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<int32_t>(info, "read_matrix_int32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<int32_t>> out;
    detail::read_matrix<int32_t>(f, info, out, count, dims);
    return out;
}

inline std::vector<std::vector<float>> read_matrix_fp32(const std::string& filepath, const DatasetInfo& info) {
    detail::validate_dims<float>(info, "read_matrix_fp32");
    std::ifstream f;
    detail::open_file(f, filepath);
    size_t count = (size_t)info.num_rows;
    size_t dims = (size_t)info.num_cols;
    std::vector<std::vector<float>> out;
    detail::read_matrix<float>(f, info, out, count, dims);
    return out;
}

// ============================================================================
// Find dataset by name
// ============================================================================

inline const DatasetInfo& find_dataset(
    const std::map<std::string, DatasetInfo>& datasets,
    const std::string& name)
{
    auto it = datasets.find(name);
    if (it == datasets.end()) {
        throw std::runtime_error("hdf5_reader: dataset '" + name + "' not found in file. "
            "Available datasets: " +
            [datasets]() {
                std::string s;
                for (const auto& [k, _] : datasets) {
                    if (!s.empty()) s += ", ";
                    s += k;
                }
                return s;
            }());
    }
    return it->second;
}

} // namespace hdf5_reader
