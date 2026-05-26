#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "hdf5_types.h"
#include "hdf5_io.h"

namespace hdf5_reader {

/**
 * @brief Read a dataset into a vector-of-vectors (one inner vector per row).
 *        Works for any element_size. Each inner vector contains num_cols * element_size bytes.
 */
inline std::vector<std::vector<std::byte>> read_dataset_as_vecs(
    const std::string& filepath,
    const DatasetInfo& info)
{
    if (info.num_cols == 0) throw std::runtime_error(
        "hdf5_reader::read_dataset_as_vecs: 1-D dataset not supported");

    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(
        "hdf5_reader: cannot open: " + filepath);

    detail::fseek(f, info.file_offset);
    const size_t bytes_per_row = (size_t)info.num_cols * info.element_size;
    std::vector<std::vector<std::byte>> vecs((size_t)info.num_rows);
    for (size_t i = 0; i < (size_t)info.num_rows; ++i) {
        vecs[i].resize(bytes_per_row);
        f.read((char*)vecs[i].data(), (std::streamsize)bytes_per_row);
        if (!f.good()) throw std::runtime_error(
            "hdf5_reader: read error at row " + std::to_string(i));
    }
    return vecs;
}

/**
 * @brief Read a dataset into a vector-of-vectors of int32 (one inner vector per row).
 * @param dims_out  number of columns
 * @param count_out number of rows
 */
inline std::vector<std::vector<int32_t>> read_dataset_as_ints(
    const std::string& filepath,
    const DatasetInfo& info,
    size_t& dims_out,
    size_t& count_out)
{
    if (info.num_cols == 0) throw std::runtime_error(
        "hdf5_reader::read_dataset_as_ints: 1-D dataset not supported");

    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(
        "hdf5_reader: cannot open: " + filepath);

    detail::fseek(f, info.file_offset);
    count_out = (size_t)info.num_rows;
    dims_out = (size_t)info.num_cols;
    const size_t bytes_per_row = info.num_cols * sizeof(int32_t);
    std::vector<std::vector<int32_t>> result((size_t)info.num_rows);
    for (size_t i = 0; i < (size_t)info.num_rows; ++i) {
        result[i].resize((size_t)info.num_cols);
        f.read(reinterpret_cast<char*>(result[i].data()),
               static_cast<std::streamsize>(bytes_per_row));
        if (!f.good()) throw std::runtime_error(
            "hdf5_reader: read error at row " + std::to_string(i));
    }
    return result;
}

/**
 * @brief Read a dataset into a vector-of-vectors of float (one inner vector per row).
 * @param dims_out  number of columns
 * @param count_out number of rows
 */
inline std::vector<std::vector<float>> read_dataset_as_floats(
    const std::string& filepath,
    const DatasetInfo& info,
    size_t& dims_out,
    size_t& count_out)
{
    if (info.num_cols == 0) throw std::runtime_error(
        "hdf5_reader::read_dataset_as_floats: 1-D dataset not supported");

    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(
        "hdf5_reader: cannot open: " + filepath);

    detail::fseek(f, info.file_offset);
    count_out = (size_t)info.num_rows;
    dims_out = (size_t)info.num_cols;
    const size_t bytes_per_row = info.num_cols * sizeof(float);
    std::vector<std::vector<float>> result((size_t)info.num_rows);
    for (size_t i = 0; i < (size_t)info.num_rows; ++i) {
        result[i].resize((size_t)info.num_cols);
        f.read(reinterpret_cast<char*>(result[i].data()),
               static_cast<std::streamsize>(bytes_per_row));
        if (!f.good()) throw std::runtime_error(
            "hdf5_reader: read error at row " + std::to_string(i));
    }
    return result;
}

/**
 * @brief Find a dataset by name in a scanned map, or throw with a clear message.
 */
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
