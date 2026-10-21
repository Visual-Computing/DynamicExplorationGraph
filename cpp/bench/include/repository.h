#pragma once

#include <stdio.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <memory>

namespace deglib {

/**
 * A contiguous repository of feature vectors with zero-copy subview support.
 */
class FeatureRepository {
  public:
    FeatureRepository() = default;

    // Owning constructor
    FeatureRepository(std::unique_ptr<std::byte[]> data, const size_t dims, const size_t count, const size_t bytes_per_dim)
        : storage_(std::move(data)), data_(storage_.get()), dims_{dims}, count_{count}, bytes_per_dim_{bytes_per_dim}, offset_{0} {}

    // Non-owning view constructor sharing storage lifetime
    FeatureRepository(std::shared_ptr<std::byte[]> storage, const std::byte* data, const size_t dims, const size_t count, const size_t bytes_per_dim, const size_t offset = 0)
        : storage_(std::move(storage)), data_(data), dims_{dims}, count_{count}, bytes_per_dim_{bytes_per_dim}, offset_{offset} {}

    // Non-owning view constructor from raw pointer
    FeatureRepository(const std::byte* data, const size_t dims, const size_t count, const size_t bytes_per_dim, const size_t offset = 0)
        : storage_(nullptr), data_(data), dims_{dims}, count_{count}, bytes_per_dim_{bytes_per_dim}, offset_{offset} {}

    size_t dims() const noexcept { return dims_; }
    size_t size() const noexcept { return count_; }
    size_t offset() const noexcept { return offset_; }
    size_t bytes_per_dim() const noexcept { return bytes_per_dim_; }

    const std::byte* getFeature(const uint32_t idx) const noexcept { 
        return &data_[idx * dims_ * bytes_per_dim_]; 
    }

    FeatureRepository subview(const size_t offset, const size_t count = (std::numeric_limits<size_t>::max)()) const noexcept {
        const size_t actual_offset = std::min(offset, count_);
        const size_t actual_count = std::min(count, count_ - actual_offset);
        const auto* sub_data = (actual_count > 0) ? getFeature(static_cast<uint32_t>(actual_offset)) : nullptr;
        return FeatureRepository(storage_, sub_data, dims_, actual_count, bytes_per_dim_, offset_ + actual_offset);
    }

    void clear() noexcept { storage_.reset(); data_ = nullptr; count_ = 0; offset_ = 0; }

  private:
    std::shared_ptr<std::byte[]> storage_;
    const std::byte* data_ = nullptr;
    size_t dims_ = 0;
    size_t count_ = 0;
    size_t bytes_per_dim_ = 0;
    size_t offset_ = 0;
};

using StaticFeatureRepository = FeatureRepository;

/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 * https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/

auto fvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
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
    if (dims == 0 || dims >= 1'000'000) {
        throw std::runtime_error("unreasonable dimension in fvecs file");
    }

    // compute number of rows
    if (file_size % ((dims + 1) * sizeof(float)) != 0) {
        throw std::runtime_error("weird file size in fvecs file");
    }
    size_t n = (size_t)file_size / ((dims + 1) * sizeof(float));
    d_out = dims;
    n_out = n;

    // read data rows (each row starts with its dimension, which is 4 bytes)
    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), file_size);
    if (!ifstream && ifstream.gcount() != static_cast<std::streamsize>(file_size)) {
        throw std::runtime_error("could not read whole file");
    }

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) std::memmove(&x[i * dims * sizeof(float)], &x[sizeof(dims) + i * (dims + 1) * sizeof(float)], dims * sizeof(float));

    ifstream.close();
    return x;
}

auto u8vecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    // get total file size
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        std::abort();
    }

    // open as binary
    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", fname);
        std::abort();
    }

    // read dimension header
    uint32_t dims = 0;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    if (dims == 0 || dims >= 1'000'000) {
        throw std::runtime_error("unreasonable dimension in u8vecs file");
    }

    // compute number of rows
    if (file_size % (dims + 4) != 0) {
        throw std::runtime_error("weird file size in u8vecs file");
    }
    size_t n = (size_t)file_size / (dims + 4);
    d_out = dims;
    n_out = n;

    // read data rows (each row starts with its dimension, which is 4 bytes)
    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), file_size);
    if (!ifstream && ifstream.gcount() != static_cast<std::streamsize>(file_size)) {
        throw std::runtime_error("could not read whole file");
    }

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) std::memmove(&x[i * dims], &x[sizeof(dims) + i * (dims + sizeof(dims))], dims);

    ifstream.close();
    return x;
}

inline bool string_ends_with(const char* str, const char* suffix) {
    size_t str_len = std::strlen(str);
    size_t suffix_len = std::strlen(suffix);

    if (suffix_len > str_len) {
        return false;
    }
    return std::strcmp(str + str_len - suffix_len, suffix) == 0;
}

inline FeatureRepository load_repository(const char* path_repository) {
    if (string_ends_with(path_repository, "fvecs")) {
        size_t dims;
        size_t count;
        auto contiguous_features = fvecs_read(path_repository, dims, count);
        return FeatureRepository(std::move(contiguous_features), dims, count, sizeof(float));
    } else if (string_ends_with(path_repository, "u8vecs")) {
        size_t dims;
        size_t count;
        auto contiguous_features = u8vecs_read(path_repository, dims, count);
        return FeatureRepository(std::move(contiguous_features), dims, count, sizeof(uint8_t));
    }

    std::fprintf(stderr, "unsupported file extension, only fvecs and u8vecs are supported, but got %s \n", path_repository);
    std::perror("");
    std::abort();
}

inline FeatureRepository load_static_repository(const char* path_repository) {
    return load_repository(path_repository);
}

}  // namespace deglib