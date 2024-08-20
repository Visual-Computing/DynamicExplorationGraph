#pragma once

#include <assert.h>
#include <stdio.h>
#include <unordered_map>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace deglib
{
/**
 * A repository of float feature vectors.
 */
class FeatureRepository
{
  public:
    virtual size_t dims() const = 0;
    virtual size_t size() const = 0;
    virtual const std::byte* getFeature(const uint32_t vertexid) const = 0;
    virtual void clear() = 0;
};

/**
 * A repository of float feature vectors. Since the repository deals
 * with static data, a single contiguous array is preserved internally.
 */
class StaticFeatureRepository : public FeatureRepository
{
  public:
    StaticFeatureRepository(std::unique_ptr<std::byte[]> contiguous_features, const size_t dims, const size_t count, const size_t bytes_per_dim)
        : bytes_per_dim_{bytes_per_dim}, dims_{dims}, count_{count}, contiguous_features_{std::move(contiguous_features), }
    {
    }

    size_t dims() const override { return dims_; }
    size_t size() const override { return count_; }
    const std::byte* getFeature(const uint32_t idx) const override { return &contiguous_features_[idx * dims_ * bytes_per_dim_]; }
    void clear() override { contiguous_features_.reset(); }

  private:
    const size_t bytes_per_dim_;
    const size_t dims_;
    const size_t count_;
    std::unique_ptr<std::byte[]> contiguous_features_;
};


/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 * https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/

auto fvecs_read(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % ((dims + 1) * sizeof(float)) == 0 || !"weird file size");
    size_t n = (size_t)file_size / ((dims + 1) * sizeof(float));

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), file_size);
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(file_size) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) std::memmove(&x[i * dims * sizeof(float)], &x[sizeof(int) + i * (dims + 1) * sizeof(float)], dims * sizeof(float));

    ifstream.close();
    return x;
}

auto u8vecs_read(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % (dims + 4) == 0 || !"weird file size");
      size_t n = (size_t)file_size / (dims + 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), file_size);
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(file_size) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) std::memmove(&x[i * dims], &x[sizeof(int) + i * (dims + sizeof(int))], dims);

    ifstream.close();
    return x;
}

bool string_ends_with(const char* str, const char* suffix) {
    size_t str_len = std::strlen(str);
    size_t suffix_len = std::strlen(suffix);
    
    if (suffix_len > str_len) {
        return false;
    }
    return std::strcmp(str + str_len - suffix_len, suffix) == 0;
}

StaticFeatureRepository load_static_repository(const char* path_repository)
{
  if (string_ends_with(path_repository, "fvecs")) {
    size_t dims;
    size_t count;
    auto contiguous_features = fvecs_read(path_repository, dims, count);
    return StaticFeatureRepository(std::move(contiguous_features), dims, count, sizeof(float));
  } else if (string_ends_with(path_repository, "u8vecs")) {
    size_t dims;
    size_t count;
    auto contiguous_features = u8vecs_read(path_repository, dims, count);
    return StaticFeatureRepository(std::move(contiguous_features), dims, count, sizeof(uint8_t));
  }

  std::fprintf(stderr, "unsupported file extension, only fvecs and u8vecs are supported, but got %s \n", path_repository);
  std::perror("");
  std::abort();
}

}  // namespace deglib