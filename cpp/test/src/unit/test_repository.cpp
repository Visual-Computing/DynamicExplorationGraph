// test_repository.cpp — Unit tests for deglib::StaticFeatureRepository

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "repository.h"
#include "gtest/gtest.h"

// Helper: create feature data as properly managed std::byte[]
static std::unique_ptr<std::byte[]> make_float_features(const std::vector<float>& data) {
    size_t total = data.size() * sizeof(float);
    auto bytes = std::make_unique<std::byte[]>(total);
    std::memcpy(bytes.get(), data.data(), total);
    return bytes;
}

static std::unique_ptr<std::byte[]> make_uint8_features(const std::vector<uint8_t>& data) {
    size_t total = data.size() * sizeof(uint8_t);
    auto bytes = std::make_unique<std::byte[]>(total);
    std::memcpy(bytes.get(), data.data(), total);
    return bytes;
}

// ---------------------------------------------------------------------------
//  StaticFeatureRepository
// ---------------------------------------------------------------------------

TEST(StaticFeatureRepository, Basic) {
    size_t dims = 4;
    size_t count = 3;
    size_t bytes_per_dim = sizeof(float);

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };

    deglib::StaticFeatureRepository repo(
        make_float_features(data), dims, count, bytes_per_dim);

    EXPECT_EQ(repo.dims(), dims);
    EXPECT_EQ(repo.size(), count);
}

TEST(StaticFeatureRepository, GetFeature) {
    size_t dims = 4;
    size_t count = 2;
    size_t bytes_per_dim = sizeof(float);

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
    };

    deglib::StaticFeatureRepository repo(
        make_float_features(data), dims, count, bytes_per_dim);

    const std::byte* f0 = repo.getFeature(0);
    const float* fp0 = reinterpret_cast<const float*>(f0);
    EXPECT_NEAR(fp0[0], 1.0f, 1e-6f);
    EXPECT_NEAR(fp0[1], 2.0f, 1e-6f);
    EXPECT_NEAR(fp0[2], 3.0f, 1e-6f);
    EXPECT_NEAR(fp0[3], 4.0f, 1e-6f);

    const std::byte* f1 = repo.getFeature(1);
    const float* fp1 = reinterpret_cast<const float*>(f1);
    EXPECT_NEAR(fp1[0], 10.0f, 1e-6f);
    EXPECT_NEAR(fp1[1], 20.0f, 1e-6f);
    EXPECT_NEAR(fp1[2], 30.0f, 1e-6f);
    EXPECT_NEAR(fp1[3], 40.0f, 1e-6f);
}

TEST(StaticFeatureRepository, Clear) {
    size_t dims = 4;
    size_t count = 2;
    size_t bytes_per_dim = sizeof(float);

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    deglib::StaticFeatureRepository repo(
        make_float_features(data), dims, count, bytes_per_dim);
    repo.clear();
}

TEST(StaticFeatureRepository, SingleVector) {
    size_t dims = 3;
    size_t count = 1;
    size_t bytes_per_dim = sizeof(float);

    std::vector<float> data = {7.0f, 8.0f, 9.0f};
    deglib::StaticFeatureRepository repo(
        make_float_features(data), dims, count, bytes_per_dim);
    EXPECT_EQ(repo.size(), 1u);
    EXPECT_EQ(repo.dims(), 3u);

    const float* fp = reinterpret_cast<const float*>(repo.getFeature(0));
    EXPECT_NEAR(fp[0], 7.0f, 1e-6f);
    EXPECT_NEAR(fp[1], 8.0f, 1e-6f);
    EXPECT_NEAR(fp[2], 9.0f, 1e-6f);
}

TEST(StaticFeatureRepository, LargeVector) {
    size_t dims = 128;
    size_t count = 10;
    size_t bytes_per_dim = sizeof(float);

    std::vector<float> data(dims * count);
    for (size_t i = 0; i < dims * count; ++i) {
        data[i] = static_cast<float>(i);
    }

    deglib::StaticFeatureRepository repo(
        make_float_features(data), dims, count, bytes_per_dim);
    EXPECT_EQ(repo.size(), count);
    EXPECT_EQ(repo.dims(), dims);

    const float* first = reinterpret_cast<const float*>(repo.getFeature(0));
    EXPECT_NEAR(first[0], 0.0f, 1e-6f);
    EXPECT_NEAR(first[dims - 1], static_cast<float>(dims - 1), 1e-6f);

    const float* last = reinterpret_cast<const float*>(repo.getFeature(count - 1));
    size_t offset = (count - 1) * dims;
    EXPECT_NEAR(last[0], static_cast<float>(offset), 1e-6f);
    EXPECT_NEAR(last[dims - 1], static_cast<float>(dims * count - 1), 1e-6f);
}

TEST(StaticFeatureRepository, Uint8Repository) {
    size_t dims = 4;
    size_t count = 2;
    size_t bytes_per_dim = sizeof(uint8_t);

    std::vector<uint8_t> data = {1, 2, 3, 4, 10, 20, 30, 40};
    deglib::StaticFeatureRepository repo(
        make_uint8_features(data), dims, count, bytes_per_dim);
    EXPECT_EQ(repo.dims(), dims);
    EXPECT_EQ(repo.size(), count);

    const uint8_t* f0 = reinterpret_cast<const uint8_t*>(repo.getFeature(0));
    EXPECT_EQ(f0[0], 1);
    EXPECT_EQ(f0[1], 2);
    EXPECT_EQ(f0[2], 3);
    EXPECT_EQ(f0[3], 4);

    const uint8_t* f1 = reinterpret_cast<const uint8_t*>(repo.getFeature(1));
    EXPECT_EQ(f1[0], 10);
    EXPECT_EQ(f1[1], 20);
    EXPECT_EQ(f1[2], 30);
    EXPECT_EQ(f1[3], 40);
}

// ---------------------------------------------------------------------------
//  string_ends_with (local copy — internal to repository.h)
// ---------------------------------------------------------------------------

static bool test_ends_with(const char* str, const char* suffix) {
    size_t str_len = std::strlen(str);
    size_t suffix_len = std::strlen(suffix);
    if (suffix_len > str_len) return false;
    return std::strcmp(str + str_len - suffix_len, suffix) == 0;
}

TEST(StringHelpers, EndsWithFvecs) {
    EXPECT_TRUE(test_ends_with("data/fvecs", "fvecs"));
}

TEST(StringHelpers, EndsWithU8vecs) {
    EXPECT_TRUE(test_ends_with("data/u8vecs", "u8vecs"));
}

TEST(StringHelpers, NoMatch) {
    EXPECT_FALSE(test_ends_with("data/text", "fvecs"));
}

TEST(StringHelpers, EmptySuffix) {
    EXPECT_TRUE(test_ends_with("anything", ""));
}

TEST(StringHelpers, SuffixLongerThanString) {
    EXPECT_FALSE(test_ends_with("ab", "abcdef"));
}
