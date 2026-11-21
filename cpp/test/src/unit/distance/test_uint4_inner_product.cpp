// test_uint4_inner_product.cpp — Unit tests for Inner Product distance on packed uint4 vectors
//
// Tests scalar and SIMD Inner Product distance implementations for packed uint4 vectors.

#include "deglib/distance/uint4_ip.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

#include <vector>

namespace {

inline std::vector<uint8_t> make_packed_uint4_vec(size_t dim, int seed = 0) {
    size_t num_bytes = (dim + 1) / 2;
    std::vector<uint8_t> v(num_bytes, 0);
    for (size_t i = 0; i < dim; ++i) {
        uint8_t nibble = static_cast<uint8_t>((seed + static_cast<int>(i)) & 0x0F);
        if (i & 1) {
            v[i / 2] |= (nibble << 4);
        } else {
            v[i / 2] |= (nibble & 0x0F);
        }
    }
    return v;
}

inline float ip_uint4_naive(const uint8_t* a, const uint8_t* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        uint8_t byte_a = a[i / 2];
        uint8_t byte_b = b[i / 2];
        uint8_t val_a = (i & 1) ? ((byte_a >> 4) & 0x0F) : (byte_a & 0x0F);
        uint8_t val_b = (i & 1) ? ((byte_b >> 4) & 0x0F) : (byte_b & 0x0F);
        sum += static_cast<float>(val_a) * static_cast<float>(val_b);
    }
    return -sum;
}

using deglib::distances::uint4_ip::InnerProductUint4;

TEST(InnerProductUint4, Identity) {
    std::vector<size_t> dims = {1, 2, 7, 8, 15, 16, 63, 64, 127, 128, 255, 256, 384};
    for (size_t dim : dims) {
        auto v = make_packed_uint4_vec(dim, 42);
        float d = InnerProductUint4::compare(v.data(), v.data(), &dim);
        EXPECT_NEAR(d, ip_uint4_naive(v.data(), v.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(InnerProductUint4, Symmetry) {
    auto a = make_packed_uint4_vec(128, 5);
    auto b = make_packed_uint4_vec(128, 99);
    size_t dim = 128;
    float ab = InnerProductUint4::compare(a.data(), b.data(), &dim);
    float ba = InnerProductUint4::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductUint4, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384, 512};
    for (size_t dim : dims) {
        auto a = make_packed_uint4_vec(dim, 1);
        auto b = make_packed_uint4_vec(dim, 7);
        float d = InnerProductUint4::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint4_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::uint4_ip::InnerProductUint4_AVX2;
using deglib::distances::uint4_ip::InnerProductUint4_AVX512;
using deglib::distances::ResidualMode;

TEST(InnerProductUint4_AVX2, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 256, 384, 512};
    for (size_t dim : dims) {
        auto a = make_packed_uint4_vec(dim, 10);
        auto b = make_packed_uint4_vec(dim, 23);
        float d_scalar = InnerProductUint4::compare(a.data(), b.data(), &dim);
        float d_avx2 = InnerProductUint4_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        float d_naive = ip_uint4_naive(a.data(), b.data(), dim);
        EXPECT_NEAR(d_avx2, d_naive, 1e-4f) << "dim=" << dim;
        EXPECT_NEAR(d_avx2, d_scalar, 1e-4f) << "dim=" << dim;
    }
}

TEST(InnerProductUint4_SelectDist, FloatSpaceDispatch) {
    std::vector<size_t> dims = {1, 2, 3, 7, 15, 16, 31, 32, 63, 64, 127, 128, 129, 253, 255, 256, 384, 512};
    std::vector<deglib::cpu::InstructionSet> isas = {deglib::cpu::InstructionSet::Auto};
    if (deglib::cpu::has_avx2()) isas.push_back(deglib::cpu::InstructionSet::AVX2);
    if (deglib::cpu::has_avx512()) isas.push_back(deglib::cpu::InstructionSet::AVX512);

    for (const size_t dim : dims) {
        auto a = make_packed_uint4_vec(dim, 12);
        auto b = make_packed_uint4_vec(dim, 45);
        const float naive = ip_uint4_naive(a.data(), b.data(), dim);
        const float scalar = deglib::distances::compute_distance(
            deglib::distances::FloatSpace(dim, deglib::distances::Metric::Uint4_InnerProduct, deglib::cpu::InstructionSet::Scalar),
            a.data(),
            b.data()
        );
        EXPECT_NEAR(scalar, naive, 1e-4f) << "scalar dim=" << dim;

        for (const auto isa : isas) {
            deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint4_InnerProduct, isa);
            EXPECT_EQ(space.get_data_size(), (dim + 1) / 2);

            const float dist = deglib::distances::compute_distance(space, a.data(), b.data());
            EXPECT_NEAR(dist, naive, 1e-4f) << "dim=" << dim << " isa=" << static_cast<int>(isa);
            EXPECT_FLOAT_EQ(dist, scalar) << "dim=" << dim << " isa=" << static_cast<int>(isa);
        }
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

