#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>

// Portable random distributions that produce identical sequences across
// compilers and platforms (Windows/MSVC, Linux/GCC, macOS/Clang).
//
// std::uniform_int_distribution and std::normal_distribution are NOT portable:
//   - MSVC STL, libstdc++, and libc++ each use different algorithm implementations
//   - The same seed produces different sequences on different platforms
//   - This makes regression tests and graph builds non-deterministic across CI runners
//
// Both distributions operate on std::mt19937 (which IS portable — same seed
// produces the same uniform sequence) and use deterministic algorithms:
//   - DeterministicUniformIntDistribution: bounded rejection sampling
//   - DeterministicNormalDistribution: Box-Muller transform with portable math
//
// Reference: Box, G.E.P. and Muller, M.E. (1958), "A Note on the Generation
// of Random Normal Deviates", Annals of Mathematical Statistics, 29(2), pp.610-611.

namespace deglib::random {

namespace detail {

// Portable float natural logarithm (log(x)) without platform libm dependencies.
inline float portable_log(float x) {
    uint32_t ix;
    std::memcpy(&ix, &x, sizeof(float));
    int exp = static_cast<int>((ix >> 23) & 0xFF) - 127;
    ix = (ix & 0x007FFFFF) | 0x3F800000;
    float m;
    std::memcpy(&m, &ix, sizeof(float));
    if (m > 1.41421356f) {
        m *= 0.5f;
        exp += 1;
    }
    float f = m - 1.0f;
    float s = f / (2.0f + f);
    float z = s * s;
    float w = z * z;
    float R = z * (0.6666666666666666f + w * (0.4f + w * (0.2857142857142857f + w * 0.2222222222222222f)));
    return static_cast<float>(exp) * 0.6931471805599453f + s * (2.0f + R);
}

// Portable float cosine (cos(x)) without platform libm dependencies.
inline float portable_cos(float x) {
    constexpr float TWO_PI = 6.28318530717958647692f;
    constexpr float INV_TWO_PI = 0.15915494309189533576f;
    float k = std::floor(x * INV_TWO_PI + 0.5f);
    x -= k * TWO_PI;
    float x2 = x * x;
    return 1.0f - x2 * (0.5f - x2 * (0.041666666666666664f - x2 * (0.0013888888888888889f - x2 * 0.0000248015873015873f)));
}

} // namespace detail

/// Portable uniform integer distribution using bounded rejection sampling.
/// Guarantees identical integer sequences across Windows, Linux, and macOS.
template <typename IntType = int>
class DeterministicUniformIntDistribution {
public:
    DeterministicUniformIntDistribution(IntType min_val, IntType max_val)
        : min_val_(min_val), max_val_(max_val) {
        if (min_val_ < max_val_) {
            range_ = static_cast<uint64_t>(max_val_) - static_cast<uint64_t>(min_val_) + 1;
            max_valid_ = (4294967296ULL / range_) * range_ - 1;
        } else {
            range_ = 1;
            max_valid_ = 4294967295ULL;
        }
    }

    template <typename Generator>
    IntType operator()(Generator& g) {
        if (range_ <= 1) {
            return min_val_;
        }

        uint32_t val;
        do {
            val = g();
        } while (static_cast<uint64_t>(val) > max_valid_);

        return static_cast<IntType>(min_val_ + (val % range_));
    }

private:
    IntType min_val_;
    IntType max_val_;
    uint64_t range_{1};
    uint64_t max_valid_{4294967295ULL};
};

/// Portable normal distribution using the Box-Muller transform with portable math.
/// Always consumes exactly 2 RNG values per call (no caching/spare state)
/// to ensure deterministic behavior regardless of how multiple instances
/// share the same RNG.
class DeterministicNormalDistribution {
public:
    explicit DeterministicNormalDistribution(float mean, float stddev)
        : mean_(mean), stddev_(stddev) {}

    // Generate a single normally-distributed float using the provided RNG.
    // Always consumes exactly 2 RNG values per call (no caching/spare state)
    // to ensure deterministic behavior regardless of how multiple
    // DeterministicNormalDistribution instances share the same RNG.
    float operator()(std::mt19937& rng) {
        // Box-Muller transform: generate one standard normal from two uniforms.
        // We discard the second value (z1) to avoid cross-instance state issues.
        float u1, u2;
        do {
            u1 = static_cast<float>(static_cast<double>(rng()) / (static_cast<double>(rng.max()) + 1.0));
        } while (u1 <= 0.0f);
        u2 = static_cast<float>(static_cast<double>(rng()) / (static_cast<double>(rng.max()) + 1.0));

        // Box-Muller formula: z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
        // Uses portable_log and portable_cos to avoid platform-dependent libm differences.
        float mag = stddev_ * std::sqrt(-2.0f * detail::portable_log(u1));
        return mean_ + mag * detail::portable_cos(2.0f * 3.14159265358979323846f * u2);
    }

private:
    float mean_;
    float stddev_;
};

} // namespace deglib::random
