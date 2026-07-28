#pragma once

#include <cmath>
#include <cstdint>
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
//   - DeterministicNormalDistribution: Box-Muller transform
//
// Reference: Box, G.E.P. and Muller, M.E. (1958), "A Note on the Generation
// of Random Normal Deviates", Annals of Mathematical Statistics, 29(2), pp.610-611.

namespace deglib::random {

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

/// Portable normal distribution using the Box-Muller transform.
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
        float mag = stddev_ * std::sqrt(-2.0f * std::log(u1));
        return mean_ + mag * std::cos(2.0f * 3.14159265358979323846f * u2);
    }

private:
    float mean_;
    float stddev_;
};

} // namespace deglib::random
