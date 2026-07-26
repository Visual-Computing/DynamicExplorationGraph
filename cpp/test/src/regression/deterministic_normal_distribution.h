#pragma once

#include <cmath>
#include <cstdint>
#include <random>

// Portable normal distribution using Box-Muller transform.
// std::normal_distribution is NOT portable across compilers/platforms:
//   - MSVC STL, libstdc++, and libc++ each use different algorithms
//   - The same seed produces different sequences on different platforms
//   - This makes regression tests non-deterministic across CI runners
//
// This implementation uses the Box-Muller transform with std::mt19937
// (which IS portable — same seed produces same uniform sequence) and
// a manual uniform_real_distribution to ensure portability.
//
// Reference: Box, G.E.P. and Muller, M.E. (1958), "A Note on the Generation
// of Random Normal Deviates", Annals of Mathematical Statistics, 29(2), pp.610-611.
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
