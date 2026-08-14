#pragma once

#include <cstdint>
#include <random>
#include <type_traits>

// Portable random distributions that produce identical sequences across
// compilers and platforms (Windows/MSVC, Linux/GCC, macOS/Clang).
//
// std::uniform_int_distribution is NOT portable across MSVC STL, libstdc++, and libc++.
// DeterministicUniformIntDistribution operates on std::mt19937 using bounded
// rejection sampling to guarantee identical integer sequences on all platforms.

namespace deglib::random {

/// Portable uniform integer distribution using bounded rejection sampling.
/// Guarantees identical integer sequences across Windows, Linux, and macOS.
/// Supports integer types (e.g. int, uint32_t, size_t) with ranges up to 2^32 (using 32-bit generators like std::mt19937).
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

} // namespace deglib::random
