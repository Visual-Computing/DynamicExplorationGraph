#pragma once

#include <cstdint>
#include <random>
#include <type_traits>

// Portable uniform integer distribution.
// std::uniform_int_distribution is NOT portable across compilers/platforms:
//   - MSVC STL, libstdc++, and libc++ use different algorithm implementations/rejection sampling logic.
//   - The same seed produces different sequences on different platforms.
//   - This makes regression tests non-deterministic across CI runners.
//
// This implementation uses bounded rejection sampling over std::mt19937 output to guarantee
// identical integer sequences across Windows, Linux, and macOS (x86_64 and ARM64).
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
