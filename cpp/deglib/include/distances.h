#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <cstdio>
#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define POPCNT32 __popcnt
#define POPCNT64 __popcnt64
#else
#define POPCNT32 std::popcount
#define POPCNT64 std::popcount
#endif

#include "distance/fp32_l2.h"
#include "distance/fp32_inner_product.h"
#include "distance/uint8_l2.h"
#include "distance/evp_inner_product.h"
#include "distance/fp16_inner_product.h"
#include "distance/fp16_evp_asym_inner_product.h"
#include "distance/fp16_l2.h"

namespace deglib {

enum class Metric {
    // 0x00 = float
    L2 = 0x00 | 1,
    InnerProduct = 0x00 | 2,

    // 0x10 = uint8
    L2_Uint8 = 0x10 | 1,

    // 0x20 = evp bits (ones + negative_ones, dim/8 bytes each)
    EvpBits = 0x20 | 3,

    // 0x30 = fp16 (uint16_t half-precision floats)
    FP16InnerProduct = 0x30 | 2,

    // 0x40 = fp16 query + evp bits database (asymmetric)
    FP16EvpAsymmetric = 0x40 | 3,

    // 0x50 = fp16 L2 (squared Euclidean in FP16 space)
    FP16L2 = 0x50 | 1
};

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

namespace distances {

template <typename T>
concept DistanceComparator = requires(const void* a, const void* b, const void* c) {
    { T::compare(a, b, c) } -> std::same_as<float>;
};

template <typename T, typename = void>
struct has_compare_batch : std::false_type {};

template <typename T>
struct has_compare_batch<T, std::void_t<decltype(T::compare_batch(
    std::declval<const void*>(),
    std::declval<const void* const*>(),
    std::declval<size_t>(),
    std::declval<const void*>(),
    std::declval<float*>()
))>> : std::true_type {};

template <typename COMPARATOR>
inline static void compare_batch(const void* query, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
    if constexpr (has_compare_batch<COMPARATOR>::value) {
        COMPARATOR::compare_batch(query, db_arr, count, qty_ptr, dists);
    } else {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = COMPARATOR::compare(query, db_arr[i], qty_ptr);
        }
    }
}

template <typename Functor>
auto dispatch_distance(const deglib::Metric metric, const size_t dim, Functor&& f) {
    if (metric == deglib::Metric::L2) {
        if (dim % 16 == 0)      return f.template operator()<deglib::distances::L2Float16Ext>();
        if (dim % 8 == 0)       return f.template operator()<deglib::distances::L2Float8Ext>();
        if (dim % 4 == 0)       return f.template operator()<deglib::distances::L2Float4Ext>();
        return f.template operator()<deglib::distances::FP32L2<1>>();
    } 
    else if (metric == deglib::Metric::InnerProduct) {
        if (dim % 16 == 0)      return f.template operator()<deglib::distances::InnerProductFloat16Ext>();
        if (dim % 8 == 0)       return f.template operator()<deglib::distances::InnerProductFloat8Ext>();
        if (dim % 4 == 0)       return f.template operator()<deglib::distances::InnerProductFloat4Ext>();
        return f.template operator()<deglib::distances::FP32InnerProduct<1>>();
    } 
    else if (metric == deglib::Metric::L2_Uint8) {
        if (dim % 32 == 0)      return f.template operator()<deglib::distances::L2Uint8Ext32>();
        if (dim % 16 == 0)      return f.template operator()<deglib::distances::L2Uint8Ext16>();
        return f.template operator()<deglib::distances::Uint8L2Default>();
    } 
    else if (metric == deglib::Metric::EvpBits) {
        return f.template operator()<deglib::distances::EvpInnerProduct>();
    } 
    else if (metric == deglib::Metric::FP16EvpAsymmetric) {
        return f.template operator()<deglib::distances::FP16EvpAsymInnerProduct>();
    } 
    else if (metric == deglib::Metric::FP16InnerProduct) {
        if (dim % 32 == 0)      return f.template operator()<deglib::distances::FP16InnerProductExt32>();
        if (dim % 16 == 0)      return f.template operator()<deglib::distances::FP16InnerProductExt16>();
        if (dim % 8 == 0)       return f.template operator()<deglib::distances::FP16InnerProductExt8>();
        return f.template operator()<deglib::distances::FP16InnerProduct<1>>();
    }
    else if (metric == deglib::Metric::FP16L2) {
        if (dim % 32 == 0)      return f.template operator()<deglib::distances::FP16L2Ext32>();
        if (dim % 16 == 0)      return f.template operator()<deglib::distances::FP16L2Ext16>();
        if (dim % 8 == 0)       return f.template operator()<deglib::distances::FP16L2Ext8>();
        return f.template operator()<deglib::distances::FP16L2Default>();
    }

    std::fprintf(stderr, "Unsupported metric %u for dispatch_distance\n", static_cast<int>(metric));
    std::abort();
}

} // namespace distances

class FloatSpace {
    static DISTFUNC<float> select_dist_func(const size_t dim, const deglib::Metric metric) {
        return distances::dispatch_distance(metric, dim, []<typename COMPARATOR>() -> DISTFUNC<float> {
            return COMPARATOR::compare;
        });
    }

    static size_t calculate_data_size(const size_t dim, const deglib::Metric metric) {
        if (metric == deglib::Metric::EvpBits || metric == deglib::Metric::FP16EvpAsymmetric) {
            return 2 * dim / 8;
        }
        if (metric == deglib::Metric::FP16InnerProduct || metric == deglib::Metric::FP16L2) {
            return dim * sizeof(uint16_t);
        }

        return (static_cast<int>(metric) & 0x10) ? dim * sizeof(uint8_t) : dim * sizeof(float);
    }

    const DISTFUNC<float> fstdistfunc_;
    const size_t data_size_;
    const size_t dim_;
    const deglib::Metric metric_;

public:
    FloatSpace(const size_t dim, const deglib::Metric metric)
        : fstdistfunc_(select_dist_func(dim, metric)), data_size_(calculate_data_size(dim, metric)), dim_(dim), metric_(metric) {}

    const size_t dim() const { return dim_; }

    const deglib::Metric metric() const { return metric_; }

    const size_t get_data_size() const { return data_size_; }

    const DISTFUNC<float> get_dist_func() const { return fstdistfunc_; }

    const void* get_dist_func_param() const { return &dim_; }

    ~FloatSpace() {}
};

namespace distances {

template <typename Functor>
auto dispatch_distance(const deglib::FloatSpace& space, Functor&& f) {
    return dispatch_distance(space.metric(), space.dim(), std::forward<Functor>(f));
}

} // namespace distances

} // namespace deglib
