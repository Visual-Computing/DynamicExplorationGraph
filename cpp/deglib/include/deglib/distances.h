#pragma once

#include "deglib/config.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <variant>

// Shared residual mode for SIMD distance implementations
#include "deglib/distance/residual_mode.h"

// Modular distance headers - contain all distance function class definitions
#include "deglib/distance/evp_ip.h"
#include "deglib/distance/fp16_ip.h"
#include "deglib/distance/fp16_l2.h"
#include "deglib/distance/fp32_ip.h"
#include "deglib/distance/fp32_l2.h"
#include "deglib/distance/uint8_ip.h"
#include "deglib/distance/uint8_l2.h"
#include "deglib/distance/int8_ip.h"
#include "deglib/distance/int8_l2.h"
#include "deglib/distance/uint4_ip.h"
#include "deglib/distance/uint4_l2.h"
#include "deglib/distance/int4_ip.h"
#include "deglib/distance/int4_l2.h"
#include "deglib/distance/uint7_ip.h"
#include "deglib/distance/uint7_l2.h"

namespace deglib::distances {

enum class MetricDataType : uint8_t { FP32 = 0x00, Uint8 = 0x10, FP16 = 0x20, EVP = 0x30, Int8 = 0x40, Uint4 = 0x50, Int4 = 0x60, Uint7 = 0x70 };

enum class MetricDistanceKind : uint8_t { L2 = 0x01, InnerProduct = 0x02 };

enum class MetricType : uint8_t {
    FP32_L2 = static_cast<uint8_t>(MetricDataType::FP32) | static_cast<uint8_t>(MetricDistanceKind::L2),                        // 0x01
    FP32_InnerProduct = static_cast<uint8_t>(MetricDataType::FP32) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),    // 0x02
    Uint8_L2 = static_cast<uint8_t>(MetricDataType::Uint8) | static_cast<uint8_t>(MetricDistanceKind::L2),                      // 0x11
    Uint8_InnerProduct = static_cast<uint8_t>(MetricDataType::Uint8) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),  // 0x12
    FP16_L2 = static_cast<uint8_t>(MetricDataType::FP16) | static_cast<uint8_t>(MetricDistanceKind::L2),                        // 0x21
    FP16_InnerProduct = static_cast<uint8_t>(MetricDataType::FP16) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),    // 0x22
    EVP_InnerProduct = static_cast<uint8_t>(MetricDataType::EVP) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),      // 0x32
    Int8_L2 = static_cast<uint8_t>(MetricDataType::Int8) | static_cast<uint8_t>(MetricDistanceKind::L2),                        // 0x41
    Int8_InnerProduct = static_cast<uint8_t>(MetricDataType::Int8) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),    // 0x42
    Uint4_L2 = static_cast<uint8_t>(MetricDataType::Uint4) | static_cast<uint8_t>(MetricDistanceKind::L2),                      // 0x51
    Uint4_InnerProduct = static_cast<uint8_t>(MetricDataType::Uint4) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),  // 0x52
    Int4_L2 = static_cast<uint8_t>(MetricDataType::Int4) | static_cast<uint8_t>(MetricDistanceKind::L2),                        // 0x61
    Int4_InnerProduct = static_cast<uint8_t>(MetricDataType::Int4) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct),    // 0x62
    Uint7_L2 = static_cast<uint8_t>(MetricDataType::Uint7) | static_cast<uint8_t>(MetricDistanceKind::L2),                      // 0x71
    Uint7_InnerProduct = static_cast<uint8_t>(MetricDataType::Uint7) | static_cast<uint8_t>(MetricDistanceKind::InnerProduct)   // 0x72
};

/**
 * Metric wrapper providing type, distance kind, and string conversion methods.
 */
struct Metric {
    MetricType value;

    constexpr Metric() : value(MetricType::FP32_L2) {}
    constexpr Metric(MetricType v) : value(v) {}
    constexpr explicit Metric(uint8_t raw) : value(static_cast<MetricType>(raw)) {}
    constexpr operator MetricType() const { return value; }

    // Primary enum names in <type>_<metric> format
    static constexpr MetricType FP32_L2 = MetricType::FP32_L2;
    static constexpr MetricType FP32_InnerProduct = MetricType::FP32_InnerProduct;
    static constexpr MetricType Uint8_L2 = MetricType::Uint8_L2;
    static constexpr MetricType Uint8_InnerProduct = MetricType::Uint8_InnerProduct;
    static constexpr MetricType FP16_L2 = MetricType::FP16_L2;
    static constexpr MetricType FP16_InnerProduct = MetricType::FP16_InnerProduct;
    static constexpr MetricType EVP_InnerProduct = MetricType::EVP_InnerProduct;
    static constexpr MetricType Int8_InnerProduct = MetricType::Int8_InnerProduct;
    static constexpr MetricType Int8_L2 = MetricType::Int8_L2;
    static constexpr MetricType Uint4_L2 = MetricType::Uint4_L2;
    static constexpr MetricType Uint4_InnerProduct = MetricType::Uint4_InnerProduct;
    static constexpr MetricType Int4_L2 = MetricType::Int4_L2;
    static constexpr MetricType Int4_InnerProduct = MetricType::Int4_InnerProduct;
    static constexpr MetricType Uint7_L2 = MetricType::Uint7_L2;
    static constexpr MetricType Uint7_InnerProduct = MetricType::Uint7_InnerProduct;

    constexpr MetricDataType get_data_type() const { return static_cast<MetricDataType>(static_cast<uint8_t>(value) & 0xF0); }

    constexpr MetricDistanceKind get_distance_kind() const { return static_cast<MetricDistanceKind>(static_cast<uint8_t>(value) & 0x0F); }

    constexpr const char* get_data_type_name() const {
        switch (get_data_type()) {
            case MetricDataType::FP32:
                return "FP32";
            case MetricDataType::Uint8:
                return "Uint8";
            case MetricDataType::FP16:
                return "FP16";
            case MetricDataType::EVP:
                return "EVP";
            case MetricDataType::Int8:
                return "Int8";
            case MetricDataType::Uint4:
                return "Uint4";
            case MetricDataType::Int4:
                return "Int4";
            case MetricDataType::Uint7:
                return "Uint7";
        }
        return "Unknown";
    }

    constexpr const char* get_distance_name() const {
        switch (get_distance_kind()) {
            case MetricDistanceKind::L2:
                return "L2";
            case MetricDistanceKind::InnerProduct:
                return "InnerProduct";
        }
        return "Unknown";
    }

    std::string to_string() const { return std::string(get_data_type_name()) + "_" + get_distance_name(); }
};

/**
 * Function pointer signature for distance comparison functions.
 */
using DISTFUNC = float (*)(const void*, const void*, const void*);

/**
 * Function pointer signature for batch distance computations over an array of target pointers.
 */
using BATCH_DISTFUNC = void (*)(
    const void* query,
    const void* const* targets,
    size_t count,
    const void* param,
    float* out_distances,
    int32_t prefetch_offset,
    size_t prefetch_bytes
);

/**
 * Function pointer signature for batch distance computations over a contiguous strided target array.
 */
using STRIDED_DISTFUNC = void (*)(
    const void* query,
    const void* targets,
    size_t stride,
    size_t count,
    const void* param,
    float* out_distances,
    int32_t prefetch_offset,
    size_t prefetch_bytes
);

/**
 * Concept for distance function implementations that provide a static compare method
 * compatible with DISTFUNC.
 */
template <typename T>
concept DistanceFunction = requires(const void* a, const void* b, const void* param) {
    { T::compare(a, b, param) } -> std::same_as<float>;
    { T::get_instruction() } -> std::convertible_to<const char*>;
} && std::is_convertible_v<decltype(&T::compare), DISTFUNC>;

template <typename Dist>
concept HasCompareBatch4 = requires(const void* q, const void* t, const void* param, float* out) {
    Dist::compare_batch4(q, t, t, t, t, param, out);
};

namespace detail {

template <DistanceFunction Dist>
inline void batch_compare_impl(
    const void* query,
    const void* const* targets,
    const size_t count,
    const void* param,
    float* out_distances,
    const int32_t prefetch_offset,
    const size_t prefetch_bytes
) {
    constexpr bool has_batch4 = HasCompareBatch4<Dist>;
    if (prefetch_offset > 0 && prefetch_bytes > 0) {
        const size_t po = static_cast<size_t>(prefetch_offset);
        const size_t warmup = std::min(po, count);
        for (size_t i = 0; i < warmup; ++i) {
            deglib::memory::prefetch(static_cast<const char*>(targets[i]), prefetch_bytes);
        }
        size_t i = 0;
        if constexpr (has_batch4) {
            for (; i + 3 < count; i += 4) {
                for (size_t p = 0; p < 4 && (i + po + p) < count; ++p) {
                    deglib::memory::prefetch(static_cast<const char*>(targets[i + po + p]), prefetch_bytes);
                }
                Dist::compare_batch4(query, targets[i], targets[i + 1], targets[i + 2], targets[i + 3], param, out_distances + i);
            }
        }
        for (; i < count; ++i) {
            if (i + po < count) {
                deglib::memory::prefetch(static_cast<const char*>(targets[i + po]), prefetch_bytes);
            }
            out_distances[i] = Dist::compare(query, targets[i], param);
        }
    } else {
        size_t i = 0;
        if constexpr (has_batch4) {
            for (; i + 3 < count; i += 4) {
                Dist::compare_batch4(query, targets[i], targets[i + 1], targets[i + 2], targets[i + 3], param, out_distances + i);
            }
        }
        for (; i < count; ++i) {
            out_distances[i] = Dist::compare(query, targets[i], param);
        }
    }
}

template <DistanceFunction Dist>
inline void strided_compare_impl(
    const void* query,
    const void* targets,
    const size_t stride,
    const size_t count,
    const void* param,
    float* out_distances,
    const int32_t prefetch_offset,
    const size_t prefetch_bytes
) {
    const auto* base = static_cast<const std::byte*>(targets);
    if (prefetch_offset > 0 && prefetch_bytes > 0) {
        const size_t po = static_cast<size_t>(prefetch_offset);
        const size_t warmup = std::min(po, count);
        for (size_t i = 0; i < warmup; ++i) {
            deglib::memory::prefetch(reinterpret_cast<const char*>(base + i * stride), prefetch_bytes);
        }
        for (size_t i = 0; i < count; ++i) {
            if (i + po < count) {
                deglib::memory::prefetch(reinterpret_cast<const char*>(base + (i + po) * stride), prefetch_bytes);
            }
            out_distances[i] = Dist::compare(query, base + i * stride, param);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out_distances[i] = Dist::compare(query, base + i * stride, param);
        }
    }
}

}  // namespace detail

/**
 * Resolved function-pointer table for a concrete DistanceFunction implementation.
 * Constructed once per FloatSpace so callers avoid template fan-out over all distance types
 * while keeping Dist::compare fully inlined inside the batch loops.
 */
struct DistanceKernel {
    DISTFUNC compare = nullptr;
    BATCH_DISTFUNC compare_batch = nullptr;
    STRIDED_DISTFUNC compare_strided = nullptr;
    const char* instruction = "Unknown";
    bool has_batch4 = false;

    template <DistanceFunction Dist>
    static consteval DistanceKernel make() noexcept {
        return DistanceKernel{
            &Dist::compare,
            &detail::batch_compare_impl<Dist>,
            &detail::strided_compare_impl<Dist>,
            Dist::get_instruction(),
            HasCompareBatch4<Dist>,
        };
    }
};

namespace detail {

template <typename SubVariant>
inline DistanceKernel to_kernel(SubVariant&& sub_variant) {
    return std::visit([]<DistanceFunction Dist>(const Dist&) -> DistanceKernel { return DistanceKernel::make<Dist>(); }, sub_variant);
}

}  // namespace detail

/**
 * Represents a metric feature space for vector distance computations.
 * Manages dimension, metric type, byte size, and distance evaluation logic.
 */
class FloatSpace {
    static DistanceKernel
    select_kernel(const size_t dim, const Metric metric, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
        switch (metric) {
            case Metric::FP32_L2:
                return detail::to_kernel(deglib::distances::fp32_l2::select_dist(dim, instruction));
            case Metric::FP32_InnerProduct:
                return detail::to_kernel(deglib::distances::fp32_ip::select_dist(dim, instruction));
            case Metric::FP16_L2:
                return detail::to_kernel(deglib::distances::fp16_l2::select_dist(dim, instruction));
            case Metric::FP16_InnerProduct:
                return detail::to_kernel(deglib::distances::fp16_ip::select_dist(dim, instruction));
            case Metric::Uint8_L2:
                return detail::to_kernel(deglib::distances::uint8_l2::select_dist(dim, instruction));
            case Metric::Uint8_InnerProduct:
                return detail::to_kernel(deglib::distances::uint8_ip::select_dist(dim, instruction));
            case Metric::EVP_InnerProduct:
                return detail::to_kernel(deglib::distances::evp_ip::select_dist(dim, instruction));
            case Metric::Int8_InnerProduct:
                return detail::to_kernel(deglib::distances::int8_ip::select_dist(dim, instruction));
            case Metric::Int8_L2:
                return detail::to_kernel(deglib::distances::int8_l2::select_dist(dim, instruction));
            case Metric::Uint4_InnerProduct:
                return detail::to_kernel(deglib::distances::uint4_ip::select_dist(dim, instruction));
            case Metric::Uint4_L2:
                return detail::to_kernel(deglib::distances::uint4_l2::select_dist(dim, instruction));
            case Metric::Int4_InnerProduct:
                return detail::to_kernel(deglib::distances::int4_ip::select_dist(dim, instruction));
            case Metric::Int4_L2:
                return detail::to_kernel(deglib::distances::int4_l2::select_dist(dim, instruction));
            case Metric::Uint7_L2:
                return detail::to_kernel(deglib::distances::uint7_l2::select_dist(dim, instruction));
            case Metric::Uint7_InnerProduct:
                return detail::to_kernel(deglib::distances::uint7_ip::select_dist(dim, instruction));
            default:
                throw std::invalid_argument("Unsupported metric type in select_kernel");
        }
    }

    static size_t calculate_data_size(const size_t dim, const Metric metric) {
        switch (metric.get_data_type()) {
            case MetricDataType::FP32:
                return dim * sizeof(float);
            case MetricDataType::Uint8:
                return dim * sizeof(uint8_t);
            case MetricDataType::FP16:
                return dim * sizeof(uint16_t);
            case MetricDataType::EVP:
                return 2 * ((dim + 7) / 8);  // EVP: [ones ((dim+7)/8 bytes)][negs ((dim+7)/8 bytes)]
            case MetricDataType::Int8:
                return dim * sizeof(int8_t);
            case MetricDataType::Uint7:
                return dim * sizeof(int8_t);
            case MetricDataType::Uint4:
            case MetricDataType::Int4:
                return (dim + 1) / 2;
            default:
                throw std::invalid_argument("Unsupported metric data type in calculate_data_size");
        }
    }

    static size_t calculate_query_data_size(const size_t dim, const Metric metric) { return calculate_data_size(dim, metric); }

    DistanceKernel kernel_;
    size_t data_size_ = 0;
    size_t query_data_size_ = 0;
    size_t dim_ = 0;
    Metric metric_;

  public:
    FloatSpace() : FloatSpace(0, Metric{}) {}

    FloatSpace(const size_t dim, const Metric metric, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto)
        : kernel_(select_kernel(dim, metric, instruction)),
          data_size_(calculate_data_size(dim, metric)),
          query_data_size_(calculate_query_data_size(dim, metric)),
          dim_(dim),
          metric_(metric) {}

    FloatSpace(const size_t dim, const Metric metric, DistanceKernel kernel)
        : kernel_(kernel), data_size_(calculate_data_size(dim, metric)), query_data_size_(calculate_query_data_size(dim, metric)), dim_(dim), metric_(metric) {}

    /**
     * Returns the dimension of feature vectors in this space.
     */
    const size_t dim() const { return dim_; }

    /**
     * Returns the metric type used for distance computations.
     */
    const Metric metric() const { return metric_; }

    /**
     * Returns the size in bytes of a single feature vector in this space.
     */
    const size_t get_data_size() const { return data_size_; }

    /**
     * Returns the size in bytes of a query vector for this space.
     */
    const size_t get_query_data_size() const { return query_data_size_; }

    /**
     * Returns a function pointer to the selected distance comparison function.
     */
    DISTFUNC get_dist_func() const noexcept { return kernel_.compare; }

    /**
     * Returns a function pointer for batch distance computation over target pointers.
     */
    BATCH_DISTFUNC get_batch_dist_func() const noexcept { return kernel_.compare_batch; }

    /**
     * Returns a function pointer for batch distance computation over contiguous strided targets.
     */
    STRIDED_DISTFUNC get_strided_dist_func() const noexcept { return kernel_.compare_strided; }

    /**
     * Returns true if the selected distance kernel provides a 4-way batch comparison (compare_batch4).
     */
    bool has_batch4() const noexcept { return kernel_.has_batch4; }

    /**
     * Returns the instruction set name (e.g. "AVX512", "AVX2", "Scalar") used by the distance function.
     */
    const char* get_instruction() const noexcept { return kernel_.instruction; }

    /**
     * Returns the parameter required by the distance function (pointer to vector dimension).
     */
    const void* get_dist_func_param() const noexcept { return &dim_; }

    ~FloatSpace() {}
};

/**
 * Compute distance between two feature vectors in the given FloatSpace.
 */
inline float compute_distance(const FloatSpace& space, const void* vec1, const void* vec2) {
    if (vec1 == nullptr || vec2 == nullptr) {
        throw std::invalid_argument("compute_distance: vec1 and vec2 must not be null");
    }
    const auto dist_func = space.get_dist_func();
    const auto param = space.get_dist_func_param();
    return dist_func(vec1, vec2, param);
}

/**
 * Compute distances between queries and targets using the given FloatSpace.
 */
inline void
compute_distances(const FloatSpace& space, const void* queries, size_t num_queries, const void* targets, size_t num_targets, float* result_distances) {
    if (queries == nullptr || targets == nullptr || result_distances == nullptr) {
        throw std::invalid_argument("compute_distances: arguments must not be null");
    }

    const size_t byte_stride = space.get_data_size();
    const uint8_t* q_ptr = static_cast<const uint8_t*>(queries);
    const auto strided_dist_func = space.get_strided_dist_func();
    const auto param = space.get_dist_func_param();

    for (size_t q = 0; q < num_queries; ++q) {
        strided_dist_func(q_ptr + q * byte_stride, targets, byte_stride, num_targets, param, result_distances + q * num_targets, 0, 0);
    }
}

/**
 * Compute distances between queries and targets using the given FloatSpace, returning a vector of floats.
 */
inline std::vector<float> compute_distances(const FloatSpace& space, const void* queries, size_t num_queries, const void* targets, size_t num_targets) {
    std::vector<float> result_distances(num_queries * num_targets);
    compute_distances(space, queries, num_queries, targets, num_targets, result_distances.data());
    return result_distances;
}
}  // namespace deglib::distances
