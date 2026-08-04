#pragma once

#include <cstddef>
#include <variant>
#include <concepts>
#include <stdexcept>

#include "config.h"

// Shared residual mode for SIMD distance implementations
#include "distance/residual_mode.h"

// Modular distance headers - contain all distance function class definitions
#include "distance/fp32_l2.h"
#include "distance/fp32_ip.h"
#include "distance/fp16_ip.h"
#include "distance/uint8_l2.h"
#include "distance/evp_inner_product.h"

namespace deglib {

    enum class MetricDataType : uint8_t {
        FP32  = 0x00,
        Uint8 = 0x10,
        FP16  = 0x20,
        EVP   = 0x30
    };

    enum class MetricDistanceKind : uint8_t {
        L2           = 0x01,
        InnerProduct = 0x02
    };

    enum class MetricType : uint8_t {
        FP32_L2            = static_cast<uint8_t>(MetricDataType::FP32)  | static_cast<uint8_t>(MetricDistanceKind::L2),           // 0x01
        FP32_InnerProduct  = static_cast<uint8_t>(MetricDataType::FP32)  | static_cast<uint8_t>(MetricDistanceKind::InnerProduct), // 0x02
        Uint8_L2          = static_cast<uint8_t>(MetricDataType::Uint8) | static_cast<uint8_t>(MetricDistanceKind::L2),           // 0x11
        FP16_InnerProduct = static_cast<uint8_t>(MetricDataType::FP16)  | static_cast<uint8_t>(MetricDistanceKind::InnerProduct), // 0x22
        EVP_InnerProduct  = static_cast<uint8_t>(MetricDataType::EVP)   | static_cast<uint8_t>(MetricDistanceKind::InnerProduct)  // 0x32
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
        static constexpr MetricType FP16_InnerProduct = MetricType::FP16_InnerProduct;
        static constexpr MetricType EVP_InnerProduct = MetricType::EVP_InnerProduct;

        // Legacy metric name aliases for 100% backward compatibility
        static constexpr MetricType L2 = MetricType::FP32_L2;
        static constexpr MetricType InnerProduct = MetricType::FP32_InnerProduct;
        static constexpr MetricType L2_Uint8 = MetricType::Uint8_L2;
        static constexpr MetricType FP16InnerProduct = MetricType::FP16_InnerProduct;
        static constexpr MetricType EVPInnerProduct = MetricType::EVP_InnerProduct;

        constexpr MetricDataType get_data_type() const {
            return static_cast<MetricDataType>(static_cast<uint8_t>(value) & 0xF0);
        }

        constexpr MetricDistanceKind get_distance_kind() const {
            return static_cast<MetricDistanceKind>(static_cast<uint8_t>(value) & 0x0F);
        }

        constexpr const char* get_data_type_name() const {
            switch (get_data_type()) {
                case MetricDataType::FP32:  return "FP32";
                case MetricDataType::Uint8: return "Uint8";
                case MetricDataType::FP16:  return "FP16";
                case MetricDataType::EVP:   return "EVP";
            }
            return "Unknown";
        }

        constexpr const char* get_distance_name() const {
            switch (get_distance_kind()) {
                case MetricDistanceKind::L2:           return "L2";
                case MetricDistanceKind::InnerProduct: return "InnerProduct";
            }
            return "Unknown";
        }

        std::string to_string() const {
            return std::string(get_data_type_name()) + "_" + get_distance_name();
        }
    };

    /**
     * Function pointer signature for distance comparison functions.
     */
    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

    /**
     * Concept for distance function implementations that provide a static compare method
     * compatible with DISTFUNC<float>.
     */
    template <typename T>
    concept DistanceFunction = requires(const void* a, const void* b, const void* param) {
        { T::compare(a, b, param) } -> std::same_as<float>;
    } && std::is_convertible_v<decltype(&T::compare), DISTFUNC<float>>;

    /**
     * Metaprogrammierung: Fügt mehrere std::variants zu EINEM flachen std::variant zusammen.
     * Das Ergebnis ist ein flacher Variant -> 0 Laufzeit-Overhead, einstufiger std::visit.
     */
    template <typename... Variants>
    struct variant_concat;
    template <typename... Ts1, typename... Ts2, typename... Rest>
    struct variant_concat<std::variant<Ts1...>, std::variant<Ts2...>, Rest...>
        : variant_concat<std::variant<Ts1..., Ts2...>, Rest...> {};
    template <typename... Ts>
    struct variant_concat<std::variant<Ts...>> {
        using type = std::variant<Ts...>;
    };
    template <typename... Variants>
    using variant_concat_t = typename variant_concat<Variants...>::type;

    /**
     * Variant containing all supported concrete distance function implementations.
     * Automatically flattened from sub-namespace variants at compile time.
     */
    using DistanceVariant = variant_concat_t<
        deglib::distances::fp32_l2::DistanceVariant,
        deglib::distances::fp32_ip::DistanceVariant,
        deglib::distances::fp16_ip::DistanceVariant,
        deglib::distances::uint8_l2::DistanceVariant,
        deglib::distances::evp_ip::DistanceVariant
    >;

    // Compile-time verification that every type in DistanceVariant fulfills the DistanceFunction concept
    static_assert([]<typename... Ts>(std::variant<Ts...>*) {
        return (deglib::DistanceFunction<Ts> && ...);
    }(static_cast<DistanceVariant*>(nullptr)), "All types in DistanceVariant must satisfy DistanceFunction concept");

    /**
     * Extracts the DISTFUNC<float> function pointer from a DistanceVariant object.
     */
    inline DISTFUNC<float> to_dist_func(const DistanceVariant& variant) {
        return std::visit([](auto&& dist) -> DISTFUNC<float> {
            using DistType = std::decay_t<decltype(dist)>;
            static_assert(deglib::DistanceFunction<DistType>, "Selected distance variant must satisfy DistanceFunction concept");
            return &DistType::compare;
        }, variant);
    }

    /**
     * Extracts the instruction set string name from a DistanceVariant object.
     */
    inline const char* to_instruction(const DistanceVariant& variant) {
        return std::visit([](auto&& dist) -> const char* {
            using DistType = std::decay_t<decltype(dist)>;
            return DistType::get_instruction();
        }, variant);
    }

    /**
     * Converts a sub-namespace DistanceVariant into the flat DistanceVariant.
     * Uses std::visit to extract the concrete distance type and return it
     * as the flattened variant — zero runtime overhead, single-level visit.
     */
    template <typename SubVariant>
    inline DistanceVariant to_flat_variant(SubVariant&& sub_variant) {
        return std::visit([](const auto& concrete_dist) -> DistanceVariant {
            return DistanceVariant{concrete_dist};
        }, sub_variant);
    }

    /**
     * Represents a metric feature space for vector distance computations.
     * Manages dimension, metric type, byte size, and distance evaluation logic.
     */
    class FloatSpace  {

        static DistanceVariant select_dist_variant(const size_t dim, const deglib::Metric metric, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
            switch (metric) {
                case deglib::Metric::FP32_L2:
                    return to_flat_variant(deglib::distances::fp32_l2::select_dist(dim, instruction));
                case deglib::Metric::FP32_InnerProduct:
                    return to_flat_variant(deglib::distances::fp32_ip::select_dist(dim, instruction));
                case deglib::Metric::FP16_InnerProduct:
                    return to_flat_variant(deglib::distances::fp16_ip::select_dist(dim, instruction));
                case deglib::Metric::Uint8_L2:
                    return to_flat_variant(deglib::distances::uint8_l2::select_dist(dim, instruction));
                case deglib::Metric::EVP_InnerProduct:
                    return to_flat_variant(deglib::distances::evp_ip::select_dist(dim, instruction));
                default:
                    throw std::invalid_argument("Unsupported metric type in select_dist_variant");
            }
        }

        static size_t calculate_data_size(const size_t dim, const deglib::Metric metric) {
            switch (metric.get_data_type()) {
                case deglib::MetricDataType::FP32:  return dim * sizeof(float);
                case deglib::MetricDataType::Uint8: return dim * sizeof(uint8_t);
                case deglib::MetricDataType::FP16:  return dim * sizeof(uint16_t);
                case deglib::MetricDataType::EVP:   return 2 * (dim / 8);  // EVP: [ones (dim/8 bytes)][negs (dim/8 bytes)]
                default: throw std::invalid_argument("Unsupported metric data type in calculate_data_size");
            }
        }

        const DistanceVariant dist_variant_;
        const size_t data_size_;
        const size_t dim_;
        const deglib::Metric metric_;

    public:
        FloatSpace(const size_t dim, const deglib::Metric metric, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) 
            : dist_variant_(select_dist_variant(dim, metric, instruction)),
              data_size_(calculate_data_size(dim, metric)),
              dim_(dim),
              metric_(metric) {
        }

        FloatSpace(const size_t dim, const deglib::Metric metric, DistanceVariant dist_variant)
            : dist_variant_(std::move(dist_variant)),
              data_size_(calculate_data_size(dim, metric)),
              dim_(dim),
              metric_(metric) {
        }

        /**
         * Returns the dimension of feature vectors in this space.
         */
        const size_t dim() const {
            return dim_;
        }

        /**
         * Returns the metric type used for distance computations.
         */
        const deglib::Metric metric() const {
            return metric_;
        }

        /**
         * Returns the size in bytes of a single feature vector in this space.
         */
        const size_t get_data_size() const {
            return data_size_;
        }

        /**
         * Returns a function pointer to the selected distance comparison function.
         */
        const DISTFUNC<float> get_dist_func() const {
            return to_dist_func(dist_variant_);
        }

        /**
         * Returns the instruction set name (e.g. "AVX512", "AVX2", "Scalar") used by the distance function.
         */
        const char* get_instruction() const {
            return to_instruction(dist_variant_);
        }

        /**
         * Returns the parameter required by the distance function (pointer to vector dimension).
         */
        const void *get_dist_func_param() const {
            return &dim_;
        }

        /**
         * Executes a visitor function with the concrete compile-time DistanceFunction type
         * selected for this feature space (static dispatch).
         */
        template <typename Visitor>
        decltype(auto) compute(Visitor&& visitor) const {
            return std::visit(std::forward<Visitor>(visitor), dist_variant_);
        }

        ~FloatSpace() {}
    };

}  // end namespace deglib
