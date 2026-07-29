#pragma once

#include <variant>
#include <concepts>
#include <stdexcept>

#include "config.h"

// Modular distance headers - contain all distance function class definitions
#include "distance/fp32_l2.h"
#include "distance/fp32_ip.h"
#include "distance/fp16_ip.h"
#include "distance/uint8_l2.h"

namespace deglib {

    enum class Metric {
        // 0x00 = float
        //L1 = 0x00 | 0,
        L2 = 0x00 | 1,
        InnerProduct = 0x00 | 2,
        FP16InnerProduct = 0x00 | 3,

        // 0x10 = uint8
        L2_Uint8 = 0x10 | 1
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
        deglib::distances::uint8_l2::DistanceVariant
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
     * Converts a sub-namespace DistanceVariant into the flat DistanceVariant.
     * Uses std::visit to extract the concrete distance type and return it
     * as the flattened variant — zero runtime overhead, single-level visit.
     */
    template <typename SubVariant>
    inline DistanceVariant to_flat_variant(SubVariant&& sub_variant) {
        return std::visit([](auto&& concrete_dist) -> DistanceVariant {
            return concrete_dist;
        }, std::forward<SubVariant>(sub_variant));
    }

    /**
     * Represents a metric feature space for vector distance computations.
     * Manages dimension, metric type, byte size, and distance evaluation logic.
     */
    class FloatSpace  {

        static DistanceVariant select_dist_variant(const size_t dim, const deglib::Metric metric) {
            switch (metric) {
                case deglib::Metric::L2:
                    return to_flat_variant(deglib::distances::fp32_l2::select_dist(dim));
                case deglib::Metric::InnerProduct:
                    return to_flat_variant(deglib::distances::fp32_ip::select_dist(dim));
                case deglib::Metric::FP16InnerProduct:
                    return to_flat_variant(deglib::distances::fp16_ip::select_dist(dim));
                case deglib::Metric::L2_Uint8:
                    return to_flat_variant(deglib::distances::uint8_l2::select_dist(dim));
                default:
                    throw std::invalid_argument("Unsupported metric type in select_dist_variant");
            }
        }

        static size_t calculate_data_size(const size_t dim, const deglib::Metric metric) {
            if (metric == deglib::Metric::FP16InnerProduct)
                return dim * sizeof(uint16_t);
            return (static_cast<int>(metric) & 0x10) ? dim * sizeof(uint8_t) : dim * sizeof(float);
        }

        const DistanceVariant dist_variant_;
        const size_t data_size_;
        const size_t dim_;
        const deglib::Metric metric_;

    public:
        FloatSpace(const size_t dim, const deglib::Metric metric) 
            : dist_variant_(select_dist_variant(dim, metric)),
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
