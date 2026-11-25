#pragma once

//
// Python bindings for Dynamic Exploration Graph using nanobind
//

#include <deglib/deglib.h>
#include <deglib/concurrent.h>
#include <deglib/distance/fp16.h>
#include <deglib/optimization.h>
#include <deglib/optimization/pruning.h>
#include <deglib/transformation.h>
#include <deglib/tuning.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

// ============================================================================
// NDArray Type Aliases & Creation / Inspection Helpers
// ============================================================================

using InArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
using InFloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using InUint16Array = nb::ndarray<const uint16_t, nb::c_contig, nb::device::cpu>;
using InUint32Array = nb::ndarray<const uint32_t, nb::c_contig, nb::device::cpu>;
using InIntArray = nb::ndarray<const int, nb::c_contig, nb::device::cpu>;
using InStridedUint32Array = nb::ndarray<const uint32_t, nb::device::cpu>;

inline constexpr nb::dlpack::dtype kDtypeFloat16{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};

template <typename T>
inline nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig> make_numpy_1d(size_t count, T*& out_ptr) {
    T* data = new T[count > 0 ? count : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[1] = {count};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig>(data, 1, shape, owner);
}

template <typename T>
inline nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig> make_numpy_2d(size_t rows, size_t cols, T*& out_ptr) {
    const size_t total = rows * cols;
    T* data = new T[total > 0 ? total : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[2] = {rows, cols};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig>(data, 2, shape, owner);
}

inline nb::ndarray<nb::numpy, nb::c_contig> make_numpy_fp16(size_t ndim, const size_t* shape, uint16_t*& out_ptr) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= shape[i];
    }
    uint16_t* data = new uint16_t[total > 0 ? total : 1];
    out_ptr = data;
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint16_t*>(p); });
    return nb::ndarray<nb::numpy, nb::c_contig>(data, ndim, shape, owner, nullptr, kDtypeFloat16);
}

inline bool is_dtype_float16_only(nb::dlpack::dtype dt) noexcept { return dt == kDtypeFloat16; }

inline bool is_dtype_fp16_or_u16(nb::dlpack::dtype dt) noexcept {
    return dt.bits == 16 && dt.lanes == 1 &&
           (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) ||
            dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int));
}

inline bool is_dtype_float32(nb::dlpack::dtype dt) noexcept { return dt == nb::dtype<float>(); }

inline std::pair<size_t, uint32_t> get_2d_shape(const InArray& arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected 2D array, got {} dimensions", arr.ndim()));
    }
    return {arr.shape(0), static_cast<uint32_t>(arr.shape(1))};
}

inline std::vector<float> convert_array_to_float32(const InArray& arr) {
    const size_t n = arr.size();
    std::vector<float> out(n);
    if (n == 0) return out;
    const auto dt = arr.dtype();
    const void* src = arr.data();
    if (dt.lanes != 1) {
        throw std::invalid_argument("Unsupported array dtype with lanes != 1");
    }
    const auto code = static_cast<nb::dlpack::dtype_code>(dt.code);
    if (code == nb::dlpack::dtype_code::Float) {
        if (dt.bits == 32) {
            std::memcpy(out.data(), src, n * sizeof(float));
            return out;
        }
        if (dt.bits == 16) {
            deglib::distances::fp16::fp16_to_floats(static_cast<const uint16_t*>(src), out.data(), n);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const double*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::Int) {
        if (dt.bits == 8) {
            const auto* p = static_cast<const int8_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 16) {
            const auto* p = static_cast<const int16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 32) {
            const auto* p = static_cast<const int32_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const int64_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::UInt) {
        if (dt.bits == 8) {
            const auto* p = static_cast<const uint8_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 16) {
            const auto* p = static_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 32) {
            const auto* p = static_cast<const uint32_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
        if (dt.bits == 64) {
            const auto* p = static_cast<const uint64_t*>(src);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            return out;
        }
    } else if (code == nb::dlpack::dtype_code::Bool && dt.bits == 8) {
        const auto* p = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n; ++i) out[i] = p[i] ? 1.0f : 0.0f;
        return out;
    }
    throw std::invalid_argument(std::format("Unsupported array dtype (code: {}, bits: {}) for float32 conversion", dt.code, dt.bits));
}

inline const void* extract_custom_features_ptr(
    const deglib::DynamicExplorationGraph& graph,
    const std::optional<deglib::distances::FloatSpace>& custom_feature_space,
    const std::optional<InArray>& custom_features
) {
    if (!custom_features.has_value() || !custom_features->is_valid()) {
        return nullptr;
    }
    const void* feat_ptr = custom_features->data();
    if (custom_feature_space.has_value()) {
        const size_t expected_bytes = graph.size() * custom_feature_space->get_data_size();
        const size_t actual_bytes = custom_features->size() * custom_features->itemsize();
        if (actual_bytes != expected_bytes) {
            throw std::invalid_argument(
                std::format("custom_features buffer size ({} bytes) does not match expected size for graph ({} bytes)", actual_bytes, expected_bytes)
            );
        }
    }
    return feat_ptr;
}

inline deglib::distances::MetricDataType parse_quant_data_type(const nb::object& quant) {
    if (nb::isinstance<deglib::distances::MetricType>(quant)) {
        return deglib::distances::Metric(nb::cast<deglib::distances::MetricType>(quant)).get_data_type();
    }
    if (nb::isinstance<nb::int_>(quant)) {
        return deglib::distances::Metric(static_cast<deglib::distances::MetricType>(nb::cast<uint8_t>(quant))).get_data_type();
    }
    if (nb::isinstance<nb::str>(quant)) {
        std::string q_lower = nb::cast<std::string>(quant);
        for (char& c : q_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (q_lower == "uint4") return deglib::distances::MetricDataType::Uint4;
        if (q_lower == "int4") return deglib::distances::MetricDataType::Int4;
        if (q_lower == "uint7") return deglib::distances::MetricDataType::Uint7;
        if (q_lower == "uint8") return deglib::distances::MetricDataType::Uint8;
        if (q_lower == "int8") return deglib::distances::MetricDataType::Int8;
        if (q_lower == "fp16") return deglib::distances::MetricDataType::FP16;
        if (q_lower == "evp") return deglib::distances::MetricDataType::EVP;
        throw std::invalid_argument("Unsupported quantizer data type: " + q_lower);
    }
    throw std::invalid_argument("quant must be a Metric or data type string (uint4, int4, uint7, uint8, int8, fp16, evp)");
}

inline deglib::search::RecallTable::Mode parse_recall_mode_py(const std::string& mode) {
    if (mode == "ef") {
        return deglib::search::RecallTable::Mode::Ef;
    }
    if (mode == "eps") {
        return deglib::search::RecallTable::Mode::Eps;
    }
    throw std::invalid_argument(std::format("RecallTable mode must be 'ef' or 'eps', got '{}'", mode));
}

nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig> presort_wrapper(
    const InArray& vectors,
    const deglib::distances::FloatSpace& space,
    float radius_decay = 0.9f,
    size_t threads = 0,
    nb::object callback = nb::none()
);

namespace deglib_py {
void bind_cpu_and_distances(nb::module_& m);
void bind_optimization(nb::module_& m);
void bind_transformation(nb::module_& m);
void bind_search(nb::module_& m);
void bind_graph_and_builder(nb::module_& m);
void bind_tuning(nb::module_& m);
}  // namespace deglib_py
