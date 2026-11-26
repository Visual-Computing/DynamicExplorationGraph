#include "bindings_common.h"

// ============================================================================
// Transformation Binding Helpers
// ============================================================================

namespace deglib_py {
namespace detail {

/**
 * Loads one FP32 or FP16 vector as an FP32 span, using `scratch` only for FP16 conversion.
 */
inline std::span<const float> load_as_f32_span(const std::byte* src, uint32_t dim, bool src_fp16, std::vector<float>& scratch) {
    if (src_fp16) {
        if (scratch.size() < dim) scratch.resize(dim);
        deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), scratch.data(), dim);
        return std::span<const float>(scratch.data(), dim);
    }
    return std::span<const float>(reinterpret_cast<const float*>(src), dim);
}

/**
 * Decides whether an InnerProduct graph needs a MipsTransform (dim -> dim + 1) and fits its max_norm.
 *
 * Why scan all `n` vectors:
 *   MipsTransform maps x -> [x / M, sqrt(1 - ||x / M||^2)] where M = max_i ||x_i||.
 *   A subsample could miss high-norm outliers and produce ||x / M|| > 1 (NaN in sqrt).
 * Why skip MipsTransform for unit-norm vectors:
 *   When all vectors already have ||x|| ~= 1, Inner Product ordering equals L2 ordering
 *   (||a - b||^2 = 2 - 2*dot(a,b)), so an extra MIPS dimension is unnecessary.
 * Why use FloatSpace(InnerProduct):
 *   Reuses the SIMD AVX2/AVX512 FP16/FP32 inner-product kernel (dist = 1 - dot(v, v))
 *   instead of a manual scalar loop or FP16-to-FP32 scratch buffer.
 */
template <typename InternalGraphT>
inline std::shared_ptr<deglib::transformation::MipsTransform> fit_mips_transform_from_graph(
    const InternalGraphT& internal, uint32_t dim, bool src_fp16
) {
    const auto ip_metric = src_fp16 ? deglib::distances::Metric::FP16_InnerProduct : deglib::distances::Metric::FP32_InnerProduct;
    const deglib::distances::FloatSpace ip_space(dim, ip_metric);
    const auto ip_dist = ip_space.get_dist_func();
    const void* dist_param = ip_space.get_dist_func_param();

    float min_norm_sq = std::numeric_limits<float>::max();
    float max_norm_sq = 0.0f;
    const size_t n = internal.size();
    for (size_t i = 0; i < n; ++i) {
        const std::byte* src = internal.getFeatureVector(static_cast<uint32_t>(i));
        const float norm_sq = 1.0f - ip_dist(src, src, dist_param);
        if (norm_sq < min_norm_sq) min_norm_sq = norm_sq;
        if (norm_sq > max_norm_sq) max_norm_sq = norm_sq;
    }

    const float tol = src_fp16 ? 1e-2f : 1e-3f;
    const bool is_unit_norm = (std::abs(min_norm_sq - 1.0f) <= tol) && (std::abs(max_norm_sq - 1.0f) <= tol);
    if (is_unit_norm) {
        return nullptr;
    }
    return std::make_shared<deglib::transformation::MipsTransform>(dim, std::sqrt(max_norm_sq));
}

}  // namespace detail

/**
 * Calibrates a quantizer pipeline directly from an FP32/FP16 graph's stored feature vectors.
 * When `mips` is true, checks whether stored vectors are already unit-norm; if not, fits and
 * prepends a `MipsTransform` (`dim -> dim + 1`). When `normalize` is true, prepends `L2NormalizeTransform`.
 */
template <typename InternalGraphT>
inline std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> make_quantizer_pipeline_from_graph(
    const InternalGraphT& internal,
    deglib::distances::MetricDataType target_dt,
    float quantile = -1.0f,
    bool use_rerank = false,
    size_t num_threads = 0,
    uint64_t rotation_seed = 7,
    bool normalize = false,
    bool mips = false
) {
    if (internal.size() == 0) {
        throw std::invalid_argument("Cannot calibrate quantizer pipeline from an empty graph");
    }
    if (normalize && mips) {
        throw std::invalid_argument("Cannot enable both normalize and mips in make_quantizer_pipeline_from_graph");
    }
    const auto& src_space = internal.getFeatureSpace();
    const auto src_dt = src_space.metric().get_data_type();
    if (src_dt != deglib::distances::MetricDataType::FP32 && src_dt != deglib::distances::MetricDataType::FP16) {
        throw std::invalid_argument("Cannot calibrate quantizer pipeline from a non-float graph without external features");
    }

    const size_t n = internal.size();
    const uint32_t dim = static_cast<uint32_t>(src_space.dim());
    const size_t max_sample = (target_dt == deglib::distances::MetricDataType::EVP) ? size_t{65536} : size_t{10000};
    const size_t sample_count = std::min(n, max_sample);
    const bool src_fp16 = (src_dt == deglib::distances::MetricDataType::FP16);

    auto l2n = normalize ? std::make_shared<deglib::transformation::L2NormalizeTransform>(dim) : nullptr;
    auto mips_tr = mips ? detail::fit_mips_transform_from_graph(internal, dim, src_fp16) : nullptr;

    // Downstream quantizers (rotation, scalar/binary quantizers) must calibrate on the
    // post-prefix vectors (`sample_f32`), not on raw vectors. The full graph is transformed
    // later when the caller applies the returned pipeline.
    const bool use_mips = static_cast<bool>(mips_tr);
    const uint32_t prep_dim = use_mips ? (dim + 1) : dim;
    std::vector<float> sample_f32(sample_count * prep_dim);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, sample_count, num_threads,
        [&](size_t begin, size_t end, size_t) {
            std::vector<float> fp16_scratch;
            for (size_t s = begin; s < end; ++s) {
                const uint32_t v_idx = static_cast<uint32_t>((sample_count == n) ? s : (s * n) / sample_count);
                const std::byte* src = internal.getFeatureVector(v_idx);
                float* dst = sample_f32.data() + s * prep_dim;
                auto dst_bytes = std::as_writable_bytes(std::span<float>(dst, prep_dim));

                if (use_mips) {
                    auto raw_f32 = detail::load_as_f32_span(src, dim, src_fp16, fp16_scratch);
                    mips_tr->transform_database(raw_f32, dst_bytes);
                } else {
                    if (src_fp16) {
                        deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), dst, dim);
                    } else {
                        std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
                    }
                    if (l2n) {
                        l2n->transform_database(std::span<const float>(dst, dim), dst_bytes);
                    }
                }
            }
        },
        64
    );

    auto pipeline = deglib::transformation::make_quantizer_pipeline(
        std::span<const float>(sample_f32), prep_dim, target_dt, quantile, use_rerank, num_threads, rotation_seed
    );
    if (l2n) {
        pipeline.insert(pipeline.begin(), std::move(l2n));
    } else if (mips_tr) {
        pipeline.insert(pipeline.begin(), std::move(mips_tr));
    }
    return pipeline;
}

}  // namespace deglib_py

template <typename QuantT, typename OutElemT, bool Packed4Bit = false>
nb::ndarray<nb::numpy, OutElemT, nb::ndim<2>, nb::c_contig> quantize_scalar_2d(const QuantT& q, const InArray& vectors, size_t threads) {
    auto [count, dim] = get_2d_shape(vectors);
    const size_t out_cols = Packed4Bit ? ((static_cast<size_t>(dim) + 1) / 2) : static_cast<size_t>(dim);
    const auto dt = vectors.dtype();
    const bool is_fp16 = is_dtype_fp16_or_u16(dt);
    const bool is_fp32 = is_dtype_float32(dt);
    if (!is_fp16 && !is_fp32) {
        throw std::invalid_argument("vectors must be float32 or float16");
    }

    OutElemT* out_ptr = nullptr;
    auto output = make_numpy_2d<OutElemT>(count, out_cols, out_ptr);
    std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
    const void* src = vectors.data();

    nb::gil_scoped_release release;
    if (is_fp16) {
        q.transform_database(
            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * out_cols * sizeof(OutElemT)), threads
        );
    } else {
        q.transform_database(
            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * out_cols * sizeof(OutElemT)), threads
        );
    }
    return output;
}

template <typename T>
std::shared_ptr<T> deserialize_transform(nb::bytes b) {
    std::string s(b.c_str(), b.size());
    std::istringstream ss(s, std::ios::binary);
    return std::shared_ptr<T>(T::deserialize(ss));
}

template <typename QuantT>
std::shared_ptr<QuantT> make_scalar_quantizer_impl(const InArray& vectors, float quantile) {
    auto [count, dim] = get_2d_shape(vectors);
    if (count == 0 || dim == 0) {
        throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
    }
    const auto dt = vectors.dtype();
    const bool is_fp16 = is_dtype_fp16_or_u16(dt);
    const bool is_fp32 = is_dtype_float32(dt);
    if (!is_fp16 && !is_fp32) {
        throw std::invalid_argument("vectors must be float32 or float16");
    }
    const void* src = vectors.data();
    auto q = std::make_shared<QuantT>(dim);
    nb::gil_scoped_release release;
    if (is_fp16) {
        q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), quantile);
    } else {
        q->fit(std::span<const float>(static_cast<const float*>(src), count * dim), quantile);
    }
    return q;
}


namespace deglib_py {
void bind_transformation(nb::module_& m) {
    nb::module_ transformation_module = m.def_submodule("transformation", "Feature transformation and quantization");

    auto apply_transformations_array = [](const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations,
                                          const InArray& in_arr, bool is_database, size_t threads = 0) -> nb::object {
        if (transformations.empty()) {
            return nb::cast(in_arr);
        }
        const size_t ndim = in_arr.ndim();
        if (ndim != 1 && ndim != 2) {
            throw std::invalid_argument(std::format("Expected 1D or 2D array, got {} dimensions", ndim));
        }
        const bool is_1d = (ndim == 1);
        const size_t count = is_1d ? 1 : in_arr.shape(0);
        const size_t dim = is_1d ? in_arr.shape(0) : in_arr.shape(1);
        if (dim != transformations.front()->input_dim()) {
            throw std::invalid_argument(
                std::format("Input dimension ({}) does not match transformation input_dim ({})", dim, transformations.front()->input_dim())
            );
        }

        const auto dt = in_arr.dtype();
        const bool is_fp16 = is_dtype_fp16_or_u16(dt);
        const bool is_fp32 = is_dtype_float32(dt);
        std::vector<float> f32_converted;
        const float* f_ptr = nullptr;
        const uint16_t* u16_ptr = nullptr;
        if (is_database && is_fp16) {
            u16_ptr = static_cast<const uint16_t*>(in_arr.data());
        } else if (is_fp32) {
            f_ptr = static_cast<const float*>(in_arr.data());
        } else {
            f32_converted = convert_array_to_float32(in_arr);
            f_ptr = f32_converted.data();
        }

        const size_t out_stride = transformations.back()->output_byte_size();
        auto run_transform = [&](std::byte* out_ptr) {
            nb::gil_scoped_release release;
            std::span<std::byte> out_span(out_ptr, count * out_stride);
            if (is_database) {
                if (u16_ptr != nullptr) {
                    deglib::transformation::apply_on_database(transformations, std::span<const uint16_t>(u16_ptr, count * dim), out_span, threads);
                } else {
                    deglib::transformation::apply_on_database(transformations, std::span<const float>(f_ptr, count * dim), out_span, threads);
                }
            } else {
                deglib::transformation::apply_on_query(transformations, std::span<const float>(f_ptr, count * dim), out_span);
            }
        };

        auto emit_typed = [&](auto type_tag, size_t cols) -> nb::object {
            using OutT = typename decltype(type_tag)::type;
            OutT* ptr = nullptr;
            if (is_1d) {
                auto out = make_numpy_1d<OutT>(cols, ptr);
                run_transform(reinterpret_cast<std::byte*>(ptr));
                return out.cast();
            }
            auto out = make_numpy_2d<OutT>(count, cols, ptr);
            run_transform(reinterpret_cast<std::byte*>(ptr));
            return out.cast();
        };

        const uint32_t tag = transformations.back()->tag();
        if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) {
            return emit_typed(std::type_identity<int8_t>{}, out_stride);
        } else if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
                   tag == deglib::transformation::TAG_SQI4) {
            return emit_typed(std::type_identity<uint8_t>{}, out_stride);
        } else if (tag == deglib::transformation::TAG_SQ16) {
            uint16_t* ptr = nullptr;
            const size_t out_dim = transformations.back()->output_dim();
            if (is_1d) {
                const size_t shape[1] = {out_dim};
                auto out = make_numpy_fp16(1, shape, ptr);
                run_transform(reinterpret_cast<std::byte*>(ptr));
                return out.cast();
            }
            const size_t shape[2] = {count, out_dim};
            auto out = make_numpy_fp16(2, shape, ptr);
            run_transform(reinterpret_cast<std::byte*>(ptr));
            return out.cast();
        } else {
            return emit_typed(std::type_identity<float>{}, transformations.back()->output_dim());
        }
    };

    nb::enum_<deglib::transformation::TransformationType>(transformation_module, "TransformationType")
        .value("Spatial", deglib::transformation::TransformationType::Spatial)
        .value("Rotation", deglib::transformation::TransformationType::Rotation)
        .value("Quantizer", deglib::transformation::TransformationType::Quantizer);

    nb::class_<deglib::transformation::FeatureTransformation>(transformation_module, "FeatureTransformation")
        .def_prop_ro("name", [](const deglib::transformation::FeatureTransformation& t) { return std::string(t.name()); })
        .def_prop_ro("type", &deglib::transformation::FeatureTransformation::type)
        .def_prop_ro("tag", &deglib::transformation::FeatureTransformation::tag)
        .def_prop_ro("input_dim", &deglib::transformation::FeatureTransformation::input_dim)
        .def_prop_ro("output_dim", &deglib::transformation::FeatureTransformation::output_dim)
        .def_prop_ro("input_byte_size", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_prop_ro("output_byte_size", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_prop_ro("input_bytes", &deglib::transformation::FeatureTransformation::input_byte_size)
        .def_prop_ro("output_bytes", &deglib::transformation::FeatureTransformation::output_byte_size)
        .def_prop_ro("is_fitted", &deglib::transformation::FeatureTransformation::is_fitted)
        .def_prop_ro(
            "output_dtype",
            [](const deglib::transformation::FeatureTransformation& t) -> nb::object {
                const uint32_t tag = t.tag();
                nb::module_ np = nb::module_::import_("numpy");
                if (tag == deglib::transformation::TAG_SQI8 || tag == deglib::transformation::TAG_SQU7) return nb::object(np.attr("int8"));
                if (tag == deglib::transformation::TAG_SQU8 || tag == deglib::transformation::TAG_EVPQ || tag == deglib::transformation::TAG_SQU4 ||
                    tag == deglib::transformation::TAG_SQI4)
                    return nb::object(np.attr("uint8"));
                if (tag == deglib::transformation::TAG_SQ16) return nb::object(np.attr("float16"));
                return nb::object(np.attr("float32"));
            }
        )
        .def(
            "transform_database",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def(
            "quantize",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr, size_t threads) {
                return apply_transformations_array({std::move(t)}, in_arr, true, threads);
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def(
            "transform_query",
            [apply_transformations_array](std::shared_ptr<deglib::transformation::FeatureTransformation> t, const InArray& in_arr) {
                return apply_transformations_array({std::move(t)}, in_arr, false, 1);
            },
            nb::arg("queries")
        )
        .def(
            "dequantize",
            [](const deglib::transformation::FeatureTransformation& t, const InArray& quantized, size_t threads) -> nb::object {
                const size_t ndim = quantized.ndim();
                const bool is_1d = (ndim == 1);
                if (ndim != 1 && ndim != 2) {
                    throw std::invalid_argument(std::format("Expected 1D or 2D array, got {} dimensions", ndim));
                }
                const size_t count = is_1d ? 1 : quantized.shape(0);
                const size_t row_bytes = (is_1d ? quantized.shape(0) : quantized.shape(1)) * quantized.itemsize();
                if (row_bytes != t.output_byte_size()) {
                    throw std::invalid_argument(std::format("quantized row bytes ({}) does not match expected ({})", row_bytes, t.output_byte_size()));
                }
                const uint32_t out_dim = t.output_dim();
                const std::byte* src = static_cast<const std::byte*>(quantized.data());
                auto run_deq = [&](float* dst) {
                    if (count > 0) {
                        nb::gil_scoped_release release;
                        deglib::concurrent::parallel_batch_for(
                            size_t{0}, count, threads,
                            [&](size_t begin, size_t end, size_t) {
                                for (size_t i = begin; i < end; ++i) {
                                    t.dequantize({src + i * row_bytes, row_bytes}, {dst + i * out_dim, out_dim});
                                }
                            },
                            16
                        );
                    }
                };
                float* dst = nullptr;
                if (is_1d) {
                    auto output = make_numpy_1d<float>(out_dim, dst);
                    run_deq(dst);
                    return output.cast();
                }
                auto output = make_numpy_2d<float>(count, out_dim, dst);
                run_deq(dst);
                return output.cast();
            },
            nb::arg("quantized"), nb::arg("threads") = 0, "Dequantizes transformed vectors back to approximate float32 vectors."
        )
        .def("serialize", [](const deglib::transformation::FeatureTransformation& t) -> nb::bytes {
            std::ostringstream ss(std::ios::binary);
            t.serialize(ss);
            std::string s = ss.str();
            return nb::bytes(s.data(), s.size());
        });

    nb::class_<deglib::transformation::MipsTransform, deglib::transformation::FeatureTransformation>(transformation_module, "MipsTransform")
        .def_prop_ro("max_norm", &deglib::transformation::MipsTransform::max_norm)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::MipsTransform>);

    nb::class_<deglib::transformation::EvpQuantizer, deglib::transformation::FeatureTransformation>(transformation_module, "EvpQuantizer")
        .def_prop_ro("dim", &deglib::transformation::EvpQuantizer::dim, "Dimension of vectors handled by this quantizer")
        .def_prop_ro("non_zeros", &deglib::transformation::EvpQuantizer::non_zeros, "Number of non-zero bits retained per vector")
        .def_prop_ro("is_fitted", &deglib::transformation::EvpQuantizer::is_fitted, "True if non_zeros is calibrated/set and mean is computed")
        .def_prop_ro(
            "mean",
            [](const deglib::transformation::EvpQuantizer& q) -> nb::object {
                if (!q.has_mean()) return nb::none();
                auto m = q.mean();
                float* ptr = nullptr;
                auto arr = make_numpy_1d<float>(m.size(), ptr);
                if (!m.empty()) {
                    std::memcpy(ptr, m.data(), m.size() * sizeof(float));
                }
                return arr.cast();
            },
            "Current 1D mean vector subtracted during quantization, or None if not set"
        )
        .def(
            "quantize",
            [](deglib::transformation::EvpQuantizer& q, const InArray& vectors, size_t threads) {
                auto [count, dim] = get_2d_shape(vectors);
                if (q.dim() != 0 && q.dim() != dim) {
                    throw std::invalid_argument(std::format("EvpQuantizer: vector dim ({}) does not match quantizer dim ({})", dim, q.dim()));
                }
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                if (!q.is_fitted()) {
                    throw std::logic_error("EvpQuantizer: quantize called before fit()");
                }

                const size_t evp_bytes = 2 * ((dim + 7) / 8);
                uint8_t* out_ptr = nullptr;
                auto output = make_numpy_2d<uint8_t>(count, evp_bytes, out_ptr);
                std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
                const void* src = vectors.data();

                {
                    nb::gil_scoped_release release;
                    if (is_fp16) {
                        q.transform_database(
                            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * evp_bytes), threads
                        );
                    } else {
                        q.transform_database(
                            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * evp_bytes), threads
                        );
                    }
                }
                return output;
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def_static("deserialize", &deserialize_transform<deglib::transformation::EvpQuantizer>);

    nb::class_<deglib::transformation::ScalarQuantizerInt8, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerInt8")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerInt8::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerInt8::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerInt8::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerInt8::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerInt8, int8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerInt8>);

    nb::class_<deglib::transformation::ScalarQuantizerUint8, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint8")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint8::dim)
        .def_prop_ro("min_val", &deglib::transformation::ScalarQuantizerUint8::min_val)
        .def_prop_ro("max_val", &deglib::transformation::ScalarQuantizerUint8::max_val)
        .def_prop_ro("dif", &deglib::transformation::ScalarQuantizerUint8::dif)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint8::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint8::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint8, uint8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint8>);

    nb::class_<deglib::transformation::ScalarQuantizerInt4, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerInt4")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerInt4::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerInt4::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerInt4::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerInt4::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerInt4, uint8_t, true>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerInt4>);

    nb::class_<deglib::transformation::ScalarQuantizerUint7, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint7")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint7::dim)
        .def_prop_ro("min", &deglib::transformation::ScalarQuantizerUint7::min)
        .def_prop_ro("max", &deglib::transformation::ScalarQuantizerUint7::max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint7::scale)
        .def_prop_ro("inv_scale", &deglib::transformation::ScalarQuantizerUint7::inv_scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint7::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint7, int8_t>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint7>);

    nb::class_<deglib::transformation::ScalarQuantizerUint4, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerUint4")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerUint4::dim)
        .def_prop_ro("min_val", &deglib::transformation::ScalarQuantizerUint4::min_val)
        .def_prop_ro("max_val", &deglib::transformation::ScalarQuantizerUint4::max_val)
        .def_prop_ro("dif", &deglib::transformation::ScalarQuantizerUint4::dif)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerUint4::scale)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerUint4::is_fitted)
        .def("quantize", &quantize_scalar_2d<deglib::transformation::ScalarQuantizerUint4, uint8_t, true>, nb::arg("vectors"), nb::arg("threads") = 0)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerUint4>);

    nb::class_<deglib::transformation::ScalarQuantizerFP16, deglib::transformation::FeatureTransformation>(transformation_module, "ScalarQuantizerFP16")
        .def_prop_ro("dim", &deglib::transformation::ScalarQuantizerFP16::dim)
        .def_prop_ro("abs_max", &deglib::transformation::ScalarQuantizerFP16::abs_max)
        .def_prop_ro("scale", &deglib::transformation::ScalarQuantizerFP16::scale)
        .def_prop_ro("inv_scale", &deglib::transformation::ScalarQuantizerFP16::inv_scale)
        .def_prop_ro("quantile", &deglib::transformation::ScalarQuantizerFP16::quantile)
        .def_prop_ro("is_fitted", &deglib::transformation::ScalarQuantizerFP16::is_fitted)
        .def(
            "quantize",
            [](const deglib::transformation::ScalarQuantizerFP16& q, const InArray& vectors, size_t threads) {
                auto [count, dim] = get_2d_shape(vectors);
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (!is_fp16 && !is_fp32) {
                    throw std::invalid_argument("vectors must be float32 or float16");
                }
                const size_t shape[2] = {count, static_cast<size_t>(dim)};
                uint16_t* out_ptr = nullptr;
                auto output = make_numpy_fp16(2, shape, out_ptr);
                std::byte* dst = reinterpret_cast<std::byte*>(out_ptr);
                const void* src = vectors.data();
                {
                    nb::gil_scoped_release release;
                    if (is_fp16) {
                        q.transform_database(
                            std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t))
                        );
                    } else {
                        q.transform_database(
                            std::span<const float>(static_cast<const float*>(src), count * dim), std::span<std::byte>(dst, count * dim * sizeof(uint16_t)),
                            threads
                        );
                    }
                }
                return output;
            },
            nb::arg("vectors"), nb::arg("threads") = 0
        )
        .def_static("deserialize", &deserialize_transform<deglib::transformation::ScalarQuantizerFP16>);

    nb::class_<deglib::transformation::L2NormalizeTransform, deglib::transformation::FeatureTransformation>(transformation_module, "L2NormalizeTransform")
        .def_prop_ro("dim", &deglib::transformation::L2NormalizeTransform::dim)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::L2NormalizeTransform>);

    nb::class_<deglib::transformation::FastHadamardRotation, deglib::transformation::FeatureTransformation>(transformation_module, "FastHadamardRotation")
        .def_prop_ro("rounds", &deglib::transformation::FastHadamardRotation::rounds)
        .def_prop_ro("dim", &deglib::transformation::FastHadamardRotation::dim)
        .def_prop_ro("seed", &deglib::transformation::FastHadamardRotation::seed)
        .def_prop_ro("is_enabled", &deglib::transformation::FastHadamardRotation::is_enabled)
        .def_prop_rw("enabled", &deglib::transformation::FastHadamardRotation::is_enabled, &deglib::transformation::FastHadamardRotation::set_enabled)
        .def_prop_ro("crest_factor", &deglib::transformation::FastHadamardRotation::crest_factor)
        .def_prop_ro("excess_kurtosis", &deglib::transformation::FastHadamardRotation::excess_kurtosis)
        .def_static("deserialize", &deserialize_transform<deglib::transformation::FastHadamardRotation>);

    transformation_module.def(
        "make_scalar_quantizer_int8", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerInt8>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint8", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint8>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_int4", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerInt4>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint7", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint7>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f
    );
    transformation_module.def(
        "make_scalar_quantizer_uint4", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerUint4>, nb::arg("vectors"),
        nb::arg("quantile") = -1.0f, "Factory function creating a ScalarQuantizerUint4 calibrated on vectors. If quantile < 0 (default), auto-optimizes."
    );
    transformation_module.def(
        "make_scalar_quantizer_fp16", &make_scalar_quantizer_impl<deglib::transformation::ScalarQuantizerFP16>, nb::arg("vectors"), nb::arg("quantile") = -1.0f
    );

    transformation_module.def(
        "make_quantizer_pipeline",
        [](const InArray& vectors, nb::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_quantizer_pipeline: vectors must be a non-empty 2D array with dim > 0");
            }
            const auto data_type = parse_quant_data_type(quant);
            const auto dt = vectors.dtype();
            const bool is_fp16 = is_dtype_fp16_or_u16(dt);
            const bool is_fp32 = is_dtype_float32(dt);
            if (!is_fp16 && !is_fp32) {
                std::vector<float> f_vec = convert_array_to_float32(vectors);
                nb::gil_scoped_release release;
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const float>(f_vec.data(), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank, threads, seed
                );
            }

            const void* src = vectors.data();
            nb::gil_scoped_release release;
            if (is_fp16) {
                return deglib::transformation::make_quantizer_pipeline(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank,
                    threads, seed
                );
            }
            return deglib::transformation::make_quantizer_pipeline(
                std::span<const float>(static_cast<const float*>(src), count * dim), static_cast<uint32_t>(dim), data_type, quantile, use_rerank, threads, seed
            );
        },
        nb::arg("vectors"), nb::arg("quant"), nb::arg("quantile") = -1.0f, nb::arg("use_rerank") = false, nb::arg("threads") = 0, nb::arg("seed") = 7,
        "Builds a transformation pipeline [FastHadamardRotation, ScalarQuantizer] or [ScalarQuantizer] based on kurtosis."
    );

    transformation_module.def(
        "make_quantizer_pipeline_from_graph",
        [](const deglib::DynamicExplorationGraph& graph, nb::object quant, float quantile, bool use_rerank, size_t threads, uint64_t seed, bool normalize, bool mips) {
            const auto target_dt = parse_quant_data_type(quant);
            nb::gil_scoped_release release;
            return deglib_py::make_quantizer_pipeline_from_graph(
                graph.internal(), target_dt, quantile, use_rerank, threads, seed, normalize, mips
            );
        },
        nb::arg("graph"), nb::arg("quant"), nb::arg("quantile") = -1.0f, nb::arg("use_rerank") = false, nb::arg("threads") = 0, nb::arg("seed") = 7, nb::arg("normalize") = false, nb::arg("mips") = false,
        "Calibrate a quantizer pipeline directly from an FP32/FP16 graph's stored vectors."
    );

    transformation_module.def(
        "make_evp_quantizer",
        [](const InArray& vectors, uint32_t non_zeros, size_t threads) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("vectors must be a non-empty 2D array with dim > 0");
            }
            const auto dt = vectors.dtype();
            const bool is_fp16 = is_dtype_fp16_or_u16(dt);
            const bool is_fp32 = is_dtype_float32(dt);
            if (!is_fp16 && !is_fp32) {
                throw std::invalid_argument("vectors must be float32 or float16");
            }
            const void* src = vectors.data();
            auto q = std::make_shared<deglib::transformation::EvpQuantizer>(dim, non_zeros);
            nb::gil_scoped_release release;
            if (is_fp16) {
                q->fit(std::span<const uint16_t>(static_cast<const uint16_t*>(src), count * dim), threads);
            } else {
                q->fit(std::span<const float>(static_cast<const float*>(src), count * dim), threads);
            }
            return q;
        },
        nb::arg("vectors"), nb::arg("non_zeros") = 0, nb::arg("threads") = 0, "Factory function creating an EvpQuantizer calibrated on vectors."
    );

    transformation_module.def(
        "make_mips_transform",
        [](const InArray& vectors, float max_norm) {
            auto [count, dim] = get_2d_shape(vectors);
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_mips_transform: vectors must be a non-empty 2D array with dim > 0");
            }
            auto t = std::make_shared<deglib::transformation::MipsTransform>(dim, max_norm);
            if (max_norm <= 0.0f) {
                const auto dt = vectors.dtype();
                const bool is_fp16 = is_dtype_fp16_or_u16(dt);
                const bool is_fp32 = is_dtype_float32(dt);
                if (is_fp32) {
                    const float* f_ptr = static_cast<const float*>(vectors.data());
                    nb::gil_scoped_release release;
                    t->fit(std::span<const float>(f_ptr, count * dim));
                } else if (is_fp16) {
                    const uint16_t* u_ptr = static_cast<const uint16_t*>(vectors.data());
                    std::vector<float> fp16_converted(count * dim);
                    nb::gil_scoped_release release;
                    deglib::distances::fp16::fp16_to_floats(u_ptr, fp16_converted.data(), count * dim);
                    t->fit(std::span<const float>(fp16_converted.data(), count * dim));
                } else {
                    std::vector<float> f_vec = convert_array_to_float32(vectors);
                    nb::gil_scoped_release release;
                    t->fit(std::span<const float>(f_vec.data(), count * dim));
                }
            }
            return t;
        },
        nb::arg("vectors"), nb::arg("max_norm") = 0.0f, "Factory function creating a MipsTransform (d -> d+1 spherical lift) fitted on vectors."
    );

    transformation_module.def(
        "make_l2_normalize_transform",
        [](uint32_t dim) {
            if (dim == 0) {
                throw std::invalid_argument("make_l2_normalize_transform: dim must be > 0");
            }
            return std::make_shared<deglib::transformation::L2NormalizeTransform>(dim);
        },
        nb::arg("dim"), "Factory function creating an L2NormalizeTransform."
    );

    transformation_module.def(
        "make_fast_hadamard_rotation",
        [](const InFloatArray& vectors, uint64_t seed, uint32_t rounds, bool auto_detect) {
            if (vectors.ndim() != 2) {
                throw std::invalid_argument(std::format("Expected 2D array, got {} dimensions", vectors.ndim()));
            }
            const size_t count = vectors.shape(0);
            const uint32_t dim = static_cast<uint32_t>(vectors.shape(1));
            if (count == 0 || dim == 0) {
                throw std::invalid_argument("make_fast_hadamard_rotation: vectors must be a non-empty 2D array with dim > 0");
            }
            const float* f_ptr = vectors.data();
            auto t = std::make_shared<deglib::transformation::FastHadamardRotation>(dim, seed, rounds);
            {
                nb::gil_scoped_release release;
                t->fit(std::span<const float>(f_ptr, count * dim), auto_detect);
            }
            return t;
        },
        nb::arg("vectors"), nb::arg("seed") = 7, nb::arg("rounds") = 0, nb::arg("auto_detect") = true,
        "Factory function creating a FastHadamardRotation fitted on vectors."
    );

    transformation_module.def(
        "apply_on_database",
        [apply_transformations_array](
            const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, const InArray& in_arr, size_t threads
        ) { return apply_transformations_array(transformations, in_arr, true, threads); },
        nb::arg("transformations"), nb::arg("vectors"), nb::arg("threads") = 0
    );

    transformation_module.def(
        "apply_on_query",
        [apply_transformations_array](
            const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>& transformations, const InArray& in_arr
        ) { return apply_transformations_array(transformations, in_arr, false); },
        nb::arg("transformations"), nb::arg("queries")
    );

    m.attr("FeatureTransformation") = transformation_module.attr("FeatureTransformation");
    m.attr("make_l2_normalize_transform") = transformation_module.attr("make_l2_normalize_transform");
    m.attr("make_fast_hadamard_rotation") = transformation_module.attr("make_fast_hadamard_rotation");
    m.attr("make_mips_transform") = transformation_module.attr("make_mips_transform");
    m.attr("make_quantizer_pipeline") = transformation_module.attr("make_quantizer_pipeline");
    m.attr("make_evp_quantizer") = transformation_module.attr("make_evp_quantizer");
    m.attr("make_scalar_quantizer_int8") = transformation_module.attr("make_scalar_quantizer_int8");
    m.attr("make_scalar_quantizer_uint8") = transformation_module.attr("make_scalar_quantizer_uint8");
    m.attr("make_scalar_quantizer_uint4") = transformation_module.attr("make_scalar_quantizer_uint4");
    m.attr("make_scalar_quantizer_int4") = transformation_module.attr("make_scalar_quantizer_int4");
    m.attr("make_scalar_quantizer_uint7") = transformation_module.attr("make_scalar_quantizer_uint7");
    m.attr("make_scalar_quantizer_fp16") = transformation_module.attr("make_scalar_quantizer_fp16");

    // search submodule

}
}  // namespace deglib_py
