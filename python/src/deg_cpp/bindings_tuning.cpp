#include "bindings_common.h"


namespace deglib_py {
void bind_tuning(nb::module_& m) {
    // tuning submodule
    nb::module_ tuning_module = m.def_submodule("tuning", "Pre-build hyperparameter prediction and dataset profiling");

    nb::class_<deglib::tuning::DatasetProfile>(tuning_module, "DatasetProfile", "Base-only geometric profile of a vector dataset.")
        .def(
            "__init__",
            [](deglib::tuning::DatasetProfile* self, uint32_t sample_count, uint32_t dim, float participation_ratio, float rel_contrast_1, float pp_iqr,
               float d16_skew, float ball_cnt_p90_p10_r15, float hadamard_kurt, deglib::distances::MetricType metric) {
                new (self) deglib::tuning::DatasetProfile{sample_count,        dim,          participation_ratio, rel_contrast_1, pp_iqr,
                                                          d16_skew, ball_cnt_p90_p10_r15, hadamard_kurt,      deglib::distances::Metric(metric)};
            },
            nb::arg("sample_count") = 0, nb::arg("dim") = 0, nb::arg("participation_ratio") = 0.0f, nb::arg("rel_contrast_1") = 0.0f, nb::arg("pp_iqr") = 0.0f,
            nb::arg("d16_skew") = 0.0f, nb::arg("ball_cnt_p90_p10_r15") = 0.0f, nb::arg("hadamard_kurt") = 0.0f,
            nb::arg("metric") = deglib::distances::MetricType::FP32_L2
        )
        .def_ro("sample_count", &deglib::tuning::DatasetProfile::sample_count)
        .def_ro("dim", &deglib::tuning::DatasetProfile::dim)
        .def_ro("participation_ratio", &deglib::tuning::DatasetProfile::participation_ratio)
        .def_ro("rel_contrast_1", &deglib::tuning::DatasetProfile::rel_contrast_1)
        .def_ro("pp_iqr", &deglib::tuning::DatasetProfile::pp_iqr)
        .def_ro("d16_skew", &deglib::tuning::DatasetProfile::d16_skew)
        .def_ro("ball_cnt_p90_p10_r15", &deglib::tuning::DatasetProfile::ball_cnt_p90_p10_r15)
        .def_ro("hadamard_kurt", &deglib::tuning::DatasetProfile::hadamard_kurt)
        .def_prop_ro("metric", [](const deglib::tuning::DatasetProfile& p) -> deglib::distances::MetricType { return p.metric.value; })
        .def("__repr__", [](const deglib::tuning::DatasetProfile& p) {
            return std::format(
                "DatasetProfile(sample_count={}, dim={}, participation_ratio={:.6f}, rel_contrast_1={:.6f}, pp_iqr={:.6f}, d16_skew={:.6f}, "
                "ball_cnt_p90_p10_r15={:.6f}, hadamard_kurt={:.6f}, metric={})",
                p.sample_count, p.dim, p.participation_ratio, p.rel_contrast_1, p.pp_iqr, p.d16_skew, p.ball_cnt_p90_p10_r15, p.hadamard_kurt,
                p.metric.to_string()
            );
        });

    nb::class_<deglib::tuning::TargetPrediction>(tuning_module, "TargetPrediction", "Predicted graph construction target and expected QPS advantage.")
        .def_ro("target", &deglib::tuning::TargetPrediction::target)
        .def_ro("lowlid_advantage_pct", &deglib::tuning::TargetPrediction::lowlid_advantage_pct)
        .def_ro("profile", &deglib::tuning::TargetPrediction::profile)
        .def("__repr__", [](const deglib::tuning::TargetPrediction& r) {
            const char* name = (r.target == deglib::builder::OptimizationTarget::LowLID)
                                   ? "LowLID"
                                   : ((r.target == deglib::builder::OptimizationTarget::HighLID) ? "HighLID" : "StreamingData");
            return std::format("TargetPrediction(target={}, lowlid_advantage_pct={:+.2f}%)", name, r.lowlid_advantage_pct);
        });

    nb::class_<deglib::tuning::QuantizationPrediction>(
        tuning_module, "QuantizationPrediction", "Predicted quantizer storage metric (Uint7_L2 vs Int8_InnerProduct) and expected QPS advantage."
    )
        .def_prop_ro("metric", [](const deglib::tuning::QuantizationPrediction& r) -> deglib::distances::MetricType { return r.metric.value; })
        .def_ro("uint7_advantage_pct", &deglib::tuning::QuantizationPrediction::uint7_advantage_pct)
        .def_ro("profile", &deglib::tuning::QuantizationPrediction::profile)
        .def("__repr__", [](const deglib::tuning::QuantizationPrediction& r) {
            return std::format("QuantizationPrediction(metric={}, uint7_advantage_pct={:+.2f}%)", r.metric.to_string(), r.uint7_advantage_pct);
        });

    auto measure_profile_py = [](const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed,
                                 size_t threads) -> deglib::tuning::DatasetProfile {
        auto [count, dim] = get_2d_shape(data);
        const auto dt = data.dtype();
        const bool is_fp32 = is_dtype_float32(dt);
        const bool is_fp16 = is_dtype_float16_only(dt);
        if (!is_fp32 && !is_fp16) {
            throw std::invalid_argument("data must be a 2D float32 or float16 array");
        }
        const void* ptr = data.data();
        const deglib::distances::Metric resolved_metric(metric);
        nb::gil_scoped_release release;
        if (is_fp16) {
            return deglib::tuning::measure_dataset_profile(
                std::span<const uint16_t>(static_cast<const uint16_t*>(ptr), count * static_cast<size_t>(dim)), dim, resolved_metric, sample_count, query_count,
                seed, threads
            );
        }
        return deglib::tuning::measure_dataset_profile(
            std::span<const float>(static_cast<const float*>(ptr), count * static_cast<size_t>(dim)), dim, resolved_metric, sample_count, query_count, seed,
            threads
        );
    };

    tuning_module.def(
        "measure_dataset_profile", measure_profile_py, nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2,
        nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount, nb::arg("query_count") = deglib::tuning::kDefaultQueryCount,
        nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure the base-only geometric DatasetProfile from a 2D float32 or float16 array."
    );
    tuning_module.def(
        "predict_optimization_target", [](const deglib::tuning::DatasetProfile& profile) { return deglib::tuning::predict_optimization_target(profile); },
        nb::arg("profile"), "Predict OptimizationTarget from a precomputed DatasetProfile."
    );
    tuning_module.def(
        "predict_optimization_target",
        [measure_profile_py](
            const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed, size_t threads
        ) {
            const auto profile = measure_profile_py(data, metric, sample_count, query_count, seed, threads);
            return deglib::tuning::predict_optimization_target(profile);
        },
        nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount,
        nb::arg("query_count") = deglib::tuning::kDefaultQueryCount, nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure a DatasetProfile and predict OptimizationTarget from a 2D float32 or float16 array."
    );
    tuning_module.def(
        "predict_quantization", [](const deglib::tuning::DatasetProfile& profile) { return deglib::tuning::predict_quantization(profile); }, nb::arg("profile"),
        "Predict quantizer Metric (Uint7_L2 vs Int8_InnerProduct) from a precomputed DatasetProfile."
    );
    tuning_module.def(
        "predict_quantization",
        [measure_profile_py](
            const InArray& data, deglib::distances::MetricType metric, size_t sample_count, size_t query_count, uint32_t seed, size_t threads
        ) {
            const auto profile = measure_profile_py(data, metric, sample_count, query_count, seed, threads);
            return deglib::tuning::predict_quantization(profile);
        },
        nb::arg("data"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount,
        nb::arg("query_count") = deglib::tuning::kDefaultQueryCount, nb::arg("seed") = deglib::tuning::kDefaultSeed, nb::arg("threads") = 0,
        "Measure a DatasetProfile and predict quantizer Metric (Uint7_L2 vs Int8_InnerProduct) from a 2D float32 or float16 array."
    );
    tuning_module.def(
        "predict_quantization",
        [](const deglib::DynamicExplorationGraph& graph, size_t sample_count, uint32_t seed, size_t threads) {
            const auto src_metric = graph.getFeatureSpace().metric();
            if (src_metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct) {
                deglib::tuning::DatasetProfile p{};
                p.dim = static_cast<uint32_t>(graph.getFeatureSpace().dim());
                p.metric = src_metric;
                return deglib::tuning::QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, p};
            }
            const size_t n = graph.size();
            const uint32_t dim = static_cast<uint32_t>(graph.getFeatureSpace().dim());
            const auto dt = src_metric.get_data_type();
            if (n < 2 || dim == 0 || (dt != deglib::distances::MetricDataType::FP32 && dt != deglib::distances::MetricDataType::FP16)) {
                deglib::tuning::DatasetProfile p{};
                p.dim = dim;
                p.metric = src_metric;
                return deglib::tuning::QuantizationPrediction{deglib::distances::Metric::Int8_InnerProduct, 0.0f, p};
            }

            bool has_l2n = false;
            for (const auto& t : graph.getTransformations()) {
                if (t && t->name() == "L2NormalizeTransform") {
                    has_l2n = true;
                    break;
                }
            }

            nb::gil_scoped_release release;
            const auto& internal = graph.internal();
            const auto indices = deglib::tuning::detail::sample_distinct_indices(n, sample_count, seed);
            const size_t actual_n = indices.size();
            std::vector<float> sample_buf(actual_n * static_cast<size_t>(dim));
            const bool is_fp16 = (dt == deglib::distances::MetricDataType::FP16);

            const size_t norm_check_n = has_l2n ? 0 : std::min<size_t>(actual_n, 1024);
            bool unit_ok = true;
            for (size_t i = 0; i < actual_n; ++i) {
                const std::byte* src = internal.getFeatureVector(static_cast<uint32_t>(indices[i]));
                float* dst = sample_buf.data() + i * static_cast<size_t>(dim);
                if (is_fp16) {
                    deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), dst, dim);
                } else {
                    std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
                }
                if (i < norm_check_n && unit_ok) {
                    float norm_sq = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        norm_sq += dst[d] * dst[d];
                    }
                    if (!std::isfinite(norm_sq) || std::abs(norm_sq - 1.0f) > 0.02f) {
                        unit_ok = false;
                    }
                }
            }

            if (!(has_l2n || unit_ok)) {
                deglib::tuning::DatasetProfile p{};
                p.sample_count = static_cast<uint32_t>(actual_n);
                p.dim = dim;
                p.metric = src_metric;
                return deglib::tuning::QuantizationPrediction{deglib::distances::Metric::Int8_InnerProduct, 0.0f, p};
            }

            return deglib::tuning::predict_quantization(
                std::span<const float>(sample_buf), dim, src_metric, 0, 0, seed, threads
            );
        },
        nb::arg("graph"), nb::arg("sample_count") = deglib::tuning::kDefaultSampleCount, nb::arg("seed") = deglib::tuning::kDefaultSeed,
        nb::arg("threads") = 0,
        "Predict quantizer Metric (Uint7_L2 vs Int8_InnerProduct) directly from a float32/float16 DynamicExplorationGraph."
    );

}
}  // namespace deglib_py
