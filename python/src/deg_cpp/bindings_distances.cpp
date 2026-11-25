#include "bindings_common.h"

// ============================================================================
// Distance Computations
// ============================================================================

float float_space_compute_distance(const deglib::distances::FloatSpace& space, const InArray& vec1, const InArray& vec2) {
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();
    const size_t vec1_bytes = vec1.size() * vec1.itemsize();
    const size_t vec2_bytes = vec2.size() * vec2.itemsize();
    if (vec1_bytes != query_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec1 size in bytes ({}) does not match required "
                "query size ({})",
                vec1_bytes, query_bytes
            )
        );
    }
    if (vec2_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Vector vec2 size in bytes ({}) does not match required "
                "feature space data size ({})",
                vec2_bytes, req_bytes
            )
        );
    }
    return deglib::distances::compute_distance(space, vec1.data(), vec2.data());
}

nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>
float_space_compute_distances(const deglib::distances::FloatSpace& space, const InArray& query, const InArray& targets, size_t threads) {
    const size_t req_bytes = space.get_data_size();
    const size_t query_bytes = space.get_query_data_size();

    size_t num_queries = 0;
    size_t query_stride_bytes = 0;
    if (query.ndim() == 1) {
        num_queries = 1;
        query_stride_bytes = query.size() * query.itemsize();
    } else if (query.ndim() == 2) {
        num_queries = query.shape(0);
        query_stride_bytes = query.shape(1) * query.itemsize();
    } else {
        throw std::invalid_argument("Query array must be 1D or 2D NumPy array");
    }

    if (query_stride_bytes != query_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match required "
                "query size ({})",
                query_stride_bytes, query_bytes
            )
        );
    }

    size_t num_targets = 0;
    size_t target_stride_bytes = 0;
    if (targets.ndim() == 1) {
        num_targets = 1;
        target_stride_bytes = targets.size() * targets.itemsize();
    } else if (targets.ndim() == 2) {
        num_targets = targets.shape(0);
        target_stride_bytes = targets.shape(1) * targets.itemsize();
    } else {
        throw std::invalid_argument("Targets array must be 1D or 2D NumPy array");
    }

    if (target_stride_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Target vector size in bytes ({}) does not match required "
                "feature space data size ({})",
                target_stride_bytes, req_bytes
            )
        );
    }

    float* out = nullptr;
    auto result = make_numpy_2d<float>(num_queries, num_targets, out);
    const size_t query_stride = query_stride_bytes;
    const void* qptr = query.data();
    const void* tptr = targets.data();

    if (num_queries == 1) {
        nb::gil_scoped_release release;
        deglib::distances::compute_distances(space, qptr, 1, tptr, num_targets, out);
    } else {
        nb::gil_scoped_release release;
        deglib::concurrent::parallel_for(0, num_queries, threads, [&](size_t q, size_t) {
            deglib::distances::compute_distances(space, static_cast<const uint8_t*>(qptr) + q * query_stride, 1, tptr, num_targets, out + q * num_targets);
        });
    }

    return result;
}


// ============================================================================
// FP16 Conversion Bindings
// ============================================================================

nb::ndarray<nb::numpy, uint16_t, nb::ndim<1>, nb::c_contig> floats_to_fp16_wrapper(const InFloatArray& floats) {
    if (floats.itemsize() != sizeof(float)) {
        throw std::invalid_argument("Input array must contain 32-bit floats (itemsize == 4)");
    }
    const size_t count = floats.size();
    uint16_t* out_ptr = nullptr;
    auto output = make_numpy_1d<uint16_t>(count, out_ptr);
    deglib::distances::fp16::floats_to_fp16(floats.data(), out_ptr, count);
    return output;
}

nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig> fp16_to_floats_wrapper(const InUint16Array& fp16_vals) {
    if (fp16_vals.itemsize() != sizeof(uint16_t)) {
        throw std::invalid_argument("Input array must contain 16-bit uint16/FP16 values (itemsize == 2)");
    }
    const size_t count = fp16_vals.size();
    float* out_ptr = nullptr;
    auto output = make_numpy_1d<float>(count, out_ptr);
    deglib::distances::fp16::fp16_to_floats(fp16_vals.data(), out_ptr, count);
    return output;
}


namespace deglib_py {
void bind_cpu_and_distances(nb::module_& m) {
    // cpu submodule
    nb::module_ cpu_module = m.def_submodule("cpu", "CPU feature detection and InstructionSet");
    nb::enum_<deglib::cpu::InstructionSet>(cpu_module, "InstructionSet", nb::is_arithmetic())
        .value("Auto", deglib::cpu::InstructionSet::Auto)
        .value("Scalar", deglib::cpu::InstructionSet::Scalar)
        .value("AVX2", deglib::cpu::InstructionSet::AVX2)
        .value("AVX2_VNNI", deglib::cpu::InstructionSet::AVX2_VNNI)
        .value("AVX512", deglib::cpu::InstructionSet::AVX512)
        .value("AVX512_VNNI", deglib::cpu::InstructionSet::AVX512_VNNI);

    cpu_module.def("has_avx2", &deglib::cpu::has_avx2, "Returns whether AVX2 instructions are available");
    cpu_module.def("has_avx2_vnni", &deglib::cpu::has_avx2_vnni, "Returns whether AVX2-VNNI (256-bit) instructions are available");
    cpu_module.def("has_avx512", &deglib::cpu::has_avx512, "Returns whether AVX512 instructions are available");
    cpu_module.def("has_avx512_vnni", &deglib::cpu::has_avx512_vnni, "Returns whether AVX-512 VNNI instructions are available");
    cpu_module.def("has_avx512_fp16", &deglib::cpu::has_avx512_fp16, "Returns whether AVX-512 FP16 instructions are available");
    cpu_module.def("has_fp16", &deglib::cpu::has_fp16, "Returns whether FP16 instructions are available");

    // distances submodule
    nb::module_ distances_module = m.def_submodule("distances", "Distance metrics and feature spaces");
    nb::enum_<deglib::distances::MetricType>(distances_module, "Metric", nb::is_arithmetic())
        .value("FP32_L2", deglib::distances::MetricType::FP32_L2)
        .value("FP32_InnerProduct", deglib::distances::MetricType::FP32_InnerProduct)
        .value("Uint8_L2", deglib::distances::MetricType::Uint8_L2)
        .value("Uint8_InnerProduct", deglib::distances::MetricType::Uint8_InnerProduct)
        .value("FP16_L2", deglib::distances::MetricType::FP16_L2)
        .value("FP16_InnerProduct", deglib::distances::MetricType::FP16_InnerProduct)
        .value("EVP_InnerProduct", deglib::distances::MetricType::EVP_InnerProduct)
        .value("Int8_InnerProduct", deglib::distances::MetricType::Int8_InnerProduct)
        .value("Int8_L2", deglib::distances::MetricType::Int8_L2)
        .value("Uint4_InnerProduct", deglib::distances::MetricType::Uint4_InnerProduct)
        .value("Uint4_L2", deglib::distances::MetricType::Uint4_L2)
        .value("Int4_InnerProduct", deglib::distances::MetricType::Int4_InnerProduct)
        .value("Int4_L2", deglib::distances::MetricType::Int4_L2)
        .value("Uint7_InnerProduct", deglib::distances::MetricType::Uint7_InnerProduct)
        .value("Uint7_L2", deglib::distances::MetricType::Uint7_L2);

    nb::class_<deglib::distances::FloatSpace>(distances_module, "FloatSpace")
        .def(
            "__init__",
            [](deglib::distances::FloatSpace* self, const size_t dim, const deglib::distances::MetricType metric,
               const deglib::cpu::InstructionSet instruction) {
                new (self) deglib::distances::FloatSpace(dim, deglib::distances::Metric(metric), instruction);
            },
            nb::arg("dim"), nb::arg("metric") = deglib::distances::MetricType::FP32_L2, nb::arg("instruction") = deglib::cpu::InstructionSet::Auto
        )
        .def("dim", &deglib::distances::FloatSpace::dim)
        .def("metric", [](const deglib::distances::FloatSpace& fs) { return fs.metric().value; })
        .def("get_data_size", &deglib::distances::FloatSpace::get_data_size)
        .def("get_query_data_size", &deglib::distances::FloatSpace::get_query_data_size)
        .def("get_instruction", &deglib::distances::FloatSpace::get_instruction)
        .def("compute_distance", &float_space_compute_distance, nb::arg("vec1"), nb::arg("vec2"))
        .def("compute_distances", &float_space_compute_distances, nb::arg("query"), nb::arg("targets"), nb::arg("threads") = 0);

    // quantization functions in distances submodule
    distances_module.def("floats_to_fp16", &floats_to_fp16_wrapper, "Convert float32 array to FP16 (uint16_t)");
    distances_module.def("fp16_to_floats", &fp16_to_floats_wrapper, "Convert FP16 (uint16_t) array to float32");

    // optimization submodule

}
}  // namespace deglib_py
