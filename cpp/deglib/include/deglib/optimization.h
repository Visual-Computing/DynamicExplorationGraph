#pragma once

// Fast Linear Assignment Sorter (FLAS) algorithms & solvers
#include "deglib/optimization/flas/junker_volgenant_solver.h"
#include "deglib/optimization/flas/fast_linear_assignment_sorter.h"
#include "deglib/optimization/flas/fast_linear_assignment_sorter_mt.h"

// Vector Quantization techniques
#include "deglib/optimization/quantization/evp_quantize.h"

namespace deglib::optimization {

    /**
     * Quantize a single FP32 vector using EVP quantization.
     */
    inline std::vector<std::byte> quantize_evp_single(const float* embedding, uint32_t dim, uint32_t non_zeros) {
        return deglib::quantization::evp::quantize_single(embedding, dim, non_zeros);
    }

    /**
     * Quantize a single FP16 (uint16_t) vector using EVP quantization.
     */
    inline std::vector<std::byte> quantize_evp_single(const uint16_t* embedding, uint32_t dim, uint32_t non_zeros) {
        return deglib::quantization::evp::quantize_single(embedding, dim, non_zeros);
    }

    /**
     * Quantize a batch of FP32 vectors using EVP quantization.
     */
    inline std::vector<std::byte> quantize_evp_batch(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
        return deglib::quantization::evp::quantize_batch(data, count, dim, non_zeros, numThreads);
    }

    /**
     * Quantize a batch of FP16 (uint16_t) vectors using EVP quantization.
     */
    inline std::vector<std::byte> quantize_evp_batch(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
        return deglib::quantization::evp::quantize_batch(data, count, dim, non_zeros, numThreads);
    }

} // namespace deglib::optimization

