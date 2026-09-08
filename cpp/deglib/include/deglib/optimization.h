#pragma once

// Fast Linear Assignment Sorter (FLAS) algorithms & solvers
#include "deglib/optimization/flas/fast_linear_assignment_sorter.h"
#include "deglib/optimization/flas/fast_linear_assignment_sorter_mt.h"
#include "deglib/optimization/flas/junker_volgenant_solver.h"

// Vector Quantization techniques
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/optimization/quantization/scalar_quantize.h"

// Graph pruning techniques
#include "deglib/optimization/pruning.h"

// Vector transformation techniques (MIPS L2 transformation)
#include "deglib/optimization/transform.h"

// Builder (needed for optimize_edges)
#include "deglib/builder.h"

namespace deglib::optimization {

/**
 * @brief Perform 1D pre-sorting of dataset feature vectors using FLAS.
 * Supports all deglib FP32 metric types.
 *
 * @param data Pointer to contiguous FP32 feature vectors array (shape: count x space.dim()).
 * @param count Number of feature vectors.
 * @param space FloatSpace defining dimension and distance metric.
 * @param radius_decay Decay factor per iteration for neighborhood radius (default 0.9).
 * @param numThreads Number of worker threads (0 = use hardware concurrency).
 * @param callback Optional progress callback returning true to cancel early.
 * @return std::vector<uint32_t> Permutation array of original vector indices [0..count-1] in sorted order.
 */
inline std::vector<uint32_t> presort(
    const float* data,
    size_t count,
    const deglib::distances::FloatSpace& space,
    float radius_decay = 0.9f,
    size_t numThreads = 0,
    std::function<bool(float)> callback = nullptr
) {
    if (count == 0) return {};

    const int dim = static_cast<int>(space.dim());
    std::vector<flas::MapField> map_fields = flas::make_map_fields(data, static_cast<int>(count), dim);
    flas::FlasSettings settings;
    settings.radius_decay = radius_decay;

    std::mt19937 rng(42);
    auto cb = callback ? callback : [](float) { return false; };

    if (numThreads != 1) {
        flas::do_sorting_1d(map_fields, space, settings, rng, cb, static_cast<int>(numThreads));
    } else {
        flas::do_sorting_1d(map_fields, space, settings, rng, cb);
    }

    std::vector<uint32_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = static_cast<uint32_t>(map_fields[i].id);
    }
    return result;
}

// ========================================================================
// EVP Quantizer Factory Method
// ========================================================================

/**
 * Make an EvpQuantizer holding the shared non_zeros setting.
 */
inline deglib::quantization::evp::EvpQuantizer make_evp_quantizer(uint32_t non_zeros) {
    return deglib::quantization::evp::EvpQuantizer(non_zeros);
}

// ========================================================================
// Scalar Quantization Factory Methods (make and fit Quantizer instance)
// ========================================================================

/**
 * Make and fit a ScalarQuantizerInt8 from FP32 dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerInt8 make_scalar_quantizer_int8(
    const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerInt8 quantizer;
    quantizer.fit(data, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerInt8 from FP16 (uint16_t) dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerInt8 make_scalar_quantizer_int8(
    const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerInt8 quantizer;
    quantizer.fit(data_fp16, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerInt8PerDim from FP32 dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerInt8PerDim make_scalar_quantizer_int8_perdim(
    const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerInt8PerDim quantizer;
    quantizer.fit(data, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerInt8PerDim from FP16 (uint16_t) dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerInt8PerDim make_scalar_quantizer_int8_perdim(
    const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerInt8PerDim quantizer;
    quantizer.fit(data_fp16, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerUint8 from FP32 dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerUint8 make_scalar_quantizer_uint8(
    const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerUint8 quantizer;
    quantizer.fit(data, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerUint8 from FP16 (uint16_t) dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerUint8 make_scalar_quantizer_uint8(
    const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerUint8 quantizer;
    quantizer.fit(data_fp16, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerUint8PerDim from FP32 dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerUint8PerDim make_scalar_quantizer_uint8_perdim(
    const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerUint8PerDim quantizer;
    quantizer.fit(data, count, dim, drop_ratio);
    return quantizer;
}

/**
 * Make and fit a ScalarQuantizerUint8PerDim from FP16 (uint16_t) dataset.
 */
inline deglib::quantization::scalar::ScalarQuantizerUint8PerDim make_scalar_quantizer_uint8_perdim(
    const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f
) {
    deglib::quantization::scalar::ScalarQuantizerUint8PerDim quantizer;
    quantizer.fit(data_fp16, count, dim, drop_ratio);
    return quantizer;
}

// ========================================================================
// Graph Pruning API
// ========================================================================

/**
 * @brief Prune the worst (highest-weight) neighbors of each vertex.
 *
 * Replaces the `prune_worst` highest-weight neighbors of each vertex with
 * self-loops. Multi-threaded.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param prune_worst Number of worst neighbors to replace with self-loops per vertex.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 */
inline void prune_worst_edges(deglib::graph::MutableGraph& graph, const uint8_t prune_worst, const size_t numThreads = 0) {
    deglib::optimization::pruning::prune_worst_edges(graph, prune_worst, numThreads);
}

/**
 * @brief Remove all edges that do not satisfy the RNG condition.
 *
 * Parallelized across hardware threads. For each vertex, checks each neighbor
 * using the RNG condition and removes non-conforming edges.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_rng_edges(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    return deglib::optimization::pruning::prune_non_rng_edges(graph, numThreads);
}

/**
 * @brief Remove non-RNG edges using a weight-sorted global strategy.
 *
 * Collects all non-RNG edges, sorts them by weight (ascending), then removes
 * them in that order. Collection is multi-threaded; removal is single-threaded.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use for collection (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_rng_edges_weight_sorted(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    return deglib::optimization::pruning::prune_non_rng_edges_weight_sorted(graph, numThreads);
}

/**
 * @brief Remove non-RNG edges using an iterative per-vertex strategy.
 *
 * For each vertex, iteratively removes non-RNG edges in a do-while loop until
 * no more edges can be removed. This accounts for cascading effects where
 * removing one edge may make another edge RNG-conform. Multi-threaded per vertex.
 *
 * @param graph Reference to the MutableGraph to be processed.
 * @param numThreads Number of threads to use (0 = use hardware concurrency).
 * @return Number of edges removed.
 */
inline uint32_t prune_non_rng_edges_iterative(deglib::graph::MutableGraph& graph, const size_t numThreads = 0) {
    return deglib::optimization::pruning::prune_non_rng_edges_iterative(graph, numThreads);
}

// ========================================================================
// Edge Optimization API
// ========================================================================

/**
 * @brief Optimizes the edges of the graph using the builder's improvement routines.
 *
 * This function creates a builder and repeatedly attempts to improve the graph's edges for a given number of iterations.
 * It reports progress and statistics during the optimization process.
 *
 * @param graph Reference to the MutableGraph to be optimized.
 * @param k_opt Number of neighbors to consider during optimization.
 * @param eps_opt Epsilon value for neighbor search during optimization.
 * @param i_opt Number of improvement attempts per build step.
 * @param iterations Number of optimization iterations to perform.
 */
inline void optimize_edges(deglib::graph::MutableGraph& graph, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t iterations) {
    auto rnd = std::mt19937(7);

    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, deglib::builder::StreamingData, 0, 0.0f, k_opt, eps_opt, i_opt, 1);

    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if (status.step % (iterations / 10) == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 100);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);

            auto duration = duration_ms / 1000;
            std::cout << std::setw(7) << status.step << " step, " << std::setw(5) << duration << "s, AEW: " << std::fixed << std::setprecision(2)
                      << std::setw(4) << avg_edge_weight << ", " << (connected ? "" : "not") << " connected, " << (valid_weights ? "valid" : "invalid") << "\n";
            start = std::chrono::steady_clock::now();
        }

        if (status.step > iterations) builder.stop();
    };

    builder.build(improvement_callback, true);
}

}  // namespace deglib::optimization
