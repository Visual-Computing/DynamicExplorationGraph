#pragma once

#include <cstddef>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/concurrent.h"
#include "deglib/graph/internal_graph.h"

namespace deglib::search {

using ObjectDistance = deglib::graph::ObjectDistance;
using ResultSet = deglib::graph::ResultSet;
using UncheckedSet = deglib::graph::UncheckedSet;

/**
 * Rerank candidate neighbor indices for queries using exact FloatSpace distances.
 *
 * @param space                FloatSpace distance calculator instance
 * @param queries              Pointer to [num_queries x dim] query vectors
 * @param num_queries          Number of query vectors
 * @param base_vectors         Pointer to [num_base_vectors x dim] target/base vectors (if null, queries are used as targets)
 * @param num_base_vectors     Number of base vectors
 * @param base_candidates      Pointer to [num_queries x candidates_per_query] candidate indices
 * @param candidates_per_query Number of candidate indices provided per query
 * @param k_top                Number of top candidates to output per query (0 = all)
 * @param num_threads          Number of worker threads (0 = auto-detect)
 * @param out_result_indices   Output pointer to [num_queries x k_top] uint32_t indices
 */
inline void rerank(
    const deglib::FloatSpace& space,
    const void* queries,
    size_t num_queries,
    const void* base_vectors,
    size_t num_base_vectors,
    const uint32_t* base_candidates,
    size_t candidates_per_query,
    size_t k_top,
    size_t num_threads,
    uint32_t* out_result_indices
) {
    if (queries == nullptr || base_candidates == nullptr || out_result_indices == nullptr) {
        throw std::invalid_argument("rerank: queries, base_candidates, and out_result_indices must not be null");
    }

    if (k_top == 0 || k_top > candidates_per_query) {
        k_top = candidates_per_query;
    }

    const void* target_vectors = (base_vectors != nullptr) ? base_vectors : queries;
    const size_t target_count = (base_vectors != nullptr) ? num_base_vectors : num_queries;

    const size_t byte_stride_query = space.get_data_size();
    const size_t byte_stride_target = space.get_data_size();

    const uint8_t* q_ptr = static_cast<const uint8_t*>(queries);
    const uint8_t* t_ptr = static_cast<const uint8_t*>(target_vectors);

    const auto param = space.get_dist_func_param();
    // Resolve variant type ONCE outside loops via compile-time static dispatch
    space.compute([&](const auto& dist_func_obj) {
        using DistType = std::decay_t<decltype(dist_func_obj)>;
        deglib::concurrent::parallel_for(0, num_queries, num_threads, [&](size_t i, size_t) {
            const uint8_t* query_ptr = q_ptr + i * byte_stride_query;
            const uint32_t* cand_row = base_candidates + i * candidates_per_query;

            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(k_top + 1);

            for (size_t j = 0; j < candidates_per_query; ++j) {
                uint32_t cand_idx = cand_row[j];
                if (cand_idx >= target_count) {
                    continue;
                }

                const uint8_t* cand_ptr = t_ptr + cand_idx * byte_stride_target;
                float dist = DistType::compare(query_ptr, cand_ptr, param);

                if (heap.size() < k_top) {
                    heap.push_back({dist, cand_idx});
                    std::push_heap(heap.begin(), heap.end());
                } else if (dist < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {dist, cand_idx};
                    std::push_heap(heap.begin(), heap.end());
                }
            }

            std::sort_heap(heap.begin(), heap.end());

            uint32_t* out_row = out_result_indices + i * k_top;
            size_t actual_k = heap.size();
            for (size_t k = 0; k < actual_k; ++k) {
                out_row[k] = heap[k].second;
            }
            for (size_t k = actual_k; k < k_top; ++k) {
                out_row[k] = static_cast<uint32_t>(i);
            }
        });
    });
}

} // namespace deglib::search

