#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "deglib/distances.h"
#include "deglib/filter.h"
#include "deglib/concurrent.h"
#include "deglib/graph/internal_graph.h"

namespace deglib::search {

using ObjectDistance = deglib::graph::ObjectDistance;
using ResultSet = deglib::graph::ResultSet;

/**
 * Rerank candidate neighbor indices for queries using exact FloatSpace distances.
 *
 * Returns a vector of ResultSet objects (one per query). Each ResultSet contains the top-k nearest
 * candidates maintained in a max-heap property. Note that ResultSet elements are in heap order (unsorted);
 * use top() and pop(), or call sort() on the ResultSet if sorted order is required.
 *
 * @param space                FloatSpace distance calculator instance
 * @param queries              Pointer to [num_queries x dim] query vectors
 * @param num_queries          Number of query vectors
 * @param base_vectors         Pointer to [num_base_vectors x dim] target/base vectors (if null, queries are used as targets)
 * @param num_base_vectors     Number of base vectors
 * @param base_candidates      Pointer to [num_queries x candidates_per_query] candidate indices
 * @param candidates_per_query Number of candidate indices provided per query
 * @param k_top                Number of top candidates to keep per query (0 = all)
 * @param num_threads          Number of worker threads (0 = auto-detect)
 * @return std::vector<ResultSet> containing the top-k result sets per query (unsorted heap order)
 */
inline std::vector<ResultSet> rerank(
    const deglib::distances::FloatSpace& space,
    const void* queries,
    size_t num_queries,
    const void* base_vectors,
    size_t num_base_vectors,
    const uint32_t* base_candidates,
    size_t candidates_per_query,
    size_t k_top = 0,
    size_t num_threads = 0
) {
    if (queries == nullptr || base_candidates == nullptr) {
        throw std::invalid_argument("rerank: queries and base_candidates must not be null");
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

    std::vector<ResultSet> results(num_queries);

    space.compute([&](const auto& dist_func_obj) {
        deglib::concurrent::parallel_for(0, num_queries, num_threads, [&](size_t i, size_t) {
            const uint8_t* query_ptr = q_ptr + i * byte_stride_query;
            const uint32_t* cand_row = base_candidates + i * candidates_per_query;

            ResultSet heap;
            heap.reserve(k_top);
            float max_dist = std::numeric_limits<float>::max();

            for (size_t j = 0; j < candidates_per_query; ++j) {
                uint32_t cand_idx = cand_row[j];
                if (cand_idx >= target_count) {
                    continue;
                }

                const uint8_t* cand_ptr = t_ptr + cand_idx * byte_stride_target;
                float dist = dist_func_obj.compare(query_ptr, cand_ptr, param);

                // Keep candidates in max-heap of capacity k_top.
                // Until the heap reaches k_top elements, add all valid candidates.
                if (heap.size() < k_top) {
                    heap.emplace(cand_idx, dist);
                    if (heap.size() == k_top) {
                        max_dist = heap.top().getDistance();
                    }
                } 
                // Once full, only consider candidates strictly closer than the worst (max_dist).
                // replace_top replaces the root in O(log k) and returns a reference to the NEW root element (new max distance).
                else if (dist < max_dist) {
                    max_dist = heap.replace_top(cand_idx, dist).getDistance();
                }
            }

            results[i] = std::move(heap);
        });
    });

    return results;
}

} // namespace deglib::search
