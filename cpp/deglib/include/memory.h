#pragma once

#include "config.h"

#include <cstddef>

#if defined(USE_AVX) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

namespace deglib {
namespace memory {

static const size_t L1_CACHE_LINE_SIZE = 64;

inline static void prefetch(const char* ptr, const size_t size = 128) {
#if defined(USE_AVX) || defined(USE_SSE)
    size_t pos = 0;
    while (pos < size) {
        _mm_prefetch(ptr + pos, _MM_HINT_T0);
        pos += L1_CACHE_LINE_SIZE;
    }
#endif
}

// =============================================================================
// Configurable feature vector prefetch for compare_batch
// =============================================================================
//
// Control:
//   PREFETCH_COUNT  - number of cache lines (64B each) to prefetch per vector (0-16)
//   Default: 2
//
// =============================================================================

#ifndef PREFETCH_COUNT
#define PREFETCH_COUNT 2
#endif

inline void prefetch_feature(const void* ptr) {
#if PREFETCH_COUNT > 0 && (defined(USE_AVX) || defined(USE_AVX512) || defined(USE_SSE))
    const char* p = reinterpret_cast<const char*>(ptr);

    #if PREFETCH_COUNT >= 1
    _mm_prefetch(p + 0 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 2
    _mm_prefetch(p + 1 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 3
    _mm_prefetch(p + 2 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 4
    _mm_prefetch(p + 3 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 5
    _mm_prefetch(p + 4 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 6
    _mm_prefetch(p + 5 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 7
    _mm_prefetch(p + 6 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 8
    _mm_prefetch(p + 7 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 9
    _mm_prefetch(p + 8 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 10
    _mm_prefetch(p + 9 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 11
    _mm_prefetch(p + 10 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 12
    _mm_prefetch(p + 11 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 13
    _mm_prefetch(p + 12 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 14
    _mm_prefetch(p + 13 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 15
    _mm_prefetch(p + 14 * 64, _MM_HINT_T0);
    #endif
    #if PREFETCH_COUNT >= 16
    _mm_prefetch(p + 15 * 64, _MM_HINT_T0);
    #endif
#else
    (void)ptr;
#endif
}

}  // namespace memory
}  // namespace deglib
