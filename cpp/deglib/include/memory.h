#pragma once

#include "config.h"

#include <cstddef>

#if defined(USE_AVX) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

namespace deglib {
namespace memory {

inline static void prefetch(const void* ptr, ...) {
#if defined(USE_AVX) || defined(USE_AVX512) || defined(USE_SSE)
    const char* p = reinterpret_cast<const char*>(ptr);
    _mm_prefetch(p + 0, _MM_HINT_T0);
    _mm_prefetch(p + 64, _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

}  // namespace memory
}  // namespace deglib
