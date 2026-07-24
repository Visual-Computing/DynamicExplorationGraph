#pragma once

#include "config.h"

#include <cstddef>

#if defined(USE_AVX2) || defined(USE_SSE42) || defined(USE_AVX512)
#include <immintrin.h>
#endif

namespace deglib::memory {

  static const size_t L1_CACHE_LINE_SIZE = 64;

  inline static void prefetch(const char *ptr, const size_t size = 128) {
    #if defined(USE_AVX2) || defined(USE_SSE42) || defined(USE_AVX512)
    size_t pos = 0;
    while(pos < size) {
      _mm_prefetch(ptr+pos, _MM_HINT_T0);
      pos += L1_CACHE_LINE_SIZE;
    }
    #endif
  }

}  // namespace deglib::memory
