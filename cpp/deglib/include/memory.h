#pragma once

#include "config.h"

namespace deglib
{
  class MemoryCache {
    public:
      inline static void prefetch(const char *ptr) {
        #if defined(USE_AVX) || defined(USE_SSE)
          _mm_prefetch(ptr, _MM_HINT_T0);
        #endif
      }
  };

}  // namespace deglib
