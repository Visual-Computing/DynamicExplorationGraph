#pragma once

#include "deglib/config.h"

#include <cstddef>

namespace deglib::memory {

  static const size_t L1_CACHE_LINE_SIZE = 64;

  inline static void prefetch(const char *ptr, const size_t size = 128) {
    size_t pos = 0;
    while(pos < size) {
#if defined(DEGLIB_X86)
      _mm_prefetch(ptr+pos, _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(ptr+pos, 0, 3);
#endif
      pos += L1_CACHE_LINE_SIZE;
    }
  }

}  // namespace deglib::memory
