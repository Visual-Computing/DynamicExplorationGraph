#pragma once

#include "deglib/config.h"

#include <cstddef>

namespace deglib::memory {

static const size_t L1_CACHE_LINE_SIZE = 64;

inline static void prefetch(const char* ptr, const size_t size = 128) {
    size_t pos = 0;
    while (pos < size) {
#if defined(DEGLIB_X86)
        _mm_prefetch(ptr + pos, _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr + pos, 0, 3);
#endif
        pos += L1_CACHE_LINE_SIZE;
    }
}

inline static void prefetch_lines(const void* ptr, int lines) {
    const char* p = static_cast<const char*>(ptr);
    switch (lines) {
        case 8:
            prefetch(p + 7 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 7:
            prefetch(p + 6 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 6:
            prefetch(p + 5 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 5:
            prefetch(p + 4 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 4:
            prefetch(p + 3 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 3:
            prefetch(p + 2 * L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 2:
            prefetch(p + L1_CACHE_LINE_SIZE, L1_CACHE_LINE_SIZE);
            [[fallthrough]];
        case 1:
            prefetch(p, L1_CACHE_LINE_SIZE);
            break;
        default:
            prefetch(p, L1_CACHE_LINE_SIZE);
            break;
    }
}

}  // namespace deglib::memory
