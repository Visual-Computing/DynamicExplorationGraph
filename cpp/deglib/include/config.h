#pragma once

#ifndef NO_MANUAL_VECTORIZATION
// Microsoft Visual C++ does not define __SSE__ or __SSE2__ but _M_IX86_FP instead
// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
#ifdef _MSC_VER
#if defined(_M_AMD64) || defined(_M_X64)
#define __SSE4_1__
#define __SSE4_2__
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#ifndef __FMA__
#define __FMA__
#endif
#endif
#endif

// Ensure only one SIMD variant is defined at a time
// AVX-512 implies AVX2 and AVX, but we want exclusive selection
// MSVC defines __AVX2__ and __AVX512F__ from /arch: flags
#if defined(__AVX512F__)
#define USE_AVX512
#define USE_AVX2
#define USE_SSE42
#elif defined(__AVX2__)
#define USE_AVX2
#define USE_SSE42
#elif defined(__SSE4_2__) || defined(__SSE4_1__)
#define USE_SSE42
#endif

#if !defined(USE_AVX2) && !defined(USE_SSE42) && !defined(USE_AVX512)
#ifdef _MSC_VER
#pragma message("warning: neither SSE4.2, AVX2 nor AVX512 are defined")
#else
#warning "neither SSE4.2, AVX2 nor AVX512 are defined"
#endif
#elif !defined(__FMA__)
#ifdef _MSC_VER
#pragma message("warning: no FMA support or compile flag is missing")
#else
#warning "no FMA support or compile flag is missing"
#endif
#endif

// #undef USE_AVX512  // for testing arm processors
// #undef USE_AVX2
// #undef USE_SSE42
#endif

// TODO switch to only #include <immintrin.h>
// https://stackoverflow.com/questions/11228855/header-files-for-x86-intrin
#if defined(USE_AVX2) || defined(USE_SSE42) || defined(USE_AVX512)
#ifdef _MSC_VER
#include <intrin.h>

#include <stdexcept>
#else
#include <x86intrin.h>
#endif
#endif
