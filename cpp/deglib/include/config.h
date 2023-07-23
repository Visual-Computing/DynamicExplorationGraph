#pragma once

#ifndef NO_MANUAL_VECTORIZATION

  // Microsoft Visual C++ does not define __SSE__ or __SSE2__ but _M_IX86_FP instead
  // https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
  #ifdef _MSC_VER
    #if (defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86_FP) == 2)
      #define __SSE__
      #define __SSE2__
    #elif defined(_M_IX86_FP) == 1
      #define __SSE__
    #endif
  #endif

  #ifdef __SSE__
    #define USE_SSE
  #endif

  #ifdef __AVX__
    #define USE_AVX
  #endif

  #ifdef __AVX512F__
    #define USE_AVX512
  #endif

#endif

// TODO switch to only #include <immintrin.h>
// https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
#if defined(USE_AVX) || defined(USE_SSE)
  #ifdef _MSC_VER
    #include <intrin.h>
    #include <stdexcept>
  #else
    #include <x86intrin.h>
  #endif
#endif

//#ifdef _WINDOWS
//  #include <malloc.h>
//#else
//  #include <alloca.h>
//#endif
//#define vla(dtype, size) static_cast<type*>(alloca(sizeof(dtype) * size))

#ifdef _WINDOWS
  #include <malloc.h>  
  #define vla(var_name, dtype, size) auto var_name = (dtype*) _malloca(size*sizeof(dtype));
  #define free_vla(arr) _freea(arr);
#else
  #define vla(var_name, dtype, size) dtype var_name[size];
  #define free_vla(arr) 
#endif
