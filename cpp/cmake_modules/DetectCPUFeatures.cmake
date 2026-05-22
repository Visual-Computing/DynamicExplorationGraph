include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})

# We modify these test programs to ensure they return 0 on successful execution.
# Using 'volatile' tells the compiler not to optimize the vector instructions away.
set(SSE4PROG "
#include <smmintrin.h>
int main() {
    __m128 x = _mm_set1_ps(0.5f);
    x = _mm_dp_ps(x, x, 0x77);
    volatile float d = _mm_cvtss_f32(x);
    (void)d;
    return 0;
}")

set(AVXPROG "
#include <immintrin.h>
int main() {
    __m128 x = _mm_set1_ps(0.5f);
    x = _mm_permute_ps(x, 1);
    volatile float d = _mm_cvtss_f32(x);
    (void)d;
    return 0;
}")

set(AVX2PROG "
#include <immintrin.h>
int main() {
    __m256i x = _mm256_set1_epi32(5);
    x = _mm256_add_epi32(x, x);
    volatile int d = _mm256_movemask_epi8(x);
    (void)d;
    return 0;
}")

# Test AVX512F — minimal: just setzero + reduce_add (no DQ needed)
set(AVX512F_MIN_PROG "
#include <immintrin.h>
int main() {
    __m512 v = _mm512_set1_ps(1.0f);
    volatile float res = _mm512_reduce_add_ps(v);
    (void)res;
    return 0;
}")

# Helper macro to check compiler capability AND runtime execution
macro(check_cpu_feature CODE FLAGS VAR)
    set(CMAKE_REQUIRED_FLAGS "${FLAGS}")
    if(CMAKE_CROSSCOMPILING)
        # If cross-compiling, we can only verify if the compiler compiles the code
        check_cxx_source_compiles("${CODE}" ${VAR})
    else()
        # If native compiling, we also verify that the compiled code runs successfully
        check_cxx_source_runs("${CODE}" ${VAR})
    endif()
endmacro()

if(MSVC)
    check_cpu_feature("${SSE4PROG}" "/EHsc /arch:SSE2" SUPPORT_SSE42)
    message(STATUS "SUPPORT_SSE42 ${SUPPORT_SSE42}")
    check_cpu_feature("${AVXPROG}" "/EHsc /arch:AVX" SUPPORT_AVX)
    message(STATUS "SUPPORT_AVX ${SUPPORT_AVX}")
    check_cpu_feature("${AVX2PROG}" "/EHsc /arch:AVX2" SUPPORT_AVX2)
    message(STATUS "SUPPORT_AVX2 ${SUPPORT_AVX2}")
    check_cpu_feature("${AVX512F_MIN_PROG}" "/EHsc /arch:AVX512" SUPPORT_AVX512F)
    message(STATUS "SUPPORT_AVX512F ${SUPPORT_AVX512F}")
else()
    check_cpu_feature("${SSE4PROG}" "-march=native -msse4.2" SUPPORT_SSE42)
    check_cpu_feature("${AVXPROG}" "-march=native -mavx" SUPPORT_AVX)
    check_cpu_feature("${AVX2PROG}" "-march=native -mavx2" SUPPORT_AVX2)
    check_cpu_feature("${AVX512F_MIN_PROG}" "-march=native -mavx512f -mfma" SUPPORT_AVX512F)
    message(STATUS "SUPPORT_AVX512F ${SUPPORT_AVX512F}")
endif()	

set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
