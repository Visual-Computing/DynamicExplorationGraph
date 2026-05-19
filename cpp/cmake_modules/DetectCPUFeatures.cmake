include(CheckCXXSourceCompiles)

set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})

set(SSE4PROG "

#include<smmintrin.h>
int main(){
__m128 x=_mm_set1_ps(0.5);
x=_mm_dp_ps(x,x,0x77);
return _mm_movemask_ps(x);
}")

set(AVXPROG "

#include<immintrin.h>
int main(){
__m128 x=_mm_set1_ps(0.5);
x=_mm_permute_ps(x,1);
return _mm_movemask_ps(x);
}")

set(AVX2PROG "

#include<immintrin.h>
int main(){
__m256i x=_mm256_set1_epi32(5);
x=_mm256_add_epi32(x,x);
return _mm256_movemask_epi8(x);
}")

# ============================================================================
# AVX-512 detection — based on actual intrinsics used in deglib/include/distances.h
#
# distances.h uses these AVX-512 intrinsics:
#   Float distance (AVX512F + FMA):
#     _mm512_setzero_ps, _mm512_sub_ps, _mm512_loadu_ps,
#     _mm512_fmadd_ps, _mm512_extractf32x8_ps (requires AVX512DQ)
#   Binary distance (AVX512VPOPCNTDQ):
#     _mm512_setzero_si512, _mm512_loadu_si512, _mm512_and_si512,
#     _mm512_add_epi64, _mm512_popcnt_epi64, _mm512_storeu_si512
#
# USE_AVX512 is set when AVX512F + FMA are available (float distance path).
# The binary distance path is guarded by #if defined(USE_AVX512) && defined(__AVX512VPOPCNTDQ__)
# in distances.h and only activates when the compiler defines __AVX512VPOPCNTDQ__.
# ===========================================================================

# Test AVX512F — minimal: just setzero + reduce_add (no DQ needed)
set(AVX512F_MIN_PROG "

#include<immintrin.h>
int main(){
    __m512 v = _mm512_set1_ps(1.0f);
    return _mm512_reduce_add_ps(v) > 0 ? 1 : 0;
}")

if(MSVC)
	set(CMAKE_REQUIRED_FLAGS "/EHsc /arch:SSE2")
	check_cxx_source_compiles("${SSE4PROG}" SUPPORT_SSE42)
	message(STATUS "SUPPORT_SSE42 ${SUPPORT_SSE42}")
	set(CMAKE_REQUIRED_FLAGS "/EHsc /arch:AVX")
	check_cxx_source_compiles("${AVXPROG}" SUPPORT_AVX)
	message(STATUS "SUPPORT_AVX ${SUPPORT_AVX}")
	set(CMAKE_REQUIRED_FLAGS "/EHsc /arch:AVX2")
	check_cxx_source_compiles("${AVX2PROG}" SUPPORT_AVX2)
	message(STATUS "SUPPORT_AVX2 ${SUPPORT_AVX2}")
	set(CMAKE_REQUIRED_FLAGS "/EHsc /arch:AVX512")
	check_cxx_source_compiles("${AVX512F_MIN_PROG}" SUPPORT_AVX512F)
	message(STATUS "SUPPORT_AVX512F ${SUPPORT_AVX512F}")
else()
	set(CMAKE_REQUIRED_FLAGS "-march=native -msse4.2")
	check_cxx_source_compiles("${SSE4PROG}" SUPPORT_SSE42)
	set(CMAKE_REQUIRED_FLAGS "-march=native -mavx")
	check_cxx_source_compiles("${AVXPROG}" SUPPORT_AVX)
	set(CMAKE_REQUIRED_FLAGS "-march=native -mavx2")
	check_cxx_source_compiles("${AVX2PROG}" SUPPORT_AVX2)
	set(CMAKE_REQUIRED_FLAGS "-march=native -mavx512f -mfma")
	check_cxx_source_compiles("${AVX512F_MIN_PROG}" SUPPORT_AVX512F)
	message(STATUS "SUPPORT_AVX512F ${SUPPORT_AVX512F}")
endif()	

set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
