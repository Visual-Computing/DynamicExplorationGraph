cmake_minimum_required(VERSION 3.19)

# Only interpret if() arguments as variables or keywords when unquoted. More
# details run "cmake --help-policy CMP0054"
cmake_policy(SET CMP0054 NEW)

project(
  deg_lib
  HOMEPAGE_URL "https://github.com/Neiko2002/DEG"
  DESCRIPTION
    "Dynamic Exploration Graphs for fast approximate nearest neighbor search and navigation in large image datasets"
  VERSION 0.0.1
  LANGUAGES CXX
  )

# build options
option(ENABLE_BENCHMARKS "compile benchmarks" ON)
option(FORCE_AVX2 "compile for avx2 support. Otherwise compile for this machine." OFF)

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(compile-options INTERFACE)
target_compile_features(compile-options INTERFACE cxx_std_20)
target_compile_definitions(compile-options INTERFACE $<$<PLATFORM_ID:Windows>:NOMINMAX>)

# detecting CPU instruction support
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  message(WARNING "You are compiling on an Apple M chip. At the moment deglib does not support ARM optimizations, so using deglib will be slow!")
endif()

# ==============================================================================
# 1. Compiler Flags Setup
# ==============================================================================
set(COMPILER_COMMON_FLAGS "")
set(COMPILER_ARCH_FLAGS "")

if (FORCE_AVX2)
  message(NOTICE "Compiling for avx2, ignoring native optimizations.")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND COMPILER_COMMON_FLAGS -O3 -fpic -w -pthread -ftree-vectorize)
    list(APPEND COMPILER_ARCH_FLAGS -mavx2 -mfma)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    list(APPEND COMPILER_COMMON_FLAGS -O2 -fpic)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      list(APPEND COMPILER_ARCH_FLAGS -mavx2 -mfma)
    endif()
  elseif(MSVC)
    list(APPEND COMPILER_COMMON_FLAGS /O2 /EHsc /W4)
    list(APPEND COMPILER_ARCH_FLAGS /arch:AVX2)
  else()
    message(WARNING "Unknown compiler for AVX2 Build: ${CMAKE_CXX_COMPILER_ID}")
  endif()

else()  # Native build
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
  include(DetectCPUFeatures)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND COMPILER_COMMON_FLAGS -O3 -march=native -fpic -w -pthread -ftree-vectorize)
    list(APPEND COMPILER_ARCH_FLAGS -mavx2 -mfma)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    list(APPEND COMPILER_COMMON_FLAGS -O2 -DNDEBUG -march=native -fpic)
  elseif(MSVC)
    list(APPEND COMPILER_COMMON_FLAGS /W4 /EHsc $<$<CONFIG:Release>:/Ox>)

    message(STATUS "[Native MSVC] CPU feature detection:")
    message(STATUS "  SUPPORT_AVX512F = ${SUPPORT_AVX512F}")
    message(STATUS "  SUPPORT_AVX2    = ${SUPPORT_AVX2}")
    message(STATUS "  SUPPORT_SSE42   = ${SUPPORT_SSE42}")

    if(SUPPORT_AVX512F)
      list(APPEND COMPILER_ARCH_FLAGS /arch:AVX512)
    elseif(SUPPORT_AVX2)
      list(APPEND COMPILER_ARCH_FLAGS /arch:AVX2)
    elseif(SUPPORT_SSE42)
      list(APPEND COMPILER_ARCH_FLAGS /arch:SSE2)
    else()
      list(APPEND COMPILER_ARCH_FLAGS /arch:native)
    endif()
  else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

# Apply compiler flags to compile-options interface target
target_compile_options(compile-options INTERFACE 
  ${COMPILER_COMMON_FLAGS}
  ${COMPILER_ARCH_FLAGS}
)

message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

# Core Library
add_subdirectory(deglib)

# Unit tests
add_subdirectory(test)

# Benchmarks (optional)
if (ENABLE_BENCHMARKS)
	set(FMT_MASTER_PROJECT OFF CACHE BOOL "" FORCE)
	set(FMT_TEST OFF CACHE BOOL "" FORCE)
	set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
	add_subdirectory(external/fmt)
	add_subdirectory(benchmark)
endif()
