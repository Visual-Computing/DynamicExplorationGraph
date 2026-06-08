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

# Consolidate all build outputs into the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

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
# https://github.com/lemire/FastPFor/blob/master/CMakeLists.txt
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  message(WARNING "You are compiling on an Apple M chip. At the moment deglib does not support ARM optimizations, so using deglib will be slow!")
endif()

# ==============================================================================
# 1. Platform & CPU Feature Detection
# ==============================================================================
if (NOT FORCE_AVX2)
  # This module runs test programs on the host machine to check for SSE42, AVX, AVX2, and AVX512 support.
  # It sets the variables: SUPPORT_SSE42, SUPPORT_AVX, SUPPORT_AVX2, SUPPORT_AVX512F.
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
  include(DetectCPUFeatures)
endif()

# ==============================================================================
# 2. Compile Flag Settings (Declaration Phase)
# ==============================================================================
set(COMPILER_ARCH_FLAGS "")
set(COMPILER_COMMON_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
  # Common flags for Unix/Apple toolchains
  list(APPEND COMPILER_COMMON_FLAGS 
    -fpic                           # Position Independent Code
    -w                              # Suppress warnings
    -pthread                        # Thread support
    -ftree-vectorize                # Enable tree vectorization
    -ftree-vectorizer-verbose=0     # Quiet vectorizer output
  )

  # Optimization levels (only for non-Debug builds)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(compile-options INTERFACE $<$<NOT:$<CONFIG:Debug>>:-O2>)
  else()
    target_compile_options(compile-options INTERFACE $<$<NOT:$<CONFIG:Debug>>:-O3>)
  endif()

  # Architecture / ISA extension flags selection
  if (FORCE_AVX2)
    message(STATUS "[FORCE_AVX2] Configuring compiler options for forced AVX2")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR NOT APPLE)
      list(APPEND COMPILER_ARCH_FLAGS -mavx2 -mfma -mf16c)
    endif()
  else()
    message(STATUS "[Native Build] Configuring native CPU architectures")
    list(APPEND COMPILER_ARCH_FLAGS -march=native)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      # Reinforce standard instruction set extensions
      list(APPEND COMPILER_ARCH_FLAGS -mavx2 -mfma -mf16c)
    endif()
  endif()

elseif(MSVC)
  # Common flags for MSVC
  list(APPEND COMPILER_COMMON_FLAGS 
    /EHsc                           # C++ exception handling model
    /W1                             # Warning level 1
  )

  # Optimization options (only for non-Debug builds)
  target_compile_options(compile-options INTERFACE $<$<NOT:$<CONFIG:Debug>>:/O2>)
  target_compile_options(compile-options INTERFACE $<$<CONFIG:Release>:/Ox>)

  # Architecture / ISA extension flags selection
  if (FORCE_AVX2)
    message(STATUS "[FORCE_AVX2 MSVC] Configuring compiler options for forced AVX2")
    list(APPEND COMPILER_ARCH_FLAGS /arch:AVX2)
  else()
    message(STATUS "[Native MSVC] CPU feature detection (from DetectCPUFeatures.cmake):")
    message(STATUS "  SUPPORT_AVX512F = ${SUPPORT_AVX512F}")
    message(STATUS "  SUPPORT_AVX2    = ${SUPPORT_AVX2}")
    message(STATUS "  SUPPORT_SSE42   = ${SUPPORT_SSE42}")

    if(SUPPORT_AVX512F)
      list(APPEND COMPILER_ARCH_FLAGS /arch:AVX512)
    elseif(SUPPORT_AVX2)
      list(APPEND COMPILER_ARCH_FLAGS /arch:AVX2)
    elseif(SUPPORT_SSE42)
      # MSVC x64 always natively supports SSE2, so we use /arch:SSE2 or fallback
      list(APPEND COMPILER_ARCH_FLAGS /arch:SSE2)
    else()
      list(APPEND COMPILER_ARCH_FLAGS /arch:native)
    endif()
  endif()

else()
  message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# ==============================================================================
# 3. Application Phase
# ==============================================================================
message(STATUS "Applying compiler flags:")
message(STATUS "  Common Flags: ${COMPILER_COMMON_FLAGS}")
message(STATUS "  Arch Flags:   ${COMPILER_ARCH_FLAGS}")

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

# fmt is only needed by benchmarks and evp — not by the core deglib library.
if (ENABLE_BENCHMARKS)
	set(FMT_MASTER_PROJECT OFF)
	set(FMT_TEST OFF)
	set(FMT_INSTALL OFF)
	add_subdirectory(external/fmt)
	add_subdirectory(benchmark)
	add_subdirectory(sisap)
endif()
