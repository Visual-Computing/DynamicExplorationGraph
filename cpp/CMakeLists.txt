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

if (FORCE_AVX2)
  message(NOTICE "Compiling for avx2, ignoring native optimizations.")

  # setup compiler flags
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "[FORCE_AVX2 Clang/GNU] Compiler: ${CMAKE_CXX_COMPILER_ID}")
    target_compile_options(
      compile-options
      INTERFACE -O3
                -mavx2
                -mfma
                -fpic
                -w
                -fopenmp
                -pthread
                -ftree-vectorize
                -ftree-vectorizer-verbose=0
                )
    message(STATUS "  -> Using: -O3 -mavx2 -mfma -fopenmp ...")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	target_compile_options(compile-options INTERFACE -O2 -fpic)

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        target_compile_options(compile-options INTERFACE -mavx2 -mfma)
        message(STATUS "[FORCE_AVX2 AppleClang x86_64] -> Using: -O2 -fpic -mavx2 -mfma")
    else()
        message(STATUS "[FORCE_AVX2 AppleClang non-x86_64] -> Using: -O2 -fpic (no AVX2)")
    endif()
  elseif(MSVC)
    message(STATUS "[FORCE_AVX2 MSVC] -> Using: /O2 /arch:AVX2 /W1 /openmp /EHsc")
    target_compile_options(
	  compile-options
	  INTERFACE /O2
				/arch:AVX2
				/W1
				/openmp
				/EHsc)
  else()
    message(WARNING "Unknown compiler for AVX2 Build: ${CMAKE_CXX_COMPILER_ID}")
  endif()

else()  # Native build
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
  include(DetectCPUFeatures)

  # setup compiler flags
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "[Native Clang/GNU] Compiler: ${CMAKE_CXX_COMPILER_ID}")
    target_compile_options(
      compile-options
      INTERFACE -O3
                -march=native
                -mavx2
                -mfma
                -fpic
                -w
                -fopenmp
                -pthread
                -ftree-vectorize
                -ftree-vectorizer-verbose=0)
    message(STATUS "  -> Using: -O3 -march=native -mavx2 -mfma -fopenmp ...")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "[Native AppleClang] Compiler: ${CMAKE_CXX_COMPILER_ID}")
    target_compile_options(
      compile-options
      INTERFACE -O2
                -DNDEBUG
                -march=native
                -fpic)
    message(STATUS "  -> Using: -O2 -DNDEBUG -march=native -fpic")
  elseif(MSVC)
    # warning level 4 (show all)
    target_compile_options(compile-options INTERFACE /W4)

    # SUPPORT_AVX2 and SUPPORT_SSE2 are already set by DetectCPUFeatures.cmake
    # (via actual compiler tests, not cmake_host_system_information)
    message(STATUS "[MSVC native] CPU feature detection (from DetectCPUFeatures.cmake):")
    message(STATUS "  SUPPORT_AVX512F = ${SUPPORT_AVX512F}")
    message(STATUS "  SUPPORT_AVX2 = ${SUPPORT_AVX2}")
    message(STATUS "  SUPPORT_SSE2 = ${SUPPORT_SSE2}")
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
    message(STATUS "  System processor: ${CMAKE_SYSTEM_PROCESSOR}")

    if(SUPPORT_AVX512F)
      message(STATUS "  -> Using /arch:AVX512")
      target_compile_options(compile-options INTERFACE /arch:AVX512)
    elseif(SUPPORT_AVX2)
      message(STATUS "  -> Using /arch:AVX2")
      target_compile_options(compile-options INTERFACE /arch:AVX2)
    elseif(SUPPORT_SSE2)
      message(STATUS "  -> Using /arch:SSE2")
      target_compile_options(compile-options INTERFACE /arch:SSE2)
    else()
      message(STATUS "  -> Using /arch:native")
      target_compile_options(compile-options INTERFACE /arch:native)
    endif()

    # disable string optimizations and function level linking
    # https://stackoverflow.com/questions/5063334/what-is-the-difference-between-the-ox-and-o2-compiler-options
    target_compile_options(compile-options INTERFACE $<$<CONFIG:Release>:/Ox>)
  else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_subdirectory(deglib)


# fmt is only needed by benchmarks — not by deglib/evp.
if (ENABLE_BENCHMARKS)
	set(FMT_MASTER_PROJECT OFF)
	set(FMT_TEST OFF)
	set(FMT_INSTALL OFF)
	add_subdirectory(external/fmt)
	add_subdirectory(benchmark)
endif()

# Unit tests
add_subdirectory(test)
