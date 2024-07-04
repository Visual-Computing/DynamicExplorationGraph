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

option(ENABLE_BENCHMARKS "compile benchmarks" ON)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
include(DetectCPUFeatures)

# detecting openmp
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(OPENMP_FOUND)
  message("Found OpenMP")
  target_link_libraries(compile-options INTERFACE OpenMP::OpenMP_CXX)
endif()

# setup compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    compile-options
    INTERFACE -march=native
              -fpic
              -fopenmp
              -mfma)
elseif(MSVC)
  
  # warning level 4 (show all)
  target_compile_options(compile-options INTERFACE /W4)

  # detecting SUPPORT_AVX2 or SSE2 support
  cmake_host_system_information(RESULT SUPPORT_SSE2 QUERY HAS_SSE2)
  if(SUPPORT_AVX2)
    target_compile_options(compile-options INTERFACE /arch:AVX2)
  elseif(SUPPORT_SSE2)
    target_compile_options(compile-options INTERFACE /arch:SSE2)
  endif()

  # disable string optimizations and function level linking
  # https://stackoverflow.com/questions/5063334/what-is-the-difference-between-the-ox-and-o2-compiler-options
  target_compile_options(compile-options INTERFACE $<$<CONFIG:Release>:/Ox>)
else()
  message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_subdirectory(deglib)

if (ENABLE_BENCHMARKS)
	add_subdirectory(external/fmt)
	add_subdirectory(benchmark)
endif()
