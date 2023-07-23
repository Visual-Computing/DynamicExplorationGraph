cmake_minimum_required(VERSION 3.19)

# Only interpret if() arguments as variables or keywords when unquoted. More
# details run "cmake --help-policy CMP0054"
cmake_policy(SET CMP0054 NEW)

project(
  deg_lib
  HOMEPAGE_URL "https://github.com/Neiko2002/DEG"
  DESCRIPTION
    "Dynamic Exploration Graphs for fast approximate nearest neighbor search and navigation in large image datasets"
  VERSION 0.0.1)

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(compile-options INTERFACE)
target_compile_features(compile-options INTERFACE cxx_std_20)
target_compile_definitions(compile-options INTERFACE $<$<PLATFORM_ID:Windows>:NOMINMAX>)

# detecting CPU instruction support
# https://github.com/lemire/FastPFor/blob/master/CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
include(DetectCPUFeatures)

# setup compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    compile-options
    INTERFACE -DNDEBUG
              -march=native
              -fpic
              -fopenmp)
elseif(MSVC)
  # https://stackoverflow.com/questions/54114287/visual-studio-not-recognizing-avx2-or-avx/54137338#54137338
  # use /fsanitize=address if needed
  target_compile_options(compile-options INTERFACE /W4)

  option(KEEP_DEBUG_SYMBOLS "Whether to compile with /Zi and /DEBUG" off)
  if(KEEP_DEBUG_SYMBOLS)
    target_compile_options(compile-options INTERFACE /Zi)
    target_link_options(compile-options INTERFACE /DEBUG)
  endif()

  # detecting SUPPORT_AVX2 or SSE2 support
  cmake_host_system_information(RESULT SUPPORT_SSE2 QUERY HAS_SSE2)
  if(SUPPORT_AVX2)
    target_compile_options(compile-options INTERFACE /arch:AVX2)
  elseif(SUPPORT_SSE2)
    target_compile_options(compile-options INTERFACE /arch:SSE2)
  else()
    target_compile_options(compile-options INTERFACE /arch:native)
  endif()

  # disable string optimizations and function level linking
  # https://stackoverflow.com/questions/5063334/what-is-the-difference-between-the-ox-and-o2-compiler-options
  target_compile_options(compile-options INTERFACE $<$<CONFIG:Release>:/Ox>)
endif()

message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_subdirectory(external/fmt)
add_subdirectory(external/robin-map)

add_subdirectory(deglib)
add_subdirectory(benchmark)
