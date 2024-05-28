cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(DATA_PATH "" CACHE PATH "Path to data sets")

if(DATA_PATH STREQUAL "")
    message(WARNING "DATA_PATH is empty")
endif()

# build DEG benchmark files
find_package(OpenMP REQUIRED)

add_executable(deglib_anns_bench ${PROJECT_SOURCE_DIR}/src/deglib_anns_bench.cpp)
target_link_libraries(deglib_anns_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only)
target_compile_definitions(deglib_anns_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_explore_bench ${PROJECT_SOURCE_DIR}/src/deglib_explore_bench.cpp)
target_link_libraries(deglib_explore_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only)
target_compile_definitions(deglib_explore_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_build_bench ${PROJECT_SOURCE_DIR}/src/deglib_build_bench.cpp)
target_link_libraries(deglib_build_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_build_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
