# test/ subdirectory: unit tests for deglib library
# Each test executable links against gtest (bundled with fmt) and deglib headers.

cmake_minimum_required(VERSION 3.19)

set(GTEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest/gmock-gtest-all.cc)
set(GTEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest)
set(DEGLIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)

# Helper macro: creates a test executable with shared gtest + deglib setup
macro(add_deglib_test name source)
    add_executable(${name} ${TEST_MAIN} ${source})
    target_include_directories(${name} PRIVATE ${DEGLIB_INCLUDE})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE})
    target_sources(${name} PRIVATE ${GTEST_SOURCE})
    target_link_libraries(${name} PRIVATE compile-options)
    set_target_properties(${name} PROPERTIES INTERMEDIATE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj/${name}")
endmacro()

# CPU Features
add_deglib_test(test_cpu_features src/test_cpu_features.cpp)

# Distance Functions & FP16 Precision
add_deglib_test(test_fp16 src/distance/test_fp16.cpp)
add_deglib_test(test_fp32_l2 src/distance/test_fp32_l2.cpp)
add_deglib_test(test_fp32_inner_product src/distance/test_fp32_inner_product.cpp)
add_deglib_test(test_fp16_inner_product src/distance/test_fp16_inner_product.cpp)
add_deglib_test(test_uint8_l2 src/distance/test_uint8_l2.cpp)
add_deglib_test(test_config_cascade src/distance/test_config_cascade.cpp)
add_deglib_test(test_distances src/test_distances.cpp)

# EVP Quantization & Asymmetric EVP Distances
add_deglib_test(test_evp_quantize src/quantization/test_evp_quantize.cpp)
add_deglib_test(test_evp_inner_product src/distance/test_evp_inner_product.cpp)
add_deglib_test(test_fp16_evp_asym_inner_product src/distance/test_fp16_evp_asym_inner_product.cpp)

# Graph Structures
add_deglib_test(test_readonly_graph_external src/graph/test_readonly_graph_external.cpp)
add_deglib_test(test_sizebounded_graph src/graph/test_sizebounded_graph.cpp)

# Core Deglib Components (Builder, Search, Repository, Filter, VisitedListPool)
add_deglib_test(test_repository src/test_repository.cpp)
add_deglib_test(test_filter src/test_filter.cpp)
add_deglib_test(test_visited_list_pool src/test_visited_list_pool.cpp)
add_deglib_test(test_search src/test_search.cpp)
add_deglib_test(test_builder src/test_builder.cpp)

# Fast Linear Assignment Sorter (FLAS) Utility
add_deglib_test(test_flas src/flas/test_flas.cpp)

# Synthetic Performance Regression Test
add_deglib_test(test_regression src/test_regression.cpp)
