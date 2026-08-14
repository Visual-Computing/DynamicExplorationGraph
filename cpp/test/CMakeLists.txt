# test/ subdirectory: unit tests for deglib headers
# Each test executable links against gtest (bundled with fmt) and deglib headers.

cmake_minimum_required(VERSION 3.19)

set(GTEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest/gmock-gtest-all.cc)
set(GTEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest)
set(DEGLIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)

# Static library for GoogleTest to avoid compiling gmock-gtest-all.cc multiple times
add_library(gtest STATIC ${GTEST_SOURCE})
target_include_directories(gtest SYSTEM PUBLIC ${GTEST_INCLUDE})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gtest PRIVATE -Wno-character-conversion)
endif()

# Helper macro: creates a test executable with shared gtest + deglib setup
macro(add_deglib_test name source)
    add_executable(${name} ${TEST_MAIN} ${source})
    target_include_directories(${name} PRIVATE ${DEGLIB_INCLUDE})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} PRIVATE gtest compile-options)
    set_target_properties(${name} PROPERTIES INTERMEDIATE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj/${name}")
    add_test(NAME ${name} COMMAND ${name})
endmacro()


# Helper macro: creates a test executable with integration test include path
macro(add_deglib_integration_test name source)
    add_executable(${name} ${TEST_MAIN} ${source})
    target_include_directories(${name} PRIVATE ${DEGLIB_INCLUDE})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} PRIVATE gtest compile-options)
    set_target_properties(${name} PROPERTIES INTERMEDIATE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj/${name}")
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# Helper macro: creates a test executable without registering it with CTest
# (used for regression tests that are built manually but excluded from CI)
macro(add_deglib_test_no_ctest name source)
    add_executable(${name} ${TEST_MAIN} ${source})
    target_include_directories(${name} PRIVATE ${DEGLIB_INCLUDE})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} PRIVATE gtest compile-options)
    set_target_properties(${name} PROPERTIES INTERMEDIATE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj/${name}")
endmacro()

# Unit tests
add_deglib_test(test_config src/unit/test_config.cpp)
add_deglib_test(test_junker_volgenant_solver src/unit/optimization/flas/test_junker_volgenant_solver.cpp)
add_deglib_test(test_flas src/unit/optimization/flas/test_fast_linear_assignment_sorter.cpp)
add_deglib_test(test_fp16 src/unit/distance/test_fp16.cpp)
add_deglib_test(test_fp16_inner_product src/unit/distance/test_fp16_inner_product.cpp)
add_deglib_test(test_internal_graph src/unit/graph/test_internal_graph.cpp)
add_deglib_test(test_search src/unit/test_search.cpp)
add_deglib_test(test_evp_quantize src/unit/optimization/quantization/test_evp_quantize.cpp)
add_deglib_test(test_concurrent src/unit/test_concurrent.cpp)
add_deglib_test(test_filter src/unit/test_filter.cpp)
add_deglib_test(test_visited_list_pool src/unit/graph/test_visited_list_pool.cpp)
add_deglib_test(test_builder src/unit/test_builder.cpp)
add_deglib_test(test_fp32_l2 src/unit/distance/test_fp32_l2.cpp)
add_deglib_test(test_fp32_inner_product src/unit/distance/test_fp32_inner_product.cpp)
add_deglib_test(test_uint8_l2 src/unit/distance/test_uint8_l2.cpp)
add_deglib_test(test_sizebounded_graph src/unit/graph/test_sizebounded_graph.cpp)
add_deglib_test(test_readonly_graph src/unit/graph/test_readonly_graph.cpp)
add_deglib_test(test_graph src/unit/test_graph.cpp)
add_deglib_test(test_pruning src/unit/optimization/test_pruning.cpp)
add_deglib_test(test_transform src/unit/optimization/test_transform.cpp)
add_deglib_test(test_analysis src/unit/test_analysis.cpp)
add_deglib_test(test_evp_inner_product src/unit/distance/test_evp_inner_product.cpp)

# Integration tests (fast, recall-only, registered with CTest for CI)
add_deglib_integration_test(test_builder_integration src/integration/test_builder_integration.cpp)
add_deglib_integration_test(test_distances_integration src/integration/test_distances_integration.cpp)

# Regression tests (100k vectors, performance benchmarking — built but NOT registered with CTest)
add_deglib_test_no_ctest(test_builder_regression src/regression/test_builder_regression.cpp)
add_deglib_test_no_ctest(test_flas_regression src/regression/optimization/flas/test_fast_linear_assignment_sorter_regression.cpp)
add_deglib_test_no_ctest(test_sizebounded_graph_regression src/regression/graph/test_sizebounded_graph_regression.cpp)
add_deglib_test_no_ctest(test_readonly_graph_regression src/regression/graph/test_readonly_graph_regression.cpp)
add_deglib_test_no_ctest(test_evp_quantize_regression src/regression/optimization/quantization/test_evp_quantize_regression.cpp)
add_deglib_test_no_ctest(test_search_regression src/regression/test_search_regression.cpp)
add_deglib_test_no_ctest(test_filter_regression src/regression/test_filter_regression.cpp)

