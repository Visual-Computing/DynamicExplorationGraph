# test/ subdirectory: unit tests for deglib headers
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

# Phase 1 Test Target: CPU Features & Build Verification
add_deglib_test(test_cpu_features src/test_cpu_features.cpp)

# Phase 2 Test Targets: FP16 & Distance Functions
add_deglib_test(test_fp16 src/distance/test_fp16.cpp)
add_deglib_test(test_fp32_l2 src/distance/test_fp32_l2.cpp)
add_deglib_test(test_fp32_inner_product src/distance/test_fp32_inner_product.cpp)
add_deglib_test(test_fp16_inner_product src/distance/test_fp16_inner_product.cpp)
add_deglib_test(test_uint8_l2 src/distance/test_uint8_l2.cpp)
add_deglib_test(test_config_cascade src/distance/test_config_cascade.cpp)
add_deglib_test(test_distances src/test_distances.cpp)
