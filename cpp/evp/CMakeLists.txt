cmake_minimum_required(VERSION 3.19)

# evp subdirectory: EVP (EvpBits) implementation

# Data path configuration
set(EVP_DATA_PATH "" CACHE PATH "Path to SISAP data sets")

if(EVP_DATA_PATH STREQUAL "")
    message(WARNING "EVP_DATA_PATH is empty - using default D:/Data/DEG/SISAP")
endif()

# Unit tests
add_executable(test_evp test_evp.cpp)
target_include_directories(test_evp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
target_link_libraries(test_evp PRIVATE compile-options)
set_target_properties(test_evp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")

# Benchmark / SISAP evaluation
add_executable(run_evp run_evp.cpp)
target_include_directories(run_evp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
target_link_libraries(run_evp PRIVATE compile-options)
target_compile_definitions(run_evp PRIVATE "DATA_PATH=\"${EVP_DATA_PATH}\"")
set_target_properties(run_evp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
