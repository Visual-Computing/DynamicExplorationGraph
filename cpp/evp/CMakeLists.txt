cmake_minimum_required(VERSION 3.19)

# evp subdirectory: EVP (EvpBits) implementation

# Data path configuration
# Uses the global DATA_PATH from CMakePresets.json
if(DATA_PATH STREQUAL "")
    message(WARNING "DATA_PATH is empty")
endif()

# Unit tests
add_executable(test_evp test_evp.cpp)
target_include_directories(test_evp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
target_link_libraries(test_evp PRIVATE compile-options)


# Benchmark / SISAP evaluation
add_executable(run_evp run_evp.cpp)
target_include_directories(run_evp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../deglib/include)
target_link_libraries(run_evp PRIVATE compile-options)
target_compile_definitions(run_evp PRIVATE "DATA_PATH=\"${DATA_PATH}\"")

