cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(DATA_PATH "" CACHE PATH "Path to data sets")

if(DATA_PATH STREQUAL "")
    message(WARNING "DATA_PATH is empty")
endif()

# build DEG benchmark files
find_package(OpenMP REQUIRED)

add_executable(deglib_build_and_test ${PROJECT_SOURCE_DIR}/src/deglib_build_and_test.cpp)
target_link_libraries(deglib_build_and_test PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_build_and_test PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
set_property(TARGET deglib_build_and_test PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_build_and_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_build_and_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_build_and_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_build_and_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/benchmark")

add_executable(deglib_test ${PROJECT_SOURCE_DIR}/src/deglib_test.cpp)
target_link_libraries(deglib_test PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_test PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
set_property(TARGET deglib_test PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_test PROPERTY RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/benchmark")

add_executable(deglib_setup_dataset ${PROJECT_SOURCE_DIR}/src/deglib_setup_dataset.cpp)
target_link_libraries(deglib_setup_dataset PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_setup_dataset PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
set_property(TARGET deglib_setup_dataset PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_setup_dataset PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_setup_dataset PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_setup_dataset PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/benchmark")
set_property(TARGET deglib_setup_dataset PROPERTY RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/benchmark")
