cmake_minimum_required(VERSION 3.19)

find_package(OpenMP REQUIRED)

# Build deglib_evp_task1 target
add_executable(deglib_evp_task1 task1.cpp)
target_include_directories(deglib_evp_task1 PRIVATE ${CMAKE_SOURCE_DIR}/benchmark/include)
target_link_libraries(deglib_evp_task1 PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_evp_task1 PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# ---------------------------------------------------------------------------
# HDF5 module unit tests (header-only, uses gtest bundled with fmt)
# ---------------------------------------------------------------------------

set(GTEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest/gmock-gtest-all.cc)
set(GTEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt/test/gtest)
set(HDF5_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/hdf5)
set(HDF5_TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/hdf5/test/test_main.cpp)

macro(add_hdf5_test name source)
    add_executable(${name} ${HDF5_TEST_MAIN} ${source})
    target_include_directories(${name} PRIVATE ${HDF5_INCLUDE})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE})
    target_sources(${name} PRIVATE ${GTEST_SOURCE})
    target_link_libraries(${name} PRIVATE compile-options)
    set_target_properties(${name} PROPERTIES INTERMEDIATE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obj/${name}")
endmacro()

add_hdf5_test(test_hdf5_types hdf5/test/test_types.cpp)
add_hdf5_test(test_hdf5_io hdf5/test/test_io.cpp)
add_hdf5_test(test_hdf5_structural hdf5/test/test_structural.cpp)
add_hdf5_test(test_hdf5_ohdr hdf5/test/test_ohdr.cpp)
add_hdf5_test(test_hdf5_integration hdf5/test/test_integration.cpp)
add_hdf5_test(test_dataset_dims hdf5/test/test_dataset_dims.cpp)
add_hdf5_test(test_sanity hdf5/test/test_sanity.cpp)
add_hdf5_test(test_dataset_dims_large hdf5/test/test_dataset_dims_large.cpp)
add_hdf5_test(test_sanity_large hdf5/test/test_sanity_large.cpp)
add_hdf5_test(test_integration_large hdf5/test/test_integration_large.cpp)


