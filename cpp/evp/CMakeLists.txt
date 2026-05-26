cmake_minimum_required(VERSION 3.19)

find_package(OpenMP REQUIRED)

# Build deglib_evp_test target
add_executable(deglib_evp_test deglib_evp_test.cpp)
target_include_directories(deglib_evp_test PRIVATE ${CMAKE_SOURCE_DIR}/benchmark/include)
target_link_libraries(deglib_evp_test PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_evp_test PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# Build deglib_evp_microbench target
add_executable(deglib_evp_microbench deglib_evp_microbench.cpp)
target_include_directories(deglib_evp_microbench PRIVATE ${CMAKE_SOURCE_DIR}/benchmark/include)
target_link_libraries(deglib_evp_microbench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only OpenMP::OpenMP_CXX)
target_compile_definitions(deglib_evp_microbench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")




