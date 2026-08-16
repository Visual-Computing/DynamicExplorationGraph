cmake_minimum_required(VERSION 3.19)

project(benchmarks)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(bench_static_data ${PROJECT_SOURCE_DIR}/src/bench_static_data.cpp)
target_include_directories(bench_static_data PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bench_static_data PUBLIC DEG_LIB compile-options fmt::fmt)

add_executable(bench_dynamic_data ${PROJECT_SOURCE_DIR}/src/bench_dynamic_data.cpp)
target_include_directories(bench_dynamic_data PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bench_dynamic_data PUBLIC DEG_LIB compile-options fmt::fmt)

add_executable(bench_edge_optimization ${PROJECT_SOURCE_DIR}/src/bench_edge_optimization.cpp)
target_include_directories(bench_edge_optimization PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bench_edge_optimization PUBLIC DEG_LIB compile-options fmt::fmt)
