# python-bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.19)
project(deglib_cpp)

# Add python TODO: maybe remove
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Add pybind11
find_package(pybind11 CONFIG REQUIRED)

# disable benchmark building
set(ENABLE_BENCHMARKS OFF CACHE BOOL "Disable benchmarks" FORCE)

# Add cpp-deglib
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/)
pybind11_add_module(deglib_cpp src/deg_cpp/deglib_cpp.cpp)

# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/lib/deglib/include)

target_link_libraries(deglib_cpp PRIVATE compile-options)
