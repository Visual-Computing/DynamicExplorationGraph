# python-bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.19)
project(deglib_cpp)

# Add python executable
if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Add pybind11
find_package(pybind11 CONFIG REQUIRED)

# disable benchmark and test building
set(ENABLE_BENCHMARKS OFF CACHE BOOL "Disable benchmarks" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "Disable tests" FORCE)

# Find deglib cpp source: prefer ../cpp in repository checkouts, fallback to local lib/ (PyPI sdist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/CMakeLists.txt")
    set(DEGLIB_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/CMakeLists.txt")
    set(DEGLIB_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
else()
    message(FATAL_ERROR "Could not find deglib cpp source directory (checked ../cpp/ and lib/)")
endif()

# Add cpp-deglib
add_subdirectory(${DEGLIB_CPP_DIR} deglib_cpp_build)
pybind11_add_module(deglib_cpp src/deg_cpp/deglib_cpp.cpp)

# Specify the include directories
include_directories(${DEGLIB_CPP_DIR}/deglib/include)

target_link_libraries(deglib_cpp PRIVATE compile-options)
