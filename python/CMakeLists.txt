# python-bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.19)
project(python-bindings)

# Add pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/python-bindings/external/pybind11/)

# Specify the source files
set(PYBIND_SOURCES deg_cpp/deglib_cpp.cpp)

# Specify the include directories
include_directories(${CMAKE_SOURCE_DIR}/deglib/include)

# Create a Python module
pybind11_add_module(deglib_cpp MODULE ${PYBIND_SOURCES})

target_link_libraries(deglib_cpp PRIVATE compile-options)

# Link necessary libraries (if any)
# target_link_libraries(deglib_cpp PRIVATE <libraries>)
