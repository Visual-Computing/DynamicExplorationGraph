//
// Created by Bruno Schilling on 29.05.24.
//
#include <pybind11/pybind11.h>
#include <distances.h>

bool avx_usable() {
    #if defined(USE_AVX)
        return true;
    #else
        return false;
    #endif
}

bool sse_usable() {
    #if defined(USE_SSE)
        return true;
    #else
        return false;
    #endif
}

float distance_float_l2(char* pVect1v, char* pVect2v, float c) {
    return (float)pVect1v[0] + (float)pVect2v[0];
    // size_t size = 3;
    // return deglib::distances::L2Float::compare(pVect1v, pVect2v, &size);
}

PYBIND11_MODULE(deglib_cpp, m) {
    m.doc() = "Python bindings for Dynamic Exploration Graph";

    m.def("avx_usable", &avx_usable, "Returns whether AVX instructions are available");
    m.def("sse_usable", &sse_usable, "Returns whether SSE instructions are available");

    m.def("distance_float_l2", &distance_float_l2, "l2 float distances");
}
