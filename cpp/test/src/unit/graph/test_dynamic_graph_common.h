#pragma once

#include "deglib/graph/dynamic_graph.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {

std::vector<float> make_vec_4d(float x, float y, float z, float w) { return {x, y, z, w}; }

std::unique_ptr<std::byte[]> make_float_bytes(const std::vector<float>& v) {
    auto bytes = std::make_unique<std::byte[]>(v.size() * sizeof(float));
    std::memcpy(bytes.get(), v.data(), v.size() * sizeof(float));
    return bytes;
}

}  // anonymous namespace
