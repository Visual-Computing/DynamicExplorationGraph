// test_main.cpp — Google Test entry point shared by all test executables.
// Each test executable must link against gmock-gtest-all.cc exactly once.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
