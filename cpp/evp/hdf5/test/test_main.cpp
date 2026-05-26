// test_main.cpp — Google Test entry point shared by all HDF5 test executables.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
