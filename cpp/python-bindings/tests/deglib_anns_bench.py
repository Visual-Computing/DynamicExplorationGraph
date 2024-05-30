"""
Mirror of the c++ benchmark "cpp/benchmark/src/deglib_anns_bench.cpp".
"""


import deglib


def test_deglib_anns_bench():
    print("Testing ...")

    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    arg1 = b'arg1'
    arg2 = b'arg2'
    a = deglib.distance_float_l2(arg1, arg2)
    print('a:', a)


if __name__ == '__main__':
    test_deglib_anns_bench()  # TODO: replace with test framework
