import deglib
import deglib_cpp


def test_cpu_functions_exist_and_return_bool():
    assert isinstance(deglib.cpu.has_avx2(), bool)
    assert isinstance(deglib.cpu.has_avx2_vnni(), bool)
    assert isinstance(deglib.cpu.has_avx512(), bool)
    assert isinstance(deglib.cpu.has_avx512_vnni(), bool)
    assert isinstance(deglib.cpu.has_avx512_fp16(), bool)
    assert isinstance(deglib.cpu.has_fp16(), bool)


def test_cpp_bindings_direct():
    assert deglib_cpp.cpu.has_fp16() == deglib.cpu.has_fp16()
    assert deglib_cpp.cpu.has_avx512_fp16() == deglib.cpu.has_avx512_fp16()
    assert deglib_cpp.cpu.has_avx2() == deglib.cpu.has_avx2()
    assert deglib_cpp.cpu.has_avx2_vnni() == deglib.cpu.has_avx2_vnni()
    assert deglib_cpp.cpu.has_avx512() == deglib.cpu.has_avx512()
    assert deglib_cpp.cpu.has_avx512_vnni() == deglib.cpu.has_avx512_vnni()
