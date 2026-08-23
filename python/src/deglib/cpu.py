import enum

import deglib_cpp.cpu as cpp_cpu


class InstructionSet(enum.IntEnum):
    """
    CPU instruction set used for vector distance calculations.

    - ``Auto`` (0): Automatically selects the fastest available instruction set supported by the host CPU.
    - ``Scalar`` (1): Standard non-vectorized fallback implementation.
    - ``AVX2`` (2): Uses 256-bit vector extensions when available.
    - ``AVX2_VNNI`` (3): Uses 256-bit vector extensions with VNNI acceleration when available.
    - ``AVX512`` (4): Uses 512-bit vector extensions when available.
    - ``AVX512_VNNI`` (5): Uses 512-bit vector extensions with VNNI acceleration when available.
    """

    Auto = cpp_cpu.InstructionSet.Auto
    Scalar = cpp_cpu.InstructionSet.Scalar
    AVX2 = cpp_cpu.InstructionSet.AVX2
    AVX2_VNNI = cpp_cpu.InstructionSet.AVX2_VNNI
    AVX512 = cpp_cpu.InstructionSet.AVX512
    AVX512_VNNI = cpp_cpu.InstructionSet.AVX512_VNNI


def has_avx2() -> bool:
    """
    Check whether the host CPU supports AVX2 extensions.

    :return: True if AVX2 is supported, False otherwise.
    """
    return cpp_cpu.has_avx2()


def has_avx_vnni() -> bool:
    """
    Check whether the host CPU supports AVX-VNNI (256-bit) extensions.

    :return: True if AVX-VNNI is supported, False otherwise.
    """
    return cpp_cpu.has_avx_vnni()


def has_avx512() -> bool:
    """
    Check whether the host CPU supports AVX-512 extensions.

    :return: True if AVX-512 is supported, False otherwise.
    """
    return cpp_cpu.has_avx512()


def has_avx512_vnni() -> bool:
    """
    Check whether the host CPU supports AVX-512 VNNI extensions.

    :return: True if AVX-512 VNNI is supported, False otherwise.
    """
    return cpp_cpu.has_avx512_vnni()
