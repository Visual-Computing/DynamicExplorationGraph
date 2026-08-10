import enum

import deglib_cpp.cpu as cpp_cpu


class InstructionSet(enum.IntEnum):
    Auto = cpp_cpu.InstructionSet.Auto
    Scalar = cpp_cpu.InstructionSet.Scalar
    AVX2 = cpp_cpu.InstructionSet.AVX2
    AVX512 = cpp_cpu.InstructionSet.AVX512

    def to_cpp(self) -> cpp_cpu.InstructionSet:
        if self == InstructionSet.Auto:
            return cpp_cpu.InstructionSet.Auto
        elif self == InstructionSet.Scalar:
            return cpp_cpu.InstructionSet.Scalar
        elif self == InstructionSet.AVX2:
            return cpp_cpu.InstructionSet.AVX2
        elif self == InstructionSet.AVX512:
            return cpp_cpu.InstructionSet.AVX512
        else:
            raise ValueError(f"unknown instruction set: {self}")


def has_avx2() -> bool:
    """Returns whether AVX2 instructions are available."""
    return cpp_cpu.has_avx2()


def has_avx512() -> bool:
    """Returns whether AVX512 instructions are available."""
    return cpp_cpu.has_avx512()
