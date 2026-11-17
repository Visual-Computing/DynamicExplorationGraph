import enum
import deglib_cpp

cpp_cpu = deglib_cpp.cpu


class InstructionSet(enum.IntEnum):
    """
    SIMD Instruction Set selection for DEGLib distance computations.

    - ``Auto``: Automatically select the fastest available instruction set supported by the CPU.
    - ``Scalar``: Use fallback scalar CPU instructions (no SIMD).
    - ``AVX2``: Force 256-bit AVX2 instructions.
    - ``AVX2_VNNI``: Force 256-bit AVX2 with Vector Neural Network Instructions.
    - ``AVX512``: Force 512-bit AVX-512 instructions.
    - ``AVX512_VNNI``: Force 512-bit AVX-512 with Vector Neural Network Instructions.
    """

    Auto = cpp_cpu.InstructionSet.Auto
    Scalar = cpp_cpu.InstructionSet.Scalar
    AVX2 = cpp_cpu.InstructionSet.AVX2
    AVX2_VNNI = cpp_cpu.InstructionSet.AVX2_VNNI
    AVX512 = cpp_cpu.InstructionSet.AVX512
    AVX512_VNNI = cpp_cpu.InstructionSet.AVX512_VNNI

    @classmethod
    def resolve(cls, instruction: "InstructionSet | str | int") -> "InstructionSet":
        """
        Resolve an ``InstructionSet`` from an enum member, integer value, or string name.

        Case-insensitive string names such as ``"Auto"``, ``"Scalar"``, ``"AVX2"``,
        ``"AVX2_VNNI"``, ``"AVX512"``, and ``"AVX512_VNNI"`` (as well as ``"none"``
        as an alias for ``Scalar``) are supported.
        """
        if isinstance(instruction, cls):
            return instruction
        if isinstance(instruction, cpp_cpu.InstructionSet):
            return cls(int(instruction))
        if isinstance(instruction, int) and not isinstance(instruction, bool):
            return cls(instruction)
        if isinstance(instruction, str):
            key = instruction.strip().lower().replace("-", "_").replace(" ", "_")
            alias_map = {
                "auto": cls.Auto,
                "scalar": cls.Scalar,
                "none": cls.Scalar,
                "none_": cls.Scalar,
                "avx2": cls.AVX2,
                "avx2_vnni": cls.AVX2_VNNI,
                "avx2vnni": cls.AVX2_VNNI,
                "avx512": cls.AVX512,
                "avx_512": cls.AVX512,
                "avx512_vnni": cls.AVX512_VNNI,
                "avx512vnni": cls.AVX512_VNNI,
                "avx_512_vnni": cls.AVX512_VNNI,
            }
            if key in alias_map:
                return alias_map[key]
        raise ValueError(f"Unknown InstructionSet: {instruction!r}. Available: {[m.name for m in cls]}")


def has_avx2() -> bool:
    """
    Check whether the host CPU supports AVX2 (256-bit SIMD) instructions.
    """
    return cpp_cpu.has_avx2()


def has_avx2_vnni() -> bool:
    """
    Check whether the host CPU supports AVX2-VNNI (256-bit) extensions.
    """
    return cpp_cpu.has_avx2_vnni()


def has_avx512() -> bool:
    """
    Check whether the host CPU supports AVX-512 (512-bit SIMD) instructions.
    """
    return cpp_cpu.has_avx512()


def has_avx512_vnni() -> bool:
    """
    Check whether the host CPU supports AVX-512 VNNI extensions.
    """
    return cpp_cpu.has_avx512_vnni()


def has_avx512_fp16() -> bool:
    """
    Check whether the host CPU supports AVX-512 FP16 (native arithmetic) extensions.
    """
    return cpp_cpu.has_avx512_fp16()


def has_fp16() -> bool:
    """
    Check whether the host CPU supports FP16 (half-precision) extensions.
    """
    return cpp_cpu.has_fp16()


__all__ = [
    "InstructionSet",
    "has_avx2",
    "has_avx2_vnni",
    "has_avx512",
    "has_avx512_vnni",
    "has_avx512_fp16",
    "has_fp16",
]
