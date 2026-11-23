import enum
import numpy as np
import deglib_cpp

from deglib.cpu import InstructionSet

cpp_distances = deglib_cpp.distances
cpp_cpu = deglib_cpp.cpu

__all__ = ["Metric", "InstructionSet", "floats_to_fp16", "fp16_to_floats"]


class _MetricMeta(enum.EnumMeta):
    def __instancecheck__(cls, instance):
        return super().__instancecheck__(instance) or isinstance(instance, _BoundMetric)


def _metric_data_size(metric: "Metric", dim: int) -> int:
    if dim < 0:
        raise ValueError(f"dim must be non-negative, got {dim}")
    m = Metric(int(metric))
    if m in (Metric.FP32_L2, Metric.FP32_InnerProduct):
        return dim * 4
    if m in (Metric.FP16_L2, Metric.FP16_InnerProduct):
        return dim * 2
    if m in (
        Metric.Uint8_L2,
        Metric.Uint8_InnerProduct,
        Metric.Int8_L2,
        Metric.Int8_InnerProduct,
        Metric.Uint7_L2,
        Metric.Uint7_InnerProduct,
    ):
        return dim
    if m in (
        Metric.Uint4_L2,
        Metric.Uint4_InnerProduct,
        Metric.Int4_L2,
        Metric.Int4_InnerProduct,
    ):
        return (dim + 1) // 2
    if m == Metric.EVP_InnerProduct:
        return 2 * ((dim + 7) // 8)
    raise ValueError(f"Unsupported metric: {metric}")


def _metric_dtype(metric: "Metric") -> np.dtype:
    m = Metric(int(metric))
    if m in (Metric.FP32_L2, Metric.FP32_InnerProduct):
        return np.dtype(np.float32)
    if m in (Metric.FP16_L2, Metric.FP16_InnerProduct):
        return np.dtype(np.uint16)
    if m in (
        Metric.Int8_L2,
        Metric.Int8_InnerProduct,
        Metric.Uint7_L2,
        Metric.Uint7_InnerProduct,
    ):
        return np.dtype(np.int8)
    if m in (
        Metric.Uint8_L2,
        Metric.Uint8_InnerProduct,
        Metric.Uint4_L2,
        Metric.Uint4_InnerProduct,
        Metric.Int4_L2,
        Metric.Int4_InnerProduct,
        Metric.EVP_InnerProduct,
    ):
        return np.dtype(np.uint8)
    raise ValueError(f"Unsupported metric: {metric}")


def _metric_accepted_dtypes(metric: "Metric") -> tuple[np.dtype, ...]:
    m = Metric(int(metric))
    if m in (Metric.FP16_L2, Metric.FP16_InnerProduct):
        return (np.dtype(np.uint16), np.dtype(np.float16))
    return (_metric_dtype(m),)


def _infer_dim_from_array(metric: "Metric", arr: np.ndarray) -> int:
    if arr.ndim == 0:
        raise ValueError("Input array must be at least 1D to infer feature dimensionality")
    width = int(arr.shape[-1])
    m = Metric(int(metric))
    if m in (
        Metric.Uint4_L2,
        Metric.Uint4_InnerProduct,
        Metric.Int4_L2,
        Metric.Int4_InnerProduct,
    ):
        return width * 2
    if m == Metric.EVP_InnerProduct:
        if width % 2 != 0:
            raise ValueError(f"EVP_InnerProduct packed width must be even, got {width}")
        return width * 4
    return width


def _prepare_distance_array(metric: "Metric", arr: np.ndarray, name: str) -> np.ndarray:
    accepted = _metric_accepted_dtypes(metric)
    if not isinstance(arr, np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=accepted[0])
    else:
        if arr.dtype not in accepted:
            expected_names = " or ".join(d.name for d in accepted)
            raise TypeError(f"{name} has dtype {arr.dtype}, expected {expected_names} for metric {metric.name}")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    return arr


def _make_cpp_space(metric: "Metric", dim: int, instruction: InstructionSet | None = None) -> cpp_distances.FloatSpace:
    if instruction is not None:
        inst = instruction
    elif isinstance(metric, _BoundMetric):
        inst = metric._instruction
    else:
        inst = InstructionSet.Auto
    return cpp_distances.FloatSpace(
        int(dim),
        cpp_distances.Metric(int(metric)),
        cpp_cpu.InstructionSet(int(inst)),
    )


class Metric(enum.IntEnum, metaclass=_MetricMeta):
    """
    Distance metric and storage format used by deglib graphs, searchers, and distance computations.
    """

    FP32_L2 = cpp_distances.Metric.FP32_L2
    FP32_InnerProduct = cpp_distances.Metric.FP32_InnerProduct
    Uint8_L2 = cpp_distances.Metric.Uint8_L2
    Uint8_InnerProduct = cpp_distances.Metric.Uint8_InnerProduct
    FP16_L2 = cpp_distances.Metric.FP16_L2
    FP16_InnerProduct = cpp_distances.Metric.FP16_InnerProduct
    EVP_InnerProduct = cpp_distances.Metric.EVP_InnerProduct
    Int8_InnerProduct = cpp_distances.Metric.Int8_InnerProduct
    Int8_L2 = cpp_distances.Metric.Int8_L2
    Uint4_InnerProduct = cpp_distances.Metric.Uint4_InnerProduct
    Uint4_L2 = cpp_distances.Metric.Uint4_L2
    Int4_InnerProduct = cpp_distances.Metric.Int4_InnerProduct
    Int4_L2 = cpp_distances.Metric.Int4_L2
    Uint7_L2 = cpp_distances.Metric.Uint7_L2
    Uint7_InnerProduct = cpp_distances.Metric.Uint7_InnerProduct

    def get_dtype(self) -> np.dtype:
        """
        Return the default NumPy storage dtype associated with this metric.
        """
        return _metric_dtype(self)

    def get_accepted_dtypes(self) -> tuple[np.dtype, ...]:
        """
        Return all NumPy dtypes directly accepted for raw storage with this metric.
        """
        return _metric_accepted_dtypes(self)

    def get_data_size(self, dim: int) -> int:
        """
        Return the byte size of one stored vector of dimension ``dim`` for this metric.
        """
        return _metric_data_size(self, dim)

    def get_query_data_size(self, dim: int) -> int:
        """
        Return the byte size of one query vector of dimension ``dim`` for this metric.
        """
        return _metric_data_size(self, dim)

    def get_instruction(self, dim: int = 16) -> InstructionSet:
        """
        Return the active SIMD ``InstructionSet`` for this metric at dimension ``dim``.
        """
        space = _make_cpp_space(self, dim, InstructionSet.Auto)
        return InstructionSet.resolve(space.get_instruction())

    def with_instruction(self, instruction: InstructionSet | str) -> "Metric":
        """
        Return a ``Metric`` handle bound to a specific SIMD ``InstructionSet``.
        """
        resolved = InstructionSet.resolve(instruction)
        return _BoundMetric(Metric(int(self)), resolved)

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute distance between two 1D feature vectors (dimension inferred automatically).
        """
        v1 = _prepare_distance_array(self, vec1, "vec1")
        v2 = _prepare_distance_array(self, vec2, "vec2")
        dim = _infer_dim_from_array(self, v1)
        space = _make_cpp_space(self, dim)
        return float(space.compute_distance(v1, v2))

    def compute_distances(self, query: np.ndarray, targets: np.ndarray, threads: int = 0) -> np.ndarray:
        """
        Compute distances between query vector(s) [D] or [Q, D] and target vector(s) [N, D].
        """
        q = _prepare_distance_array(self, query, "query")
        t = _prepare_distance_array(self, targets, "targets")
        dim = _infer_dim_from_array(self, q)
        space = _make_cpp_space(self, dim)
        return space.compute_distances(q, t, threads=threads)

    def _to_cpp_space(self, dim: int) -> cpp_distances.FloatSpace:
        return _make_cpp_space(self, dim)

    def _to_cpp_metric(self) -> cpp_distances.Metric:
        return cpp_distances.Metric(int(self))

    @classmethod
    def resolve(
        cls,
        metric: "Metric | str | int",
        distance_kind: str | None = None,
        instruction: InstructionSet | str | None = None,
    ) -> "Metric":
        """
        Resolve a ``Metric`` from an enum member or case-insensitive name/parts.

        Examples:
            Metric.resolve("Uint8_L2")
            Metric.resolve("int8", "ip")
            Metric.resolve("fp32", "cosine")
            Metric.resolve("fp32", "l2", instruction="AVX2")
            Metric.resolve("evp")
        """
        bound_inst: InstructionSet | None = None
        base_metric: Metric | None = None

        if isinstance(metric, _BoundMetric):
            base_metric = metric._metric
            bound_inst = metric._instruction
        elif isinstance(metric, cls):
            base_metric = metric
        elif isinstance(metric, cpp_distances.Metric):
            base_metric = cls(int(metric))
        elif isinstance(metric, int) and not isinstance(metric, bool):
            base_metric = cls(metric)
        elif isinstance(metric, str):
            raw = metric.strip()
            if distance_kind is not None:
                dtype_part = raw
                kind_part = distance_kind.strip()
            elif raw.lower() == "evp":
                base_metric = cls.EVP_InnerProduct
                dtype_part = ""
                kind_part = ""
            elif "_" in raw:
                parts = raw.split("_", 1)
                dtype_part, kind_part = parts[0], parts[1]
            else:
                raise ValueError(
                    f"Cannot resolve metric {metric!r} without distance_kind (e.g. 'FP32_L2' or Metric.resolve('int8', 'ip'))"
                )

            if base_metric is None:
                dtype_norm_map = {
                    "fp32": "FP32",
                    "float32": "FP32",
                    "f32": "FP32",
                    "fp16": "FP16",
                    "float16": "FP16",
                    "f16": "FP16",
                    "uint8": "Uint8",
                    "u8": "Uint8",
                    "int8": "Int8",
                    "i8": "Int8",
                    "uint7": "Uint7",
                    "u7": "Uint7",
                    "uint4": "Uint4",
                    "u4": "Uint4",
                    "int4": "Int4",
                    "i4": "Int4",
                    "evp": "EVP",
                }
                kind_norm_map = {
                    "l2": "L2",
                    "euclidean": "L2",
                    "sqeuclidean": "L2",
                    "ip": "InnerProduct",
                    "innerproduct": "InnerProduct",
                    "inner_product": "InnerProduct",
                    "dot": "InnerProduct",
                    "mips": "InnerProduct",
                    "cosine": "InnerProduct",
                }
                d_key = dtype_part.strip().lower().replace("-", "").replace("_", "")
                k_key = kind_part.strip().lower().replace("-", "_").replace(" ", "_")
                d_canon = dtype_norm_map.get(d_key)
                k_canon = kind_norm_map.get(k_key)
                if d_canon is None or k_canon is None:
                    raise ValueError(
                        f"Unknown metric specification: metric={metric!r}, distance_kind={distance_kind!r}"
                    )
                full_name = f"{d_canon}_{k_canon}"
                if not hasattr(cls, full_name):
                    raise ValueError(f"Unsupported Metric combination: {full_name}")
                base_metric = getattr(cls, full_name)
        else:
            raise TypeError(f"metric must be Metric, str, or int, got {type(metric).__name__}")

        if distance_kind is not None and not isinstance(metric, str):
            # Re-resolve distance kind on an enum metric if explicitly overridden
            dtype_prefix = base_metric.name.split("_", 1)[0]
            base_metric = cls.resolve(dtype_prefix, distance_kind)

        target_inst = (
            InstructionSet.resolve(instruction)
            if instruction is not None
            else (bound_inst if bound_inst is not None else InstructionSet.Auto)
        )
        if target_inst == InstructionSet.Auto:
            return base_metric
        return _BoundMetric(base_metric, target_inst)


class _BoundMetric(int):
    """
    A ``Metric`` handle paired with a non-default ``InstructionSet`` override.
    Behaves seamlessly as an ``int``/``Metric`` across comparisons and bindings.
    """

    def __new__(cls, metric: Metric, instruction: InstructionSet):
        obj = super().__new__(cls, int(metric))
        obj._metric = Metric(int(metric))
        obj._instruction = instruction
        return obj

    @property
    def name(self) -> str:
        return self._metric.name

    @property
    def value(self) -> int:
        return int(self._metric)

    @property
    def metric(self) -> Metric:
        return self._metric

    def get_dtype(self) -> np.dtype:
        return _metric_dtype(self._metric)

    def get_accepted_dtypes(self) -> tuple[np.dtype, ...]:
        return _metric_accepted_dtypes(self._metric)

    def get_data_size(self, dim: int) -> int:
        return _metric_data_size(self._metric, dim)

    def get_query_data_size(self, dim: int) -> int:
        return _metric_data_size(self._metric, dim)

    def get_instruction(self, dim: int = 16) -> InstructionSet:
        space = _make_cpp_space(self._metric, dim, self._instruction)
        return InstructionSet.resolve(space.get_instruction())

    def with_instruction(self, instruction: InstructionSet | str) -> "Metric":
        return self._metric.with_instruction(instruction)

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        v1 = _prepare_distance_array(self._metric, vec1, "vec1")
        v2 = _prepare_distance_array(self._metric, vec2, "vec2")
        dim = _infer_dim_from_array(self._metric, v1)
        space = _make_cpp_space(self._metric, dim, self._instruction)
        return float(space.compute_distance(v1, v2))

    def compute_distances(self, query: np.ndarray, targets: np.ndarray, threads: int = 0) -> np.ndarray:
        q = _prepare_distance_array(self._metric, query, "query")
        t = _prepare_distance_array(self._metric, targets, "targets")
        dim = _infer_dim_from_array(self._metric, q)
        space = _make_cpp_space(self._metric, dim, self._instruction)
        return space.compute_distances(q, t, threads=threads)

    def _to_cpp_space(self, dim: int) -> cpp_distances.FloatSpace:
        return _make_cpp_space(self._metric, dim, self._instruction)

    def _to_cpp_metric(self) -> cpp_distances.Metric:
        return cpp_distances.Metric(int(self._metric))

    def __repr__(self) -> str:
        return f"<Metric.{self._metric.name}: {int(self._metric)} [{self._instruction.name}]>"

    def __str__(self) -> str:
        return f"Metric.{self._metric.name}"


def floats_to_fp16(floats: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy float32 array to FP16 bit-patterns stored as uint16.
    """
    if floats.dtype != np.float32:
        raise ValueError("Input array must have dtype float32")
    if not floats.flags.c_contiguous:
        floats = np.ascontiguousarray(floats)
    res = cpp_distances.floats_to_fp16(floats)
    return res.reshape(floats.shape)


def fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy uint16 array storing FP16 bit-patterns back to float32.
    """
    if fp16_vals.dtype != np.uint16:
        raise ValueError("Input array must have dtype uint16")
    if not fp16_vals.flags.c_contiguous:
        fp16_vals = np.ascontiguousarray(fp16_vals)
    res = cpp_distances.fp16_to_floats(fp16_vals)
    return res.reshape(fp16_vals.shape)
