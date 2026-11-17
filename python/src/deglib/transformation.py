from collections.abc import Sequence
import numpy as np
import deglib_cpp

from deglib.distances import Metric

cpp_transformation = deglib_cpp.transformation

FeatureTransformation = cpp_transformation.FeatureTransformation


def _normalize_transforms(
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None,
) -> list[FeatureTransformation]:
    if transformations is None:
        return []
    if isinstance(transformations, FeatureTransformation):
        return [transformations]
    return list(transformations)


def _resolve_quant_arg(quant: Metric | str | int) -> int | str:
    if isinstance(quant, str):
        if "_" in quant:
            return int(Metric.resolve(quant))
        return quant
    return int(Metric.resolve(quant)) if isinstance(quant, Metric) else int(quant)


def make_l2_normalize_transform(vectors_or_dim: np.ndarray | int) -> FeatureTransformation:
    """
    Create a validated L2 unit-normalization transformation (``x -> x / ||x||_2``).

    Args:
        vectors_or_dim: Sample feature matrix (2D ``[N, D]`` or 1D ``[D]``) or positive integer ``dim``.
    """
    if isinstance(vectors_or_dim, np.ndarray):
        if vectors_or_dim.ndim == 2:
            if vectors_or_dim.shape[0] == 0 or vectors_or_dim.shape[1] == 0:
                raise ValueError("make_l2_normalize_transform: vectors must be non-empty with dim > 0")
            dim = int(vectors_or_dim.shape[1])
        elif vectors_or_dim.ndim == 1:
            if vectors_or_dim.shape[0] == 0:
                raise ValueError("make_l2_normalize_transform: vectors must be non-empty")
            dim = int(vectors_or_dim.shape[0])
        else:
            raise ValueError(f"make_l2_normalize_transform: expected 1D or 2D array, got {vectors_or_dim.ndim}D")
    elif isinstance(vectors_or_dim, (int, np.integer)) and not isinstance(vectors_or_dim, bool):
        dim = int(vectors_or_dim)
        if dim <= 0:
            raise ValueError(f"make_l2_normalize_transform: dim must be > 0, got {dim}")
    else:
        raise TypeError(f"vectors_or_dim must be a NumPy array or positive int, got {type(vectors_or_dim).__name__}")
    return cpp_transformation.make_l2_normalize_transform(dim=dim)


def make_mips_transform(vectors: np.ndarray, max_norm: float = 0.0) -> FeatureTransformation:
    """
    Create and fit a spherical MIPS-to-L2 transformation (``d -> d + 1``).

    On database vectors, appends ``sqrt(M^2 - ||x||^2)`` (with ``M`` calibrated as max norm when
    ``max_norm <= 0``); on query vectors, appends ``0.0``.
    """
    return cpp_transformation.make_mips_transform(vectors=vectors, max_norm=float(max_norm))


def make_fast_hadamard_rotation(
    vectors: np.ndarray,
    seed: int = 42,
    rounds: int = 0,
    auto_detect: bool = True,
) -> FeatureTransformation:
    """
    Create and fit a randomized Fast Walsh-Hadamard rotation to spread outlier coordinates before quantization.
    """
    return cpp_transformation.make_fast_hadamard_rotation(
        vectors=vectors,
        seed=seed,
        rounds=rounds,
        auto_detect=auto_detect,
    )


def make_quantizer_pipeline(
    vectors: np.ndarray,
    quant: Metric | str,
    quantile: float = -1.0,
    use_rerank: bool = False,
    threads: int = 0,
    seed: int = 42,
) -> list[FeatureTransformation]:
    """
    Build a calibrated transformation pipeline (optional Hadamard rotation + quantizer) from ``vectors``.

    Args:
        vectors: 2D ``float32`` or ``float16`` array of shape ``(N, dim)``.
        quant: Target ``Metric`` (e.g. ``Metric.Uint8_L2``, ``Metric.EVP_InnerProduct``) or quantizer string
            (``"uint4"``, ``"int4"``, ``"uint7"``, ``"uint8"``, ``"int8"``, ``"fp16"``, ``"evp"``).
        quantile: Clipping quantile override in ``(0, 1]``; ``< 0`` auto-tunes quantile.
        use_rerank: Whether quantization will be followed by exact reranking (affects auto-quantile).
        threads: Worker threads for calibration (``0`` = all CPU cores).
        seed: Random seed for optional rotation calibration.
    """
    return cpp_transformation.make_quantizer_pipeline(
        vectors=vectors,
        quant=_resolve_quant_arg(quant),
        quantile=quantile,
        use_rerank=use_rerank,
        threads=threads,
        seed=seed,
    )


def make_evp_quantizer(
    vectors: np.ndarray,
    non_zeros: int = 0,
    threads: int = 0,
) -> FeatureTransformation:
    """
    Create and calibrate an ``EvpQuantizer`` from 2D ``float32`` or ``float16`` ``vectors``.
    """
    return cpp_transformation.make_evp_quantizer(
        vectors=vectors,
        non_zeros=non_zeros,
        threads=threads,
    )


def make_scalar_quantizer_int8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate a symmetric ``int8`` scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_int8(vectors=vectors, quantile=quantile)


def make_scalar_quantizer_uint8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate an affine ``uint8`` scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_uint8(vectors=vectors, quantile=quantile)


def make_scalar_quantizer_uint7(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate an affine 7-bit ``uint7`` (stored in ``int8``) scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_uint7(vectors=vectors, quantile=quantile)


def make_scalar_quantizer_uint4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate a packed 4-bit unsigned ``uint4`` scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_uint4(vectors=vectors, quantile=quantile)


def make_scalar_quantizer_int4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate a packed 4-bit signed ``int4`` scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_int4(vectors=vectors, quantile=quantile)


def make_scalar_quantizer_fp16(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation:
    """
    Create and calibrate a power-of-two scaled ``FP16`` scalar quantizer.
    """
    return cpp_transformation.make_scalar_quantizer_fp16(vectors=vectors, quantile=quantile)


def apply_on_database(
    transformations: FeatureTransformation | Sequence[FeatureTransformation],
    vectors: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    """
    Apply a transformation or sequence of transformations to 1D/2D database vectors.
    """
    t_list = _normalize_transforms(transformations)
    if not t_list:
        return np.ascontiguousarray(vectors)
    return cpp_transformation.apply_on_database(
        transformations=t_list,
        vectors=vectors,
        threads=threads,
    )


def apply_on_query(
    transformations: FeatureTransformation | Sequence[FeatureTransformation],
    queries: np.ndarray,
) -> np.ndarray:
    """
    Apply a transformation or sequence of transformations to 1D/2D query vectors.
    """
    t_list = _normalize_transforms(transformations)
    if not t_list:
        return np.ascontiguousarray(queries)
    return cpp_transformation.apply_on_query(
        transformations=t_list,
        queries=queries,
    )


__all__ = [
    "FeatureTransformation",
    "make_l2_normalize_transform",
    "make_mips_transform",
    "make_fast_hadamard_rotation",
    "make_quantizer_pipeline",
    "make_evp_quantizer",
    "make_scalar_quantizer_int8",
    "make_scalar_quantizer_uint8",
    "make_scalar_quantizer_uint7",
    "make_scalar_quantizer_uint4",
    "make_scalar_quantizer_int4",
    "make_scalar_quantizer_fp16",
    "apply_on_database",
    "apply_on_query",
]
