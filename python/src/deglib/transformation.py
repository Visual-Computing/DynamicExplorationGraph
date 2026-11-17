from collections.abc import Sequence
from typing import TYPE_CHECKING

import deglib_cpp
import numpy as np

if TYPE_CHECKING:
    from .distances import Metric

# Core FeatureTransformation and Quantizer Classes
FeatureTransformation = deglib_cpp.transformation.FeatureTransformation
MipsTransform = deglib_cpp.transformation.MipsTransform
L2NormalizeTransform = deglib_cpp.transformation.L2NormalizeTransform
FastHadamardRotation = deglib_cpp.transformation.FastHadamardRotation
EvpQuantizer = deglib_cpp.transformation.EvpQuantizer
ScalarQuantizerInt8 = deglib_cpp.transformation.ScalarQuantizerInt8
ScalarQuantizerUint8 = deglib_cpp.transformation.ScalarQuantizerUint8
ScalarQuantizerUint4 = deglib_cpp.transformation.ScalarQuantizerUint4
ScalarQuantizerInt4 = deglib_cpp.transformation.ScalarQuantizerInt4
ScalarQuantizerUInt7 = deglib_cpp.transformation.ScalarQuantizerUInt7
ScalarQuantizerFP16 = deglib_cpp.transformation.ScalarQuantizerFP16


# Factory Functions
def make_l2_normalize_transform(dim: int = 0) -> L2NormalizeTransform:
    """
    Create an ``L2NormalizeTransform`` that normalizes vectors to unit L2 length.

    :param dim: Expected vector dimensionality (``0`` infers dimension from input).
    :return: Configured ``L2NormalizeTransform`` instance.
    """
    return deglib_cpp.transformation.make_l2_normalize_transform(dim=dim)


def make_fast_hadamard_rotation(
    vectors: np.ndarray,
    seed: int = 42,
    rounds: int = 0,
    auto_detect: bool = True,
) -> FastHadamardRotation:
    """
    Create and fit a ``FastHadamardRotation`` on training vectors.

    :param vectors: 2D ``float32`` NumPy array of shape ``(N, dim)``.
    :param seed: PRNG seed for diagonal Rademacher sign flips.
    :param rounds: Number of Walsh-Hadamard transform rounds (``0`` uses library default).
    :param auto_detect: If ``True``, enables rotation based on measured excess kurtosis.
    :return: Fitted ``FastHadamardRotation`` instance.
    """
    return deglib_cpp.transformation.make_fast_hadamard_rotation(
        vectors, seed=seed, rounds=rounds, auto_detect=auto_detect
    )


def make_quantizer_pipeline(
    vectors: np.ndarray,
    quant: "Metric | int | str",
    quantile: float = -1.0,
    use_rerank: bool = False,
    num_threads: int = 0,
    seed: int = 42,
) -> list[FeatureTransformation]:
    """
    Build a calibrated quantization pipeline for any supported target quantizer.

    For scalar quantizers (``uint4``, ``int4``, ``uint7``, ``uint8``, ``int8``),
    excess kurtosis is measured on a sample of ``vectors`` and a ``FastHadamardRotation``
    is automatically prepended when outliers exceed the threshold. For ``fp16`` and ``evp``,
    no rotation is prepended.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quant: Target quantizer as a :class:`~deglib.distances.Metric` (e.g. ``Metric.Int8_L2``,
                  ``Metric.EVP_InnerProduct``) or string (``"uint4"``, ``"int4"``, ``"uint7"``,
                  ``"uint8"``, ``"int8"``, ``"fp16"``, ``"evp"``).
    :param quantile: Clipping quantile for scalar quantizers. Values ``< 0`` (default) auto-optimize.
                     Ignored for ``fp16`` and ``evp``.
    :param use_rerank: Adjusts the kurtosis rotation threshold when a high-precision reranker is used.
    :param num_threads: Worker threads for calibration/rotation (``0`` uses hardware concurrency).
    :param seed: PRNG seed for ``FastHadamardRotation``.
    :return: List of fitted :class:`FeatureTransformation` objects (1 or 2 elements).
    """
    return deglib_cpp.transformation.make_quantizer_pipeline(
        vectors, quant, quantile=quantile, use_rerank=use_rerank, num_threads=num_threads, seed=seed
    )


def make_evp_quantizer(
    vectors: np.ndarray,
    non_zeros: int = 0,
    num_threads: int = 0,
) -> EvpQuantizer:
    """
    Create and calibrate an ``EvpQuantizer`` (Equi-Voronoi Polytopes) on training vectors.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param non_zeros: Number of extreme non-zero components retained per vector.
                      ``0`` (default) auto-calibrates the optimal count during ``fit()``.
    :param num_threads: Worker threads for calibration (``0`` uses hardware concurrency).
    :return: Fitted ``EvpQuantizer`` instance.
    """
    return deglib_cpp.transformation.make_evp_quantizer(vectors, non_zeros=non_zeros, num_threads=num_threads)


def make_scalar_quantizer_int8(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerInt8:
    """
    Create and calibrate a symmetric ``ScalarQuantizerInt8`` (``[-127, 127]``) on ``vectors``.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Clipping quantile; values ``< 0`` (default) auto-optimize.
    :return: Fitted ``ScalarQuantizerInt8`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_int8(vectors, quantile=quantile)


def make_scalar_quantizer_uint8(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerUint8:
    """
    Create and calibrate an affine ``ScalarQuantizerUint8`` (``[0, 255]``) on ``vectors``.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Clipping quantile; values ``< 0`` (default) auto-optimize.
    :return: Fitted ``ScalarQuantizerUint8`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_uint8(vectors, quantile=quantile)


def make_scalar_quantizer_uint4(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerUint4:
    """
    Create and calibrate a packed 4-bit ``ScalarQuantizerUint4`` (``[0, 15]``) on ``vectors``.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Clipping quantile; values ``< 0`` (default) auto-optimize.
    :return: Fitted ``ScalarQuantizerUint4`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_uint4(vectors, quantile=quantile)


def make_scalar_quantizer_int4(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerInt4:
    """
    Create and calibrate a packed symmetric 4-bit ``ScalarQuantizerInt4`` (``[-7, 7]``) on ``vectors``.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Clipping quantile; values ``< 0`` (default) auto-optimize.
    :return: Fitted ``ScalarQuantizerInt4`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_int4(vectors, quantile=quantile)


def make_scalar_quantizer_uint7(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerUInt7:
    """
    Create and calibrate an affine 7-bit ``ScalarQuantizerUInt7`` (``[0, 127]``) on ``vectors``.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Clipping quantile; values ``< 0`` (default) auto-optimize.
    :return: Fitted ``ScalarQuantizerUInt7`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_uint7(vectors, quantile=quantile)


def make_scalar_quantizer_fp16(vectors: np.ndarray, quantile: float = -1.0) -> ScalarQuantizerFP16:
    """
    Create a ``ScalarQuantizerFP16`` that casts vectors to IEEE 754 float16.

    :param vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
    :param quantile: Unused clipping parameter kept for uniform factory signature.
    :return: Fitted ``ScalarQuantizerFP16`` instance.
    """
    return deglib_cpp.transformation.make_scalar_quantizer_fp16(vectors, quantile=quantile)


# Transformation Application
def apply_on_database(
    transformations: Sequence[FeatureTransformation],
    vectors: np.ndarray,
    num_threads: int = 0,
) -> np.ndarray:
    """
    Apply a transformation pipeline sequentially to database vectors.

    :param transformations: Sequence of fitted ``FeatureTransformation`` instances.
    :param vectors: 2D NumPy array of shape ``(N, input_dim)``.
    :param num_threads: Worker threads (``0`` uses hardware concurrency).
    :return: Transformed 2D NumPy array ready for graph storage.
    """
    return deglib_cpp.transformation.apply_on_database(list(transformations), vectors, num_threads=num_threads)


def apply_on_query(
    transformations: Sequence[FeatureTransformation],
    queries: np.ndarray,
) -> np.ndarray:
    """
    Apply a transformation pipeline sequentially to query vectors.

    :param transformations: Sequence of fitted ``FeatureTransformation`` instances.
    :param queries: 1D or 2D NumPy array of query vectors.
    :return: Transformed query array.
    """
    return deglib_cpp.transformation.apply_on_query(list(transformations), queries)


def mips_l2_transform(database: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Transform database vectors from d-dimensional space to (d+1)-dimensional space
    for Maximum Inner Product Search (MIPS) using L2 distance.

    For each vector :math:`x_i \\in \\mathbb{R}^d`:

    .. math::

        x'_i = [x_i, \\sqrt{M^2 - \\|x_i\\|^2}] \\in \\mathbb{R}^{d+1}

    where :math:`M^2 = \\max_i \\|x_i\\|^2`.

    :param database: 2D float32 NumPy array of shape (N, d).
    :return: Tuple of (transformed_database array of shape (N, d+1), max_norm float M).
    """
    db_f32 = np.ascontiguousarray(database, dtype=np.float32)
    return deglib_cpp.transformation.mips_l2_transform(db_f32)


def mips_l2_transform_query(queries: np.ndarray) -> np.ndarray:
    """
    Pad query vectors from d-dimensional space to (d+1)-dimensional space
    for MIPS queries against an L2-transformed database.

    For each query vector :math:`q_i \\in \\mathbb{R}^d`:

    .. math::

        q'_i = [q_i, 0] \\in \\mathbb{R}^{d+1}

    :param queries: 1D or 2D float32 NumPy array.
    :return: Transformed queries array of shape (d+1) or (Q, d+1).
    """
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    return deglib_cpp.transformation.mips_l2_transform_query(queries_f32)


__all__ = [
    "FeatureTransformation",
    "MipsTransform",
    "L2NormalizeTransform",
    "FastHadamardRotation",
    "EvpQuantizer",
    "ScalarQuantizerInt8",
    "ScalarQuantizerUint8",
    "ScalarQuantizerFP16",
    "ScalarQuantizerUint4",
    "ScalarQuantizerInt4",
    "ScalarQuantizerUInt7",
    "make_scalar_quantizer_uint4",
    "make_scalar_quantizer_int4",
    "make_scalar_quantizer_uint7",
    "make_l2_normalize_transform",
    "make_fast_hadamard_rotation",
    "make_quantizer_pipeline",
    "make_evp_quantizer",
    "make_scalar_quantizer_int8",
    "make_scalar_quantizer_uint8",
    "make_scalar_quantizer_fp16",
    "apply_on_database",
    "apply_on_query",
    "mips_l2_transform",
    "mips_l2_transform_query",
]
