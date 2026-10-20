import numpy as np
import deglib_cpp

# Core FeatureTransformation and Quantizer Classes
FeatureTransformation = deglib_cpp.transformation.FeatureTransformation
MipsTransform = deglib_cpp.transformation.MipsTransform
EvpQuantizer = deglib_cpp.transformation.EvpQuantizer
ScalarQuantizerInt8 = deglib_cpp.transformation.ScalarQuantizerInt8
ScalarQuantizerUint8 = deglib_cpp.transformation.ScalarQuantizerUint8
ScalarQuantizerInt8PerDim = deglib_cpp.transformation.ScalarQuantizerInt8PerDim
ScalarQuantizerUint8PerDim = deglib_cpp.transformation.ScalarQuantizerUint8PerDim

# Factory Functions
make_evp_quantizer = deglib_cpp.transformation.make_evp_quantizer
make_scalar_quantizer_int8 = deglib_cpp.transformation.make_scalar_quantizer_int8
make_scalar_quantizer_int8_perdim = deglib_cpp.transformation.make_scalar_quantizer_int8_perdim
make_scalar_quantizer_uint8 = deglib_cpp.transformation.make_scalar_quantizer_uint8
make_scalar_quantizer_uint8_perdim = deglib_cpp.transformation.make_scalar_quantizer_uint8_perdim

# Transformation Serialization & Application
serialize_transformations = deglib_cpp.transformation.serialize_transformations
deserialize_transformations = deglib_cpp.transformation.deserialize_transformations
apply_database = deglib_cpp.transformation.apply_database
apply_query = deglib_cpp.transformation.apply_query


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
    "EvpQuantizer",
    "ScalarQuantizerInt8",
    "ScalarQuantizerUint8",
    "ScalarQuantizerInt8PerDim",
    "ScalarQuantizerUint8PerDim",
    "make_evp_quantizer",
    "make_scalar_quantizer_int8",
    "make_scalar_quantizer_int8_perdim",
    "make_scalar_quantizer_uint8",
    "make_scalar_quantizer_uint8_perdim",
    "serialize_transformations",
    "deserialize_transformations",
    "apply_database",
    "apply_query",
    "mips_l2_transform",
    "mips_l2_transform_query",
]
