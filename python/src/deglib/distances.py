import enum
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp.distances as cpp_distances
import deglib_cpp.cpu as cpp_cpu
from deglib.cpu import InstructionSet

__all__ = ["Metric", "FloatSpace", "InstructionSet", "floats_to_fp16", "fp16_to_floats"]


class Metric(enum.IntEnum):
    """
    Distance metric used to calculate similarity or distance between feature vectors.

    - ``FP32_L2``: Euclidean (L2) distance for 32-bit floating point vectors.
    - ``FP32_InnerProduct``: Inner product (dot product) distance for 32-bit floating point vectors.
    - ``Uint8_L2``: Euclidean (L2) distance for 8-bit unsigned integer vectors.
    - ``Uint8_InnerProduct``: Inner product distance for 8-bit unsigned integer vectors.
    - ``FP16_L2``: Euclidean (L2) distance for 16-bit half-precision floating point vectors.
    - ``FP16_InnerProduct``: Inner product distance for 16-bit half-precision floating point vectors.
    - ``EVP_InnerProduct``: Inner product distance for quantized byte-packed (EVP) vectors.
    - ``Int8_InnerProduct``: Inner product distance for 8-bit signed integer vectors.
    - ``Int8_L2``: Euclidean (L2) distance for 8-bit signed integer vectors.
    - ``Uint4_L2``: Euclidean (L2) distance for 4-bit unsigned integer vectors (packed 2 per uint8).
    - ``Uint4_InnerProduct``: Inner product distance for 4-bit unsigned integer vectors (packed 2 per uint8).
    - ``Int4_L2``: Euclidean (L2) distance for 4-bit signed integer vectors (packed 2 per uint8).
    - ``Int4_InnerProduct``: Inner product distance for 4-bit signed integer vectors (packed 2 per uint8).
    - ``UInt7_InnerProduct``: Inner product distance for UInt7 codes in [0, 127].
    - ``UInt7_L2``: Euclidean (L2) distance for UInt7 codes in [0, 127].
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
    Uint4_L2 = cpp_distances.Metric.Uint4_L2
    Uint4_InnerProduct = cpp_distances.Metric.Uint4_InnerProduct
    Int4_L2 = cpp_distances.Metric.Int4_L2
    Int4_InnerProduct = cpp_distances.Metric.Int4_InnerProduct
    UInt7_L2 = cpp_distances.Metric.UInt7_L2
    UInt7_InnerProduct = cpp_distances.Metric.UInt7_InnerProduct

    def get_dtype(self):
        """
        Return the corresponding NumPy dtype for feature data with this metric.
        """
        if self in (Metric.FP32_L2, Metric.FP32_InnerProduct):
            return np.float32
        if self in (Metric.Uint8_L2, Metric.Uint8_InnerProduct, Metric.Uint4_L2, Metric.Uint4_InnerProduct, Metric.Int4_L2, Metric.Int4_InnerProduct):
            return np.uint8
        if self in (Metric.Int8_InnerProduct, Metric.Int8_L2, Metric.UInt7_InnerProduct, Metric.UInt7_L2):
            return np.int8
        if self in (Metric.FP16_L2, Metric.FP16_InnerProduct):
            return np.uint16
        if self == Metric.EVP_InnerProduct:
            return np.uint8
        return np.float32


class FloatSpace:
    """
    Configuration and computational space for vector distance calculations.

    Defines the dimensionality, metric, and CPU instruction set used
    to evaluate distances between vectors.
    """

    def __init__(self, float_space_cpp: cpp_distances.FloatSpace):
        self.float_space_cpp = float_space_cpp

    @classmethod
    def create(cls, dim: int, metric: Metric, instruction: InstructionSet = InstructionSet.Auto) -> "FloatSpace":
        """
        Create a new FloatSpace instance.

        :param dim: Dimensionality of the vector space.
        :param metric: Distance metric to use for comparisons.
        :param instruction: CPU instruction set (defaults to ``InstructionSet.Auto``).
        :return: Configured FloatSpace instance.
        """
        return FloatSpace(cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(metric)), cpp_cpu.InstructionSet(int(instruction))))

    def dim(self) -> int:
        """
        Return the dimensionality of the vector space.
        """
        return self.float_space_cpp.dim()

    def metric(self) -> cpp_distances.Metric:
        """
        Return the distance metric configured for this space.
        """
        return Metric(int(self.float_space_cpp.metric()))

    def get_data_size(self) -> int:
        """
        Return the memory size in bytes required per feature vector.
        """
        return self.float_space_cpp.get_data_size()

    def get_query_data_size(self) -> int:
        """
        Return the memory size in bytes required per query vector.
        """
        return self.float_space_cpp.get_query_data_size()

    def get_instruction(self) -> InstructionSet:
        """
        Return the active CPU instruction set.
        """
        return InstructionSet[self.float_space_cpp.get_instruction()]

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the distance between two feature vectors.

        :param vec1: First feature vector (1D NumPy array).
        :param vec2: Second feature vector (1D NumPy array).
        :return: Computed distance value.
        """
        return float(self.float_space_cpp.compute_distance(vec1, vec2))

    def compute_distances(self, query: np.ndarray, targets: np.ndarray, threads: int = 0) -> np.ndarray:
        """
        Calculate distances between query vector(s) and multiple target vectors.

        :param query: 1D query vector [D] or 2D batch of queries [Q, D] (C-contiguous).
        :param targets: 2D array of target vectors (shape: N x dim, C-contiguous).
        :param threads: Worker threads for the query batch (0 = all CPU cores, 1 = serial).
        :return: 2D NumPy array of distances (shape Q, N); a 1D query yields shape (1, N).
        """
        return self.float_space_cpp.compute_distances(query, targets, threads)

    def __repr__(self):
        return f"FloatSpace(size={self.get_data_size()} dim={self.dim()}, metric={self.metric()})"


def floats_to_fp16(floats: np.ndarray) -> np.ndarray:
    """
    Convert a 32-bit float NumPy array into half-precision (16-bit float stored as uint16).

    :param floats: NumPy array of float32 values.
    :return: NumPy array of uint16 values with the same shape.
    """
    arr = np.ascontiguousarray(floats, dtype=np.float32)
    res = cpp_distances.floats_to_fp16(arr)
    return res.reshape(arr.shape)


def fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit half-precision array (stored as uint16) back into a 32-bit float array.

    :param fp16_vals: NumPy array of uint16 values representing FP16 floats.
    :return: NumPy array of float32 values with the same shape.
    """
    res = cpp_distances.fp16_to_floats(fp16_vals)
    return res.reshape(fp16_vals.shape)
