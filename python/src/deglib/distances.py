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
    - ``FP16_InnerProduct``: Inner product distance for 16-bit half-precision floating point vectors.
    - ``EVP_InnerProduct``: Inner product distance for quantized byte-packed (EVP) vectors.
    """

    FP32_L2 = cpp_distances.Metric.FP32_L2
    FP32_InnerProduct = cpp_distances.Metric.FP32_InnerProduct
    Uint8_L2 = cpp_distances.Metric.Uint8_L2
    FP16_InnerProduct = cpp_distances.Metric.FP16_InnerProduct
    EVP_InnerProduct = cpp_distances.Metric.EVP_InnerProduct

    def get_dtype(self):
        """
        Return the corresponding NumPy dtype for feature data with this metric.
        """
        if self in (Metric.FP32_L2, Metric.FP32_InnerProduct):
            return np.float32
        elif self == Metric.Uint8_L2:
            return np.uint8
        elif self == Metric.FP16_InnerProduct:
            return np.uint16
        elif self == Metric.EVP_InnerProduct:
            return np.uint8
        else:
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
        return FloatSpace(
            cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(metric)), cpp_cpu.InstructionSet(int(instruction)))
        )

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

    def compute_distances(self, query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculate distances between a query vector and multiple target vectors.

        :param query: 1D query vector.
        :param targets: 2D array of target vectors (shape: N x dim).
        :return: 1D NumPy array of computed distances.
        """
        return self.float_space_cpp.compute_distances(query, targets)

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
