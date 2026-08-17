import enum
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp.distances as cpp_distances
from deglib.cpu import InstructionSet

__all__ = ["Metric", "FloatSpace", "InstructionSet", "floats_to_fp16", "fp16_to_floats"]


class Metric(enum.IntEnum):
    FP32_L2 = cpp_distances.Metric.FP32_L2
    FP32_InnerProduct = cpp_distances.Metric.FP32_InnerProduct
    Uint8_L2 = cpp_distances.Metric.Uint8_L2
    FP16_InnerProduct = cpp_distances.Metric.FP16_InnerProduct
    EVP_InnerProduct = cpp_distances.Metric.EVP_InnerProduct

    def to_cpp(self) -> cpp_distances.Metric:
        return cpp_distances.Metric(int(self))

    def get_dtype(self):
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
    def __init__(self, float_space_cpp: cpp_distances.FloatSpace):
        """
        Create a FloatSpace.

        :param float_space_cpp: The cpp implementation of a float space
        """
        self.float_space_cpp = float_space_cpp

    @classmethod
    def create(cls, dim: int, metric: Metric, instruction: InstructionSet = InstructionSet.Auto) -> "FloatSpace":
        """
        Create a FloatSpace.

        :param dim: The dimension of the space
        :param metric: Metric to calculate distances between features
        :param instruction: CPU Instruction Set to use for SIMD vector distance computation
        """
        return FloatSpace(cpp_distances.FloatSpace(dim, metric.to_cpp(), instruction.to_cpp()))

    def dim(self) -> int:
        """
        :return: the dimensionality of the space
        """
        return self.float_space_cpp.dim()

    def metric(self) -> cpp_distances.Metric:
        """
        :return: the metric that can be used to calculate distances between features
        """
        return Metric(int(self.float_space_cpp.metric()))

    def get_data_size(self) -> int:
        """
        :returns: number of features.
        """
        return self.float_space_cpp.get_data_size()

    def get_instruction(self) -> InstructionSet:
        """
        :return: the CPU instruction set used for SIMD distance computation
        """
        return InstructionSet[self.float_space_cpp.get_instruction()]

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the distance between two feature vectors using the configured metric and SIMD instruction set.
        """
        return float(self.float_space_cpp.compute_distance(vec1, vec2))

    def compute_distances(self, query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates a 1D NumPy array of distances between query vector and a batch/matrix of target vectors.
        """
        return self.float_space_cpp.compute_distances(query, targets)

    def to_cpp(self) -> cpp_distances.FloatSpace:
        return self.float_space_cpp

    def __repr__(self):
        return f"FloatSpace(size={self.get_data_size()} dim={self.dim()}, metric={self.metric()})"


def floats_to_fp16(floats: np.ndarray) -> np.ndarray:
    """
    Convert float32 array to FP16 (uint16_t) representation in C++.
    """
    arr = np.ascontiguousarray(floats, dtype=np.float32)
    res = cpp_distances.floats_to_fp16(arr)
    return res.reshape(arr.shape)


def fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray:
    """
    Convert FP16 (uint16_t) array to float32 representation in C++.
    """
    res = cpp_distances.fp16_to_floats(fp16_vals)
    return res.reshape(fp16_vals.shape)
