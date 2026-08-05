import enum
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp


class Metric(enum.IntEnum):
    FP32_L2 = deglib_cpp.Metric.FP32_L2
    FP32_InnerProduct = deglib_cpp.Metric.FP32_InnerProduct
    Uint8_L2 = deglib_cpp.Metric.Uint8_L2
    FP16_InnerProduct = deglib_cpp.Metric.FP16_InnerProduct
    EVP_InnerProduct = deglib_cpp.Metric.EVP_InnerProduct

    def to_cpp(self) -> deglib_cpp.Metric:
        return deglib_cpp.Metric(int(self))

    def get_dtype(self):
        if self in (Metric.FP32_L2, Metric.FP32_InnerProduct):
            return np.float32
        elif self == Metric.Uint8_L2:
            return np.uint8
        elif self == Metric.FP16_InnerProduct:
            return np.float16
        else:
            return np.float32


class InstructionSet(enum.IntEnum):
    Auto = deglib_cpp.InstructionSet.Auto
    Scalar = deglib_cpp.InstructionSet.Scalar
    AVX2 = deglib_cpp.InstructionSet.AVX2
    AVX512 = deglib_cpp.InstructionSet.AVX512

    def to_cpp(self) -> deglib_cpp.InstructionSet:
        if self == InstructionSet.Auto:
            return deglib_cpp.InstructionSet.Auto
        elif self == InstructionSet.Scalar:
            return deglib_cpp.InstructionSet.Scalar
        elif self == InstructionSet.AVX2:
            return deglib_cpp.InstructionSet.AVX2
        elif self == InstructionSet.AVX512:
            return deglib_cpp.InstructionSet.AVX512
        else:
            raise ValueError(f"unknown instruction set: {self}")


class SpaceInterface(ABC):
    @abstractmethod
    def dim(self) -> int:
        return NotImplemented()

    @abstractmethod
    def metric(self) -> Metric:
        return NotImplemented()

    @abstractmethod
    def get_data_size(self) -> int:
        return NotImplemented()


class FloatSpace(SpaceInterface):
    def __init__(self, float_space_cpp: deglib_cpp.FloatSpace):
        """
        Create a FloatSpace.

        :param float_space_cpp: The cpp implementation of a float space
        """
        self.float_space_cpp = float_space_cpp

    @classmethod
    def create(cls, dim: int, metric: Metric, instruction: InstructionSet = InstructionSet.Auto) -> 'FloatSpace':
        """
        Create a FloatSpace.

        :param dim: The dimension of the space
        :param metric: Metric to calculate distances between features
        :param instruction: CPU Instruction Set to use for SIMD vector distance computation
        """
        return FloatSpace(deglib_cpp.FloatSpace(dim, metric.to_cpp(), instruction.to_cpp()))

    def dim(self) -> int:
        """
        :return: the dimensionality of the space
        """
        return self.float_space_cpp.dim()

    def metric(self) -> deglib_cpp.Metric:
        """
        :return: the metric that can be used to calculate distances between features
        """
        return Metric(int(self.float_space_cpp.metric()))

    def get_data_size(self) -> int:
        """
        :returns: number of features.
        """
        return self.float_space_cpp.get_data_size()

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

    def to_cpp(self) -> deglib_cpp.FloatSpace:
        return self.float_space_cpp

    def __repr__(self):
        return f'FloatSpace(size={self.get_data_size()} dim={self.dim()}, metric={self.metric()})'
