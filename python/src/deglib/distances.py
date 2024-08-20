import enum
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

import deglib_cpp


class Metric(enum.IntEnum):
    L2 = deglib_cpp.Metric.L2
    InnerProduct = deglib_cpp.Metric.InnerProduct
    L2_Uint8 = deglib_cpp.Metric.L2_Uint8

    def to_cpp(self) -> deglib_cpp.Metric:
        if self == Metric.L2:
            return deglib_cpp.Metric.L2
        elif self == Metric.InnerProduct:
            return deglib_cpp.Metric.InnerProduct
        elif self == Metric.L2_Uint8:
            return deglib_cpp.Metric.L2_Uint8

    def get_dtype(self):
        if self in (Metric.L2, Metric.InnerProduct):
            dtype = np.float32
        elif self == Metric.L2_Uint8:
            dtype = np.uint8
        else:
            raise ValueError('unknown metric: {}'.format(self))
        return dtype


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
    def create(cls, dim: int, metric: Metric) -> Self:
        """
        Create a FloatSpace.

        :param dim: The dimension of the space
        :param metric: Metric to calculate distances between features
        """
        return FloatSpace(deglib_cpp.FloatSpace(dim, metric.to_cpp()))

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

    def to_cpp(self) -> deglib_cpp.FloatSpace:
        return self.float_space_cpp

    def __repr__(self):
        return f'FloatSpace(size={self.get_data_size()} dim={self.dim()}, metric={self.metric()})'
