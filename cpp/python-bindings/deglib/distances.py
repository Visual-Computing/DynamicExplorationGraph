import enum

import deglib_cpp


class Metric(enum.IntEnum):
    L2 = 1
    InnerProduct = 2

    def to_cpp(self) -> deglib_cpp.Metric:
        if self == Metric.L2:
            return deglib_cpp.Metric.L2
        elif self == Metric.InnerProduct:
            return deglib_cpp.Metric.InnerProduct


class FloatSpace:
    def __init__(self, dim: int, metric: Metric, float_space_cpp=None):
        if float_space_cpp is None:
            float_space_cpp = deglib_cpp.FloatSpace(dim, metric.to_cpp())
        self.float_space_cpp = float_space_cpp

    def dim(self) -> int:
        return self.float_space_cpp.dim()
