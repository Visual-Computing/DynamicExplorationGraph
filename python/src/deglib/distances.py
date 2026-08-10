import enum
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp.distances as cpp_distances
from deglib.cpu import InstructionSet


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
    def create(cls, dim: int, metric: Metric, instruction: InstructionSet = InstructionSet.Auto) -> 'FloatSpace':
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

    def rerank(
        self,
        queries: np.ndarray,
        candidate_indices: np.ndarray,
        base_vectors: np.ndarray | None = None,
        k_top: int = 0,
        num_threads: int = 0
    ) -> np.ndarray:
        """
        Reranks candidate vectors for queries using SIMD distance computations in C++.

        :param queries: 2D array of query vectors (shape: N_queries x D)
        :param candidate_indices: 2D uint32 array of candidate feature IDs for each query (shape: N_queries x K_cand)
        :param base_vectors: 2D array of dataset feature vectors (shape: N_base x D). If None, defaults to `queries`.
        :param k_top: Number of top nearest candidates to return per query (default 0 returns all K_cand sorted).
        :param num_threads: Number of threads (0 for hardware concurrency).
        :return: 2D uint32 NumPy array of shape (N_queries, k_top) containing top candidate IDs sorted by similarity.
        """
        return self.float_space_cpp.rerank(
            queries,
            candidate_indices.astype(np.uint32, copy=False),
            base_vectors,
            k_top,
            num_threads
        )

    def to_cpp(self) -> cpp_distances.FloatSpace:
        return self.float_space_cpp

    def __repr__(self):
        return f'FloatSpace(size={self.get_data_size()} dim={self.dim()}, metric={self.metric()})'


def quantize_batch(vectors: np.ndarray, non_zeros: int, num_threads: int = 0) -> np.ndarray:
    """
    Quantize float32 or float16/uint16 vectors to byte-packed EVP format using C++ multi-threading.
    """
    if vectors.dtype == np.float16:
        vectors = vectors.view(np.uint16)
    return cpp_distances.quantize_batch(vectors, non_zeros, num_threads)


def floats_to_fp16(floats: np.ndarray) -> np.ndarray:
    """
    Convert float32 array to FP16 (uint16_t) representation in C++.
    """
    return cpp_distances.floats_to_fp16(floats)


def fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray:
    """
    Convert FP16 (uint16_t) array to float32 representation in C++.
    """
    return cpp_distances.fp16_to_floats(fp16_vals)


