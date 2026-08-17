from typing import Optional, Union
import numpy as np

import deglib_cpp.search as cpp_search

from deglib.distances import FloatSpace
from deglib.utils import assure_contiguous


class Filter:
    def __init__(self, valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1):
        """
        Creates an object that can be used to limit the set of possible results.

        :param valid_labels: A numpy array with dtype int32, that contains all labels that can be returned.
                              All other labels will not be included in the result set.
        :param max_value: The maximum value in valid_labels. Will be computed automatically, if set to -1.
        :param max_label_count: The size of the whole dataset. If not set, the size of the search graph is assumed.
        """
        self.valid_labels = valid_labels
        if max_value < 0:
            max_value = np.max(valid_labels)
        self.max_value = max_value
        self.max_label_count = max_label_count

    def create_filter_obj(self, graph_size: int) -> cpp_search.Filter:
        """
        Only for internal use.
        Creates a filter object that can be used to limit the set of possible results.
        """
        valid_labels = assure_contiguous(self.valid_labels.astype(np.int32, copy=False), "filter_labels")
        filter_obj = None
        if valid_labels is not None:
            max_label_count = self.max_label_count
            if max_label_count <= 0:
                max_label_count = graph_size
            filter_obj = cpp_search.create_filter(valid_labels, self.max_value, max_label_count)

        return filter_obj

    @staticmethod
    def create_filter(filter_labels: Union[None, np.ndarray, "Filter"], graph_size: int) -> Optional[cpp_search.Filter]:
        if filter_labels is None:
            return None
        if isinstance(filter_labels, np.ndarray):
            filter_labels = Filter(filter_labels)
        if not isinstance(filter_labels, Filter):
            raise TypeError("filter_labels must be a None, numpy array or Filter, got {}".format(type(filter_labels)))
        return filter_labels.create_filter_obj(graph_size)


def rerank(
    space: FloatSpace,
    queries: np.ndarray,
    candidate_indices: np.ndarray,
    base_vectors: np.ndarray | None = None,
    k_top: int = 0,
    num_threads: int = 0,
    return_distances: bool = False,
    unsorted: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Reranks candidate vectors for queries using SIMD distance computations in C++.

    :param space: FloatSpace used to compute distances (metric + SIMD instruction set).
    :param queries: 2D array of query vectors (shape: N_queries x D)
    :param candidate_indices: 2D uint32 array of candidate feature IDs for each query (shape: N_queries x K_cand)
    :param base_vectors: 2D array of dataset feature vectors (shape: N_base x D). If None, defaults to `queries`.
    :param k_top: Number of top nearest candidates to return per query (default 0 returns all K_cand sorted).
    :param num_threads: Number of threads (0 for hardware concurrency).
    :param return_distances: If True, returns a tuple `(indices, distances)` containing candidate IDs and distance scores.
    :param unsorted: If True, skips sorting the resulting candidates.
    :return: 2D uint32 NumPy array of candidate IDs, or tuple `(indices, distances)` if return_distances is True.
    """
    cpp_space = space.to_cpp() if hasattr(space, "to_cpp") else space
    return cpp_search.rerank(
        cpp_space,
        queries,
        candidate_indices.astype(np.uint32, copy=False),
        base_vectors,
        k_top,
        num_threads,
        return_distances,
        unsorted,
    )


__all__ = ["Filter", "rerank"]
