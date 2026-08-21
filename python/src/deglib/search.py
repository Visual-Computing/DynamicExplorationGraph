from typing import Optional, Union
import numpy as np

import deglib_cpp.search as cpp_search

from deglib.distances import FloatSpace
from deglib.utils import assure_contiguous


class Filter:
    """
    Search filter used to restrict nearest-neighbor search results to a subset of valid labels.

    :param valid_labels: 1D NumPy array of valid int32 labels/IDs that are allowed in the result set.
    :param max_value: Maximum label value in `valid_labels`. Computed automatically if negative.
    :param max_label_count: Total size of dataset. Defaults to the search graph size if not provided.
    """

    def __init__(self, valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1):
        self.valid_labels = valid_labels
        if max_value < 0:
            max_value = np.max(valid_labels)
        self.max_value = max_value
        self.max_label_count = max_label_count

    def create_filter_obj(self, graph_size: int) -> cpp_search.Filter:
        """
        Create a backend filter representation for the specified graph size.
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
        """
        Helper method to construct a filter object from an array, Filter instance, or None.
        """
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
    Re-evaluate and rank candidate vector indices for each query.

    Useful in two-stage retrieval pipelines where a fast first-stage search provides
    a candidate pool that is re-scored with exact distance calculations.

    :param space: FloatSpace instance defining the distance metric and dimensionality.
    :param queries: 2D NumPy array of query vectors (shape: N_queries x dim).
    :param candidate_indices: 2D uint32 NumPy array of candidate indices per query (shape: N_queries x K_candidates).
    :param base_vectors: 2D array of dataset vectors (shape: N_base x dim). If None, defaults to `queries`.
    :param k_top: Number of top nearest candidates to return per query (0 returns all candidates sorted).
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    :param return_distances: If True, returns a tuple ``(indices, distances)``.
    :param unsorted: If True, skips sorting candidate results by distance.
    :return: 2D uint32 NumPy array of candidate IDs, or tuple ``(indices, distances)`` if `return_distances` is True.
    """
    cpp_space = space.float_space_cpp if hasattr(space, "float_space_cpp") else space
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
