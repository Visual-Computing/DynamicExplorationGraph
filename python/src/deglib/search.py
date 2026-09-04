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

class Searcher:
    """
    High-performance Stateless Searcher for fast single-query and batch-query execution.

    Performs end-to-end query transformation (quantization), graph search, and exact distance
    candidate refinement within C++ in a single native invocation, eliminating Python interpreter
    and bridge overhead in query loops.

    :param graph: DynamicExplorationGraph (Mutable or ReadOnly) instance.
    :param quantizer: Optional scalar or EVP quantizer instance (or integer non_zeros for EVP).
    :param refine_space: Optional FloatSpace for exact candidate reranking.
    :param refine_data: Optional base feature matrix for candidate reranking.
    """

    def __init__(
        self,
        graph,
        quantizer=None,
        refine_space: FloatSpace | None = None,
        refine_data: np.ndarray | None = None,
    ):
        cpp_graph = graph.dynamic_exploration_graph_cpp if hasattr(graph, "dynamic_exploration_graph_cpp") else graph
        cpp_refine_space = (
            refine_space.float_space_cpp if (refine_space is not None and hasattr(refine_space, "float_space_cpp")) else refine_space
        )
        cpp_quantizer = quantizer.quantizer_cpp if (quantizer is not None and hasattr(quantizer, "quantizer_cpp")) else quantizer

        self.searcher_cpp = cpp_search.Searcher(
            graph=cpp_graph,
            quantizer=cpp_quantizer,
            rerank_space=cpp_refine_space,
            base_vectors=refine_data,
        )

    def search(
        self,
        query: np.ndarray,
        k: int,
        eps: float = 0.1,
        rerank_factor: float = 1.0,
        threads: int = 1,
        return_distances: bool = False,
        unsorted: bool = False,
        ef: int = 0,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors of a single query vector (1D or 2D with 1 vector) or a batch of query vectors (2D).

        - If `query` represents a single vector (`ndim == 1` or `(1, dim)` or `(dim, 1)`):
          returns 1D array of shape `(k,)` (or tuple of 1D arrays if `return_distances` is True).
        - If `query` is a batch (`(N, dim)` with N > 1):
          returns 2D array of shape `(N, k)` (or tuple of 2D arrays if `return_distances` is True).

        :param query: Query array (1D vector or 2D batch).
        :param k: Number of top nearest neighbors to return per query.
        :param eps: Search expansion factor (controls speed vs. recall trade-off, default: 0.1).
        :param rerank_factor: Candidate expansion factor for exact reranking (default: 1.0).
        :param threads: Number of worker threads for batch search (default: 1).
        :param return_distances: If True, returns a tuple ``(indices, distances)``.
        :param unsorted: If True, returns candidates in heap order instead of ascending distance order.
        :return: NumPy array of nearest neighbor IDs, or tuple ``(indices, distances)`` if `return_distances` is True.
        """
        if query.ndim == 1 or (query.ndim == 2 and (query.shape[0] == 1 or query.shape[1] == 1)):
            flat_query = np.ascontiguousarray(query.ravel())
            return self.searcher_cpp.search(
                flat_query, int(k), float(eps), float(rerank_factor), return_distances, unsorted, int(ef)
            )
        elif query.ndim == 2:
            contiguous_queries = np.ascontiguousarray(query)
            return self.searcher_cpp.search_batch(
                contiguous_queries, int(k), float(eps), float(rerank_factor), int(threads), return_distances, unsorted, int(ef)
            )
        else:
            raise ValueError(f"query must be 1D or 2D NumPy array, got ndim={query.ndim} with shape {query.shape}")


def create_searcher(
    graph,
    quantizer=None,
    refine_space: FloatSpace | None = None,
    refine_data: np.ndarray | None = None,
) -> Searcher:
    """
    Factory function to create a Searcher instance.
    """
    return Searcher(
        graph=graph,
        quantizer=quantizer,
        refine_space=refine_space,
        refine_data=refine_data,
    )


__all__ = ["Filter", "rerank", "Searcher", "create_searcher"]
