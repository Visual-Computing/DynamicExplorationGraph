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


class Reranker:
    """
    High-performance candidate reranker re-evaluating graph candidates with exact distances.

    Indexing Contract:
    The Reranker evaluates candidate index `c` directly as row `base_vectors[c]`.

    1. Normal case (Static Dataset with Searcher):
       Row i corresponds to the i-th vector given to the graph builder. Simply pass the same
       feature matrix (or its FP16 cast) that was passed to `build_from_data`. No reordering needed!
    2. Dynamic or Custom Labels case:
       If external labels do not match builder input rows (e.g. dynamic deletions or arbitrary IDs),
       row i must contain the vector for label `graph.get_labels()[i]`.
    3. Standalone usage (without Searcher):
       `base_vectors` matches whatever candidate ID space you pass to `rerank()`.

    :param space: FloatSpace defining the distance metric and vector dimensionality.
    :param base_vectors: 2D NumPy array of dataset vectors (shape: N_base x dim).
    """

    def __init__(
        self,
        space: FloatSpace,
        base_vectors: np.ndarray,
        transformations: list | None = None,
    ):
        cpp_space = space.float_space_cpp if hasattr(space, "float_space_cpp") else space
        contiguous_base = np.ascontiguousarray(base_vectors)
        cpp_transformations = []
        if transformations:
            for t in transformations:
                cpp_transformations.append(t.transformation_cpp if hasattr(t, "transformation_cpp") else t)
        self.reranker_cpp = cpp_search.Reranker(cpp_space, contiguous_base, cpp_transformations)
        self.base_vectors = contiguous_base

    def get_num_base_vectors(self) -> int:
        """Number of base vectors the reranker scores candidates against."""
        return self.reranker_cpp.get_num_base_vectors()

    def get_po(self) -> int:
        """Prefetch lookahead offset for candidate feature vectors (0 disables rerank prefetching)."""
        return self.reranker_cpp.get_po()

    def get_pl(self) -> int:
        """Cache lines prefetched per feature vector (0 prefetches the whole vector)."""
        return self.reranker_cpp.get_pl()

    def set_prefetch(self, po: int, pl: int) -> None:
        """Override the rerank prefetch parameters, bypassing auto-tuning.

        Prefetching is enabled by default (``po=8``, ``pl=0``), which is the right starting point
        whenever the base vectors do not fit in cache. Pass ``po=0`` to score candidates without
        issuing any prefetch, for example when the working set is already cache-resident.

        :param po: Lookahead offset in candidates; values <= 0 disable prefetching.
        :param pl: Cache lines per candidate; values <= 0 prefetch the whole vector.
        """
        self.reranker_cpp.set_prefetch(int(po), int(pl))

    def optimize(self, sample_count: int = 50, num_candidates: int = 100, k: int = 10) -> None:
        """Auto-tune the reranker's prefetch parameters (po, pl) by timing reranking over sampled base vectors.

        The reranker samples its own base vectors as queries, so no external query buffer is needed.
        Call once after construction to tune for the expected operating point before reranking.

        :param sample_count: Number of base vectors used as sample queries (capped at the base count).
        :param num_candidates: Candidates scored per query (clamped to the base count).
        :param k: Candidates kept per query after reranking (clamped to num_candidates).
        """
        self.reranker_cpp.optimize(int(sample_count), int(num_candidates), int(k))

    def rerank(
        self,
        queries: np.ndarray,
        candidate_indices: np.ndarray,
        k_top: int = 0,
        num_threads: int = 0,
        return_distances: bool = False,
        unsorted: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Re-evaluate and rank candidate vector indices for each query against the base vectors.

        :param queries: 2D NumPy array of query vectors (shape: N_queries x dim).
        :param candidate_indices: 2D uint32 NumPy array of candidate indices per query (shape: N_queries x K_candidates).
        :param k_top: Number of top nearest candidates to return per query (0 returns all candidates).
        :param num_threads: Number of worker threads (0 uses all available CPU cores).
        :param return_distances: If True, returns a tuple ``(indices, distances)``.
        :param unsorted: If True, skips sorting candidate results by distance.
        :return: 2D uint32 NumPy array of candidate IDs, or tuple ``(indices, distances)`` if `return_distances` is True.
        """
        return self.reranker_cpp.rerank(
            np.ascontiguousarray(queries),
            candidate_indices.astype(np.uint32, copy=False),
            int(k_top),
            int(num_threads),
            bool(return_distances),
            bool(unsorted),
        )


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
    :param candidate_indices: 2D uint32 NumPy array of candidate indices per query (shape: N_queries x K_candidates). Each value ``c`` is a row of ``base_vectors``.
    :param base_vectors: 2D array of dataset vectors (shape: N_base x dim). If None, defaults to `queries`. Candidate ``c`` is scored as ``base_vectors[c]``.
    :param k_top: Number of top nearest candidates to return per query (0 returns all candidates sorted).
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    :param return_distances: If True, returns a tuple ``(indices, distances)``.
    :param unsorted: If True, skips sorting candidate results by distance.
    :return: 2D uint32 NumPy array of candidate IDs, or tuple ``(indices, distances)`` if `return_distances` is True.
    """
    target_vectors = queries if base_vectors is None else base_vectors
    return Reranker(space, target_vectors).rerank(
        queries, candidate_indices, k_top, num_threads, return_distances, unsorted
    )


class Searcher:
    """
    High-performance Stateless Searcher for fast single-query and batch-query execution.

    Performs end-to-end query transformation (quantization), graph search, and exact distance
    candidate reranking within C++ in a single native invocation, eliminating Python interpreter
    and bridge overhead in query loops. Any transformations or quantization are derived directly
    from the attached graph.

    :param graph: DynamicExplorationGraph (Mutable or ReadOnly) instance.
    :param rerank_space: Optional FloatSpace for exact candidate reranking.
    :param rerank_data: Optional base feature matrix in the graph's internal index order (0 .. size()-1, see ``graph.get_labels()``). Valid for static graphs.
    :param rerank_transformations: Optional list of transformations applied to queries before candidate reranking.
    :param normalize_query: If True, input queries are L2-normalized once before graph search and reranking.
    """

    def __init__(
        self,
        graph,
        rerank_space: FloatSpace | None = None,
        rerank_data: np.ndarray | None = None,
        rerank_transformations: list | None = None,
        normalize_query: bool = False,
    ):
        cpp_graph = graph.dynamic_exploration_graph_cpp if hasattr(graph, "dynamic_exploration_graph_cpp") else graph
        cpp_rerank_space = (
            rerank_space.float_space_cpp
            if (rerank_space is not None and hasattr(rerank_space, "float_space_cpp"))
            else rerank_space
        )
        cpp_rerank_transforms = []
        if rerank_transformations:
            for t in rerank_transformations:
                cpp_rerank_transforms.append(t.transformation_cpp if hasattr(t, "transformation_cpp") else t)

        self.searcher_cpp = cpp_search.Searcher(
            graph=cpp_graph,
            rerank_space=cpp_rerank_space,
            base_vectors=rerank_data,
            rerank_transformations=cpp_rerank_transforms,
            normalize_query=normalize_query,
        )

    @property
    def normalize_query(self) -> bool:
        """Whether input queries are L2-normalized prior to search and rerank."""
        return self.searcher_cpp.normalize_query

    @normalize_query.setter
    def normalize_query(self, val: bool) -> None:
        self.searcher_cpp.normalize_query = bool(val)

    def optimize(
        self,
        n_clusters: int = 256,
        n_iter: int = 20,
        sample_size: int = 0,
        k: int = 100,
        seed: int = 7,
        num_threads: int = 1,
    ) -> None:
        """Select entry vertices via k-means medoids.

        :param n_clusters: Number of entry vertices to select (maximum 512).
        :param n_iter: Number of k-means iterations.
        :param sample_size: Number of vertices sampled for clustering, 0 selects 3% of the graph size.
        :param k: Expected result count; sets the reranker and traversal prefetch tuning operating point.
        :param seed: Random seed for sampling and centroid init.
        :param num_threads: Number of worker threads, 0 selects a library default.
        """
        self.searcher_cpp.optimize(int(n_clusters), int(n_iter), int(sample_size), int(k), int(seed), int(num_threads))

    def search(
        self,
        query: np.ndarray,
        k: int,
        eps_or_ef: float = 0.1,
        rerank_factor: float = 1.0,
        threads: int = 1,
        return_distances: bool = False,
        unsorted: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors of a single query vector (1D or 2D with 1 vector) or a batch of query vectors (2D).

        - If `query` represents a single vector (`ndim == 1` or `(1, dim)` or `(dim, 1)`):
          returns 1D array of shape `(k,)` (or tuple of 1D arrays if `return_distances` is True).
        - If `query` is a batch (`(N, dim)` with N > 1):
          returns 2D array of shape `(N, k)` (or tuple of 2D arrays if `return_distances` is True).

        Entry vertices come solely from ``optimize()``.

        :param query: Query array (1D vector or 2D batch).
        :param k: Number of top nearest neighbors to return per query.
        :param eps_or_ef: Exploration parameter (controls speed vs. recall trade-off, default: 0.1).
            Values >= 1.0 are treated as fixed candidate pool size (``ef``, HNSW-style search).
            Values < 1.0 are treated as relative exploration distance margin (``eps``).
        :param rerank_factor: Candidate expansion factor for exact reranking (default: 1.0).
            The initial graph search retrieves ``ceil(k * rerank_factor)`` candidates.
        :param threads: Number of worker threads for batch search (default: 1).
        :param return_distances: If True, returns a tuple ``(indices, distances)``.
        :param unsorted: If True, returns candidates in heap order instead of ascending distance order.
        :return: NumPy array of nearest neighbor IDs, or tuple ``(indices, distances)`` if `return_distances` is True.
        """
        return self.searcher_cpp.search(
            query,
            int(k),
            float(eps_or_ef),
            float(rerank_factor),
            int(threads),
            bool(return_distances),
            bool(unsorted),
        )


def create_searcher(
    graph,
    rerank_space: FloatSpace | None = None,
    rerank_data: np.ndarray | None = None,
    rerank_transformations: list | None = None,
    normalize_query: bool = False,
) -> Searcher:
    """
    Factory function to create a Searcher instance.

    :param graph: DynamicExplorationGraph (Mutable or ReadOnly) instance.
    :param rerank_space: Optional FloatSpace for exact candidate reranking.
    :param rerank_data: Optional base feature matrix. In standard static workflows, simply pass the same array used for building. If labels were permuted/deleted, pass features aligned with ``graph.get_labels()``.
    :param rerank_transformations: Optional list of transformations applied to queries before candidate reranking.
    :param normalize_query: If True, input queries are L2-normalized once before graph search and reranking.
    """
    return Searcher(
        graph=graph,
        rerank_space=rerank_space,
        rerank_data=rerank_data,
        rerank_transformations=rerank_transformations,
        normalize_query=normalize_query,
    )


__all__ = ["Filter", "Reranker", "rerank", "Searcher", "create_searcher"]
