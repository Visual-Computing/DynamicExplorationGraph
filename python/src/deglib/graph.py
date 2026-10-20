import os
import multiprocessing
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import deglib_cpp
import deglib_cpp.distances as cpp_distances
import deglib_cpp.search as cpp_search
import pathlib

from .distances import FloatSpace, Metric
from .cpu import InstructionSet
from .search import Filter
from .utils import assure_array, InvalidShapeException


class DynamicExplorationGraph:
    """
    Public DynamicExplorationGraph Facade for End Users.
    All public methods accept and return external_labels (User Object IDs).
    Internal calls map transparently between external_label and internal_index.

    This is the single, minimal graph class. Graph construction and mutation
    must be performed via GraphBuilder.
    """

    def __init__(self, graph_cpp: deglib_cpp.DynamicExplorationGraph):
        self.dynamic_exploration_graph_cpp = graph_cpp

    def size(self) -> int:
        """
        :return: the number of vertices in the graph
        """
        return self.dynamic_exploration_graph_cpp.size()

    def get_edges_per_vertex(self) -> int:
        """
        :return: the number of edges of each vertex
        """
        return self.dynamic_exploration_graph_cpp.get_edges_per_vertex()

    def get_po(self) -> int:
        """
        :return: the traversal feature prefetch offset (candidates prefetched ahead)
        """
        return self.dynamic_exploration_graph_cpp.get_po()

    def get_pl(self) -> int:
        """
        :return: the traversal feature prefetch look-ahead (cache lines per feature)
        """
        return self.dynamic_exploration_graph_cpp.get_pl()

    def get_nl(self) -> int:
        """
        :return: the traversal neighbor prefetch look-ahead
        """
        return self.dynamic_exploration_graph_cpp.get_nl()

    def get_feature_space(self) -> FloatSpace:
        """
        :return: the feature space
        """
        return FloatSpace(float_space_cpp=self.dynamic_exploration_graph_cpp.get_feature_space())

    def has_vertex(self, external_label: int) -> bool:
        """
        :param external_label: The external label to check for existence.
        :returns: whether the given external label is present in the graph.
        """
        return self.dynamic_exploration_graph_cpp.has_vertex(external_label)

    def is_mutable(self) -> bool:
        """
        :return: whether the graph is mutable (can be modified via GraphBuilder)
        """
        return self.dynamic_exploration_graph_cpp.is_mutable()

    def get_neighbors(
        self, external_label: Union[int, np.ndarray, list, tuple], threads: int = 1
    ) -> Union[List[int], np.ndarray]:
        """
        Get the neighbor external labels of the given external label(s).

        :param external_label: The external label to get the neighbors of, or an array/list of external labels.
        :param threads: The number of worker threads to use for batch retrieval.
        :returns: List of neighbor external labels (for single label) or 2D array of shape (N, edges) for batch.
        """
        if isinstance(external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(external_label, dtype=np.uint32)
            if arr.ndim == 2:
                arr = arr[:, 0]
            threads = get_num_useful_threads(threads, arr.shape[0])
            return self.dynamic_exploration_graph_cpp.get_neighbors_batch(arr, threads)
        return self.dynamic_exploration_graph_cpp.get_neighbors(int(external_label))

    def get_neighbors_batch(
        self, external_labels: Union[np.ndarray, list, tuple], threads: int = 1
    ) -> np.ndarray:
        """
        Get neighbor external labels for a batch of external labels.

        :param external_labels: 1D array or sequence of external labels.
        :param threads: The number of worker threads to use.
        :returns: 2D numpy array of shape (N, edges_per_vertex).
        """
        arr = np.ascontiguousarray(external_labels, dtype=np.uint32)
        if arr.ndim == 2:
            arr = arr[:, 0]
        threads = get_num_useful_threads(threads, arr.shape[0])
        return self.dynamic_exploration_graph_cpp.get_neighbors_batch(arr, threads)

    def has_transformations(self) -> bool:
        """
        :return: whether the graph has associated feature transformations / quantizers
        """
        return self.dynamic_exploration_graph_cpp.has_transformations()

    def get_transformations(self):
        """
        :return: the list of associated FeatureTransformation objects
        """
        return self.dynamic_exploration_graph_cpp.get_transformations()

    def search(
        self,
        query: np.ndarray,
        eps_or_ef: float = 0.0,
        k: int = 10,
        filter_labels: Union[None, np.ndarray, Filter] = None,
        max_distance_computation_count: int = 0,
        threads: int = 0,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Search for nearest neighbors of query vector(s).

        :param query: Query feature vector(s) as numpy array.
        :param eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),
                          values < 1.0 are treated as relative distance margin (eps). Default: 0.0.
        :param k: The number of results to return per query.
        :param filter_labels: Filter for labels to include.
        :param max_distance_computation_count: Distance computation budget limit.
        :param threads: Number of parallel worker threads.
        :param return_distances: If True, returns (indices, distances). If False, returns only indices.
        :param unsorted: If True, returns candidates in unsorted order instead of ascending distance order.
        :returns: (indices, distances) tuple if return_distances is True, otherwise indices array.
                  If fewer than k results are found, unfilled elements are padded with
                  np.iinfo(np.uint32).max (for indices) and np.nan / infinity (for distances).
        """
        filter_obj = None if filter_labels is None else Filter.create_filter(filter_labels, self.size())

        if isinstance(query, np.ndarray) and query.flags["C_CONTIGUOUS"]:
            contiguous_query = query
        else:
            contiguous_query = np.ascontiguousarray(query)

        if contiguous_query.ndim == 1:
            return self.dynamic_exploration_graph_cpp.search(
                contiguous_query, eps_or_ef, k, filter_obj, max_distance_computation_count, return_distances, unsorted
            )
        elif contiguous_query.ndim == 2:
            threads = get_num_useful_threads(threads, contiguous_query.shape[0])
            return self.dynamic_exploration_graph_cpp.search_batch(
                contiguous_query, eps_or_ef, k, filter_obj, max_distance_computation_count, threads, return_distances, unsorted
            )
        else:
            raise InvalidShapeException("invalid query shape: {}".format(query.shape))

    def explore(
        self,
        entry_external_label: Union[int, np.ndarray, list],
        k: int,
        max_distance_computation_count: int = 0,
        eps_or_ef: float = 0.0,
        include_entry: bool = True,
        threads: int = 1,
        filter_labels: Union[None, np.ndarray, Filter] = None,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        An exploration for similar elements, limited by max_distance_computation_count.

        Handles both single entry labels (int) and batch entry arrays (ndarray/list).

        :param entry_external_label: The external label of the vertex to start exploration from.
        :param k: The number of similar feature vectors to return.
        :param max_distance_computation_count: Limit the number of distance calculations.
        :param eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),
                          values < 1.0 are treated as relative distance margin (eps). Default: 0.0.
        :param include_entry: If True, the entry vertex is included in the result set.
        :param threads: The number of threads to use for parallel processing.
        :param filter_labels: Labels filter.
        :param return_distances: If True, returns (indices, distances). If False, returns only indices.
        :param unsorted: If True, returns candidates in order instead of ascending distance order.
        :returns: (indices, distances) tuple if return_distances is True, otherwise indices array.
                  If fewer than k results are found, unfilled elements are padded with
                  np.iinfo(np.uint32).max (for indices) and np.nan / infinity (for distances).
        """
        if k > self.size():
            warnings.warn(
                "k={} is larger than number of vertices in graph={}. Setting k={}".format(k, self.size(), self.size())
            )
            k = self.size()

        if isinstance(entry_external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(entry_external_label, dtype=np.uint32)
            if arr.ndim == 2:
                arr = arr[:, 0]
            filter_obj = Filter.create_filter(filter_labels, self.size())
            threads = get_num_useful_threads(threads, arr.shape[0])
            return self.dynamic_exploration_graph_cpp.explore_batch(
                arr,
                k,
                max_distance_computation_count,
                eps_or_ef,
                include_entry,
                filter_obj,
                threads,
                return_distances,
                unsorted,
            )
        else:
            indices, distances = self.dynamic_exploration_graph_cpp.explore(
                int(entry_external_label), k, max_distance_computation_count, eps_or_ef, include_entry
            )
            if return_distances:
                return indices, distances
            return indices

    def has_path(
        self,
        entry_external_labels: Union[int, np.ndarray, list],
        to_external_label: int,
        eps: float = 0.0,
        k: int = 1,
        return_distances: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Greedy best-first traversal towards a stored vertex, returning the traversed path.

        The target vertex's own feature vector is used as the query, so the traversal follows the
        graph's greedy structure from the entry vertices until the target is reached. With
        eps = 0.0 and k = 1 this is a pure greedy descent; larger eps widen the exploration radius
        and allow escaping local optima at the cost of more distance computations.

        :param entry_external_labels: External label(s) of the vertices to start the traversal from.
        :param to_external_label: External label of the vertex the path should end at.
        :param eps: Exploration factor widening the search radius. Default: 0.0 (greedy descent).
        :param k: Size of the result list driving the exploration radius. Default: 1.
        :param return_distances: If True, returns (path, distances). If False, returns only the path.
        :returns: (path, distances) tuple if return_distances is True, otherwise the path array.
                  The path is ordered from the target back to the entry vertex. Both arrays are
                  empty if the target was not reached within the given budget.
        """
        if isinstance(entry_external_labels, (int, np.integer)):
            entries = [int(entry_external_labels)]
        else:
            entries = [int(label) for label in entry_external_labels]
        return self.dynamic_exploration_graph_cpp.has_path(
            entries, int(to_external_label), float(eps), int(k), return_distances
        )

    def save_graph(self, path: pathlib.Path | str):
        """
        Save graph to specified file. Creates necessary directories.

        :param path: The path where to save the file.
        """
        self.dynamic_exploration_graph_cpp.save_graph(str(path))

    def get_entry_vertex_indices(self) -> list[int]:
        """Returns the list of internal entry vertex indices."""
        return self.dynamic_exploration_graph_cpp.get_entry_vertex_indices()

    def set_entry_vertex_indices(self, indices: list[int] | np.ndarray):
        """Sets the list of internal entry vertex indices."""
        self.dynamic_exploration_graph_cpp.set_entry_vertex_indices(list(indices))

    def optimize(self, sample_count: int = 50, k: int = 100, ef: int = 200, seed: int = 7) -> None:
        """Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.

        The graph self-samples its own stored features as queries, so no external query buffer is
        needed. Call once after construction to tune traversal for the expected operating point.

        :param sample_count: Number of vertices sampled as queries (capped at the graph size).
        :param k: Result count per traversal (the expected query operating point).
        :param ef: Beam width per traversal; clamped to at least k.
        :param seed: Random seed for query sampling.
        """
        self.dynamic_exploration_graph_cpp.optimize(int(sample_count), int(k), int(ef), int(seed))

    def to_readonly(
        self, feature_space: Optional[FloatSpace] = None, custom_features: Optional[np.ndarray] = None
    ) -> "DynamicExplorationGraph":
        """
        Create a read-only graph from the given graph by only keeping information
        that is useful for searching.

        Optionally overrides feature space and replaces feature vectors with a custom features buffer
        (e.g., swapping FP32 (d+1) features for FP16 d features while preserving graph topology).

        :param feature_space: Optional FloatSpace specifying the target feature space.
        :param custom_features: Optional NumPy array containing replacement feature vectors.
        :return: A new read-only DynamicExplorationGraph.
        """
        fs_cpp = feature_space.float_space_cpp if feature_space is not None else None
        feat_cpp = np.ascontiguousarray(custom_features) if custom_features is not None else None
        return DynamicExplorationGraph(self.dynamic_exploration_graph_cpp.to_readonly(fs_cpp, feat_cpp))

    def to_mutable(
        self,
        feature_space: Optional[FloatSpace] = None,
        custom_features: Optional[np.ndarray] = None,
        capacity: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Create a mutable SizeBoundedGraph from the given graph (ReadOnly or SizeBounded),
        optionally overriding the feature space and feature vectors.
        All edge weights are recalculated using the new feature space.

        :param feature_space: Optional FloatSpace specifying the target feature space.
                              If None, uses the graph's existing feature space.
        :param custom_features: Optional NumPy array containing replacement feature vectors.
        :param capacity: If > 0, sets the capacity to max(capacity, graph.size()).
                         If 0, capacity equals graph.size().
        :return: A new mutable DynamicExplorationGraph.
        """
        fs_cpp = feature_space.float_space_cpp if feature_space is not None else None
        feat_cpp = np.ascontiguousarray(custom_features) if custom_features is not None else None
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_mutable(
                custom_feature_space=fs_cpp, custom_features=feat_cpp, new_max_size=capacity
            )
        )

    def to_dynamic(
        self,
        feature_space: Optional[FloatSpace] = None,
        custom_features: Optional[np.ndarray] = None,
        chunk_size: int = 1024,
    ) -> "DynamicExplorationGraph":
        """
        Create a mutable DynamicGraph with chunk-based dynamic memory allocation
        from the given graph (ReadOnly, SizeBounded, or Dynamic),
        optionally overriding the feature space and feature vectors.
        All edge weights are recalculated using the new feature space.

        :param feature_space: Optional FloatSpace specifying the target feature space.
                              If None, uses the graph's existing feature space.
        :param custom_features: Optional NumPy array containing replacement feature vectors.
        :param chunk_size: Target number of vertices per memory chunk (default = 1024).
                           Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
        :return: A new mutable DynamicExplorationGraph.
        """
        fs_cpp = feature_space.float_space_cpp if feature_space is not None else None
        feat_cpp = np.ascontiguousarray(custom_features) if custom_features is not None else None
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_dynamic(
                custom_feature_space=fs_cpp, custom_features=feat_cpp, chunk_size=chunk_size
            )
        )

    def __repr__(self) -> str:
        return (
            f"DynamicExplorationGraph(size={self.size()} edges_per_vertex={self.get_edges_per_vertex()} "
            f"dim={self.get_feature_space().dim()})"
        )


def get_num_useful_threads(requested: int, max_limit: int):
    if requested == 0:
        requested = multiprocessing.cpu_count()
    return min(requested, max_limit)  # dont use more threads than queries


__all__ = [
    "DynamicExplorationGraph",
    "create_empty",
    "create_dynamic_empty",
    "create_random_graph",
    "load_graph",
    "load_readonly_graph",
    "load_dynamic_graph",
    "load_mutable_graph",
]


def _normalize_transformations(transformations):
    if transformations is None:
        return []
    if isinstance(transformations, (list, tuple)):
        return list(transformations)
    return [transformations]


def create_empty(
    capacity: int, feature_space: FloatSpace, edges_per_vertex: int = 32, transformations=None
) -> DynamicExplorationGraph:
    """
    Create an empty mutable DynamicExplorationGraph (SizeBoundedGraph).

    :param capacity: The maximal number of vertices of this graph.
    :param feature_space: A FloatSpace object defining dimensionality, metric, and instruction set.
    :param edges_per_vertex: Number of neighbors for each vertex. Defaults to 32.
    :param transformations: Optional list of FeatureTransformation objects (or single FeatureTransformation) attached to the graph.
    :return: A new mutable DynamicExplorationGraph.
    """
    xfrms = _normalize_transformations(transformations)
    graph_cpp = deglib_cpp.create_size_bounded_graph(
        capacity, edges_per_vertex, feature_space.float_space_cpp, xfrms
    )
    return DynamicExplorationGraph(graph_cpp)


def create_dynamic_empty(
    feature_space: FloatSpace, edges_per_vertex: int = 32, chunk_size: int = 1024, transformations=None
) -> DynamicExplorationGraph:
    """
    Create an empty mutable DynamicExplorationGraph with dynamic chunk-based memory allocation (DynamicGraph).

    :param feature_space: A FloatSpace object defining dimensionality, metric, and instruction set.
    :param edges_per_vertex: Number of neighbors for each vertex (must be even). Defaults to 32.
    :param chunk_size: Target number of vertices per memory chunk (default = 1024).
                       Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
    :param transformations: Optional list of FeatureTransformation objects (or single FeatureTransformation) attached to the graph.
    :return: A new mutable DynamicExplorationGraph.
    """
    xfrms = _normalize_transformations(transformations)
    graph_cpp = deglib_cpp.create_dynamic_graph(edges_per_vertex, feature_space.float_space_cpp, chunk_size, xfrms)
    return DynamicExplorationGraph(graph_cpp)


def create_random_graph(
    features: np.ndarray, feature_space: FloatSpace, edges_per_vertex: int = 32, seed: int = 7, transformations=None
) -> DynamicExplorationGraph:
    """
    Create a random exploration graph from the given feature data.

    The graph is built by first fully connecting the initial (edges_per_vertex + 1)
    vertices, then iteratively inserting each remaining vertex by connecting it
    to edges_per_vertex neighbors chosen from existing vertices.

    :param features: A 2D NumPy array of feature vectors (shape: vertex_count x dim).
    :param feature_space: A FloatSpace object defining dimensionality, metric, and instruction set.
    :param edges_per_vertex: Number of neighbors for each vertex. Must be even. Defaults to 32.
    :param seed: Random seed for deterministic graph construction. Defaults to 7.
    :param transformations: Optional list of FeatureTransformation objects (or single FeatureTransformation) attached to the graph.
    :return: A new DynamicExplorationGraph with a random exploration graph.
    """
    valid_dtype = feature_space.metric().get_dtype()
    features = assure_array(features, "features", valid_dtype)
    xfrms = _normalize_transformations(transformations)
    graph_cpp = deglib_cpp.create_random_graph(
        features, edges_per_vertex, feature_space.float_space_cpp, seed, xfrms
    )
    return DynamicExplorationGraph(graph_cpp)


def load_graph(path: pathlib.Path | str) -> DynamicExplorationGraph:
    """
    Load a saved DynamicExplorationGraph from disk (.deg container format or legacy v1).
    Automatically restores internal graph, prefetch configuration, entry vertices,
    and feature transformation / quantization pipeline.

    :param path: The path where to look for the file
    :raises FileNotFoundError: If the given file does not exist
    :return: A DynamicExplorationGraph.
    """
    return load_readonly_graph(path)


def load_readonly_graph(path: pathlib.Path | str) -> DynamicExplorationGraph:
    """
    Read a saved ReadOnlyGraph from given file.

    :param path: The path where to look for the file
    :raises FileNotFoundError: If the given file does not exist
    :return: A read-only DynamicExplorationGraph.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('File "{}" could not be found'.format(path))
    graph_cpp = deglib_cpp.load_readonly_graph(str(path))
    return DynamicExplorationGraph(graph_cpp)


def load_dynamic_graph(path: pathlib.Path | str, chunk_size: int = 1024) -> DynamicExplorationGraph:
    """
    Read a saved graph as a mutable DynamicGraph with chunk-based allocation.

    :param path: The path where to look for the file
    :param chunk_size: Target number of vertices per memory chunk (default = 1024).
                       Will be automatically rounded up to the nearest power of 2 (e.g. 600 -> 1024).
    :raises FileNotFoundError: If the given file does not exist
    :return: A mutable DynamicExplorationGraph.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('File "{}" could not be found'.format(path))
    graph_cpp = deglib_cpp.load_dynamic_graph(str(path), chunk_size)
    return DynamicExplorationGraph(graph_cpp)


def load_mutable_graph(path: pathlib.Path | str, capacity: int = 0) -> DynamicExplorationGraph:
    """
    Read a saved graph as a mutable SizeBoundedGraph without recomputing edge weights.

    :param path: The path where to look for the file
    :param capacity: Optional capacity upper bound. If 0 or <= file size, capacity equals graph size in file.
    :raises FileNotFoundError: If the given file does not exist
    :return: A mutable DynamicExplorationGraph.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('File "{}" could not be found'.format(path))
    graph_cpp = deglib_cpp.load_sizebounded_graph(str(path), capacity)
    return DynamicExplorationGraph(graph_cpp)
