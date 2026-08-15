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

    @classmethod
    def create_empty(
            cls, capacity: int, feature_space: FloatSpace, edges_per_vertex: int = 32
    ) -> 'DynamicExplorationGraph':
        """
        Create an empty mutable DynamicExplorationGraph.

        :param capacity: The maximal number of vertices of this graph.
        :param feature_space: A FloatSpace object defining dimensionality, metric, and instruction set.
        :param edges_per_vertex: Number of neighbors for each vertex. Defaults to 32.
        :return: A new mutable DynamicExplorationGraph.
        """
        graph_cpp = deglib_cpp.create_size_bounded_graph(capacity, edges_per_vertex, feature_space.to_cpp())
        return cls(graph_cpp)

    @classmethod
    def create_random_graph(
            cls,
            features: np.ndarray,
            feature_space: FloatSpace,
            edges_per_vertex: int = 32,
            seed: int = 7
    ) -> 'DynamicExplorationGraph':
        """
        Create a random exploration graph from the given feature data.

        The graph is built by first fully connecting the initial (edges_per_vertex + 1)
        vertices, then iteratively inserting each remaining vertex by connecting it
        to edges_per_vertex neighbors chosen from existing vertices.

        :param features: A 2D NumPy array of feature vectors (shape: vertex_count x dim).
        :param feature_space: A FloatSpace object defining dimensionality, metric, and instruction set.
        :param edges_per_vertex: Number of neighbors for each vertex. Must be even. Defaults to 32.
        :param seed: Random seed for deterministic graph construction. Defaults to 7.
        :return: A new DynamicExplorationGraph with a random exploration graph.
        """
        valid_dtype = feature_space.metric().get_dtype()
        features = assure_array(features, 'features', valid_dtype)
        graph_cpp = deglib_cpp.create_random_graph(
            features, edges_per_vertex, feature_space.to_cpp(), seed
        )
        return cls(graph_cpp)

    @classmethod
    def load_readonly_graph(cls, path: pathlib.Path | str) -> 'DynamicExplorationGraph':
        """
        Read a saved ReadOnlyGraph from given file.

        :param path: The path where to look for the file
        :raises FileNotFoundError: If the given file does not exist
        :return: A read-only DynamicExplorationGraph.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError('File "{}" could not be found'.format(path))
        graph_cpp = deglib_cpp.load_readonly_graph(str(path))
        return cls(graph_cpp)

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

    def get_neighbors(self, external_label: int) -> List[int]:
        """
        Get the neighbor external labels of the given external label.

        :param external_label: The external label to get the neighbors of
        :returns: The external labels of the neighbors
        """
        return self.dynamic_exploration_graph_cpp.get_neighbors(external_label)

    def search(
            self, query: np.ndarray, eps: float = 0.0, k: int = 10,
            filter_labels: Union[None, np.ndarray, Filter] = None,
            max_distance_computation_count: int = 0, threads: int = 0,
            return_distances: bool = True, unsorted: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Search for nearest neighbors of query vector(s).

        :param query: Query feature vector(s) as numpy array.
        :param eps: Controls how many nodes are checked during search.
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
        # handle query shapes
        single_query = len(query.shape) == 1
        if single_query:
            query = query.reshape(1, -1)
        if len(query.shape) != 2:
            raise InvalidShapeException('invalid query shape: {}'.format(query.shape))

        if k > self.size():
            warnings.warn(
                'k={} is smaller than number of vertices in graph={}. Setting k={}'.format(k, self.size(), self.size()))
            k = self.size()

        valid_dtype = self.get_feature_space().metric().get_dtype()
        query = assure_array(query, 'query', valid_dtype)
        filter_obj = Filter.create_filter(filter_labels, self.size())
        threads = get_num_useful_threads(threads, query.shape[0])
        indices_or_tuple = self.dynamic_exploration_graph_cpp.search_batch(
            query, eps, k, filter_obj, max_distance_computation_count, threads, return_distances, unsorted
        )

        if return_distances:
            indices, distances = indices_or_tuple
            if single_query:
                return indices[0], distances[0]
            return indices, distances
        else:
            indices = indices_or_tuple
            if single_query:
                return indices[0]
            return indices

    def explore(
            self, entry_external_label: Union[int, np.ndarray, list], k: int,
            max_distance_computation_count: int = 0, eps: float = 0.0,
            include_entry: bool = True, threads: int = 1,
            filter_labels: Union[None, np.ndarray, Filter] = None,
            return_distances: bool = True, unsorted: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        An exploration for similar elements, limited by max_distance_computation_count.

        Handles both single entry labels (int) and batch entry arrays (ndarray/list).

        :param entry_external_label: The external label of the vertex to start exploration from.
        :param k: The number of similar feature vectors to return.
        :param max_distance_computation_count: Limit the number of distance calculations.
        :param eps: Controls how many nodes are checked during search.
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
                'k={} is larger than number of vertices in graph={}. Setting k={}'.format(k, self.size(), self.size()))
            k = self.size()

        if isinstance(entry_external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(entry_external_label, dtype=np.uint32)
            if arr.ndim == 2:
                arr = arr[:, 0]
            filter_obj = Filter.create_filter(filter_labels, self.size())
            threads = get_num_useful_threads(threads, arr.shape[0])
            return self.dynamic_exploration_graph_cpp.explore_batch(
                arr, k, max_distance_computation_count, eps, include_entry, filter_obj, threads, return_distances, unsorted
            )
        else:
            indices, distances = self.dynamic_exploration_graph_cpp.explore(
                int(entry_external_label), k, max_distance_computation_count, eps, include_entry
            )
            if return_distances:
                return indices, distances
            return indices

    def save_graph(self, path: pathlib.Path | str):
        """
        Save graph to specified file. Creates necessary directories.

        :param path: The path where to save the file.
        """
        self.dynamic_exploration_graph_cpp.save_graph(str(path))

    def to_readonly(
        self,
        feature_space: Optional[FloatSpace] = None,
        custom_features: Optional[np.ndarray] = None
    ) -> 'DynamicExplorationGraph':
        """
        Create a read-only graph from the given graph by only keeping information
        that is useful for searching.

        Optionally overrides feature space and replaces feature vectors with a custom features buffer
        (e.g., swapping FP32 (d+1) features for FP16 d features while preserving graph topology).

        :param feature_space: Optional FloatSpace specifying the target feature space.
        :param custom_features: Optional NumPy array containing replacement feature vectors.
        :return: A new read-only DynamicExplorationGraph.
        """
        fs_cpp = feature_space.to_cpp() if feature_space is not None else None
        feat_cpp = np.ascontiguousarray(custom_features) if custom_features is not None else None
        graph_cpp = deglib_cpp.read_only_graph_from_graph(
            self.dynamic_exploration_graph_cpp, fs_cpp, feat_cpp
        )
        return DynamicExplorationGraph(graph_cpp)

    def to_mutable(
        self,
        feature_space: Optional[FloatSpace] = None,
        custom_features: Optional[np.ndarray] = None,
        capacity: int = 0
    ) -> 'DynamicExplorationGraph':
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
        fs_cpp = feature_space.to_cpp() if feature_space is not None else None
        feat_cpp = np.ascontiguousarray(custom_features) if custom_features is not None else None
        graph_cpp = deglib_cpp.size_bounded_graph_from_graph(
            self.dynamic_exploration_graph_cpp, fs_cpp, feat_cpp, capacity
        )
        return DynamicExplorationGraph(graph_cpp)

    def __repr__(self) -> str:
        return (f'DynamicExplorationGraph(size={self.size()} edges_per_vertex={self.get_edges_per_vertex()} '
                f'dim={self.get_feature_space().dim()})')


def get_num_useful_threads(requested: int, max_limit: int):
    if requested == 0:
        requested = multiprocessing.cpu_count()
    return min(requested, max_limit)  # dont use more threads than queries


__all__ = ['DynamicExplorationGraph', 'load_readonly_graph']


def load_readonly_graph(path: pathlib.Path | str) -> DynamicExplorationGraph:
    """
    Read a saved ReadOnlyGraph from given file.

    :param path: The path where to look for the file
    :raises FileNotFoundError: If the given file does not exist
    """
    return DynamicExplorationGraph.load_readonly_graph(path)
