import os
import multiprocessing
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import deglib_cpp
import deglib_cpp.distances as cpp_distances
import deglib_cpp.search as cpp_search
import pathlib

from .distances import FloatSpace, Metric, InstructionSet
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
            self, query: np.ndarray, eps: float, k: int, filter_labels: Union[None, np.ndarray, Filter] = None,
            max_distance_computation_count: int = 0, threads: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.
        For lower numbers it is recommended to set eps to 0 since its very unlikely the method can make use of the
        extended the search range.

        :param query: A feature vector for which similar feature vectors should searched.
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param k: The number of results that will be returned. If k is smaller than the number of vertices in the graph,
                  k is set to the number of vertices in the graph.
        :param filter_labels: A numpy array with dtype int32, that contains all labels that can be returned or an object
                              of type Filter, that limits the possible results to a given set.
                              All other labels will not be included in the result set.
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param threads: The number of threads to use for parallel processing. It should not excel the number of queries.
                        If set to 0, the minimum of the number of cores of this machine and the number of queries is
                        used.
        :returns: A tuple containing (indices, distances) where indices is a numpy-array of shape [n_queries, k]
                  containing the external labels of the closest found neighbors to the queries.
                  Distances is a numpy-array of shape [n_queries, k] containing the distances to the closest found
                  neighbors. Unfilled positions have NaN distance.
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
        indices, distances = self.dynamic_exploration_graph_cpp.search_batch(
            query, eps, k, filter_obj, max_distance_computation_count, threads
        )
        # Check if any query returned fewer than k valid results (unfilled elements have NaN distance)
        valid_counts = np.sum(~np.isnan(distances), axis=1)
        min_valid = int(np.min(valid_counts)) if valid_counts.size > 0 else 0
        if min_valid < k:
            warnings.warn('Number of results ({}) is smaller than k ({})'.format(min_valid, k), UserWarning)
            indices = indices[:, :min_valid]
            distances = distances[:, :min_valid]
        if single_query:
            indices = indices[0]
            distances = distances[0]
        return indices, distances

    def explore(
            self, entry_external_label: Union[int, np.ndarray, list], k: int,
            max_distance_computation_count: int = 0, eps: float = 0.0,
            include_entry: bool = True, threads: int = 1,
            filter_labels: Union[None, np.ndarray, Filter] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        An exploration for similar elements, limited by max_distance_computation_count.

        Handles both single entry labels (int) and batch entry arrays (ndarray/list).

        :param entry_external_label: The external label of the vertex to start exploration from (int for single,
                                     ndarray/list for batch).
        :param k: The number of similar feature vectors to return
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param include_entry: If True, the entry vertex is included in the result set.
        :param threads: The number of threads to use for parallel processing.
        :param filter_labels: A numpy array with dtype int32, that contains all labels that can be returned or an object
                              of type Filter, that limits the possible results to a given set.
                              All other labels will not be included in the result set.
        :returns: For a single entry, a tuple of (indices, distances) as 1D numpy arrays where indices are external labels.
                  For batch, a tuple of (indices, distances) as 2D numpy arrays where indices are external labels.
        """
        if isinstance(entry_external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(entry_external_label, dtype=np.uint32)
            if arr.ndim == 2:
                arr = arr[:, 0]
            filter_obj = Filter.create_filter(filter_labels, self.size())
            threads = get_num_useful_threads(threads, arr.shape[0])
            indices, distances = self.dynamic_exploration_graph_cpp.explore_batch(
                arr, k, max_distance_computation_count, eps, include_entry, filter_obj, threads
            )
            valid_counts = np.sum(~np.isnan(distances), axis=1)
            min_valid = int(np.min(valid_counts)) if valid_counts.size > 0 else 0
            if min_valid < k:
                warnings.warn('Number of results ({}) is smaller than k ({})'.format(min_valid, k), UserWarning)
                indices = indices[:, :min_valid]
                distances = distances[:, :min_valid]
            return indices, distances
        else:
            return self.dynamic_exploration_graph_cpp.explore(
                int(entry_external_label), k, max_distance_computation_count, eps, include_entry
            )

    def save_graph(self, path: pathlib.Path | str):
        """
        Save graph to specified file. Creates necessary directories.

        :param path: The path where to save the file.
        """
        self.dynamic_exploration_graph_cpp.save_graph(str(path))

    def to_readonly(self) -> 'DynamicExplorationGraph':
        """
        Create a read-only graph from the given graph by only keeping information
        that is useful for searching.

        :return: A new read-only DynamicExplorationGraph.
        """
        graph_cpp = self.dynamic_exploration_graph_cpp.to_readonly()
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
