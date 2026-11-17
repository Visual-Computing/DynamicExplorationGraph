import multiprocessing
import os
import pathlib
import warnings

import deglib_cpp
import numpy as np

from .distances import FloatSpace, Metric
from .search import Filter
from .utils import InvalidShapeException, assure_array


def _normalize_transformations(transformations):
    if transformations is None:
        return []
    if isinstance(transformations, (list, tuple)):
        return list(transformations)
    return [transformations]


_FLOAT_METRICS = frozenset({
    Metric.FP32_L2,
    Metric.FP32_InnerProduct,
    Metric.FP16_L2,
    Metric.FP16_InnerProduct,
})


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
        self, external_label: int | np.ndarray | list | tuple, threads: int = 1
    ) -> list[int] | np.ndarray:
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

    def get_neighbors_batch(self, external_labels: np.ndarray | list | tuple, threads: int = 1) -> np.ndarray:
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

    def get_labels(self) -> np.ndarray:
        """
        Returns a 1D NumPy uint32 array of all external labels ordered by their internal graph index (0 .. size()-1).

        In static graphs, this matches the builder's insertion order. In dynamic graphs where vertices have
        been deleted, this reflects the current internal memory layout.

        Use this array to index/reorder external feature arrays (e.g. ``X[graph.get_labels()]``) so they match
        the internal contiguous order required by ``to_readonly``, ``to_mutable``, and ``create_searcher``.

        :return: 1D uint32 NumPy array of external labels of length ``size()``.
        """
        return self.dynamic_exploration_graph_cpp.get_labels()

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
        filter_labels: None | np.ndarray | Filter = None,
        max_distance_computation_count: int = 0,
        threads: int = 0,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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

        if query.ndim == 1:
            return self.dynamic_exploration_graph_cpp.search(
                query, eps_or_ef, k, filter_obj, max_distance_computation_count, return_distances, unsorted
            )
        elif query.ndim == 2:
            threads = get_num_useful_threads(threads, query.shape[0])
            return self.dynamic_exploration_graph_cpp.search_batch(
                query,
                eps_or_ef,
                k,
                filter_obj,
                max_distance_computation_count,
                threads,
                return_distances,
                unsorted,
            )
        else:
            raise InvalidShapeException(f"invalid query shape: {query.shape}")

    def explore(
        self,
        entry_external_label: int | np.ndarray | list,
        k: int,
        max_distance_computation_count: int = 0,
        eps_or_ef: float = 0.0,
        include_entry: bool = True,
        threads: int = 1,
        filter_labels: None | np.ndarray | Filter = None,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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
            warnings.warn(f"k={k} is larger than number of vertices in graph={self.size()}. Setting k={self.size()}")
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
        from_external_labels: int | np.ndarray | list,
        to_external_label: int,
        eps: float = 0.0,
        k: int = 1,
        return_distances: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Greedy best-first traversal towards a stored vertex, returning the traversed path.

        The target vertex's own feature vector is used as the query, so the traversal follows the
        graph's greedy structure from the entry vertices until the target is reached. With
        eps = 0.0 and k = 1 this is a pure greedy descent; larger eps widen the exploration radius
        and allow escaping local optima at the cost of more distance computations.

        :param from_external_labels: External label(s) of the vertices to start the traversal from.
        :param to_external_label: External label of the vertex the path should end at.
        :param eps: Exploration factor widening the search radius. Default: 0.0 (greedy descent).
        :param k: Size of the result list driving the exploration radius. Default: 1.
        :param return_distances: If True, returns (path, distances). If False, returns only the path.
        :returns: (path, distances) tuple if return_distances is True, otherwise the path array.
                  The path is ordered from the target back to the entry vertex. Both arrays are
                  empty if the target was not reached within the given budget.
        """
        if isinstance(from_external_labels, (int, np.integer)):
            entries = [int(from_external_labels)]
        else:
            entries = [int(label) for label in from_external_labels]
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

    def optimize(self, sample_count: int = 300, median_subset: int = 4096) -> None:
        """Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.

        The graph self-samples its own stored features as queries, so no external query buffer is
        needed. Call once after construction to tune traversal for the expected operating point.

        :param sample_count: Number of vertices sampled as queries (capped at the graph size; 300 for noise-stable tuning).
        :param median_subset: Subset size for computing the graph-median entry point (<= 1 keeps native entries).
        """
        self.dynamic_exploration_graph_cpp.optimize(int(sample_count), int(median_subset))

    def set_transformations(self, transformations: None | list | tuple | object) -> None:
        """
        Sets or replaces the query feature transformations attached to this graph.
        """
        xfrms = _normalize_transformations(transformations)
        self.dynamic_exploration_graph_cpp.set_transformations(xfrms)

    def _prepare_conversion_args(
        self,
        feature_space: FloatSpace | None,
        custom_features: np.ndarray | None,
        transformations: list | tuple | object | None,
    ):
        """Validate and normalize arguments for ``to_readonly``, ``to_mutable``, and ``to_dynamic``."""
        if feature_space is not None and not isinstance(feature_space, FloatSpace):
            raise TypeError(f"feature_space must be a FloatSpace or None, got {type(feature_space).__name__}")

        xfrms = _normalize_transformations(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, deglib_cpp.transformation.FeatureTransformation):
                raise TypeError(
                    f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}"
                )
            if not t.is_fitted:
                raise ValueError(f"Transformation '{t.name}' (index {i}) must be fitted before converting a graph.")
        for i in range(len(xfrms) - 1):
            if xfrms[i].output_dim != xfrms[i + 1].input_dim:
                raise ValueError(
                    f"Transformation chain dimension mismatch: '{xfrms[i].name}' output_dim={xfrms[i].output_dim} "
                    f"!= '{xfrms[i + 1].name}' input_dim={xfrms[i + 1].input_dim}."
                )

        if feature_space is not None and xfrms:
            last = xfrms[-1]
            if last.output_dim != feature_space.dim():
                raise ValueError(
                    f"Last transformation '{last.name}' output_dim ({last.output_dim}) "
                    f"does not match feature_space.dim() ({feature_space.dim()})."
                )
            if last.output_byte_size != feature_space.get_data_size():
                raise ValueError(
                    f"Last transformation '{last.name}' output_byte_size ({last.output_byte_size} bytes) "
                    f"does not match feature_space.get_data_size() ({feature_space.get_data_size()} bytes)."
                )

        src_space = self.get_feature_space()
        src_metric = src_space.metric()
        n_vertices = self.size()

        if custom_features is not None:
            feat_cpp = np.ascontiguousarray(custom_features)
            if feat_cpp.ndim != 2:
                raise ValueError(
                    f"custom_features must be a 2D array of shape ({n_vertices}, ...), got shape {feat_cpp.shape}."
                )
            if feat_cpp.shape[0] != n_vertices:
                raise ValueError(
                    f"custom_features row count ({feat_cpp.shape[0]}) does not match graph.size() ({n_vertices})."
                )
            if feature_space is not None:
                expected_bytes = feature_space.get_data_size()
            elif xfrms:
                expected_bytes = xfrms[-1].output_byte_size
            else:
                expected_bytes = src_space.get_data_size()

            if feat_cpp.nbytes != n_vertices * expected_bytes:
                raise ValueError(
                    f"custom_features size ({feat_cpp.nbytes} bytes) does not match expected size "
                    f"({n_vertices * expected_bytes} bytes for {n_vertices} vertices x {expected_bytes} bytes/vertex)."
                )
        else:
            feat_cpp = None
            if xfrms:
                if xfrms[0].input_dim != src_space.dim():
                    raise ValueError(
                        f"First transformation '{xfrms[0].name}' input_dim ({xfrms[0].input_dim}) "
                        f"does not match graph feature_space.dim() ({src_space.dim()})."
                    )
                if feature_space is not None and feature_space.metric() != src_metric and src_metric not in _FLOAT_METRICS:
                    raise ValueError(
                        f"Cannot self-transform from non-float graph metric '{src_metric.name}' without custom_features. "
                        f"Provide custom_features or convert from an FP32/FP16 graph."
                    )
            elif feature_space is not None:
                if feature_space.dim() != src_space.dim() or feature_space.get_data_size() != src_space.get_data_size():
                    raise ValueError(
                        f"Cannot change feature_space from (dim={src_space.dim()}, bytes={src_space.get_data_size()}) "
                        f"to (dim={feature_space.dim()}, bytes={feature_space.get_data_size()}) "
                        f"without providing custom_features or transformations."
                    )

        fs_cpp = feature_space.float_space_cpp if feature_space is not None else None
        return fs_cpp, feat_cpp, xfrms

    def to_readonly(
        self,
        feature_space: FloatSpace | None = None,
        custom_features: np.ndarray | None = None,
        transformations: list | tuple | object | None = None,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a memory-compact, search-optimized read-only graph.

        Supports four usage modes depending on which optional arguments are provided:

        1. **Simple conversion** (all args omitted):
           Copies the graph topology and existing feature vectors as-is.
        2. **In-place self-transformation** (``transformations`` given, ``custom_features=None``):
           Transforms/quantizes each vertex directly from the graph's stored ``FP32`` or ``FP16``
           features into the new graph's memory, avoiding an intermediate feature array allocation.
           ``feature_space`` is optional (derived automatically from ``transformations`` unless
           overriding the metric, e.g. ``Int8_InnerProduct``) and only triggers self-transformation
           when the target metric differs from the source metric.
        3. **Custom feature buffer** (``custom_features`` given, with ``feature_space`` and/or ``transformations``):
           Replaces the stored features with a pre-transformed 2D NumPy array of shape ``(size(), ...)``
           aligned to internal vertex order (``get_labels()``). Note: ``custom_features`` is copied
           directly and is **not** passed through ``transformations``.
        4. **Attach query transformations only** (``transformations`` keeps the source metric, ``custom_features=None``):
           Copies stored features as-is and attaches ``transformations`` for query-time preprocessing.

        :param feature_space: Optional target ``FloatSpace``. Required when changing metric/dtype
                              via ``custom_features`` without ``transformations``, or when overriding
                              the metric derived from ``transformations``.
        :param custom_features: Optional contiguous 2D NumPy array of replacement features aligned
                                with ``get_labels()``.
        :param transformations: Optional ``FeatureTransformation`` or sequence of fitted
                                ``FeatureTransformation`` objects to apply/attach.
        :return: A new read-only ``DynamicExplorationGraph``.
        """
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(feature_space, custom_features, transformations)
        return DynamicExplorationGraph(self.dynamic_exploration_graph_cpp.to_readonly(fs_cpp, feat_cpp, xfrms))

    def to_mutable(
        self,
        feature_space: FloatSpace | None = None,
        custom_features: np.ndarray | None = None,
        capacity: int = 0,
        transformations: list | tuple | object | None = None,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a fixed-capacity mutable ``SizeBoundedGraph``.
        All edge weights are recalculated when the feature space or features change.

        Supports the same four feature/transformation modes as :meth:`to_readonly`:

        1. **Simple conversion** (``feature_space=None, custom_features=None, transformations=None``):
           Copies topology and features as-is.
        2. **In-place self-transformation** (``transformations`` given, ``custom_features=None``):
           Transforms/quantizes directly from stored ``FP32``/``FP16`` features into the new graph
           without allocating an intermediate feature buffer (requires target metric != source metric).
        3. **Custom feature buffer** (``custom_features`` given):
           Replaces features with a 2D NumPy array aligned to ``get_labels()`` (not passed through ``transformations``).
        4. **Attach query transformations only** (``transformations`` keeps the source metric, ``custom_features=None``):
           Copies features as-is and attaches ``transformations`` for query-time use.

        :param feature_space: Optional target ``FloatSpace`` (defaults to existing space or derived from ``transformations``).
        :param custom_features: Optional contiguous 2D NumPy array of replacement features aligned with ``get_labels()``.
        :param capacity: Maximum vertex capacity. If ``0`` or ``< size()``, capacity is set to ``size()``.
        :param transformations: Optional ``FeatureTransformation`` or sequence of fitted ``FeatureTransformation`` objects.
        :return: A new mutable ``DynamicExplorationGraph``.
        """
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(feature_space, custom_features, transformations)
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_mutable(
                custom_feature_space=fs_cpp, custom_features=feat_cpp, new_max_size=capacity, transformations=xfrms
            )
        )

    def to_dynamic(
        self,
        feature_space: FloatSpace | None = None,
        custom_features: np.ndarray | None = None,
        chunk_size: int = 1024,
        transformations: list | tuple | object | None = None,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a chunk-allocated mutable ``DynamicGraph``.
        All edge weights are recalculated when the feature space or features change.

        Supports the same four feature/transformation modes as :meth:`to_readonly`:

        1. **Simple conversion** (``feature_space=None, custom_features=None, transformations=None``):
           Copies topology and features as-is.
        2. **In-place self-transformation** (``transformations`` given, ``custom_features=None``):
           Transforms/quantizes directly from stored ``FP32``/``FP16`` features into the new graph
           without allocating an intermediate feature buffer (requires target metric != source metric).
        3. **Custom feature buffer** (``custom_features`` given):
           Replaces features with a 2D NumPy array aligned to ``get_labels()`` (not passed through ``transformations``).
        4. **Attach query transformations only** (``transformations`` keeps the source metric, ``custom_features=None``):
           Copies features as-is and attaches ``transformations`` for query-time use.

        :param feature_space: Optional target ``FloatSpace`` (defaults to existing space or derived from ``transformations``).
        :param custom_features: Optional contiguous 2D NumPy array of replacement features aligned with ``get_labels()``.
        :param chunk_size: Target vertices per memory chunk (default ``1024``, rounded up to power of 2).
        :param transformations: Optional ``FeatureTransformation`` or sequence of fitted ``FeatureTransformation`` objects.
        :return: A new dynamic ``DynamicExplorationGraph``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(feature_space, custom_features, transformations)
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_dynamic(
                custom_feature_space=fs_cpp, custom_features=feat_cpp, chunk_size=chunk_size, transformations=xfrms
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
    "create_dynamic_empty",
    "create_empty",
    "create_random_graph",
    "load_dynamic_graph",
    "load_graph",
    "load_mutable_graph",
    "load_readonly_graph",
]


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
    graph_cpp = deglib_cpp.create_size_bounded_graph(capacity, edges_per_vertex, feature_space.float_space_cpp, xfrms)
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
    graph_cpp = deglib_cpp.create_random_graph(features, edges_per_vertex, feature_space.float_space_cpp, seed, xfrms)
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
        raise FileNotFoundError(f'File "{path}" could not be found')
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
        raise FileNotFoundError(f'File "{path}" could not be found')
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
        raise FileNotFoundError(f'File "{path}" could not be found')
    graph_cpp = deglib_cpp.load_sizebounded_graph(str(path), capacity)
    return DynamicExplorationGraph(graph_cpp)
