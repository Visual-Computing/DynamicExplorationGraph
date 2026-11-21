from collections.abc import Sequence
import os
import pathlib
from typing import Any, Literal
import numpy as np
import deglib_cpp

from deglib.cpu import InstructionSet
from deglib.distances import Metric, _infer_dim_from_array
from deglib.transformation import (
    FeatureTransformation,
    _normalize_transforms,
    apply_on_database,
    make_quantizer_pipeline,
)
from deglib.utils import InvalidShapeException, assure_array

cpp_transformation = deglib_cpp.transformation

_FLOAT_METRICS = frozenset(
    {
        Metric.FP32_L2,
        Metric.FP32_InnerProduct,
        Metric.FP16_L2,
        Metric.FP16_InnerProduct,
    }
)

_FP32_METRICS = frozenset({Metric.FP32_L2, Metric.FP32_InnerProduct})
_FP16_METRICS = frozenset({Metric.FP16_L2, Metric.FP16_InnerProduct})


def _metric_data_tag(metric: Metric) -> str:
    return Metric(int(metric)).name.split("_", 1)[0]


def get_num_useful_threads(requested: int, max_limit: int) -> int:
    """
    Resolve the effective thread count clamped to ``max_limit``.
    """
    if max_limit <= 0:
        return 1
    if requested <= 0:
        hw = os.cpu_count() or 1
        return min(hw, max_limit)
    return min(requested, max_limit)


class DynamicExplorationGraph:
    """
    Python wrapper around deglib's C++ DynamicExplorationGraph.
    All vertex indexing methods take and return external labels (``uint32``).
    """

    def __init__(self, graph_cpp: deglib_cpp.DynamicExplorationGraph):
        self.dynamic_exploration_graph_cpp = graph_cpp

    @property
    def _cpp(self) -> deglib_cpp.DynamicExplorationGraph:
        return self.dynamic_exploration_graph_cpp

    def size(self) -> int:
        """
        Return the number of vertices currently in the graph.
        """
        return self.dynamic_exploration_graph_cpp.size()

    def get_dim(self) -> int:
        """
        Return the dimensionality of stored feature vectors.
        """
        return int(self.dynamic_exploration_graph_cpp.get_feature_space().dim())

    def get_instruction(self) -> InstructionSet:
        """
        Return the active SIMD ``InstructionSet`` bound to this graph.
        """
        return InstructionSet.resolve(self.dynamic_exploration_graph_cpp.get_feature_space().get_instruction())

    def get_metric(self) -> Metric:
        """
        Return the active distance ``Metric`` (including its bound ``InstructionSet``).
        """
        space = self.dynamic_exploration_graph_cpp.get_feature_space()
        base = Metric(int(space.metric()))
        return base.with_instruction(InstructionSet.resolve(space.get_instruction()))

    def get_data_size(self) -> int:
        """
        Return the byte size of one stored vertex feature vector.
        """
        return int(self.dynamic_exploration_graph_cpp.get_feature_space().get_data_size())

    def get_edges_per_vertex(self) -> int:
        """
        Return the number of outgoing edges per vertex.
        """
        return self.dynamic_exploration_graph_cpp.get_edges_per_vertex()

    def get_po(self) -> int:
        """
        Return the traversal feature prefetch offset.
        """
        return self.dynamic_exploration_graph_cpp.get_po()

    def get_pl(self) -> int:
        """
        Return the traversal feature prefetch look-ahead in cache lines (0 = whole vector).
        """
        return self.dynamic_exploration_graph_cpp.get_pl()

    def get_nl(self) -> int:
        """
        Return the traversal neighbor prefetch look-ahead.
        """
        return self.dynamic_exploration_graph_cpp.get_nl()

    def has_vertex(self, external_label: int) -> bool:
        """
        Return whether ``external_label`` is present in the graph.
        """
        return self.dynamic_exploration_graph_cpp.has_vertex(int(external_label))

    def is_mutable(self) -> bool:
        """
        Return whether the graph is mutable (can be modified via ``GraphBuilder``).
        """
        return self.dynamic_exploration_graph_cpp.is_mutable()

    def get_neighbors(
        self, external_label: int | np.ndarray | Sequence[int], threads: int = 1
    ) -> list[int] | np.ndarray:
        """
        Get neighbor external labels for a single label or a batch of labels.
        """
        if isinstance(external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(external_label, dtype=np.uint32)
            return self.dynamic_exploration_graph_cpp.get_neighbors_batch(arr, threads)
        return self.dynamic_exploration_graph_cpp.get_neighbors(int(external_label))

    def get_neighbors_batch(self, external_labels: np.ndarray | Sequence[int], threads: int = 1) -> np.ndarray:
        """
        Get neighbor external labels for a batch of external labels as a 2D ``uint32`` array.
        """
        arr = np.ascontiguousarray(external_labels, dtype=np.uint32)
        return self.dynamic_exploration_graph_cpp.get_neighbors_batch(arr, threads)

    def get_labels(self) -> np.ndarray:
        """
        Return a 1D ``uint32`` array of external labels ordered by internal vertex index ``0 .. size()-1``.
        """
        return self.dynamic_exploration_graph_cpp.get_labels()

    def has_transformations(self) -> bool:
        """
        Return whether query feature transformations or quantizers are attached.
        """
        return self.dynamic_exploration_graph_cpp.has_transformations()

    def get_transformations(self) -> list[FeatureTransformation]:
        """
        Return the list of attached ``FeatureTransformation`` objects.
        """
        return self.dynamic_exploration_graph_cpp.get_transformations()

    def set_transformations(
        self,
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None,
    ) -> None:
        """
        Set or replace query feature transformations attached to this graph.
        """
        xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")
        self.dynamic_exploration_graph_cpp.set_transformations(xfrms)

    def get_entry_vertex_indices(self) -> list[int]:
        """
        Return internal entry vertex indices.
        """
        return self.dynamic_exploration_graph_cpp.get_entry_vertex_indices()

    def set_entry_vertex_indices(self, indices: Sequence[int] | np.ndarray) -> None:
        """
        Set internal entry vertex indices.
        """
        self.dynamic_exploration_graph_cpp.set_entry_vertex_indices([int(x) for x in indices])

    def optimize(self, sample_count: int = 300, median_subset: int = 4096) -> None:
        """
        Auto-tune traversal prefetch parameters (``po``, ``pl``, ``nl``) and graph-median entry point.
        """
        self.dynamic_exploration_graph_cpp.optimize(int(sample_count), int(median_subset))

    def save_graph(self, path: pathlib.Path | str) -> None:
        """
        Save graph topology, features, transformations, and prefetch config to disk (``.deg`` format).
        """
        p = pathlib.Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        self.dynamic_exploration_graph_cpp.save_graph(str(p))

    def search(
        self,
        query: np.ndarray,
        eps_or_ef: float = 0.0,
        k: int = 10,
        filter_labels: Any = None,
        max_distance_computation_count: int = 0,
        threads: int = 0,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> deglib_cpp.SearchResults:
        """
        Search the graph for the ``k`` approximate nearest neighbors of ``query`` (1D or 2D).

        Returns a ``SearchResults`` object containing zero-copy NumPy ``.indices``,
        optional ``.distances`` (or ``None`` when ``return_distances=False``),
        and traversal telemetry (``.graph_distance_count``, ``.graph_hop_count``).
        """
        if not isinstance(query, np.ndarray):
            query = np.ascontiguousarray(query, dtype=np.float32)
        elif not query.flags.c_contiguous:
            query = np.ascontiguousarray(query)

        if filter_labels is None:
            filter_obj = None
        else:
            from deglib.search import Filter

            filter_obj = Filter.create_filter(filter_labels, self.size()).filter_cpp

        if query.ndim == 1:
            return self.dynamic_exploration_graph_cpp.search(
                query,
                eps_or_ef,
                k,
                filter_obj,
                max_distance_computation_count,
                return_distances,
                unsorted,
            )
        if query.ndim == 2:
            eff_threads = get_num_useful_threads(threads, query.shape[0])
            return self.dynamic_exploration_graph_cpp.search_batch(
                query,
                eps_or_ef,
                k,
                filter_obj,
                max_distance_computation_count,
                eff_threads,
                return_distances,
                unsorted,
            )
        raise InvalidShapeException(f"Query needs to have 1 or 2 dimensions, found {query.ndim}")

    def explore(
        self,
        entry_external_label: int | np.ndarray | Sequence[int],
        k: int,
        max_distance_computation_count: int = 0,
        eps_or_ef: float = 0.0,
        include_entry: bool = True,
        threads: int = 1,
        filter_labels: Any = None,
        return_distances: bool = True,
        unsorted: bool = False,
    ) -> deglib_cpp.SearchResults:
        """
        Explore the neighborhood starting from ``entry_external_label`` (single int or 1D array/sequence).

        Returns a ``SearchResults`` object containing zero-copy NumPy ``.indices``,
        optional ``.distances`` (or ``None`` when ``return_distances=False``),
        and traversal telemetry (``.graph_distance_count``, ``.graph_hop_count``).
        """
        if filter_labels is None:
            filter_obj = None
        else:
            from deglib.search import Filter

            filter_obj = Filter.create_filter(filter_labels, self.size()).filter_cpp

        if isinstance(entry_external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(entry_external_label, dtype=np.uint32)
            eff_threads = get_num_useful_threads(threads, arr.shape[0])
            return self.dynamic_exploration_graph_cpp.explore_batch(
                arr,
                k,
                max_distance_computation_count,
                eps_or_ef,
                include_entry,
                filter_obj,
                eff_threads,
                return_distances,
                unsorted,
            )

        return self.dynamic_exploration_graph_cpp.explore(
            int(entry_external_label),
            k,
            max_distance_computation_count,
            eps_or_ef,
            include_entry,
            filter_obj,
            return_distances,
            unsorted,
        )

    def has_path(
        self,
        from_external_labels: int | np.ndarray | Sequence[int],
        to_external_label: int,
        eps: float = 0.0,
        k: int = 1,
        return_distances: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Check reachability via greedy traversal from ``from_external_labels`` to ``to_external_label``.
        """
        if isinstance(from_external_labels, (int, np.integer)):
            entries = [int(from_external_labels)]
        else:
            entries = [int(x) for x in np.asarray(from_external_labels, dtype=np.uint32).ravel()]
        return self.dynamic_exploration_graph_cpp.has_path(
            entries, int(to_external_label), float(eps), int(k), return_distances
        )

    def to_reranker(
        self,
        metric: Metric | str,
        data: np.ndarray | None = None,
        order: Literal["label", "internal"] = "label",
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        optimize: bool = True,
        threads: int = 0,
    ):
        """
        Create a standalone ``Reranker`` aligned with this graph's internal vertex order.

        This allows building a single ``Reranker`` once and sharing it across multiple quantized
        ``Searcher`` instances created via ``to_searcher(..., reranker=reranker)``.

        - ``data=None`` (default): extracts and quantizes vectors directly from an ``FP32``/``FP16`` graph.
        - ``data`` provided: aligns rows via ``order`` (``"label"`` or ``"internal"``).
        """
        if order not in ("label", "internal"):
            raise ValueError(f"order must be 'label' or 'internal', got {order!r}")

        from deglib.search import Reranker

        resolved_metric = Metric.resolve(metric)
        r_xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(r_xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")

        if data is None:
            target_dim = int(r_xfrms[-1].output_dim) if r_xfrms else self.get_dim()
            cpp_space = resolved_metric._to_cpp_space(target_dim)
            reranker_cpp = deglib_cpp.search.Reranker.from_graph(
                self.dynamic_exploration_graph_cpp,
                cpp_space,
                r_xfrms,
                int(threads),
            )
            return Reranker._from_cpp(resolved_metric, reranker_cpp, optimize=optimize)

        raw_data = np.asarray(data)
        if raw_data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got shape {raw_data.shape}")
        n_vertices = self.size()
        row_indices = None
        if order == "label":
            if n_vertices == 0:
                raw_data = raw_data[:0]
            else:
                labels = self.get_labels()
                if int(labels.max()) >= raw_data.shape[0]:
                    raise ValueError(
                        f"Max graph external label ({int(labels.max())}) >= data row count ({raw_data.shape[0]}). "
                        "Pass data indexed by external labels or set order='internal'."
                    )
                row_indices = labels
        elif raw_data.shape[0] != n_vertices:
            raise ValueError(f"data row count ({raw_data.shape[0]}) does not match graph.size() ({n_vertices}).")

        return Reranker(
            metric=resolved_metric,
            base_vectors=raw_data,
            transformations=r_xfrms,
            optimize=optimize,
            threads=threads,
            row_indices=row_indices,
        )

    def to_searcher(
        self,
        metric: Metric | str | None = None,
        reranker: Any | None = None,
        rerank_metric: Metric | str | None = None,
        rerank_data: np.ndarray | None = None,
        rerank_order: Literal["label", "internal"] = "label",
        rerank_transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        normalize_query: bool = False,
        optimize: bool = True,
        use_routing: bool = True,
        n_clusters: int = 256,
        n_iter: int = 20,
        sample_size: int = 0,
        threads: int = 0,
    ):
        """
        Create a native C++ ``Searcher`` from this graph.

        Reranking is opt-in via either a shared ``reranker`` instance or ``rerank_metric``:
        - ``reranker=None`` and ``rerank_metric=None`` (default): NO reranking is performed.
        - ``reranker`` set: reuses the prebuilt ``Reranker`` (zero-copy over its base vectors).
        - ``rerank_metric`` set, ``rerank_data=None``: extracts and quantizes rerank vectors directly
          from this ``FP32``/``FP16`` graph before converting/quantizing it to ``metric``.
        - ``rerank_metric`` set, ``rerank_data`` given: uses ``rerank_data`` aligned via ``rerank_order``.
        """
        from deglib.search import create_searcher

        return create_searcher(
            self,
            metric=metric,
            reranker=reranker,
            rerank_metric=rerank_metric,
            rerank_data=rerank_data,
            rerank_order=rerank_order,
            rerank_transformations=rerank_transformations,
            normalize_query=normalize_query,
            optimize=optimize,
            use_routing=use_routing,
            n_clusters=n_clusters,
            n_iter=n_iter,
            sample_size=sample_size,
            threads=threads,
        )

    def _prepare_conversion_args(
        self,
        metric: Metric | str | None,
        custom_features: np.ndarray | None,
        custom_features_order: Literal["label", "internal"],
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None,
        use_rerank: bool = False,
        threads: int = 0,
    ):
        if custom_features_order not in ("label", "internal"):
            raise ValueError(f"custom_features_order must be 'label' or 'internal', got {custom_features_order!r}")

        xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")
            if not t.is_fitted:
                raise ValueError(f"transformations[{i}] must be fitted before conversion")

        resolved_metric = None if metric is None else Metric.resolve(metric)
        n_vertices = self.size()

        if custom_features is not None:
            feat_arr = np.asarray(custom_features)
            if feat_arr.ndim != 2:
                raise ValueError(f"custom_features must be a 2D array, got shape {feat_arr.shape}.")

            if custom_features_order == "label":
                if n_vertices == 0:
                    aligned = np.ascontiguousarray(feat_arr[:0])
                else:
                    labels = self.get_labels()
                    if int(labels.max()) >= feat_arr.shape[0]:
                        raise ValueError(
                            f"Max graph external label ({int(labels.max())}) >= custom_features row count ({feat_arr.shape[0]}). "
                            "Pass an array indexed by external labels or use custom_features_order='internal'."
                        )
                    aligned = np.ascontiguousarray(feat_arr[labels])
            else:
                if feat_arr.shape[0] != n_vertices:
                    raise ValueError(
                        f"custom_features row count ({feat_arr.shape[0]}) does not match graph.size() ({n_vertices})."
                    )
                aligned = np.ascontiguousarray(feat_arr)

            if resolved_metric is not None:
                target_base = Metric(int(resolved_metric))
                is_float_input = aligned.dtype == np.float32 or (
                    aligned.dtype == np.float16 and target_base not in _FP16_METRICS
                )
                if len(xfrms) == 0 and target_base not in _FP32_METRICS and is_float_input:
                    target_dim = int(aligned.shape[1])
                    xfrms = make_quantizer_pipeline(
                        aligned,
                        resolved_metric,
                        use_rerank=use_rerank,
                        threads=threads,
                    )
                    aligned = apply_on_database(xfrms, aligned, threads=threads)
                elif (
                    len(xfrms) > 0
                    and aligned.dtype == np.float32
                    and aligned.shape[1] == xfrms[0].input_dim
                    and (xfrms[-1].output_dtype != np.float32 or xfrms[-1].output_dim != aligned.shape[1])
                ):
                    target_dim = int(xfrms[-1].output_dim)
                    aligned = apply_on_database(xfrms, aligned, threads=threads)
                else:
                    aligned = assure_array(aligned, "custom_features", resolved_metric.get_accepted_dtypes())
                    width = int(aligned.shape[1])
                    src_dim = self.get_dim()
                    if target_base in (
                        Metric.Uint4_L2,
                        Metric.Uint4_InnerProduct,
                        Metric.Int4_L2,
                        Metric.Int4_InnerProduct,
                    ):
                        target_dim = src_dim if (src_dim + 1) // 2 == width else width * 2
                    elif target_base == Metric.EVP_InnerProduct:
                        target_dim = src_dim if 2 * ((src_dim + 7) // 8) == width else width * 4
                    else:
                        target_dim = width

                expected_bytes = resolved_metric.get_data_size(target_dim)
                if xfrms and xfrms[-1].output_byte_size != expected_bytes:
                    raise ValueError(
                        f"transformations[-1].output_byte_size ({xfrms[-1].output_byte_size}) does not match "
                        f"target metric data size ({expected_bytes})."
                    )
                if aligned.nbytes != n_vertices * expected_bytes:
                    raise ValueError(
                        f"custom_features size ({aligned.nbytes} bytes) does not match expected size "
                        f"({n_vertices * expected_bytes} bytes for {n_vertices} vertices x {expected_bytes} bytes/vertex)."
                    )
                fs_cpp = resolved_metric._to_cpp_space(target_dim)
            else:
                src_metric = self.get_metric()
                aligned = assure_array(aligned, "custom_features", src_metric.get_accepted_dtypes())
                expected_bytes = self.get_data_size()
                if aligned.nbytes != n_vertices * expected_bytes:
                    raise ValueError(
                        f"custom_features size ({aligned.nbytes} bytes) does not match expected size "
                        f"({n_vertices * expected_bytes} bytes for {n_vertices} vertices x {expected_bytes} bytes/vertex)."
                    )
                fs_cpp = None

            return fs_cpp, aligned, xfrms

        # custom_features is None
        src_metric = self.get_metric()
        src_dim = self.get_dim()
        if xfrms and xfrms[0].input_dim != src_dim:
            raise ValueError(
                f"transformations[0].input_dim ({xfrms[0].input_dim}) does not match graph dim ({src_dim})."
            )

        if resolved_metric is not None:
            fs_cpp = resolved_metric._to_cpp_space(src_dim)
            src_tag = _metric_data_tag(src_metric)
            dst_tag = _metric_data_tag(resolved_metric)
            expected_bytes = resolved_metric.get_data_size(src_dim)
            if xfrms and xfrms[-1].output_byte_size != expected_bytes:
                raise ValueError(
                    f"transformations[-1].output_byte_size ({xfrms[-1].output_byte_size}) does not match "
                    f"target metric data size ({expected_bytes})."
                )
            if Metric(int(src_metric)) not in _FLOAT_METRICS and (src_tag != dst_tag or len(xfrms) > 0):
                raise ValueError(
                    f"Cannot self-transform from non-float metric '{src_metric.name}' to '{resolved_metric.name}' "
                    "without custom_features. Provide custom_features or convert from an FP32/FP16 graph."
                )
            if src_tag != dst_tag and len(xfrms) == 0:
                if Metric(int(src_metric)) in _FLOAT_METRICS and Metric(int(resolved_metric)) not in _FP32_METRICS:
                    xfrms = cpp_transformation.make_quantizer_pipeline_from_graph(
                        self.dynamic_exploration_graph_cpp,
                        int(resolved_metric),
                        -1.0,
                        use_rerank,
                        threads,
                        7,
                    )
                else:
                    raise ValueError(
                        f"Cannot self-transform from non-float metric '{src_metric.name}' to '{resolved_metric.name}' "
                        "without custom_features. Provide custom_features or convert from an FP32/FP16 graph."
                    )
        else:
            fs_cpp = None
            if len(xfrms) > 0 and Metric(int(src_metric)) not in _FLOAT_METRICS:
                raise ValueError(
                    f"Cannot self-transform from non-float metric '{src_metric.name}' without custom_features."
                )

        return fs_cpp, None, xfrms

    def to_readonly(
        self,
        metric: Metric | str | None = None,
        custom_features: np.ndarray | None = None,
        custom_features_order: Literal["label", "internal"] = "label",
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        use_rerank: bool = False,
        optimize: bool = True,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a compact, search-optimized ``ReadOnlyGraph``.
        """
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(
            metric,
            custom_features,
            custom_features_order,
            transformations,
            use_rerank=use_rerank,
            threads=threads,
        )
        res = DynamicExplorationGraph(self.dynamic_exploration_graph_cpp.to_readonly(fs_cpp, feat_cpp, xfrms, threads))
        if optimize and res.size() > 0:
            res.optimize()
        return res

    def to_mutable(
        self,
        metric: Metric | str | None = None,
        custom_features: np.ndarray | None = None,
        custom_features_order: Literal["label", "internal"] = "label",
        capacity: int = 0,
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a fixed-capacity mutable ``SizeBoundedGraph``.
        """
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(
            metric,
            custom_features,
            custom_features_order,
            transformations,
            use_rerank=False,
            threads=threads,
        )
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_mutable(
                custom_feature_space=fs_cpp,
                custom_features=feat_cpp,
                new_max_size=capacity,
                transformations=xfrms,
            )
        )

    def to_dynamic(
        self,
        metric: Metric | str | None = None,
        custom_features: np.ndarray | None = None,
        custom_features_order: Literal["label", "internal"] = "label",
        chunk_size: int = 1024,
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Convert this graph into a chunk-allocated mutable ``DynamicGraph``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        fs_cpp, feat_cpp, xfrms = self._prepare_conversion_args(
            metric,
            custom_features,
            custom_features_order,
            transformations,
            use_rerank=False,
            threads=threads,
        )
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_dynamic(
                custom_feature_space=fs_cpp,
                custom_features=feat_cpp,
                chunk_size=chunk_size,
                transformations=xfrms,
            )
        )

    def __repr__(self) -> str:
        return (
            f"DynamicExplorationGraph(size={self.size()} "
            f"edges_per_vertex={self.get_edges_per_vertex()} "
            f"dim={self.get_dim()} metric={self.get_metric().name})"
        )


def create_empty(
    capacity: int,
    dim: int,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph:
    """
    Create an empty fixed-capacity mutable ``SizeBoundedGraph``.
    """
    resolved_metric = Metric.resolve(metric)
    fs_cpp = resolved_metric._to_cpp_space(int(dim))
    xfrms = _normalize_transforms(transformations)
    graph_cpp = deglib_cpp.create_size_bounded_graph(capacity, edges_per_vertex, fs_cpp, xfrms)
    return DynamicExplorationGraph(graph_cpp)


def create_dynamic_empty(
    dim: int,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    chunk_size: int = 1024,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph:
    """
    Create an empty dynamically growing mutable ``DynamicGraph``.
    """
    resolved_metric = Metric.resolve(metric)
    fs_cpp = resolved_metric._to_cpp_space(int(dim))
    xfrms = _normalize_transforms(transformations)
    graph_cpp = deglib_cpp.create_dynamic_graph(edges_per_vertex, fs_cpp, chunk_size, xfrms)
    return DynamicExplorationGraph(graph_cpp)


def create_random_graph(
    features: np.ndarray,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    seed: int = 7,
    optimize: bool = True,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph:
    """
    Create a random k-regular exploration graph directly from a 2D feature array.
    """
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        raise InvalidShapeException(
            f"features must be a 2D numpy array, got {getattr(features, 'shape', type(features))}"
        )
    resolved_metric = Metric.resolve(metric)
    features = assure_array(features, "features", resolved_metric.get_accepted_dtypes())
    dim = _infer_dim_from_array(resolved_metric, features)
    fs_cpp = resolved_metric._to_cpp_space(dim)
    xfrms = _normalize_transforms(transformations)
    graph_cpp = deglib_cpp.create_random_graph(features, edges_per_vertex, fs_cpp, seed, xfrms)
    graph = DynamicExplorationGraph(graph_cpp)
    if optimize and graph.size() > 0:
        graph.optimize()
    return graph


def load_graph(path: pathlib.Path | str) -> DynamicExplorationGraph:
    """
    Load a saved ``ReadOnlyGraph`` from disk (``.deg`` container format).
    """
    return load_readonly_graph(path)


def load_readonly_graph(path: pathlib.Path | str) -> DynamicExplorationGraph:
    """
    Load a saved ``ReadOnlyGraph`` from disk.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Graph file not found: {p}")
    return DynamicExplorationGraph(deglib_cpp.load_readonly_graph(str(p)))


def load_dynamic_graph(path: pathlib.Path | str, chunk_size: int = 1024) -> DynamicExplorationGraph:
    """
    Load a saved graph as a mutable ``DynamicGraph``.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Graph file not found: {p}")
    return DynamicExplorationGraph(deglib_cpp.load_dynamic_graph(str(p), chunk_size))


def load_mutable_graph(path: pathlib.Path | str, capacity: int = 0) -> DynamicExplorationGraph:
    """
    Load a saved graph as a mutable ``SizeBoundedGraph``.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Graph file not found: {p}")
    return DynamicExplorationGraph(deglib_cpp.load_sizebounded_graph(str(p), capacity))


__all__ = [
    "DynamicExplorationGraph",
    "get_num_useful_threads",
    "create_empty",
    "create_dynamic_empty",
    "create_random_graph",
    "load_graph",
    "load_readonly_graph",
    "load_dynamic_graph",
    "load_mutable_graph",
]
