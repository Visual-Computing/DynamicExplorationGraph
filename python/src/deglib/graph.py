from collections.abc import Sequence
import os
import pathlib
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import deglib_cpp

if TYPE_CHECKING:
    from deglib.search import ParetoLine, RecallTable

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


def _resolve_recall_args(
    recall_range: tuple[float, float] | None,
    recall_step: float | None,
    target_recalls: Sequence[float] | np.ndarray | None,
) -> tuple[tuple[float, float], float, list[float] | None]:
    if target_recalls is not None:
        if recall_range is not None or recall_step is not None:
            raise ValueError("Cannot specify 'target_recalls' together with 'recall_range' or 'recall_step'.")
        arr = np.asarray(target_recalls, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"target_recalls must be 1D, got shape {arr.shape}")
        return (0.80, 0.999), 0.005, [float(x) for x in arr]

    if recall_range is None:
        r_range = (0.80, 0.999)
    else:
        if len(recall_range) != 2:
            raise ValueError(f"recall_range must be a (min, max) 2-tuple, got length {len(recall_range)}")
        r_range = (float(recall_range[0]), float(recall_range[1]))

    r_step = 0.005 if recall_step is None else float(recall_step)
    return r_range, r_step, None


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
        Retrieve neighbor external labels for a single vertex or a batch of vertices.

        Parameters
        ----------
        external_label : int, np.ndarray, or sequence of int
            Single external vertex label (returns ``list[int]``) or 1D batch of external
            labels (returns 2D ``uint32`` array of shape ``(len(external_label), edges_per_vertex)``).
        threads : int, default=1
            Number of worker threads for batch retrieval (ignored for a single label).

        Returns
        -------
        list[int] or np.ndarray
            Neighbor external labels for the requested vertex or vertices.
        """
        if isinstance(external_label, (np.ndarray, list, tuple)):
            arr = np.ascontiguousarray(external_label, dtype=np.uint32)
            return self.dynamic_exploration_graph_cpp.get_neighbors(arr, threads)
        return self.dynamic_exploration_graph_cpp.get_neighbors(int(external_label))

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

    def to_recall_table(
        self,
        k: int,
        mode: Literal["ef", "eps"],
        sample_count: int = 300,
        recall_range: tuple[float, float] | None = None,
        recall_step: float | None = None,
        target_recalls: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
    ) -> "RecallTable":
        """
        Calibrate a single-mode ``RecallTable`` (``"ef"`` or ``"eps"``) mapping target recall@k levels
        to graph search parameters.
        """
        from deglib.search import RecallTable

        if mode not in ("ef", "eps"):
            raise ValueError(f"mode must be 'ef' or 'eps', got {mode!r}")

        r_range, r_step, targets_list = _resolve_recall_args(recall_range, recall_step, target_recalls)
        return RecallTable(
            self.dynamic_exploration_graph_cpp.to_recall_table(
                int(k),
                mode,
                int(sample_count),
                r_range,
                r_step,
                targets_list,
                int(threads),
            )
        )

    def to_recall_tables(
        self,
        k: int,
        sample_count: int = 300,
        recall_range: tuple[float, float] | None = None,
        recall_step: float | None = None,
        target_recalls: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
    ) -> list["RecallTable"]:
        """
        Calibrate both ``"ef"`` and ``"eps"`` ``RecallTable`` curves in one pass sharing exact ground truth.
        """
        from deglib.search import RecallTable

        r_range, r_step, targets_list = _resolve_recall_args(recall_range, recall_step, target_recalls)
        cpp_tables = self.dynamic_exploration_graph_cpp.to_recall_tables(
            int(k),
            int(sample_count),
            r_range,
            r_step,
            targets_list,
            int(threads),
        )
        return [RecallTable(t) for t in cpp_tables]

    def to_pareto_line(
        self,
        k: int | None = None,
        recall_tables: "RecallTable | Sequence[RecallTable] | None" = None,
        sample_count: int | None = None,
        recall_range: tuple[float, float] | None = None,
        recall_step: float | None = None,
        target_recalls: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
        rounds: int = 3,
    ) -> "ParetoLine":
        """
        Build a unified ``ParetoLine`` from measured operating points across ``recall_tables`` in ``(measured_recall, latency_us)`` space.

        If ``recall_tables`` is omitted, calibrates both ``"ef"`` and ``"eps"`` tables via :meth:`to_recall_tables`.
        Pareto-dominated points are pruned so the resulting frontier is strictly monotonic in recall and latency.
        """
        from deglib.search import ParetoLine, RecallTable

        if int(rounds) < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        if recall_tables is not None:
            if k is not None or recall_range is not None or recall_step is not None or target_recalls is not None:
                raise ValueError(
                    "Cannot specify 'recall_tables' together with 'k', 'recall_range', 'recall_step', or 'target_recalls'."
                )
            if isinstance(recall_tables, RecallTable):
                tables = [recall_tables]
            elif isinstance(recall_tables, Sequence) and not isinstance(recall_tables, (str, bytes)):
                tables = list(recall_tables)
                for i, t in enumerate(tables):
                    if not isinstance(t, RecallTable):
                        raise TypeError(f"recall_tables[{i}] must be a RecallTable instance, got {type(t).__name__}")
            else:
                raise TypeError(
                    f"recall_tables must be a RecallTable or sequence of RecallTable, got {type(recall_tables).__name__}"
                )
            if sample_count is not None and int(sample_count) < 0:
                raise ValueError(f"sample_count must be >= 0, got {sample_count}")
            eff_samples = 0 if sample_count is None else int(sample_count)
            cpp_line = self.dynamic_exploration_graph_cpp.to_pareto_line(
                [t._recall_table_cpp for t in tables],
                eff_samples,
                int(rounds),
            )
            return ParetoLine(cpp_line)

        if k is None:
            raise ValueError("Must specify either 'k' or 'recall_tables'.")

        eff_samples = 300 if sample_count is None else int(sample_count)
        tables = self.to_recall_tables(
            k=int(k),
            sample_count=eff_samples,
            recall_range=recall_range,
            recall_step=recall_step,
            target_recalls=target_recalls,
            threads=threads,
        )
        cpp_line = self.dynamic_exploration_graph_cpp.to_pareto_line(
            [t._recall_table_cpp for t in tables],
            0,
            int(rounds),
        )
        return ParetoLine(cpp_line)

    def save_graph(self, path: pathlib.Path | str) -> None:
        """
        Save graph topology, features, transformations, and prefetch config to disk (``.deg`` format).
        """
        p = pathlib.Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        target = self if self.is_mutable() else self.to_mutable()
        target.dynamic_exploration_graph_cpp.save_graph(str(p))

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
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        optimize: bool = True,
        threads: int = 0,
    ):
        """
        Create a standalone ``Reranker`` aligned with this graph's internal vertex order.

        Extracts and quantizes stored vectors directly from an ``FP32`` or ``FP16`` graph
        and inherits any prefix transformations already attached to the graph (such as
        ``MipsTransform``), appending ``transformations`` if provided.

        For external datasets not stored in the graph, construct ``deglib.Reranker(...)``
        directly instead.

        Parameters
        ----------
        metric : Metric or str
            Target distance metric for the reranker (e.g. ``Metric.FP16_L2`` or
            ``Metric.FP32_InnerProduct``).
        transformations : FeatureTransformation or sequence of FeatureTransformation, optional
            Additional fitted transformations applied after the graph's own prefix
            transformations.
        optimize : bool, default=True
            If ``True``, immediately runs ``reranker.optimize()`` to tune memory prefetching.
        threads : int, default=0
            Number of worker threads used during quantization (``0`` selects all cores).

        Returns
        -------
        Reranker
            Standalone reranker aligned with ``self.get_labels()``.
        """
        from deglib.search import Reranker

        resolved_metric = Metric.resolve(metric)
        r_xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(r_xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")

        target_dim = int(r_xfrms[-1].output_dim) if r_xfrms else self.get_dim()
        cpp_space = resolved_metric._to_cpp_space(target_dim)
        reranker_cpp = deglib_cpp.search.Reranker.from_graph(
            self.dynamic_exploration_graph_cpp,
            cpp_space,
            r_xfrms,
            int(threads),
        )
        return Reranker._from_cpp(resolved_metric, reranker_cpp, optimize=optimize)

    def _prepare_metric_transform(
        self,
        resolved_metric: Metric,
        use_rerank: bool,
        threads: int,
    ) -> tuple[list[FeatureTransformation], Any]:
        """Resolve C++ transformations and target FloatSpace for a target metric."""
        src_metric = self.get_metric()
        src_dim = self.get_dim()
        same_data_type = (int(resolved_metric) & 0xF0) == (int(src_metric) & 0xF0)
        if same_data_type:
            return [], resolved_metric._to_cpp_space(src_dim)

        if Metric(int(src_metric)) not in _FLOAT_METRICS:
            raise ValueError(
                f"Cannot change metric from non-float metric '{src_metric.name}' to '{resolved_metric.name}'."
            )
        if Metric(int(resolved_metric)) in _FP32_METRICS:
            raise ValueError(
                f"Cannot switch from '{src_metric.name}' to '{resolved_metric.name}' without explicit transformations; "
                "use with_transformations() instead."
            )
        src_is_ip = Metric(int(src_metric)).name.endswith("InnerProduct")
        dst_is_l2 = not Metric(int(resolved_metric)).name.endswith("InnerProduct")
        has_spatial_prefix = any(
            t.name in ("L2NormalizeTransform", "MipsTransform") for t in self.get_transformations()
        )
        need_mips = src_is_ip and dst_is_l2 and not has_spatial_prefix
        xfrms = cpp_transformation.make_quantizer_pipeline_from_graph(
            self.dynamic_exploration_graph_cpp,
            int(resolved_metric),
            -1.0,
            use_rerank,
            threads,
            7,
            False,
            need_mips,
        )
        return xfrms, resolved_metric._to_cpp_space(int(xfrms[-1].output_dim))

    def to_searcher(
        self,
        metric: Metric | str | None = None,
        reranker: Any | None = None,
        normalize_query: bool = False,
        optimize: bool = True,
        threads: int = 0,
    ):
        """
        Create a native C++ ``Searcher`` from this graph.

        If ``metric`` differs from the graph's current metric (or requests an explicit
        SIMD instruction), a quantized/re-tagged ``ReadOnlyGraph`` is built in a single
        pass (without recomputing mutable edge weights or allocating intermediate graph
        copies) while preserving any attached prefix transformations (e.g. ``MipsTransform``).

        Parameters
        ----------
        metric : Metric or str, optional
            Optional target metric for the backing search graph. When ``None`` or equal
            to the graph's current metric/instruction, no feature transformation is run.
        reranker : Reranker, optional
            Optional prebuilt ``Reranker`` (e.g. from ``graph.to_reranker(...)`` or
            ``deglib.Reranker(...)``) for two-stage candidate reranking.
        normalize_query : bool, default=False
            If ``True``, L2-normalizes incoming float32 queries before graph traversal.
        optimize : bool, default=True
            If ``True``, calls ``searcher.optimize(threads=threads)`` to compute K-Means
            routing entry points and tune prefetching. For custom clustering parameters,
            pass ``optimize=False`` and call ``searcher.optimize(...)`` manually.
        threads : int, default=0
            Number of worker threads for quantization and optimization (``0`` = all cores).

        Returns
        -------
        Searcher
            High-throughput C++ searcher backed by a ``ReadOnlyGraph``.
        """
        from deglib.search import Searcher

        search_graph = self
        if metric is not None:
            target_metric = Metric.resolve(metric)
            src_metric = self.get_metric()
            src_inst = self.get_instruction()
            explicit_inst = getattr(target_metric, "_instruction", None)
            needs_conversion = Metric(int(target_metric)) != Metric(int(src_metric)) or (
                explicit_inst is not None and explicit_inst != src_inst
            )
            if needs_conversion:
                xfrms, target_space = self._prepare_metric_transform(
                    target_metric,
                    use_rerank=(reranker is not None),
                    threads=threads,
                )
                search_graph = DynamicExplorationGraph(
                    self.dynamic_exploration_graph_cpp._transform_readonly(xfrms, target_space, int(threads))
                )

        if search_graph.is_mutable() and search_graph.size() > 0:
            search_graph = search_graph.to_readonly()

        return Searcher(
            graph=search_graph,
            reranker=reranker,
            normalize_query=normalize_query,
            optimize=optimize,
            threads=threads,
        )

    def to_readonly(self) -> "DynamicExplorationGraph":
        """
        Convert this graph into a compact, immutable ``ReadOnlyGraph``.

        This is a pure backend copy that preserves topology, external labels, stored
        features, metric, and attached query transformations.

        Returns
        -------
        DynamicExplorationGraph
            A new read-only graph sharing no mutable state with ``self``.
        """
        return DynamicExplorationGraph(self.dynamic_exploration_graph_cpp.to_readonly())

    def to_mutable(self, max_size: int = 0, threads: int = 0) -> "DynamicExplorationGraph":
        """
        Convert this graph into a fixed-capacity mutable ``SizeBoundedGraph``.

        This is a pure backend copy that preserves topology, external labels, stored
        features, metric, and attached query transformations. Existing edge weights
        are copied directly from mutable graphs, or recomputed in parallel when
        converting from a ``ReadOnlyGraph``.

        Parameters
        ----------
        max_size : int, default=0
            Maximum vertex capacity of the new mutable graph. ``0`` preserves the
            current graph's vertex count (or capacity if larger).
        threads : int, default=0
            Number of worker threads used to recompute edge weights when converting
            from a ``ReadOnlyGraph`` (``0`` = all available cores).

        Returns
        -------
        DynamicExplorationGraph
            A new fixed-capacity mutable graph.

        Raises
        ------
        ValueError
            If ``max_size < 0``.
        """
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_mutable(max_size=int(max_size), threads=int(threads))
        )

    def to_dynamic(self, chunk_size: int = 1024, threads: int = 0) -> "DynamicExplorationGraph":
        """
        Convert this graph into a chunk-allocated mutable ``DynamicGraph``.

        This is a pure backend copy that preserves topology, external labels, stored
        features, metric, and attached query transformations. Existing edge weights
        are copied directly from mutable graphs, or recomputed in parallel when
        converting from a ``ReadOnlyGraph``.

        Parameters
        ----------
        chunk_size : int, default=1024
            Number of vertices allocated per memory chunk.
        threads : int, default=0
            Number of worker threads used to recompute edge weights when converting
            from a ``ReadOnlyGraph`` (``0`` = all available cores).

        Returns
        -------
        DynamicExplorationGraph
            A new dynamically growing mutable graph.

        Raises
        ------
        ValueError
            If ``chunk_size <= 0``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.to_dynamic(chunk_size=int(chunk_size), threads=int(threads))
        )

    def with_metric(
        self,
        metric: Metric | str,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Return a copy of this graph in ``metric``, preserving the current backend.

        When the target data type matches the graph's stored data type (e.g. ``FP32_L2``
        to ``FP32_InnerProduct``), stored features are preserved without transformation
        (edge weights are recomputed if mutable). When moving from floating-point features
        to a quantized or FP16 metric, an auto-calibrated quantizer pipeline is applied
        and appended to ``transformations_``.

        Parameters
        ----------
        metric : Metric or str
            Target distance metric (e.g. ``Metric.FP32_InnerProduct``, ``Metric.Int8_L2``).
        threads : int, default=0
            Number of worker threads for calibration, feature transformation, and
            edge-weight recomputation on mutable graphs (``0`` = all available cores).

        Returns
        -------
        DynamicExplorationGraph
            New graph in the requested ``metric`` using the same backend as ``self``.
        """
        resolved_metric = Metric.resolve(metric)
        xfrms, target_space = self._prepare_metric_transform(
            resolved_metric,
            use_rerank=False,
            threads=threads,
        )
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.with_transformations(xfrms, target_space, int(threads))
        )

    def with_transformations(
        self,
        transformations: FeatureTransformation | Sequence[FeatureTransformation],
        metric: Metric | str | None = None,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Self-transform stored floating-point features, preserving the current backend.

        Transforms the graph's stored ``FP32``/``FP16`` vectors using the fitted
        ``transformations`` and appends them to the graph's existing query pipeline.
        If ``self`` is mutable, edge weights are recomputed in the target space while
        keeping existing capacity or chunk size.

        Parameters
        ----------
        transformations : FeatureTransformation or sequence of FeatureTransformation
            Fitted transformations to apply to stored features and append to queries.
        metric : Metric or str, optional
            Optional target metric. Inferred automatically from ``transformations`` when omitted.
        threads : int, default=0
            Number of worker threads for feature transformation and edge-weight
            recomputation on mutable graphs (``0`` = all available cores).

        Returns
        -------
        DynamicExplorationGraph
            New graph using the same backend as ``self`` with transformed features and
            chained query transformations.
        """
        xfrms = _normalize_transforms(transformations)
        if len(xfrms) == 0:
            raise ValueError("with_transformations() requires at least one FeatureTransformation.")
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")
            if not t.is_fitted:
                raise ValueError(f"transformations[{i}] must be fitted before with_transformations()")

        src_dim = self.get_dim()
        if xfrms[0].input_dim != src_dim:
            raise ValueError(
                f"transformations[0].input_dim ({xfrms[0].input_dim}) does not match graph dim ({src_dim})."
            )

        target_space = None
        if metric is not None:
            resolved_metric = Metric.resolve(metric)
            expected_bytes = resolved_metric.get_data_size(int(xfrms[-1].output_dim))
            if xfrms[-1].output_byte_size != expected_bytes:
                raise ValueError(
                    f"transformations[-1].output_byte_size ({xfrms[-1].output_byte_size}) does not match "
                    f"target metric data size ({expected_bytes})."
                )
            target_space = resolved_metric._to_cpp_space(int(xfrms[-1].output_dim))

        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.with_transformations(xfrms, target_space, int(threads))
        )

    def with_features(
        self,
        features: np.ndarray,
        metric: Metric | str | None = None,
        order: Literal["label", "internal"] = "label",
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        threads: int = 0,
    ) -> "DynamicExplorationGraph":
        """
        Replace stored features with external ``features``, preserving topology, labels, and backend.

        Resets the graph's query-transformation pipeline to ``transformations`` (or an
        auto-calibrated quantizer pipeline when ``features`` is floating-point, ``metric``
        is quantized/FP16, and ``transformations`` is ``None``). Keeps the same backend
        (``ReadOnlyGraph``, ``SizeBoundedGraph`` capacity, or ``DynamicGraph`` chunk size).

        Parameters
        ----------
        features : np.ndarray
            2D array of replacement features.
        metric : Metric or str, optional
            Target metric for the new graph. Defaults to the current graph's metric.
        order : {"label", "internal"}, default="label"
            Row ordering of ``features``. ``"label"`` indexes rows by external vertex
            labels (``features[graph.get_labels()]``); ``"internal"`` expects rows to
            already match the graph's internal vertex order.
        transformations : FeatureTransformation or sequence of FeatureTransformation, optional
            Fitted transformations defining the new query pipeline (and applied to
            ``features`` if ``features`` is float32 in the input space).
        threads : int, default=0
            Number of worker threads for calibration/quantization and (for mutable
            backends) edge-weight recomputation.

        Returns
        -------
        DynamicExplorationGraph
            New graph using the same backend as ``self`` with replacement features and a
            reset query-transformation pipeline.
        """
        if order not in ("label", "internal"):
            raise ValueError(f"order must be 'label' or 'internal', got {order!r}")

        feat_arr = np.asarray(features)
        if feat_arr.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {feat_arr.shape}.")

        n_vertices = self.size()
        if order == "label":
            if n_vertices == 0:
                aligned = np.ascontiguousarray(feat_arr[:0])
            else:
                labels = self.get_labels()
                if int(labels.max()) >= feat_arr.shape[0]:
                    raise ValueError(
                        f"Max graph external label ({int(labels.max())}) >= features row count ({feat_arr.shape[0]}). "
                        "Pass an array indexed by external labels or set order='internal'."
                    )
                aligned = np.ascontiguousarray(feat_arr[labels])
        else:
            if feat_arr.shape[0] != n_vertices:
                raise ValueError(
                    f"features row count ({feat_arr.shape[0]}) does not match graph.size() ({n_vertices})."
                )
            aligned = np.ascontiguousarray(feat_arr)

        xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")
            if not t.is_fitted:
                raise ValueError(f"transformations[{i}] must be fitted before with_features()")

        resolved_metric = self.get_metric() if metric is None else Metric.resolve(metric)
        target_base = Metric(int(resolved_metric))
        is_float_input = aligned.dtype == np.float32 or (
            aligned.dtype == np.float16 and target_base not in _FP16_METRICS
        )
        if len(xfrms) == 0 and target_base not in _FP32_METRICS and is_float_input:
            target_dim = int(aligned.shape[1])
            xfrms = make_quantizer_pipeline(aligned, resolved_metric, use_rerank=False, threads=threads)
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
            aligned = assure_array(aligned, "features", resolved_metric.get_accepted_dtypes())
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
                f"features size ({aligned.nbytes} bytes) does not match expected size "
                f"({n_vertices * expected_bytes} bytes for {n_vertices} vertices x {expected_bytes} bytes/vertex)."
            )

        target_space = resolved_metric._to_cpp_space(target_dim)
        return DynamicExplorationGraph(
            self.dynamic_exploration_graph_cpp.with_features(target_space, aligned, xfrms, int(threads))
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
    Load a saved ``.deg`` graph from disk as an immutable ``ReadOnlyGraph``.

    See Also
    --------
    deglib.graph.load_readonly_graph : Explicit read-only loader.
    deglib.graph.load_dynamic_graph : Load as a chunked mutable ``DynamicGraph``.
    deglib.graph.load_mutable_graph : Load as a fixed-capacity mutable ``SizeBoundedGraph``.
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


def compress(
    graph: DynamicExplorationGraph,
    target: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str,
    threads: int = 0,
) -> DynamicExplorationGraph:
    """
    Compress ``graph`` to the specified bit-width or ``Metric``, preserving the input graph's backend type.

    Parameters
    ----------
    graph : DynamicExplorationGraph
        Source graph (must use an uncompressed ``FP32`` or ``FP16`` metric unless
        ``target="32bit"`` on an existing ``FP32`` graph).
    target : {"32bit", "16bit", "8bit", "4bit", "2bit", "auto"} or Metric or str
        Target precision or metric:

        - ``"32bit"`` (also ``"fp32"``, ``"float32"``): ``FP32_L2`` or ``FP32_InnerProduct``
          matching the graph's distance kind (returns ``graph`` if already ``FP32``).
        - ``"16bit"`` (also ``"fp16"``, ``"float16"``): ``FP16_L2`` or ``FP16_InnerProduct``
          matching the graph's distance kind.
        - ``"8bit"`` / ``"auto"``: ``Uint7_L2`` (for L2) or ``Int8_InnerProduct`` (for InnerProduct).
        - ``"4bit"``: ``Uint4_L2`` (for L2) or ``L2NormalizeTransform + Int4_InnerProduct`` (for InnerProduct).
        - ``"2bit"`` (also ``"evp"``): ``EVP_InnerProduct`` (always InnerProduct).
        - Explicit ``Metric`` or metric string (e.g. ``"Int8_L2"``, ``Metric.Uint8_L2``).
    threads : int, default=0
        Number of worker threads used during quantization (``0`` = all CPU cores).

    Returns
    -------
    DynamicExplorationGraph
        Compressed graph preserving the input graph's backend type, topology, and external labels.

    See Also
    --------
    deglib.graph.DynamicExplorationGraph.with_metric : Low-level metric conversion.
    deglib.graph.DynamicExplorationGraph.with_transformations : Explicit transformation pipeline.
    deglib.transformation.make_quantizer_pipeline : Calibrate a custom quantizer pipeline.
    """
    # See docs/python_metric_quantization_matrix.md (Section 3) for target resolution rules.
    src_metric = graph.get_metric()
    src_is_ip = Metric(int(src_metric)).name.endswith("InnerProduct")

    if isinstance(target, Metric):
        target_metric = target
    elif isinstance(target, str):
        key = target.strip().lower().replace("-", "").replace("_", "")
        if key in ("32bit", "fp32", "float32"):
            target_metric = Metric.FP32_InnerProduct if src_is_ip else Metric.FP32_L2
            if Metric(int(src_metric)) == Metric(int(target_metric)):
                return graph
        elif key in ("16bit", "fp16", "float16"):
            target_metric = Metric.FP16_InnerProduct if src_is_ip else Metric.FP16_L2
        elif key in ("8bit", "auto"):
            target_metric = Metric.Int8_InnerProduct if src_is_ip else Metric.Uint7_L2
        elif key == "4bit":
            target_metric = Metric.Int4_InnerProduct if src_is_ip else Metric.Uint4_L2
            if src_is_ip:
                has_l2n = any(t.name == "L2NormalizeTransform" for t in graph.get_transformations())
                if not has_l2n:
                    xfrms = cpp_transformation.make_quantizer_pipeline_from_graph(
                        graph.dynamic_exploration_graph_cpp,
                        int(target_metric),
                        -1.0,
                        False,
                        threads,
                        7,
                        True,
                    )
                    return graph.with_transformations(xfrms, metric=target_metric, threads=threads)
        elif key in ("2bit", "evp"):
            target_metric = Metric.EVP_InnerProduct
        elif "_" in target:
            target_metric = Metric.resolve(target)
        else:
            target_metric = Metric.resolve(target, "ip" if src_is_ip else "l2")
    else:
        target_metric = Metric.resolve(target)

    return graph.with_metric(target_metric, threads=threads)


__all__ = [
    "DynamicExplorationGraph",
    "get_num_useful_threads",
    "compress",
    "create_empty",
    "create_dynamic_empty",
    "create_random_graph",
    "load_graph",
    "load_readonly_graph",
    "load_dynamic_graph",
    "load_mutable_graph",
]
