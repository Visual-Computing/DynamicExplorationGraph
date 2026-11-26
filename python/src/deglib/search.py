from collections.abc import Sequence
from typing import Literal
import numpy as np
import deglib_cpp

from deglib.cpu import InstructionSet
from deglib.distances import Metric, _infer_dim_from_array
from deglib.graph import DynamicExplorationGraph
from deglib.transformation import (
    FeatureTransformation,
    TransformationType,
    _normalize_transforms,
)
from deglib.utils import assure_array, assure_contiguous

cpp_search = deglib_cpp.search
SearchResults = cpp_search.SearchResults


class Filter:
    """
    Bitset filter restricting graph search and exploration to a subset of external labels.
    """

    def __init__(self, valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1):
        self.valid_labels = np.asarray(valid_labels)
        if self.valid_labels.ndim != 1:
            raise ValueError(f"valid_labels must be a 1D array, got shape {self.valid_labels.shape}")
        if self.valid_labels.size == 0:
            self.max_value = 0 if max_value < 0 else max_value
            self.max_label_count = 0 if max_label_count < 0 else max_label_count
            valid_c = np.empty(0, dtype=np.int32)
        else:
            valid_c = assure_contiguous(self.valid_labels.astype(np.int32, copy=False), "valid_labels")
            self.max_value = int(valid_c.max()) if max_value < 0 else int(max_value)
            self.max_label_count = int(valid_c.size) if max_label_count < 0 else int(max_label_count)
        self.filter_cpp = cpp_search.create_filter(valid_c, self.max_value, self.max_label_count)

    @classmethod
    def create_filter(cls, valid_labels: "Filter | np.ndarray | Sequence[int]", max_value: int = -1) -> "Filter":
        if isinstance(valid_labels, cls):
            return valid_labels
        return cls(np.asarray(valid_labels), max_value=max_value)


class RecallTable:
    """
    Single-mode calibration table mapping target recall@k levels to ``ef`` or ``eps`` parameters.

    Created by :meth:`DynamicExplorationGraph.to_recall_table`, :meth:`DynamicExplorationGraph.to_recall_tables`,
    :meth:`Searcher.to_recall_table`, or :meth:`Searcher.to_recall_tables`.
    """

    def __init__(self, recall_table_cpp: cpp_search.RecallTable):
        if not isinstance(recall_table_cpp, cpp_search.RecallTable):
            raise TypeError(
                f"recall_table_cpp must be a deglib_cpp.search.RecallTable, got {type(recall_table_cpp).__name__}"
            )
        self._recall_table_cpp = recall_table_cpp

    @property
    def k(self) -> int:
        return int(self._recall_table_cpp.k)

    @property
    def sample_count(self) -> int:
        return int(self._recall_table_cpp.sample_count)

    @property
    def is_ef(self) -> bool:
        return bool(self._recall_table_cpp.is_ef)

    @property
    def is_eps(self) -> bool:
        return bool(self._recall_table_cpp.is_eps)

    @property
    def rerank_factor(self) -> float:
        return float(self._recall_table_cpp.rerank_factor)

    @property
    def recall_bounds(self) -> tuple[float, float] | None:
        bounds = self._recall_table_cpp.recall_bounds
        return None if bounds is None else (float(bounds[0]), float(bounds[1]))

    @property
    def entries(self) -> list[tuple[float, float, float]]:
        """Calibrated rows as ``(target_recall, eps_or_ef, measured_recall)`` tuples."""
        return [(float(t), float(param), float(m)) for t, param, m in self._recall_table_cpp.entries]

    def eps_or_ef_values(self) -> list[float]:
        """Return deduplicated calibrated ``eps_or_ef`` values in ascending order."""
        return [float(v) for v in self._recall_table_cpp.eps_or_ef_values()]

    def param_for(self, target_recall: float, clamp: bool = False) -> float | None:
        """Look up or linearly interpolate ``eps_or_ef`` for ``target_recall``."""
        val = self._recall_table_cpp.param_for(float(target_recall), bool(clamp))
        return None if val is None else float(val)

    def __len__(self) -> int:
        return len(self._recall_table_cpp)

    def __str__(self) -> str:
        return str(self._recall_table_cpp)

    def __repr__(self) -> str:
        return repr(self._recall_table_cpp)


class ParetoLine:
    """
    Unified recall-latency Pareto frontier combining one or more calibrated :class:`RecallTable` curves.

    Created by :meth:`DynamicExplorationGraph.to_pareto_line` or :meth:`Searcher.to_pareto_line`.
    """

    def __init__(self, pareto_line_cpp: cpp_search.ParetoLine):
        if not isinstance(pareto_line_cpp, cpp_search.ParetoLine):
            raise TypeError(
                f"pareto_line_cpp must be a deglib_cpp.search.ParetoLine, got {type(pareto_line_cpp).__name__}"
            )
        self._pareto_line_cpp = pareto_line_cpp

    @property
    def k(self) -> int:
        return int(self._pareto_line_cpp.k)

    @property
    def sample_count(self) -> int:
        return int(self._pareto_line_cpp.sample_count)

    @property
    def recall_bounds(self) -> tuple[float, float] | None:
        bounds = self._pareto_line_cpp.recall_bounds
        return None if bounds is None else (float(bounds[0]), float(bounds[1]))

    @property
    def recall_tables(self) -> list[RecallTable]:
        """Return the underlying :class:`RecallTable` curves used to construct this frontier."""
        return [RecallTable(t) for t in self._pareto_line_cpp.recall_tables]

    @property
    def entries(self) -> list[tuple[float, float, float, float, float, float]]:
        """Non-dominated frontier rows as ``(target_recall, eps_or_ef, rerank_factor, measured_recall, latency_us, qps)`` tuples."""
        return [
            (float(tgt), float(param), float(rf), float(meas), float(lat), float(qps))
            for tgt, param, rf, meas, lat, qps in self._pareto_line_cpp.entries
        ]

    def eps_or_ef_values(self) -> list[float]:
        """Return deduplicated ``eps_or_ef`` parameters along the frontier in ascending recall order."""
        return [float(v) for v in self._pareto_line_cpp.eps_or_ef_values()]

    def params(self) -> list[tuple[float, float]]:
        """Return deduplicated ``(eps_or_ef, rerank_factor)`` parameter pairs along the frontier in ascending recall order."""
        return [(float(param), float(rf)) for param, rf in self._pareto_line_cpp.params()]

    def param_for(self, target_recall: float, clamp: bool = False) -> tuple[float, float] | None:
        """Return ``(eps_or_ef, rerank_factor)`` of the frontier entry whose ``measured_recall`` is nearest to ``target_recall``."""
        val = self._pareto_line_cpp.param_for(float(target_recall), bool(clamp))
        return None if val is None else (float(val[0]), float(val[1]))

    def __len__(self) -> int:
        return len(self._pareto_line_cpp)

    def __str__(self) -> str:
        return str(self._pareto_line_cpp)

    def __repr__(self) -> str:
        return repr(self._pareto_line_cpp)


class Reranker:
    """
    Standalone owning C++ candidate reranker for exact scoring over candidate indices.
    """

    def __init__(
        self,
        metric: Metric | str,
        base_vectors: np.ndarray,
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        optimize: bool = False,
        threads: int = 0,
        row_indices: np.ndarray | None = None,
    ):
        """
        Initialize a standalone candidate ``Reranker``.

        Parameters
        ----------
        metric : Metric or str
            Distance metric used to score candidates (e.g. ``Metric.FP16_L2`` or
            ``Metric.FP32_InnerProduct``).
        base_vectors : np.ndarray
            2D array of base vectors. If ``float32``, vectors are automatically converted
            or quantized to ``metric`` using ``transformations`` (or an auto-calibrated
            quantizer). Otherwise, ``base_vectors`` must match the metric's accepted dtype.
        transformations : FeatureTransformation or sequence of FeatureTransformation, optional
            Fitted transformations applied to queries (and to ``float32`` ``base_vectors``
            if needed).
        optimize : bool, default=False
            If ``True``, tunes memory prefetching immediately after construction.
        threads : int, default=0
            Number of worker threads for quantization (``0`` = all cores).
        row_indices : np.ndarray, optional
            Optional 1D ``uint32`` index mapping (e.g. ``graph.get_labels()``) when
            ``base_vectors`` is indexed by external labels while the graph's internal
            vertex order differs.
        """
        resolved_metric = Metric.resolve(metric)
        xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")

        arr = np.asarray(base_vectors)
        if arr.ndim != 2:
            raise ValueError(f"base_vectors must be a 2D array, got shape {arr.shape}")
        arr = np.ascontiguousarray(arr)

        idx = None if row_indices is None else np.ascontiguousarray(row_indices, dtype=np.uint32)

        self.metric = resolved_metric
        if arr.dtype == np.float32:
            target_dim = int(xfrms[-1].output_dim) if xfrms else int(arr.shape[1])
            cpp_space = resolved_metric._to_cpp_space(target_dim)
            self._reranker_cpp = cpp_search.Reranker.from_float32(
                cpp_space,
                arr,
                xfrms,
                int(threads),
                idx,
            )
            self._dim = int(self._reranker_cpp.get_dim())
        elif arr.dtype == np.float16:
            target_dim = int(xfrms[-1].output_dim) if xfrms else int(arr.shape[1])
            cpp_space = resolved_metric._to_cpp_space(target_dim)
            self._reranker_cpp = cpp_search.Reranker.from_fp16(
                cpp_space,
                arr,
                xfrms,
                int(threads),
                idx,
            )
            self._dim = int(self._reranker_cpp.get_dim())
        else:
            arr = assure_array(arr, "base_vectors", resolved_metric.get_accepted_dtypes())
            dim = _infer_dim_from_array(resolved_metric, arr)
            cpp_space = resolved_metric._to_cpp_space(dim)
            self._reranker_cpp = cpp_search.Reranker.from_raw(
                cpp_space,
                arr,
                xfrms,
                int(threads),
                idx,
            )
            self._dim = dim

        if optimize and self.get_num_base_vectors() > 0:
            self.optimize()

    @classmethod
    def _from_cpp(cls, metric: Metric, reranker_cpp, optimize: bool = False) -> "Reranker":
        obj = cls.__new__(cls)
        obj.metric = metric
        obj._reranker_cpp = reranker_cpp
        obj._dim = int(reranker_cpp.get_dim())
        if optimize and obj.get_num_base_vectors() > 0:
            obj.optimize()
        return obj

    def get_num_base_vectors(self) -> int:
        return int(self._reranker_cpp.get_num_base_vectors())

    def get_instruction(self) -> InstructionSet:
        return InstructionSet.resolve(self._reranker_cpp.get_instruction())

    def get_po(self) -> int:
        return int(self._reranker_cpp.get_po())

    def get_pl(self) -> int:
        return int(self._reranker_cpp.get_pl())

    def optimize(self, sample_count: int = 300, num_candidates: int = 150) -> None:
        """
        Auto-tune candidate prefetch parameters (``po``, ``pl``).
        """
        self._reranker_cpp.optimize(int(sample_count), int(num_candidates))

    def rerank(
        self,
        queries: np.ndarray,
        candidate_indices: np.ndarray,
        k_top: int = 0,
        threads: int = 0,
        return_distances: bool = False,
        unsorted: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Score and rank ``candidate_indices`` for ``queries`` against stored base vectors.
        """
        queries = np.ascontiguousarray(queries)
        cand = np.ascontiguousarray(candidate_indices, dtype=np.uint32)
        return self._reranker_cpp.rerank(
            queries,
            cand,
            k_top=int(k_top),
            threads=int(threads),
            return_distances=bool(return_distances),
            unsorted=bool(unsorted),
        )


def rerank(
    metric: Metric | str,
    queries: np.ndarray,
    candidate_indices: np.ndarray,
    base_vectors: np.ndarray | None = None,
    k_top: int = 0,
    threads: int = 0,
    return_distances: bool = False,
    unsorted: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Stateless candidate reranking convenience function.
    Defaults ``base_vectors`` to ``queries`` (self-join) when ``None``.
    """
    target_vecs = queries if base_vectors is None else base_vectors
    ranker = Reranker(metric=metric, base_vectors=target_vecs, optimize=False, threads=threads)
    return ranker.rerank(
        queries=queries,
        candidate_indices=candidate_indices,
        k_top=k_top,
        threads=threads,
        return_distances=return_distances,
        unsorted=unsorted,
    )


class Searcher:
    """
    Native C++ search pipeline coordinating optional query normalization, feature
    transformation/quantization, K-Means routing, graph traversal, and optional reranking.
    """

    def __init__(
        self,
        graph: DynamicExplorationGraph,
        reranker: Reranker | None = None,
        normalize_query: bool = False,
        optimize: bool = False,
        threads: int = 0,
    ):
        """
        Create a ``Searcher`` wrapping a ``ReadOnlyGraph`` and an optional ``Reranker``.

        Parameters
        ----------
        graph : DynamicExplorationGraph
            Backing exploration graph (typically a ``ReadOnlyGraph``).
        reranker : Reranker, optional
            Optional standalone ``Reranker`` for two-stage candidate re-scoring.
        normalize_query : bool, default=False
            If ``True``, L2-normalizes float32 queries before graph traversal.
        optimize : bool, default=False
            If ``True``, calls ``self.optimize(threads=threads)`` after construction.
        threads : int, default=0
            Number of worker threads used when ``optimize=True``.
        """
        if reranker is not None:
            if not isinstance(reranker, Reranker):
                raise TypeError(f"reranker must be a Reranker instance, got {type(reranker).__name__}")
            if reranker.get_num_base_vectors() < graph.size():
                raise ValueError(
                    f"reranker base vector count ({reranker.get_num_base_vectors()}) is smaller than graph.size() ({graph.size()})."
                )

        self.graph = graph
        self.reranker = reranker
        cpp_graph = getattr(graph, "dynamic_exploration_graph_cpp", graph)

        self._searcher_cpp = cpp_search.Searcher(
            cpp_graph,
            reranker._reranker_cpp if reranker is not None else None,
            bool(normalize_query),
        )

        if optimize and graph.size() > 0:
            self.optimize(threads=threads)

    def __repr__(self) -> str:
        backend = "MutableGraph" if self.graph.is_mutable() else "ReadOnlyGraph"
        rr_str = self.reranker.metric.name if self.reranker is not None else "None"
        return (
            f"Searcher(graph={backend}[{self.graph.get_metric().name}, "
            f"size={self.graph.size()}, dim={self.graph.get_dim()}], "
            f"reranker={rr_str})"
        )

    @property
    def normalize_query(self) -> bool:
        return bool(self._searcher_cpp.normalize_query)

    @normalize_query.setter
    def normalize_query(self, value: bool) -> None:
        self._searcher_cpp.normalize_query = bool(value)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        eps_or_ef: float = 0.1,
        rerank_factor: float = 1.0,
        threads: int = 1,
        return_distances: bool = False,
        unsorted: bool = False,
        max_distance_computation_count: int = 0,
    ) -> SearchResults:
        """
        Search for the ``k`` nearest neighbors of a 1D query or 2D query batch.

        Returns a ``SearchResults`` object with zero-copy NumPy ``.indices``,
        optional ``.distances`` (or ``None`` when ``return_distances=False``),
        and traversal telemetry (``.graph_distance_count``, ``.rerank_distance_count``,
        ``.graph_hop_count``).
        """
        if type(query) is not np.ndarray:
            query = np.ascontiguousarray(query, dtype=np.float32)

        return self._searcher_cpp.search(
            query,
            k,
            eps_or_ef,
            rerank_factor,
            threads,
            return_distances,
            unsorted,
            max_distance_computation_count,
        )

    def optimize(
        self,
        n_clusters: int = 256,
        n_iter: int = 20,
        sample_size: int = 0,
        threads: int = 0,
        use_routing: bool = True,
    ) -> None:
        """
        Optimize K-Means routing medoids, reranker prefetch, and graph traversal prefetch.
        """
        self._searcher_cpp.optimize(
            n_clusters=int(n_clusters),
            n_iter=int(n_iter),
            sample_size=int(sample_size),
            threads=int(threads),
            use_routing=bool(use_routing),
        )

    def to_recall_table(
        self,
        k: int,
        mode: Literal["ef", "eps"],
        rerank_factor: float = 1.0,
        sample_count: int = 300,
        recall_range: tuple[float, float] | None = None,
        recall_step: float | None = None,
        target_recalls: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
    ) -> RecallTable:
        """
        Calibrate a single-mode ``RecallTable`` (``"ef"`` or ``"eps"``) for this ``Searcher`` pipeline.
        """
        from deglib.graph import _resolve_recall_args

        if mode not in ("ef", "eps"):
            raise ValueError(f"mode must be 'ef' or 'eps', got {mode!r}")

        r_range, r_step, targets_list = _resolve_recall_args(recall_range, recall_step, target_recalls)
        return RecallTable(
            self._searcher_cpp.to_recall_table(
                int(k),
                mode,
                float(rerank_factor),
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
        rerank_factors: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
    ) -> list[RecallTable]:
        """
        Calibrate ``"ef"`` and ``"eps"`` ``RecallTable`` curves across ``rerank_factors`` in a shared single pass.
        """
        from deglib.graph import _resolve_recall_args

        r_range, r_step, targets_list = _resolve_recall_args(recall_range, recall_step, target_recalls)
        factors_list = _resolve_rerank_factors(rerank_factors)
        cpp_tables = self._searcher_cpp.to_recall_tables(
            int(k),
            int(sample_count),
            r_range,
            r_step,
            targets_list,
            factors_list,
            int(threads),
        )
        return [RecallTable(t) for t in cpp_tables]

    def to_pareto_line(
        self,
        k: int | None = None,
        recall_tables: RecallTable | Sequence[RecallTable] | None = None,
        sample_count: int | None = None,
        recall_range: tuple[float, float] | None = None,
        recall_step: float | None = None,
        target_recalls: Sequence[float] | np.ndarray | None = None,
        rerank_factors: Sequence[float] | np.ndarray | None = None,
        threads: int = 0,
        rounds: int = 3,
    ) -> ParetoLine:
        """
        Build an end-to-end ``ParetoLine`` unifying measured operating points across ``recall_tables`` in ``(measured_recall, latency_us)`` space.

        If ``recall_tables`` is omitted, calibrates ``("ef", "eps") x rerank_factors`` via :meth:`to_recall_tables`.
        """
        if int(rounds) < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        if recall_tables is not None:
            if (
                k is not None
                or recall_range is not None
                or recall_step is not None
                or target_recalls is not None
                or rerank_factors is not None
            ):
                raise ValueError(
                    "Cannot specify 'recall_tables' together with 'k', 'recall_range', 'recall_step', 'target_recalls', or 'rerank_factors'."
                )
            tables = _coerce_recall_tables(recall_tables)
            if sample_count is not None and int(sample_count) < 0:
                raise ValueError(f"sample_count must be >= 0, got {sample_count}")
            eff_samples = 0 if sample_count is None else int(sample_count)
            cpp_line = self._searcher_cpp.to_pareto_line(
                [t._recall_table_cpp for t in tables],
                eff_samples,
                int(rounds),
            )
            return ParetoLine(cpp_line)

        if k is None:
            raise ValueError("Must specify either 'k' or 'recall_tables'.")

        eff_samples = 300 if sample_count is None else int(sample_count)
        if eff_samples < 0:
            raise ValueError(f"sample_count must be >= 0, got {sample_count}")
        tables = self.to_recall_tables(
            k=int(k),
            sample_count=eff_samples,
            recall_range=recall_range,
            recall_step=recall_step,
            target_recalls=target_recalls,
            rerank_factors=rerank_factors,
            threads=threads,
        )
        cpp_line = self._searcher_cpp.to_pareto_line(
            [t._recall_table_cpp for t in tables],
            0,
            int(rounds),
        )
        return ParetoLine(cpp_line)


def _resolve_rerank_metric(
    distance_kind: str,
    compress: Literal["16bit", "8bit"] | Metric | str | None,
) -> Metric:
    """Resolve target Metric for a Reranker from distance kind and compress level."""
    if compress is None:
        target = Metric.FP32_InnerProduct if distance_kind == "InnerProduct" else Metric.FP32_L2
    elif isinstance(compress, Metric):
        target = compress
    elif isinstance(compress, str):
        key = compress.strip().lower().replace("-", "").replace("_", "")
        if key in ("16bit", "fp16", "float16"):
            target = Metric.FP16_InnerProduct if distance_kind == "InnerProduct" else Metric.FP16_L2
        elif key in ("8bit", "int8"):
            target = Metric.Int8_InnerProduct if distance_kind == "InnerProduct" else Metric.Int8_L2
        elif key == "uint7":
            target = Metric.Uint7_InnerProduct if distance_kind == "InnerProduct" else Metric.Uint7_L2
        elif key == "uint8":
            target = Metric.Uint8_InnerProduct if distance_kind == "InnerProduct" else Metric.Uint8_L2
        elif key in ("32bit", "fp32", "float32"):
            target = Metric.FP32_InnerProduct if distance_kind == "InnerProduct" else Metric.FP32_L2
        elif "_" in compress:
            target = Metric.resolve(compress)
        else:
            target = Metric.resolve(compress, "ip" if distance_kind == "InnerProduct" else "l2")
    else:
        target = Metric.resolve(compress)

    return target


def reranker(
    graph: DynamicExplorationGraph,
    compress: Literal["16bit", "8bit"] | Metric | str | None = "16bit",
    optimize: bool = True,
    threads: int = 0,
) -> Reranker:
    """
    Create a ``Reranker`` directly from an uncompressed (``FP32`` or ``FP16``) ``graph``.

    Raises ``ValueError`` if ``graph`` is already quantized (use :func:`reranker_with_data` instead).

    Parameters
    ----------
    graph : DynamicExplorationGraph
        Backing uncompressed graph whose stored vectors and labels are used for reranking.
    compress : {"16bit", "8bit"} or Metric or str, optional, default="16bit"
        Target reranker precision:

        - ``None`` / ``"32bit"`` / ``"fp32"`` / ``"float32"``: ``FP32_L2`` or ``FP32_InnerProduct``.
        - ``"16bit"`` / ``"fp16"`` / ``"float16"``: ``FP16_L2`` or ``FP16_InnerProduct``.
        - ``"8bit"`` / ``"auto"``: ``Uint7_L2`` (for L2), ``Int8_InnerProduct`` (for unnormalized ``InnerProduct``),
          or auto-selected ``Uint7_L2`` vs. ``Int8_InnerProduct`` for unit-sphere ``cosine``/``normalized`` graphs.
        - ``"int8"``: ``Int8_L2`` or ``Int8_InnerProduct``.
        - ``"uint8"`` / ``"uint7"`` or any explicit ``Metric`` / metric string.
    optimize : bool, default=True
        If ``True``, runs ``reranker.optimize()`` to tune memory prefetching.
    threads : int, default=0
        Number of worker threads used during quantization/optimization (``0`` = all CPU cores).

    Returns
    -------
    Reranker
        Standalone reranker aligned with ``graph.get_labels()``.

    See Also
    --------
    deglib.Reranker : Low-level reranker class.
    deglib.DynamicExplorationGraph.to_reranker : Low-level graph-to-reranker conversion.
    deglib.search.rerank : Stateless candidate reranking helper.
    """
    base_metric = Metric(int(graph.get_metric()))
    if base_metric not in (
        Metric.FP32_L2,
        Metric.FP32_InnerProduct,
        Metric.FP16_L2,
        Metric.FP16_InnerProduct,
    ):
        raise ValueError(
            f"Cannot extract reranker from quantized graph ({base_metric.name}). "
            "Use deglib.reranker_with_data(graph, data, ...) with the original float vectors instead."
        )
    distance_kind = "InnerProduct" if base_metric.name.endswith("InnerProduct") else "L2"
    if isinstance(compress, str) and compress.strip().lower().replace("-", "").replace("_", "") in ("8bit", "auto"):
        if distance_kind == "L2":
            target_metric = Metric.Uint7_L2
        else:
            from deglib.tuning import predict_quantization

            target_metric = predict_quantization(graph, threads=threads).metric
    else:
        target_metric = _resolve_rerank_metric(distance_kind, compress)
    return graph.to_reranker(target_metric, optimize=optimize, threads=threads)


def reranker_with_data(
    graph: DynamicExplorationGraph,
    data: np.ndarray,
    labels: Sequence[int] | np.ndarray | None = None,
    compress: Literal["16bit", "8bit"] | Metric | str | None = "16bit",
    metric: Literal["l2", "cosine", "normalized", "ip"] | Metric | str | None = None,
    optimize: bool = True,
    threads: int = 0,
) -> Reranker:
    """
    Create a ``Reranker`` from external ``data`` aligned to ``graph`` (supports quantized/presorted graphs).

    Preserves any spatial prefix transformations (``MipsTransform``, ``L2NormalizeTransform``)
    attached to ``graph`` and maps ``graph.get_labels()`` to rows in ``data`` without copying ``data``.

    Parameters
    ----------
    graph : DynamicExplorationGraph
        Target graph whose internal vertex order (``graph.get_labels()``) the reranker aligns to.
    data : np.ndarray
        2D array of original ``float32`` or ``float16`` feature vectors.
    labels : sequence of int or np.ndarray, optional
        External labels corresponding to rows of ``data``. Defaults to ``0..len(data)-1``.
    compress : {"16bit", "8bit"} or Metric or str, optional, default="16bit"
        Target reranker precision (``"16bit"``, ``"8bit"``, ``"32bit"``, ``None``, or explicit
        ``Metric`` / metric string; see :func:`reranker`).
    metric : {"l2", "cosine", "normalized", "ip"} or Metric or str, optional
        Optional distance kind override (``"l2"`` / ``"euclidean"``, ``"cosine"``,
        ``"normalized"``, ``"ip"`` / ``"innerproduct"`` / ``"mips"``, or explicit ``Metric``).
        Defaults to ``graph.get_metric()``.
    optimize : bool, default=True
        If ``True``, runs ``reranker.optimize()`` to tune memory prefetching.
    threads : int, default=0
        Number of worker threads used during quantization/optimization (``0`` = all CPU cores).

    Returns
    -------
    Reranker
        Standalone reranker aligned with ``graph.get_labels()``.

    See Also
    --------
    deglib.Reranker : Low-level reranker class.
    deglib.DynamicExplorationGraph.to_reranker : Low-level graph-to-reranker conversion.
    deglib.search.rerank : Stateless candidate reranking helper.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array, got shape {arr.shape}")

    spatial_prefix: list[FeatureTransformation] = []
    for t in graph.get_transformations():
        if t.type == TransformationType.Spatial:
            spatial_prefix.append(t)
        else:
            break

    has_mips = any(t.name == "MipsTransform" for t in spatial_prefix)
    if has_mips:
        distance_kind = "L2"
    elif metric is not None:
        if isinstance(metric, str) and metric.strip().lower() in ("l2", "euclidean"):
            distance_kind = "L2"
        elif isinstance(metric, str) and metric.strip().lower() in (
            "ip",
            "innerproduct",
            "inner_product",
            "mips",
            "cosine",
        ):
            distance_kind = "InnerProduct"
        else:
            distance_kind = "InnerProduct" if Metric(int(Metric.resolve(metric))).name.endswith("InnerProduct") else "L2"
    else:
        distance_kind = "InnerProduct" if Metric(int(graph.get_metric())).name.endswith("InnerProduct") else "L2"

    if isinstance(compress, str) and compress.strip().lower().replace("-", "").replace("_", "") in ("8bit", "auto"):
        if distance_kind == "L2":
            target_metric = Metric.Uint7_L2
        else:
            from deglib.transformation import is_l2_normalized
            from deglib.tuning import predict_quantization, predict_quantization_normalized

            has_l2n = any(t.name == "L2NormalizeTransform" for t in spatial_prefix)
            is_explicit_ip = isinstance(metric, str) and metric.strip().lower() in (
                "ip",
                "innerproduct",
                "inner_product",
                "mips",
            )
            if is_explicit_ip and not has_l2n:
                target_metric = Metric.Int8_InnerProduct
            else:
                n_arr = arr.shape[0]
                is_unit = n_arr >= 2 and is_l2_normalized(arr[: min(n_arr, 1024)])
                if has_l2n:
                    target_metric = predict_quantization(
                        arr, metric=Metric.FP32_InnerProduct, threads=threads
                    ).metric
                elif is_unit:
                    target_metric = predict_quantization_normalized(
                        arr, is_cosine=False, metric=Metric.FP32_InnerProduct, threads=threads
                    ).metric
                elif isinstance(metric, str) and metric.strip().lower() in ("cosine", "normalized"):
                    target_metric = predict_quantization(
                        arr, metric=Metric.FP32_InnerProduct, threads=threads
                    ).metric
                else:
                    target_metric = Metric.Int8_InnerProduct
    else:
        target_metric = _resolve_rerank_metric(distance_kind, compress)

    graph_labels = graph.get_labels()
    if labels is None:
        row_indices = graph_labels
    else:
        labels_arr = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels, dtype=np.uint32)
        if labels_arr.ndim != 1 or labels_arr.shape[0] != arr.shape[0]:
            raise ValueError(f"labels must be 1D of length {arr.shape[0]}, got shape {labels_arr.shape}")
        if (
            labels_arr.shape[0] > 0
            and labels_arr[0] == 0
            and np.all(np.diff(labels_arr) == 1)
            and np.all(graph_labels < labels_arr.shape[0])
        ):
            row_indices = graph_labels
        elif labels_arr.shape[0] > 0 and np.all(labels_arr[:-1] <= labels_arr[1:]):
            pos = np.searchsorted(labels_arr, graph_labels)
            if np.any(pos >= labels_arr.shape[0]) or not np.all(labels_arr[pos] == graph_labels):
                raise KeyError("Some graph labels were not found in the provided labels array")
            row_indices = pos.astype(np.uint32, copy=False)
        else:
            lookup = {int(lbl): idx for idx, lbl in enumerate(labels_arr)}
            try:
                row_indices = np.array([lookup[int(g_lbl)] for g_lbl in graph_labels], dtype=np.uint32)
            except KeyError as exc:
                raise KeyError(f"Graph label {exc} was not found in the provided labels array") from exc

    xfrms: list[FeatureTransformation] | None = None
    if spatial_prefix:
        target_base = Metric(int(target_metric))
        if target_base in (Metric.FP32_L2, Metric.FP32_InnerProduct):
            xfrms = spatial_prefix
        else:
            from deglib.transformation import apply_on_database, make_quantizer_pipeline

            n_rows = int(row_indices.shape[0])
            if n_rows > 0:
                sample_count = min(n_rows, 10000)
                step = max(1, n_rows // sample_count)
                sample_idx = row_indices[::step][:sample_count]
                sample = np.ascontiguousarray(arr[sample_idx], dtype=np.float32)
                transformed_sample = apply_on_database(spatial_prefix, sample, threads=threads)
                quant_xfrms = make_quantizer_pipeline(
                    transformed_sample,
                    target_metric,
                    use_rerank=True,
                    threads=threads,
                )
                xfrms = spatial_prefix + quant_xfrms
            else:
                xfrms = spatial_prefix

    return Reranker(
        metric=target_metric,
        base_vectors=arr,
        transformations=xfrms,
        optimize=optimize,
        threads=threads,
        row_indices=row_indices,
    )


def searcher(
    graph: DynamicExplorationGraph,
    compress: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str | None = None,
    reranker: Reranker | bool | Metric | str | None = None,
    optimize: bool = True,
    threads: int = 0,
) -> Searcher:
    """
    Create an optimized ``Searcher`` for ``graph``, optionally compressing the search graph first.

    Memory & Ownership
    ------------------
    If ``graph`` is already a ``ReadOnlyGraph`` and ``compress=None``, ``searcher.graph``
    shares ``graph`` zero-copy (``searcher.graph is graph``). If ``graph`` is mutable or
    ``compress`` is set, a new ``ReadOnlyGraph`` is created for ``searcher.graph``; release
    your reference to ``graph`` (e.g. ``del graph``) if it is no longer needed in RAM.

    Parameters
    ----------
    graph : DynamicExplorationGraph
        Backing exploration graph (usually a ``ReadOnlyGraph`` returned by :func:`deglib.build`).
    compress : {"32bit", "16bit", "8bit", "4bit", "2bit", "auto"} or Metric or str, optional
        Optional compression target applied via :func:`deglib.compress` before wrapping in a
        ``Searcher``:

        - ``None``: Use ``graph`` as-is without re-quantizing.
        - ``"32bit"`` (also ``"fp32"``, ``"float32"``): ``FP32_L2`` or ``FP32_InnerProduct``.
        - ``"16bit"`` (also ``"fp16"``, ``"float16"``): ``FP16_L2`` or ``FP16_InnerProduct``.
        - ``"8bit"`` / ``"auto"``: ``Uint7_L2`` (L2), ``Int8_InnerProduct`` (unnormalized ``InnerProduct``),
          or auto-selected ``Uint7_L2`` vs. ``Int8_InnerProduct`` on unit-sphere ``cosine``/``normalized`` graphs.
        - ``"4bit"``: ``Uint4_L2`` (on L2 and ``cosine``/``normalized`` graphs) or ``Int4_InnerProduct``
          (on unnormalized ``InnerProduct`` graphs, avoiding ``d+1`` ``MipsTransform`` overhead).
        - ``"2bit"`` (also ``"evp"``): ``EVP_InnerProduct``.
        - Explicit ``Metric`` or metric string (e.g. ``Metric.Int8_L2``, ``"Uint8_L2"``).
    reranker : Reranker or bool or Metric or str, optional
        Optional reranker for two-stage candidate re-scoring:

        - ``None`` / ``False``: No reranking.
        - ``Reranker``: Prebuilt reranker (from :func:`deglib.reranker` or
          :func:`deglib.reranker_with_data`).
        - ``True``: Automatically extracts a ``"16bit"`` reranker from ``graph`` before
          compression (requires ``graph`` to use an uncompressed ``FP32`` or ``FP16`` metric).
        - ``str`` or ``Metric`` (e.g. ``"8bit"``, ``"16bit"``, ``"32bit"``, ``Metric.Int8_L2``):
          Automatically builds a reranker with that precision from ``graph`` via
          :func:`deglib.reranker` before compression.
    optimize : bool, default=True
        If ``True``, runs ``searcher.optimize(threads=threads)`` (K-Means routing entry points
        and prefetch tuning). Pass ``optimize=False`` to call ``searcher.optimize(...)`` with
        custom ``n_clusters`` / ``use_routing`` settings manually.
    threads : int, default=0
        Number of worker threads for compression and optimization (``0`` = all CPU cores).

    Returns
    -------
    Searcher
        Native C++ search pipeline ready for ``searcher.search(...)``.

    See Also
    --------
    deglib.Searcher : Low-level searcher class.
    deglib.DynamicExplorationGraph.to_searcher : Low-level graph-to-searcher conversion.
    deglib.search.RecallTable : Calibration table mapping target recall to ``ef``/``eps``.
    deglib.search.ParetoLine : Unified recall-latency Pareto frontier.
    deglib.search.Filter : External-label filter for constrained search.
    """
    base_graph = graph.to_readonly() if (compress is not None and graph.is_mutable() and graph.size() > 0) else graph

    if reranker is None or reranker is False:
        resolved_reranker = None
    elif isinstance(reranker, Reranker):
        resolved_reranker = reranker
    elif reranker is True:
        resolved_reranker = globals()["reranker"](base_graph, compress="16bit", optimize=False, threads=threads)
    elif isinstance(reranker, (str, Metric)):
        resolved_reranker = globals()["reranker"](base_graph, compress=reranker, optimize=False, threads=threads)
    else:
        raise TypeError(
            f"reranker must be a Reranker, bool, str, Metric, or None, got {type(reranker).__name__}"
        )

    if compress is not None:
        from deglib.graph import compress as compress_graph

        search_graph = compress_graph(base_graph, compress, threads=threads)
    else:
        search_graph = base_graph
    return search_graph.to_searcher(reranker=resolved_reranker, optimize=optimize, threads=threads)


def _resolve_rerank_factors(rerank_factors: Sequence[float] | np.ndarray | None) -> list[float] | None:
    if rerank_factors is None:
        return None
    if isinstance(rerank_factors, (str, bytes)):
        raise TypeError(f"rerank_factors must be a sequence of floats >= 1.0, got {type(rerank_factors).__name__}")
    factors = [float(rf) for rf in rerank_factors]
    if not factors:
        raise ValueError("rerank_factors must not be empty")
    return factors


def _coerce_recall_tables(recall_tables: RecallTable | Sequence[RecallTable]) -> list[RecallTable]:
    if isinstance(recall_tables, RecallTable):
        return [recall_tables]
    if isinstance(recall_tables, Sequence) and not isinstance(recall_tables, (str, bytes)):
        tables = list(recall_tables)
        for i, t in enumerate(tables):
            if not isinstance(t, RecallTable):
                raise TypeError(f"recall_tables[{i}] must be a RecallTable instance, got {type(t).__name__}")
        return tables
    raise TypeError(
        f"recall_tables must be a RecallTable or sequence of RecallTable, got {type(recall_tables).__name__}"
    )


__all__ = [
    "Filter",
    "Reranker",
    "rerank",
    "Searcher",
    "SearchResults",
    "RecallTable",
    "ParetoLine",
]
