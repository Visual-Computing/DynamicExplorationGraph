from collections.abc import Sequence
from typing import Literal
import numpy as np
import deglib_cpp

from deglib.cpu import InstructionSet
from deglib.distances import Metric, _infer_dim_from_array
from deglib.graph import DynamicExplorationGraph
from deglib.transformation import (
    FeatureTransformation,
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

    Created by :meth:`DynamicExplorationGraph.to_recall_table` or :meth:`DynamicExplorationGraph.to_recall_tables`.
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

    Created by :meth:`DynamicExplorationGraph.to_pareto_line`.
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
    def entries(self) -> list[tuple[float, float, float, float, float]]:
        """Non-dominated frontier rows as ``(target_recall, eps_or_ef, measured_recall, latency_us, qps)`` tuples."""
        return [
            (float(tgt), float(param), float(meas), float(lat), float(qps))
            for tgt, param, meas, lat, qps in self._pareto_line_cpp.entries
        ]

    def eps_or_ef_values(self) -> list[float]:
        """Return deduplicated ``eps_or_ef`` parameters along the frontier in ascending recall order."""
        return [float(v) for v in self._pareto_line_cpp.eps_or_ef_values()]

    def param_for(self, target_recall: float, clamp: bool = False) -> float | None:
        """Look up or same-regime interpolate ``eps_or_ef`` for ``target_recall``."""
        val = self._pareto_line_cpp.param_for(float(target_recall), bool(clamp))
        return None if val is None else float(val)

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
        rerank_metric: Metric | str | None = None,
        rerank_data: np.ndarray | None = None,
        rerank_order: Literal["label", "internal"] = "label",
        rerank_transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        normalize_query: bool = False,
        optimize: bool = False,
        use_routing: bool = True,
        n_clusters: int = 256,
        n_iter: int = 20,
        sample_size: int = 0,
        threads: int = 0,
    ):
        if rerank_order not in ("label", "internal"):
            raise ValueError(f"rerank_order must be 'label' or 'internal', got {rerank_order!r}")
        if reranker is not None:
            if not isinstance(reranker, Reranker):
                raise TypeError(f"reranker must be a Reranker instance, got {type(reranker).__name__}")
            if rerank_metric is not None or rerank_data is not None or bool(rerank_transformations):
                raise ValueError(
                    "Cannot pass 'reranker' together with 'rerank_metric', 'rerank_data', or 'rerank_transformations'."
                )
            if reranker.get_num_base_vectors() < graph.size():
                raise ValueError(
                    f"reranker base vector count ({reranker.get_num_base_vectors()}) is smaller than graph.size() ({graph.size()})."
                )
        elif rerank_data is not None and rerank_metric is None:
            raise ValueError(
                "rerank_data cannot be provided when rerank_metric is None; "
                "specify rerank_metric (e.g. Metric.FP16_InnerProduct) to enable reranking."
            )

        self.graph = graph
        cpp_graph = getattr(graph, "dynamic_exploration_graph_cpp", graph)

        if reranker is not None:
            self._searcher_cpp = cpp_search.Searcher(
                cpp_graph,
                reranker._reranker_cpp,
                bool(normalize_query),
            )
        elif rerank_metric is not None:
            built_reranker = graph.to_reranker(
                rerank_metric,
                data=rerank_data,
                order=rerank_order,
                transformations=rerank_transformations,
                optimize=False,
                threads=threads,
            )
            self._searcher_cpp = cpp_search.Searcher(
                cpp_graph,
                built_reranker._reranker_cpp,
                bool(normalize_query),
            )
        else:
            self._searcher_cpp = cpp_search.Searcher(
                cpp_graph,
                None,
                bool(normalize_query),
            )

        if optimize and graph.size() > 0:
            self.optimize(
                n_clusters=n_clusters,
                n_iter=n_iter,
                sample_size=sample_size,
                threads=threads,
                use_routing=use_routing,
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


def create_searcher(
    graph: DynamicExplorationGraph,
    metric: Metric | str | None = None,
    reranker: Reranker | None = None,
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
) -> Searcher:
    """
    Create a native C++ ``Searcher`` pipeline from ``graph``.

    Reranking is opt-in via either a shared ``reranker`` instance or ``rerank_metric``:
    - ``reranker=None`` and ``rerank_metric=None`` (default): NO reranking is performed.
    - ``reranker`` set: reuses the prebuilt ``Reranker`` (zero-copy over its base vectors).
    - ``rerank_metric`` set, ``rerank_data=None``: extracts & quantizes rerank vectors directly
      from the ``FP32``/``FP16`` source ``graph`` prior to quantizing/converting ``graph`` to ``metric``.
    - ``rerank_metric`` set, ``rerank_data`` provided: aligns via ``rerank_order``.
    """
    if rerank_order not in ("label", "internal"):
        raise ValueError(f"rerank_order must be 'label' or 'internal', got {rerank_order!r}")
    if reranker is not None:
        if not isinstance(reranker, Reranker):
            raise TypeError(f"reranker must be a Reranker instance, got {type(reranker).__name__}")
        if rerank_metric is not None or rerank_data is not None or bool(rerank_transformations):
            raise ValueError(
                "Cannot pass 'reranker' together with 'rerank_metric', 'rerank_data', or 'rerank_transformations'."
            )
        if reranker.get_num_base_vectors() < graph.size():
            raise ValueError(
                f"reranker base vector count ({reranker.get_num_base_vectors()}) is smaller than graph.size() ({graph.size()})."
            )
    elif rerank_data is not None and rerank_metric is None:
        raise ValueError(
            "rerank_data cannot be provided when rerank_metric is None; "
            "specify rerank_metric (e.g. Metric.FP16_InnerProduct) to enable reranking."
        )

    # Build Reranker from source graph BEFORE converting/quantizing `graph`.
    if reranker is None and rerank_metric is not None:
        reranker = graph.to_reranker(
            rerank_metric,
            data=rerank_data,
            order=rerank_order,
            transformations=rerank_transformations,
            optimize=False,
            threads=threads,
        )

    # Quantize / convert graph only if `metric` was specified and differs in metric or explicit instruction.
    search_graph = graph
    if metric is not None:
        target_metric = Metric.resolve(metric)
        src_metric = graph.get_metric()
        src_inst = graph.get_instruction()
        explicit_inst = getattr(target_metric, "_instruction", None)
        needs_conversion = Metric(int(target_metric)) != Metric(int(src_metric)) or (
            explicit_inst is not None and explicit_inst != src_inst
        )
        if needs_conversion:
            search_graph = graph.to_readonly(
                metric=target_metric,
                use_rerank=(reranker is not None),
                optimize=False,
                threads=threads,
            )

    return Searcher(
        graph=search_graph,
        reranker=reranker,
        normalize_query=normalize_query,
        optimize=optimize,
        use_routing=use_routing,
        n_clusters=n_clusters,
        n_iter=n_iter,
        sample_size=sample_size,
        threads=threads,
    )


__all__ = [
    "Filter",
    "Reranker",
    "rerank",
    "Searcher",
    "SearchResults",
    "RecallTable",
    "ParetoLine",
    "create_searcher",
]
