from collections.abc import Sequence
from typing import Literal
import numpy as np
import deglib_cpp

from deglib.distances import Metric, _infer_dim_from_array
from deglib.graph import DynamicExplorationGraph
from deglib.transformation import (
    FeatureTransformation,
    _normalize_transforms,
)
from deglib.utils import assure_array, assure_contiguous

cpp_search = deglib_cpp.search


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
            self._reranker_cpp = cpp_search.Reranker.from_float32(
                resolved_metric._to_cpp_metric(),
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
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Search for the ``k`` nearest neighbors of a 1D query or 2D query batch.
        """
        if not isinstance(query, np.ndarray):
            query = np.ascontiguousarray(query, dtype=np.float32)
        elif not query.flags.c_contiguous:
            query = np.ascontiguousarray(query)

        return self._searcher_cpp.search(
            query,
            k=int(k),
            eps_or_ef=float(eps_or_ef),
            rerank_factor=float(rerank_factor),
            threads=int(threads),
            return_distances=bool(return_distances),
            unsorted=bool(unsorted),
            max_distance_computation_count=int(max_distance_computation_count),
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


__all__ = ["Filter", "Reranker", "rerank", "Searcher", "create_searcher"]
