from collections.abc import Sequence
from typing import Literal
import numpy as np
import deglib_cpp

from deglib.distances import Metric, _infer_dim_from_array
from deglib.graph import DynamicExplorationGraph, _FP16_METRICS, _FP32_METRICS
from deglib.transformation import (
    FeatureTransformation,
    _normalize_transforms,
    apply_on_database,
    apply_on_query,
    make_quantizer_pipeline,
    make_scalar_quantizer_fp16,
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


def _prepare_rerank_vectors_and_transforms(
    metric: Metric,
    vectors: np.ndarray,
    transformations: list[FeatureTransformation],
    threads: int,
) -> tuple[np.ndarray, int, list[FeatureTransformation]]:
    arr = np.asarray(vectors)
    if arr.ndim != 2:
        raise ValueError(f"base_vectors must be a 2D array, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr)

    base_metric = Metric(int(metric))
    xfrms = list(transformations)

    if len(xfrms) == 0 and arr.dtype == np.float32 and base_metric not in _FP32_METRICS:
        if base_metric in _FP16_METRICS:
            q = make_scalar_quantizer_fp16(arr)
            xfrms = [q]
            arr = apply_on_database(xfrms, arr, threads=threads)
            dim = int(arr.shape[1])
        else:
            dim = int(arr.shape[1])
            xfrms = make_quantizer_pipeline(arr, metric, use_rerank=True, threads=threads)
            arr = apply_on_database(xfrms, arr, threads=threads)
    elif (
        len(xfrms) > 0
        and arr.dtype == np.float32
        and arr.shape[1] == xfrms[0].input_dim
        and (xfrms[-1].output_dtype != np.float32 or xfrms[-1].output_dim != arr.shape[1])
    ):
        arr = apply_on_database(xfrms, arr, threads=threads)
        dim = int(xfrms[-1].output_dim)
    else:
        arr = assure_array(arr, "base_vectors", metric.get_accepted_dtypes())
        dim = _infer_dim_from_array(metric, arr)

    return arr, dim, xfrms


class Reranker:
    """
    Standalone candidate reranker for exact scoring over candidate indices.
    """

    def __init__(
        self,
        metric: Metric | str,
        base_vectors: np.ndarray,
        transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
        optimize: bool = False,
        threads: int = 0,
    ):
        resolved_metric = Metric.resolve(metric)
        xfrms = _normalize_transforms(transformations)
        for i, t in enumerate(xfrms):
            if not isinstance(t, FeatureTransformation):
                raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")

        prepared_vectors, dim, prepared_xfrms = _prepare_rerank_vectors_and_transforms(
            resolved_metric, base_vectors, xfrms, threads
        )
        cpp_space = resolved_metric._to_cpp_space(dim)

        self.metric = resolved_metric
        self._dim = dim
        self._transformations = prepared_xfrms
        self._reranker_cpp = cpp_search.Reranker(cpp_space, prepared_vectors, prepared_xfrms)
        if optimize and prepared_vectors.shape[0] > 0:
            self.optimize()

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
        if not isinstance(queries, np.ndarray):
            queries = np.ascontiguousarray(queries, dtype=np.float32)
        elif not queries.flags.c_contiguous:
            queries = np.ascontiguousarray(queries)

        if self._transformations:
            expected_bytes = self.metric.get_data_size(self._dim)
            row_bytes = int(queries.shape[-1]) * queries.dtype.itemsize
            if queries.dtype == np.float32 or row_bytes != expected_bytes:
                if queries.dtype != np.float32:
                    queries = np.ascontiguousarray(queries, dtype=np.float32)
                queries = apply_on_query(self._transformations, queries)

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
        else:
            r_xfrms = _normalize_transforms(rerank_transformations)
            for i, t in enumerate(r_xfrms):
                if not isinstance(t, FeatureTransformation):
                    raise TypeError(
                        f"rerank_transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}"
                    )

            rerank_space_cpp = None
            prepared_rerank = None

            if rerank_metric is not None:
                resolved_rerank = Metric.resolve(rerank_metric)
                if rerank_data is None:
                    prepared_rerank, extracted_xfrms = deglib_cpp.extract_rerank_from_graph(
                        cpp_graph,
                        resolved_rerank._to_cpp_metric(),
                        int(threads),
                    )
                    if not r_xfrms:
                        if extracted_xfrms:
                            r_xfrms = extracted_xfrms
                        elif (
                            Metric(int(resolved_rerank)) in _FP16_METRICS
                            and Metric(int(graph.get_metric())) in _FP16_METRICS
                            and graph.has_transformations()
                        ):
                            r_xfrms = graph.get_transformations()
                    rerank_dim = graph.get_dim()
                else:
                    raw_rerank = np.asarray(rerank_data)
                    if raw_rerank.ndim != 2:
                        raise ValueError(f"rerank_data must be a 2D array, got shape {raw_rerank.shape}")
                    n_vertices = graph.size()
                    if rerank_order == "label":
                        if n_vertices == 0:
                            aligned = np.ascontiguousarray(raw_rerank[:0])
                        else:
                            labels = graph.get_labels()
                            if int(labels.max()) >= raw_rerank.shape[0]:
                                raise ValueError(
                                    f"Max graph external label ({int(labels.max())}) >= rerank_data row count ({raw_rerank.shape[0]}). "
                                    "Pass rerank_data indexed by external labels or set rerank_order='internal'."
                                )
                            aligned = np.ascontiguousarray(raw_rerank[labels])
                    else:
                        if raw_rerank.shape[0] != n_vertices:
                            raise ValueError(
                                f"rerank_data row count ({raw_rerank.shape[0]}) does not match graph.size() ({n_vertices})."
                            )
                        aligned = np.ascontiguousarray(raw_rerank)

                    prepared_rerank, rerank_dim, r_xfrms = _prepare_rerank_vectors_and_transforms(
                        resolved_rerank, aligned, r_xfrms, threads
                    )

                rerank_space_cpp = resolved_rerank._to_cpp_space(rerank_dim)

            self._searcher_cpp = cpp_search.Searcher(
                cpp_graph,
                rerank_space_cpp,
                prepared_rerank,
                r_xfrms,
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

    # Extract rerank vectors from source graph BEFORE converting/quantizing `graph`.
    resolved_rerank_metric = None if rerank_metric is None else Metric.resolve(rerank_metric)
    resolved_rerank_xfrms = _normalize_transforms(rerank_transformations)
    extracted_rerank_data = rerank_data
    effective_rerank_order: Literal["label", "internal"] = rerank_order

    if reranker is None and resolved_rerank_metric is not None and rerank_data is None:
        cpp_graph = getattr(graph, "dynamic_exploration_graph_cpp", graph)
        extracted_rerank_data, extracted_xfrms = deglib_cpp.extract_rerank_from_graph(
            cpp_graph,
            resolved_rerank_metric._to_cpp_metric(),
            int(threads),
        )
        effective_rerank_order = "internal"
        if not resolved_rerank_xfrms:
            if extracted_xfrms:
                resolved_rerank_xfrms = extracted_xfrms
            elif (
                Metric(int(resolved_rerank_metric)) in _FP16_METRICS
                and Metric(int(graph.get_metric())) in _FP16_METRICS
                and graph.has_transformations()
            ):
                resolved_rerank_xfrms = graph.get_transformations()

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
                use_rerank=(reranker is not None or resolved_rerank_metric is not None),
                optimize=False,
                threads=threads,
            )

    return Searcher(
        graph=search_graph,
        reranker=reranker,
        rerank_metric=resolved_rerank_metric,
        rerank_data=extracted_rerank_data,
        rerank_order=effective_rerank_order,
        rerank_transformations=resolved_rerank_xfrms or None,
        normalize_query=normalize_query,
        optimize=optimize,
        use_routing=use_routing,
        n_clusters=n_clusters,
        n_iter=n_iter,
        sample_size=sample_size,
        threads=threads,
    )


__all__ = ["Filter", "Reranker", "rerank", "Searcher", "create_searcher"]
