import numpy as np
import deglib
from deglib.optimization import presort

# Try relative import when executed inside VIBE framework (vibe/algorithms/deg/module.py)
try:
    from ..base.module import BaseANN
except (ImportError, ValueError):

    class BaseANN:
        pass


class DegANN(BaseANN):
    """
    VIBE BaseANN adapter for Dynamic Exploration Graph (DEG).
    Standard parameters:
      - metric: Distance metric ('euclidean', 'ip', 'normalized', 'cosine')
      - k: Degree / edges per vertex (e.g. 16, 24, 30, 40, 48)
      - extend_k: Additional neighbors checked during build (e.g. 2*k)
      - build_eps: Accuracy factor during graph construction (e.g. 0.1)
      - opt_target: 'HighLID' (for Cosine/IP) or 'LowLID' (for L2)
      - improve_k: Additional refinement neighbors (default: 0)
      - improve_eps: Additional refinement epsilon (default: 0.0)
      - threads: CPU threads during graph build
      - query_dtype: Dtype used to search graph ('float32', 'float16', 'int8', 'uint8')
      - k_rerank: Optional number of candidates to fetch from graph before reranking with FP16
      - prune_non_rng: Optional MRNG edge pruning
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "HighLID",
        extend_k: int = 60,
        build_eps: float = 0.1,
        improve_k: int = 0,
        improve_eps: float = 0.0,
        prune_non_rng: bool = False,
        threads: int = 1,
        query_dtype: str = "float32",
        k_rerank: int | None = None,
        graph_path: str | None = None,
    ):
        self.raw_metric = metric
        self.k = int(k)
        self.opt_target = str(opt_target)
        self.extend_k = int(extend_k)
        self.build_eps = float(build_eps)
        self.improve_k = int(improve_k)
        self.improve_eps = float(improve_eps)
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = int(threads)
        self.graph_path = graph_path
        self.k_rerank = int(k_rerank) if k_rerank is not None and int(k_rerank) > 0 else None
        self.query_dtype = self._normalize_dtype(query_dtype)

        self.search_eps: float = 0.1
        self.is_cosine: bool = metric.lower().strip() in ("cosine", "normalized")
        if self.is_cosine and self.query_dtype in ("uint8",):
            raise ValueError(
                "Affine UINT8 quantization is mathematically incompatible with Cosine metric "
                "(shifting by min destroys the inner product ranking of normalized vectors). "
                "Use 'int8', 'float16' or 'float32' for Cosine datasets."
            )

        self.base_metric_enum = self._map_metric(metric)
        self.query_metric_enum = self._resolve_metric_for_dtype(self.base_metric_enum, self.query_dtype)
        self.metric_enum = self.query_metric_enum
        self.opt_enum = self._map_opt_target(self.opt_target, metric)

        self.graph: deglib.ReadOnlyGraph | None = None
        self.original_features_fp16: np.ndarray | None = None
        self.rerank_space_fp16: deglib.FloatSpace | None = None
        self._batch_results: np.ndarray | None = None

    def _normalize_dtype(self, dtype_str: str) -> str:
        d = dtype_str.lower().strip()
        if d in ("float32", "fp32", "float", "f32"):
            return "float32"
        elif d in ("float16", "fp16", "half", "f16"):
            return "float16"
        elif d in ("int8", "sq8", "i8"):
            return "int8"
        elif d in ("uint8", "u8"):
            return "uint8"
        else:
            raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: 'float32', 'float16', 'int8', 'uint8'")

    def _resolve_metric_for_dtype(self, base_metric: deglib.Metric, dtype_str: str) -> deglib.Metric:
        is_l2 = base_metric in (deglib.Metric.FP32_L2, deglib.Metric.FP16_L2, deglib.Metric.Int8_L2, deglib.Metric.Uint8_L2)
        if dtype_str == "float32":
            return deglib.Metric.FP32_L2 if is_l2 else deglib.Metric.FP32_InnerProduct
        elif dtype_str == "float16":
            return deglib.Metric.FP16_L2 if is_l2 else deglib.Metric.FP16_InnerProduct
        elif dtype_str == "int8":
            return deglib.Metric.Int8_L2 if is_l2 else deglib.Metric.Int8_InnerProduct
        elif dtype_str == "uint8":
            return deglib.Metric.Uint8_L2 if is_l2 else deglib.Metric.Uint8_InnerProduct
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

    def _map_opt_target(self, opt_target: str, metric: str) -> deglib.builder.OptimizationTarget:
        target = opt_target.lower().strip()
        if target in ("streamingdata", "streaming", "streaming_data"):
            return deglib.builder.OptimizationTarget.StreamingData
        elif target == "highlid":
            return deglib.builder.OptimizationTarget.HighLID
        elif target == "lowlid":
            return deglib.builder.OptimizationTarget.LowLID
        elif target == "":
            m = metric.lower().strip()
            return (
                deglib.builder.OptimizationTarget.LowLID
                if m in ("euclidean", "l2", "fp32_l2", "int8_l2", "uint8_l2", "fp16_l2")
                else deglib.builder.OptimizationTarget.HighLID
            )
        else:
            raise ValueError(
                f"Unknown OptimizationTarget '{opt_target}'. Choose from: 'StreamingData', 'HighLID', 'LowLID', ''"
            )

    def _map_metric(self, metric: str) -> deglib.Metric:
        m = metric.lower().strip()
        if m in ("euclidean", "l2", "fp32_l2"):
            return deglib.Metric.FP32_L2
        elif m in ("ip", "innerproduct", "dot", "cosine", "normalized", "fp32_innerproduct"):
            return deglib.Metric.FP32_InnerProduct
        else:
            raise ValueError(f"Unsupported metric for DEG: {metric}")

    def _normalize(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(mat / norms, dtype=np.float32)

    def _to_float32(self, X: np.ndarray) -> np.ndarray:
        """Converts input array (FP16 or FP32) reliably to contiguous FP32."""
        if X.dtype == np.float16:
            return deglib.distances.fp16_to_floats(X.view(np.uint16)).reshape(X.shape)
        elif X.dtype == np.uint16:
            return deglib.distances.fp16_to_floats(X).reshape(X.shape)
        elif X.dtype != np.float32:
            return np.ascontiguousarray(X, dtype=np.float32)
        return np.ascontiguousarray(X, dtype=np.float32)

    def _convert_features_to_dtype(self, X_f32: np.ndarray, target_dtype: str, num_threads: int = 0) -> tuple[np.ndarray, deglib.FloatSpace]:
        target_metric = self._resolve_metric_for_dtype(self.base_metric_enum, target_dtype)
        dim = X_f32.shape[1]

        if target_dtype == "float32":
            feats = np.ascontiguousarray(X_f32, dtype=np.float32)
        elif target_dtype == "float16":
            feats = deglib.distances.floats_to_fp16(np.ascontiguousarray(X_f32, dtype=np.float32))
        elif target_dtype == "int8":
            feats = deglib.optimization.quantize_int8(X_f32, num_threads=num_threads)
        elif target_dtype == "uint8":
            feats = deglib.optimization.quantize_uint8(X_f32, num_threads=num_threads)
        else:
            raise ValueError(f"Unsupported target dtype: {target_dtype}")

        space = deglib.FloatSpace.create(dim=dim, metric=target_metric)
        return feats, space

    def _build_graph(self, X_f32: np.ndarray) -> deglib.DynamicExplorationGraph:
        """
        Phase 1: Graph Construction.
        - FLAS 1D pre-sorting is always executed in FP32 with radius_decay=0.9.
        - Graph is constructed as SizeBoundedGraph in FP32 for maximum accuracy and topology quality.
        - Saved to disk if graph_path is specified.
        """
        import time
        from pathlib import Path

        n_vectors, dims = X_f32.shape
        is_l2 = self.base_metric_enum in (deglib.Metric.FP32_L2, deglib.Metric.Int8_L2, deglib.Metric.Uint8_L2, deglib.Metric.FP16_L2)

        # 1. FLAS 1D Pre-sorting ALWAYS in float32
        flas_space = deglib.FloatSpace.create(
            dim=dims,
            metric=deglib.Metric.FP32_L2 if is_l2 else deglib.Metric.FP32_InnerProduct
        )
        print(
            f"Running FLAS 1D Pre-sorting in float32: N={n_vectors:,}, dim={dims}, threads={self.threads}..."
        )
        t_flas_start = time.perf_counter()
        last_pct = -1

        def flas_progress_cb(prog: float) -> bool:
            nonlocal last_pct
            pct = int(prog * 100.0)
            if pct != last_pct or prog >= 1.0:
                last_pct = pct
                print(f"\r  FLAS Progress: {pct:3d} %", end="", flush=True)
            if prog >= 1.0:
                print()
            return False

        sorted_indices = deglib.optimization.presort(
            X_f32,
            space=flas_space,
            radius_decay=0.9,
            threads=self.threads,
            callback=flas_progress_cb,
        )
        t_flas = time.perf_counter() - t_flas_start
        print(f"FLAS 1D Pre-sorting completed in {t_flas:.3f} s.")
        labels = sorted_indices
        ordered_f32 = X_f32[sorted_indices]

        # 2. Build graph ALWAYS in FP32
        build_metric = deglib.Metric.FP32_L2 if is_l2 else deglib.Metric.FP32_InnerProduct
        build_space = deglib.FloatSpace.create(dim=dims, metric=build_metric)
        print(f"Constructing DEG graph in float32 (Metric: {build_space.metric().name})...")

        graph_mut = deglib.create_empty(
            capacity=n_vectors,
            feature_space=build_space,
            edges_per_vertex=self.k,
        )

        builder = deglib.builder.GraphBuilder(
            graph_mut,
            seed=7,
            optimization_target=self.opt_enum,
            extend_k=self.extend_k,
            extend_eps=self.build_eps,
            improve_k=self.improve_k,
            improve_eps=self.improve_eps,
        )
        builder.set_batch_size(10, 10)
        if self.threads > 1:
            builder.set_thread_count(self.threads)

        builder.add_entry(labels, ordered_f32)
        builder.build(callback="progress")

        # Save graph in FP32 format if graph_path is given
        if self.graph_path:
            p = Path(self.graph_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            graph_mut.save_graph(str(p))
            print(f"Saved built DEG graph to: {p}")

        return graph_mut

    def _apply_post_graph_operations(self, graph_obj: deglib.DynamicExplorationGraph, X_f32: np.ndarray) -> deglib.DynamicExplorationGraph:
        """
        Phase 2: Post-Graph Operations for Search.
        - If pruning is requested, prunes non-RNG edges on SizeBoundedGraph.
        - Finalizes graph to ReadOnlyGraph with query_dtype features.
        """
        base_metric = graph_obj.get_feature_space().metric()
        self.base_metric_enum = base_metric
        target_metric = self._resolve_metric_for_dtype(base_metric, self.query_dtype)
        needs_feature_swap = (base_metric != target_metric)

        # 1. Edge Pruning on SizeBoundedGraph (MutableGraph)
        if self.prune_non_rng:
            mut_graph = graph_obj if graph_obj.is_mutable() else graph_obj.to_mutable()
            print(f"Pruning non-RNG edges on SizeBoundedGraph...")
            removed = deglib.optimization.prune_non_rng_edges(mut_graph, num_threads=self.threads)
            print(f"Pruned {removed:,} non-RNG edges.")
            graph_obj = mut_graph

        # 2. Conversion to ReadOnlyGraph + Feature Quantization
        if needs_feature_swap:
            print(f"Finalizing ReadOnlyGraph with query_dtype='{self.query_dtype}' (Metric: {target_metric.name})...")
            q_features, target_space = self._convert_features_to_dtype(X_f32, self.query_dtype, num_threads=self.threads)
            ro_graph = graph_obj.to_readonly(target_space, q_features)
            self.metric_enum = target_metric
        else:
            ro_graph = graph_obj if not graph_obj.is_mutable() else graph_obj.to_readonly()
            self.metric_enum = base_metric

        return ro_graph

    def fit(self, X: np.ndarray):
        """
        Builds or loads the DEG graph for corpus X and applies post-graph operations.
        - Accepts FP16 and FP32 inputs seamlessly.
        """
        import os

        X_f32 = self._to_float32(X)
        if self.is_cosine:
            X_f32 = self._normalize(X_f32)

        n_vectors, dims = X_f32.shape
        is_l2 = self.base_metric_enum in (deglib.Metric.FP32_L2, deglib.Metric.Int8_L2, deglib.Metric.Uint8_L2, deglib.Metric.FP16_L2)

        # Store original features in FP16 for fast and memory-efficient reranking
        if self.k_rerank is not None:
            print(f"Storing base features in FP16 for reranking ({n_vectors:,} x {dims})...")
            self.original_features_fp16 = deglib.distances.floats_to_fp16(X_f32)
            self.rerank_space_fp16 = deglib.FloatSpace.create(
                dim=dims,
                metric=deglib.Metric.FP16_L2 if is_l2 else deglib.Metric.FP16_InnerProduct
            )

        # 1. Obtain base graph in FP32 (either by loading cached file or by building)
        if self.graph_path and os.path.isfile(self.graph_path):
            if self.prune_non_rng:
                print(f"Loading cached DEG graph directly as SizeBoundedGraph from: {self.graph_path}")
                raw_graph = deglib.load_mutable_graph(str(self.graph_path))
            else:
                print(f"Loading cached DEG graph from: {self.graph_path}")
                raw_graph = deglib.load_readonly_graph(str(self.graph_path))
        else:
            raw_graph = self._build_graph(X_f32)

        # 2. Apply post-graph operations (Pruning -> ReadOnly + Feature Quantization for search)
        self.graph = self._apply_post_graph_operations(raw_graph, X_f32)

    def set_query_arguments(self, search_eps: float):
        """Sets query-time search_eps."""
        self.search_eps = float(search_eps)

    def _prepare_query_matrix(self, queries: np.ndarray, num_threads: int = 1) -> np.ndarray:
        q_f32 = self._to_float32(queries)
        if self.is_cosine:
            q_f32 = self._normalize(q_f32)

        if self.query_dtype == "float32":
            return q_f32
        elif self.query_dtype == "float16":
            return deglib.distances.floats_to_fp16(q_f32)
        elif self.query_dtype == "int8":
            return deglib.optimization.quantize_int8(q_f32, num_threads=num_threads)
        elif self.query_dtype == "uint8":
            return deglib.optimization.quantize_uint8(q_f32, num_threads=num_threads)
        else:
            return q_f32

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with optional FP16 reranking."""
        if self.graph is None:
            raise RuntimeError("Index not fitted. Call fit(X) first.")
        query_mat = self._prepare_query_matrix(v.reshape(1, -1), num_threads=1)

        fetch_k = max(n, self.k_rerank) if self.k_rerank is not None else n
        res = self.graph.search(
            query_mat, eps=self.search_eps, k=fetch_k, threads=1, return_distances=False, unsorted=True
        )
        candidates = res[0] if isinstance(res, tuple) else res

        if self.k_rerank is not None and self.original_features_fp16 is not None:
            v_f32 = self._to_float32(v.reshape(1, -1))
            if self.is_cosine:
                v_f32 = self._normalize(v_f32)
            query_fp16 = deglib.distances.floats_to_fp16(v_f32)
            reranked = deglib.search.rerank(
                space=self.rerank_space_fp16,
                queries=query_fp16,
                candidate_indices=np.ascontiguousarray(candidates, dtype=np.uint32),
                base_vectors=self.original_features_fp16,
                k_top=n,
                num_threads=1,
                unsorted=False,
            )
            return np.asarray(reranked[0], dtype=np.int64)

        return np.asarray(candidates[0][:n], dtype=np.int64)

    def batch_query(self, X: np.ndarray, n: int):
        """Batch query search on 1 thread with optional FP16 reranking."""
        if self.graph is None:
            raise RuntimeError("Index not fitted. Call fit(X) first.")
        query_mat = self._prepare_query_matrix(X, num_threads=self.threads)

        fetch_k = max(n, self.k_rerank) if self.k_rerank is not None else n
        indices, _ = self.graph.search(query_mat, eps=self.search_eps, k=fetch_k, threads=1, unsorted=True)

        if self.k_rerank is not None and self.original_features_fp16 is not None:
            X_f32 = self._to_float32(X)
            if self.is_cosine:
                X_f32 = self._normalize(X_f32)
            queries_fp16 = deglib.distances.floats_to_fp16(X_f32)
            reranked = deglib.search.rerank(
                space=self.rerank_space_fp16,
                queries=queries_fp16,
                candidate_indices=np.ascontiguousarray(indices, dtype=np.uint32),
                base_vectors=self.original_features_fp16,
                k_top=n,
                num_threads=1,
                unsorted=False,
            )
            self._batch_results = np.asarray(reranked, dtype=np.int64)
        else:
            self._batch_results = np.asarray(indices[:, :n], dtype=np.int64)

    def get_batch_results(self) -> np.ndarray:
        if self._batch_results is None:
            raise RuntimeError("No batch results available.")
        return self._batch_results

    def __str__(self) -> str:
        return (
            f"DEG(k={self.k}, extend_k={self.extend_k}, build_eps={self.build_eps}, "
            f"opt={self.opt_target}, improve_k={self.improve_k}, improve_eps={self.improve_eps}, "
            f"threads={self.threads}, query_dtype={self.query_dtype}, "
            f"k_rerank={self.k_rerank}, eps={self.search_eps})"
        )
