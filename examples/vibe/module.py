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
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        extend_k: int = 60,
        build_eps: float = 0.1,
        opt_target: str = "HighLID",
        improve_k: int = 0,
        improve_eps: float = 0.0,
        threads: int = 1,
        graph_path: str | None = None,
        use_flas: bool = False,
        flas_radius_decay: float = 0.9,
    ):
        self.raw_metric = metric
        self.k = int(k)
        self.extend_k = int(extend_k)
        self.build_eps = float(build_eps)
        self.opt_target = str(opt_target)
        self.improve_k = int(improve_k)
        self.improve_eps = float(improve_eps)
        self.threads = int(threads)
        self.graph_path = graph_path
        self.use_flas = bool(use_flas)
        self.flas_radius_decay = float(flas_radius_decay)

        self.search_eps: float = 0.1
        self.is_cosine: bool = metric.lower().strip() == "cosine"
        self.metric_enum = self._map_metric(metric)
        self.opt_enum = self._map_opt_target(self.opt_target, metric)

        self.graph: deglib.ReadOnlyGraph | None = None
        self._batch_results: np.ndarray | None = None

    def _map_opt_target(self, opt_target: str, metric: str) -> deglib.builder.OptimizationTarget:
        target = opt_target.lower().strip()
        if target in ("streamingdata", "streaming", "streaming_data"):
            return deglib.builder.OptimizationTarget.StreamingData
        elif target == "highlid":
            return deglib.builder.OptimizationTarget.HighLID
        elif target == "lowlid":
            return deglib.builder.OptimizationTarget.LowLID
        elif target == "":
            # Default fallback based on metric if empty
            m = metric.lower().strip()
            return (
                deglib.builder.OptimizationTarget.LowLID
                if m in ("euclidean", "l2", "fp32_l2")
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
        elif m in ("uint8", "uint8_l2", "u8"):
            return deglib.Metric.Uint8_L2
        elif m in ("ip", "innerproduct", "dot", "cosine", "normalized", "fp32_innerproduct"):
            return deglib.Metric.FP32_InnerProduct
        else:
            raise ValueError(f"Unsupported metric for DEG: {metric}")

    def _normalize(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(mat / norms, dtype=np.float32)

    def fit(self, X: np.ndarray):
        """
        Builds or loads the DEG graph for corpus X.
        If graph_path is provided and exists on disk, the graph is loaded directly.
        Otherwise, the graph is built and saved to graph_path if specified.
        """
        import os
        import time
        from pathlib import Path

        if self.graph_path and os.path.isfile(self.graph_path):
            print(f"Loading cached DEG graph from: {self.graph_path}")
            self.graph = deglib.load_readonly_graph(str(self.graph_path))
            return

        is_uint8 = X.dtype == np.uint8 or self.metric_enum == deglib.Metric.Uint8_L2
        if is_uint8:
            X_mat = np.ascontiguousarray(X, dtype=np.uint8)
            self.metric_enum = deglib.Metric.Uint8_L2
        else:
            X_mat = np.ascontiguousarray(X, dtype=np.float32)
            if self.is_cosine:
                X_mat = self._normalize(X_mat)

        n_vectors, dims = X_mat.shape

        space = deglib.FloatSpace.create(dim=dims, metric=self.metric_enum)

        # Optional FLAS 1D pre-sorting before graph construction
        if self.use_flas and not is_uint8:
            print(
                f"Running FLAS 1D Pre-sorting: N={n_vectors:,}, dim={dims}, decay={self.flas_radius_decay}, metric={self.metric_enum.name}, threads={self.threads}..."
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
                X_mat,
                space=space,
                radius_decay=self.flas_radius_decay,
                threads=self.threads,
                callback=flas_progress_cb,
            )
            t_flas = time.perf_counter() - t_flas_start
            print(f"FLAS 1D Pre-sorting completed in {t_flas:.3f} s.")
            labels = sorted_indices
            features_to_add = X_mat[sorted_indices]
        else:
            labels = np.arange(n_vectors, dtype=np.uint32)
            features_to_add = X_mat

        graph_mut = deglib.create_empty(
            capacity=n_vectors,
            feature_space=space,
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

        builder.add_entry(labels, features_to_add)
        builder.build(callback="progress")

        if self.graph_path:
            p = Path(self.graph_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            graph_mut.save_graph(str(p))
            print(f"Saved built DEG graph to: {p}")
            self.graph = deglib.load_readonly_graph(str(p))
        else:
            self.graph = graph_mut.to_readonly()

    def set_query_arguments(self, search_eps: float):
        """Sets query-time search_eps."""
        self.search_eps = float(search_eps)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread."""
        if self.graph is None:
            raise RuntimeError("Index not fitted. Call fit(X) first.")
        if self.metric_enum == deglib.Metric.Uint8_L2:
            query_mat = np.ascontiguousarray(v.reshape(1, -1), dtype=np.uint8)
        else:
            query_mat = np.ascontiguousarray(v.reshape(1, -1), dtype=np.float32)
            if self.is_cosine:
                query_mat = self._normalize(query_mat)

        indices, _ = self.graph.search(
            query_mat, eps=self.search_eps, k=n, threads=1, return_distances=False, unsorted=True
        )
        return np.asarray(indices[0], dtype=np.int64)

    def batch_query(self, X: np.ndarray, n: int):
        """Batch query search on 1 thread."""
        if self.graph is None:
            raise RuntimeError("Index not fitted. Call fit(X) first.")
        if self.metric_enum == deglib.Metric.Uint8_L2:
            query_mat = np.ascontiguousarray(X, dtype=np.uint8)
        else:
            query_mat = np.ascontiguousarray(X, dtype=np.float32)
            if self.is_cosine:
                query_mat = self._normalize(query_mat)

        indices, _ = self.graph.search(query_mat, eps=self.search_eps, k=n, threads=1)
        self._batch_results = np.asarray(indices, dtype=np.int64)

    def get_batch_results(self) -> np.ndarray:
        if self._batch_results is None:
            raise RuntimeError("No batch results available.")
        return self._batch_results

    def __str__(self) -> str:
        return (
            f"DEG(k={self.k}, extend_k={self.extend_k}, build_eps={self.build_eps}, "
            f"opt={self.opt_target}, improve_k={self.improve_k}, improve_eps={self.improve_eps}, "
            f"threads={self.threads}, eps={self.search_eps})"
        )
