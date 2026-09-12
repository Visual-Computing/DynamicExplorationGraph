import time
from pathlib import Path
import numpy as np
import deglib

# Try relative import when executed inside VIBE framework (vibe/algorithms/deg/module.py)
try:
    from ..base.module import BaseANN
except (ImportError, ValueError):

    class BaseANN:
        pass


_METRIC_MAP = {
    "euclidean": (deglib.Metric.FP32_L2, deglib.Metric.Int8_L2, deglib.Metric.FP16_L2),
    "cosine": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "ip": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "normalized": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
}


def _resolve_graph_cache_path(
    metric: str,
    k: int,
    opt_target: str,
    n_vectors: int,
    dims: int,
    extend_k: int | None = None,
    extend_eps: float = 0.1,
    cache_dir: Path | None = None,
) -> Path:
    """Resolves standard cache path using get_default_cache_dir()."""
    dataset_name = f"corpus_{n_vectors}_{dims}d"
    if cache_dir is None:
        try:
            from dataset import VIBE_DATASETS, get_default_cache_dir

            cache_dir = get_default_cache_dir()
            for key, meta in VIBE_DATASETS.items():
                if meta.get("dim") == dims and abs(meta.get("size", 0) - n_vectors) < 1000:
                    dataset_name = key
                    break
        except ImportError:
            cache_dir = Path.home() / ".cache" / "deg_datasets"
    else:
        try:
            from dataset import VIBE_DATASETS

            for key, meta in VIBE_DATASETS.items():
                if meta.get("dim") == dims and abs(meta.get("size", 0) - n_vectors) < 1000:
                    dataset_name = key
                    break
        except ImportError:
            pass

    deg_dir = cache_dir / dataset_name / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

    build_metric_str = deglib.Metric.FP32_L2.name if metric == "euclidean" else deglib.Metric.FP32_InnerProduct.name
    ext_k = extend_k if extend_k is not None else k * 2
    
    # 1. Preferred filename with AddK
    filename = f"{dims}D_{build_metric_str}_K{k}_AddK{ext_k}Eps{extend_eps:.1f}_{opt_target}_FLAS.deg"
    full_path = deg_dir / filename
    if full_path.exists():
        return full_path

    # 2. Check alternative without AddK
    alt_filename = f"{dims}D_{build_metric_str}_K{k}_{opt_target}_FLAS.deg"
    if (deg_dir / alt_filename).exists():
        return deg_dir / alt_filename

    # 3. Check for any AddK variations (e.g. AddK60)
    matches = list(deg_dir.glob(f"{dims}D_{build_metric_str}_K{k}_AddK*_{opt_target}_FLAS.deg"))
    if matches:
        return matches[0]

    return full_path


class DEG(BaseANN):
    """
    Dynamic Exploration Graph (DEG) using Float32 throughout.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        prune_non_rng: bool = False,
        threads: int = 1,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = int(threads)
        self.eps_or_ef = 0.1
        self.metric_enum = _METRIC_MAP[self.metric][0]
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]
        self.graph = None
        self.searcher = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds or loads the DEG graph in FP32."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_vectors, dims = X.shape

        cache_file = _resolve_graph_cache_path(
            metric=self.metric,
            k=self.k,
            opt_target=self.opt_target,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        if cache_file.exists():
            print(f"Loading cached DEG graph from {cache_file}...", flush=True)
            load_fn = deglib.load_mutable_graph if self.prune_non_rng else deglib.load_readonly_graph
            graph = load_fn(str(cache_file))
        else:
            # 1. FLAS 1D Pre-sorting
            print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
            sorted_indices = deglib.optimization.presort(
                X,
                metric=self.metric_enum,
                threads=self.threads,
                callback="progress",
            )

            # 2. Build graph in FP32
            print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
            graph = deglib.builder.build_from_data(
                data=X[sorted_indices],
                labels=sorted_indices,
                edges_per_vertex=self.k,
                metric=self.metric_enum,
                seed=7,
                optimization_target=self.opt_enum,
                thread_count=self.threads,
                callback="progress",
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))

        # 3. Optional MRNG edge pruning
        if self.prune_non_rng:
            deglib.optimization.prune_non_rng_edges(graph, num_threads=1)

        self.graph = graph.to_readonly() if graph.is_mutable() else graph
        self.searcher = deglib.search.create_searcher(graph=self.graph)

        # 4s. Optimize entry vertices and prefetch values
        t_km = time.time()
        self.searcher.optimize()  
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float | int):
        """Sets query-time parameter: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with Float32 via C++ searcher."""
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.searcher.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            eps_or_ef=self.eps_or_ef,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        if self.eps_or_ef >= 1.0:
            return f"DEG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, ef={int(round(self.eps_or_ef))})"
        return f"DEG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, eps={self.eps_or_ef})"


class QG(BaseANN):
    """
    Quantized DEG (DEG-QG): Graph search with INT8 quantized vectors and FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        prune_non_rng: bool = False,
        threads: int = 1,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = int(threads)
        self.rerank_size_factor = 1.0
        self.search_eps = 0.0
        self.ef = 0

        self.base_metric, self.int8_metric, self.fp16_metric = _METRIC_MAP[self.metric]
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]

        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.original_features_fp16 = None
        self.rerank_space_fp16 = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds DEG graph, quantizes vectors to INT8 using ScalarQuantizer, and prepares C++ searcher."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_vectors, dims = X.shape

        self.original_features_fp16 = deglib.distances.floats_to_fp16(X)
        self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=self.fp16_metric)

        cache_file = _resolve_graph_cache_path(
            metric=self.metric,
            k=self.k,
            opt_target=self.opt_target,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        if cache_file.exists():
            print(f"Loading cached DEG graph from {cache_file}...", flush=True)
            load_fn = deglib.load_mutable_graph if self.prune_non_rng else deglib.load_readonly_graph
            loaded_graph = load_fn(str(cache_file))
        else:
            # 1. FLAS 1D Pre-sorting
            print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
            sorted_indices = deglib.optimization.presort(
                X,
                metric=self.base_metric,
                threads=self.threads,
                callback="progress",
            )

            # 2. Build graph in FP32
            print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
            graph = deglib.builder.build_from_data(
                data=X[sorted_indices],
                labels=sorted_indices,
                edges_per_vertex=self.k,
                metric=self.base_metric,
                seed=7,
                optimization_target=self.opt_enum,
                thread_count=self.threads,
                callback="progress",
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))
            loaded_graph = graph

        # 3. Optional MRNG edge pruning
        if self.prune_non_rng:
            deglib.optimization.prune_non_rng_edges(loaded_graph, num_threads=1)

        # 4. Finalize ReadOnlyGraph with INT8 features using calibrated ScalarQuantizer
        self.quantizer = deglib.optimization.make_scalar_quantizer_int8(X)
        int8_features = self.quantizer.quantize(X, num_threads=1)
        target_space = deglib.FloatSpace.create(dim=dims, metric=self.int8_metric)
        self.graph = loaded_graph.to_readonly(target_space, int8_features)

        # 5. Initialize C++ Zero-overhead Searcher
        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            quantizer=self.quantizer,
            refine_space=self.rerank_space_fp16,
            refine_data=self.original_features_fp16,
        )

        # 6. Optimize entry vertices and prefetch values
        t_km = time.time()
        self.searcher.optimize()  
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float | int, rerank_size_factor: float = 1.0):
        """Sets query-time parameters: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with INT8 search and FP16 reranking directly in C++."""
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.searcher.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        if self.eps_or_ef >= 1.0:
            return (
                f"DEG-QG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, "
                f"rerank_factor={self.rerank_size_factor}, ef={int(round(self.eps_or_ef))})"
            )
        return (
            f"DEG-QG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.eps_or_ef})"
        )

