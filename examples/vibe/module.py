import time
from pathlib import Path

import deglib
import numpy as np

# Try relative import when executed inside VIBE framework (vibe/algorithms/deg/module.py)
try:
    from ..base.module import BaseANN
except (ImportError, ValueError):

    class BaseANN:
        pass


_METRIC_MAP = {
    "euclidean": (
        deglib.Metric.FP32_L2,
        deglib.Metric.Int8_L2,
        deglib.Metric.FP16_L2,
        deglib.Metric.Uint4_L2,
        deglib.Metric.Int4_L2,
        deglib.Metric.UInt7_L2,
    ),
    "cosine": (
        deglib.Metric.FP32_InnerProduct,
        deglib.Metric.Int8_InnerProduct,
        deglib.Metric.FP16_InnerProduct,
        deglib.Metric.Uint4_InnerProduct,
        deglib.Metric.Int4_InnerProduct,
        deglib.Metric.UInt7_InnerProduct,
    ),
    "ip": (
        deglib.Metric.FP32_InnerProduct,
        deglib.Metric.Int8_InnerProduct,
        deglib.Metric.FP16_InnerProduct,
        deglib.Metric.Uint4_InnerProduct,
        deglib.Metric.Int4_InnerProduct,
        deglib.Metric.UInt7_InnerProduct,
    ),
    "normalized": (
        deglib.Metric.FP32_InnerProduct,
        deglib.Metric.Int8_InnerProduct,
        deglib.Metric.FP16_InnerProduct,
        deglib.Metric.Uint4_InnerProduct,
        deglib.Metric.Int4_InnerProduct,
        deglib.Metric.UInt7_InnerProduct,
    ),
}


_FP16_METRICS = {
    "l2": deglib.Metric.FP16_L2,
    "ip": deglib.Metric.FP16_InnerProduct,
}

_QUANT_METRICS = {
    "l2": {
        "int8-int8": deglib.Metric.Int8_L2,
        "uint7-uint7": deglib.Metric.UInt7_L2,
        "uint4-uint4": deglib.Metric.Uint4_L2,
    },
    "ip": {
        "int8-int8": deglib.Metric.Int8_InnerProduct,
        "uint7-uint7": deglib.Metric.UInt7_InnerProduct,
        "uint4-uint4": deglib.Metric.Uint4_InnerProduct,
    },
}


def _resolve_presort_cache_path(
    metric: str,
    n_vectors: int,
    dims: int,
    cache_dir: Path | None = None,
) -> Path:
    """Resolves cache path for presort indices (one per dataset+metric)."""
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

    presort_metric = "l2" if metric == "euclidean" else "ip"
    filename = f"{dims}D_{presort_metric}_presort.npy"
    return deg_dir / filename


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
        threads: int = 1,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
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

        presort_file = _resolve_presort_cache_path(
            metric=self.metric,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        if cache_file.exists():
            print(f"Loading cached DEG graph from {cache_file}...", flush=True)
            graph = deglib.load_readonly_graph(str(cache_file))
        else:
            # 1. FLAS 1D Pre-sorting
            if presort_file.exists():
                print(f"Loading cached presort from {presort_file}...", flush=True)
                sorted_indices = np.load(str(presort_file))
            else:
                print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
                sorted_indices = deglib.optimization.presort(
                    X,
                    metric=self.metric_enum,
                    threads=self.threads,
                    show_progress=True,
                )
                print(f"Saving presort to cache {presort_file}...", flush=True)
                np.save(str(presort_file), sorted_indices)

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
                show_progress=True,
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))

        self.graph = graph.to_readonly() if graph.is_mutable() else graph
        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            normalize_query=(self.metric == "cosine"),
        )

        # 4s. Optimize entry vertices and prefetch values
        t_km = time.time()
        self.searcher.optimize()
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float):
        """Sets query-time parameter: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with Float32 via C++ searcher."""
        return self.searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        if self.eps_or_ef >= 1.0:
            return f"DEG(k={self.k}, opt={self.opt_target}, ef={int(round(self.eps_or_ef))})"
        return f"DEG(k={self.k}, opt={self.opt_target}, eps={self.eps_or_ef})"


class QG(BaseANN):
    """
    Quantized DEG (DEG-QG): Graph search with INT8 or UINT4 quantized vectors and FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        threads: int = 1,
        quant: str = "int8-int8",
        query_metric: str = "auto",
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.threads = int(threads)
        self.quant = (quant or "int8-int8").lower().strip()
        qmetric = (query_metric or "auto").lower().strip()
        if qmetric == "auto":
            self.query_metric = "l2" if self.metric == "euclidean" else "ip"
        elif qmetric == "l2":
            self.query_metric = "l2"
        elif qmetric == "innerproduct":
            self.query_metric = "ip"
        else:
            raise ValueError(f"Unsupported query_metric '{query_metric}'. Choose from: ['L2', 'InnerProduct']")
        self.rerank_size_factor = 1.0
        self.search_eps = 0.0
        self.ef = 0
        self.eps_or_ef = 0.0

        self.base_metric, self.int8_metric, self.fp16_metric, self.uint4_metric, self.int4_metric, self.int7_metric = (
            _METRIC_MAP[self.metric]
        )
        self.search_fp16_metric = _FP16_METRICS[self.query_metric]
        self.target_metric = _QUANT_METRICS[self.query_metric][self.quant]
        self.normalize_queries = self.query_metric == "ip" or self.metric == "cosine"
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]

        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.rerank_space_fp16 = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds DEG graph, quantizes vectors to INT8 using ScalarQuantizer, and prepares C++ searcher."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_vectors, dims = X.shape

        self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=self.search_fp16_metric)

        cache_file = _resolve_graph_cache_path(
            metric=self.metric,
            k=self.k,
            opt_target=self.opt_target,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        presort_file = _resolve_presort_cache_path(
            metric=self.metric,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        if cache_file.exists():
            print(f"Loading cached DEG graph from {cache_file}...", flush=True)
            loaded_graph = deglib.load_readonly_graph(str(cache_file))
            # Cache: Reconstruct graph-ordered features from original un-permuted X using saved labels
            X_graph_order = X[loaded_graph.get_labels()]
        else:
            # 1. FLAS 1D Pre-sorting
            if presort_file.exists():
                print(f"Loading cached presort from {presort_file}...", flush=True)
                sorted_indices = np.load(str(presort_file))
            else:
                print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
                sorted_indices = deglib.optimization.presort(
                    X,
                    metric=self.base_metric,
                    threads=self.threads,
                    show_progress=True,
                )
                print(f"Saving presort to cache {presort_file}...", flush=True)
                np.save(str(presort_file), sorted_indices)

            # Standard static workflow: sort data once so builder, quantizer, and reranker all share the same array
            X_graph_order = X[sorted_indices]

            # 2. Build graph in FP32
            graph_data = X_graph_order
            graph_metric = self.base_metric

            print(
                f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...",
                flush=True,
            )
            graph = deglib.builder.build_from_data(
                data=graph_data,
                labels=sorted_indices,
                edges_per_vertex=self.k,
                metric=graph_metric,
                seed=7,
                optimization_target=self.opt_enum,
                thread_count=self.threads,
                show_progress=True,
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))
            loaded_graph = graph

        # 3. Finalize ReadOnlyGraph with quantized features using calibrated ScalarQuantizer
        # Both features and rerank_data are derived directly from X_graph_order (row i = vertex i).
        target_space = deglib.FloatSpace.create(dim=dims, metric=self.target_metric)
        if self.quant == "uint4-uint4":
            self.quantizer = deglib.transformation.make_scalar_quantizer_uint4(X)
            quant_features = self.quantizer.quantize(X_graph_order, num_threads=self.threads)
        elif self.quant == "uint7-uint7":
            self.quantizer = deglib.transformation.make_scalar_quantizer_uint7(X)
            quant_features = self.quantizer.quantize(X_graph_order, num_threads=self.threads)
        elif self.quant == "int8-int8":
            self.quantizer = deglib.transformation.make_scalar_quantizer_int8(X)
            quant_features = self.quantizer.quantize(X_graph_order, num_threads=self.threads)
        else:
            raise ValueError(f"Unsupported quant '{self.quant}'")

        # Convert the FP32 graph into a readonly graph that uses the quantized features and quantizer for the graph-based search
        self.graph = loaded_graph.to_readonly(target_space, quant_features, transformations=[self.quantizer])

        # 5. Initialize a Searcher with FP16 Reranker & ScalarQuantizerFP16
        self.rerank_quantizer = deglib.transformation.make_scalar_quantizer_fp16(X)
        rerank_features = self.rerank_quantizer.quantize(X_graph_order, num_threads=self.threads)

        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            rerank_space=self.rerank_space_fp16,
            rerank_data=rerank_features,
            rerank_transformations=[self.rerank_quantizer],
            normalize_query=self.normalize_queries,
        )

        # 6. Full searcher optimization
        t_km = time.time()
        self.searcher.optimize(k=self.k, use_routing=True, n_clusters=256, num_threads=self.threads)
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float, rerank_size_factor: float = 1.0):
        """Sets query-time parameters: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with INT8 search and FP16 reranking directly in C++."""
        return self.searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        qname = self.quant.upper()
        if self.eps_or_ef >= 1.0:
            return (
                f"DEG-QG[{qname}](k={self.k}, opt={self.opt_target}, query_metric={self.query_metric.upper()}, "
                f"rerank_factor={self.rerank_size_factor}, ef={int(round(self.eps_or_ef))})"
            )
        return (
            f"DEG-QG[{qname}](k={self.k}, opt={self.opt_target}, query_metric={self.query_metric.upper()}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.eps_or_ef})"
        )


class EVP(BaseANN):
    """
    Equi-Voronoi Polytopes Quantized DEG (DEG-EVP): Graph search with EVP (1-bit/ternary) vectors and FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        threads: int = 1,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.threads = int(threads)
        self.rerank_size_factor = 1.0
        self.search_eps = 0.0
        self.ef = 0
        self.eps_or_ef = 0.0

        self.base_metric, _, self.fp16_metric = _METRIC_MAP[self.metric]
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]

        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.rerank_space_fp16 = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds DEG graph, quantizes vectors using EVP Quantizer, and prepares C++ searcher."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_vectors, dims = X.shape

        self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=self.fp16_metric)

        cache_file = _resolve_graph_cache_path(
            metric=self.metric,
            k=self.k,
            opt_target=self.opt_target,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        presort_file = _resolve_presort_cache_path(
            metric=self.metric,
            n_vectors=n_vectors,
            dims=dims,
            cache_dir=cache_dir,
        )

        if cache_file.exists():
            print(f"Loading cached DEG graph from {cache_file}...", flush=True)
            loaded_graph = deglib.load_readonly_graph(str(cache_file))
            X_graph_order = X[loaded_graph.get_labels()]
        else:
            # 1. FLAS 1D Pre-sorting
            if presort_file.exists():
                print(f"Loading cached presort from {presort_file}...", flush=True)
                sorted_indices = np.load(str(presort_file))
            else:
                print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
                sorted_indices = deglib.optimization.presort(
                    X,
                    metric=self.base_metric,
                    threads=self.threads,
                    show_progress=True,
                )
                print(f"Saving presort to cache {presort_file}...", flush=True)
                np.save(str(presort_file), sorted_indices)

            X_graph_order = X[sorted_indices]

            # 2. Build graph in FP32
            print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
            graph = deglib.builder.build_from_data(
                data=X_graph_order,
                labels=sorted_indices,
                edges_per_vertex=self.k,
                metric=self.base_metric,
                seed=7,
                optimization_target=self.opt_enum,
                thread_count=self.threads,
                show_progress=True,
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))
            loaded_graph = graph

        # 3. Finalize ReadOnlyGraph with EVP features
        # EvpQuantizer::fit() in C++ automatically calibrates the optimal nz value across threads.
        self.quantizer = deglib.transformation.make_evp_quantizer(X, num_threads=self.threads)
        print(
            f"Quantizing with EVP (dim={dims}, non_zeros={self.quantizer.non_zeros} [~{self.quantizer.non_zeros / dims * 100:.1f}%])...",
            flush=True,
        )
        evp_features = self.quantizer.quantize(X_graph_order, num_threads=self.threads)
        target_space = deglib.FloatSpace.create(dim=dims, metric=deglib.Metric.EVP_InnerProduct)
        # self.graph = loaded_graph.to_readonly(target_space, evp_features, transformations=[self.quantizer])
        self.graph = loaded_graph.to_readonly(target_space, evp_features, transformations=[self.quantizer])

        # 5. Initialize C++ Zero-overhead Searcher with FP16 base vectors & ScalarQuantizerFP16
        self.rerank_quantizer = deglib.transformation.make_scalar_quantizer_fp16(X)
        rerank_features = self.rerank_quantizer.quantize(X_graph_order, num_threads=self.threads)

        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            rerank_space=self.rerank_space_fp16,
            rerank_data=rerank_features,
            rerank_transformations=[self.rerank_quantizer],
            normalize_query=(self.metric == "cosine"),
        )

        # 6. Optimize entry vertices and prefetch values
        t_km = time.time()
        self.searcher.optimize()
        print(f"Optimized Searcher for EVP graph in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float, rerank_size_factor: float = 1.0):
        """Sets query-time parameters: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with EVP search and FP16 reranking."""
        return self.searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        nz = self.quantizer.non_zeros if self.quantizer else 0
        if self.eps_or_ef >= 1.0:
            return (
                f"DEG-EVP(k={self.k}, nz={nz}, opt={self.opt_target}, "
                f"rerank_factor={self.rerank_size_factor}, ef={int(round(self.eps_or_ef))})"
            )
        return (
            f"DEG-EVP(k={self.k}, nz={nz}, opt={self.opt_target}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.eps_or_ef})"
        )
