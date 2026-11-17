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


_VIBE_METRIC_KIND = {
    "euclidean": "L2",
    "cosine": "InnerProduct",
    "ip": "InnerProduct",
    "normalized": "InnerProduct",
}

_QUANT_PREFIXES = {
    "int8": "Int8",
    "uint8": "Uint8",
    "uint7": "UInt7",
    "uint4": "Uint4",
    "int4": "Int4",
    "fp16": "FP16",
    "evp": "EVP",
}


def _get_metric(data_type: str, metric_kind: str) -> deglib.Metric:
    try:
        return deglib.Metric[f"{data_type}_{metric_kind}"]
    except KeyError as exc:
        raise ValueError(f"Unsupported deglib metric '{data_type}_{metric_kind}'") from exc


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
        if self.metric not in _VIBE_METRIC_KIND:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_VIBE_METRIC_KIND.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.threads = int(threads)
        self.eps_or_ef = 0.1
        self.build_metric = _get_metric("FP32", _VIBE_METRIC_KIND[self.metric])
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
                sorted_indices = deglib.presort(
                    X,
                    metric=self.build_metric,
                    threads=self.threads,
                    show_progress=True,
                )
                print(f"Saving presort to cache {presort_file}...", flush=True)
                np.save(str(presort_file), sorted_indices)

            # 2. Build graph in FP32
            print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
            graph = deglib.build(
                X,
                presort=sorted_indices,
                edges_per_vertex=self.k,
                metric=self.build_metric,
                seed=7,
                optimization_target=self.opt_enum,
                optimize=False,
                threads=self.threads,
                show_progress=True,
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            graph.save_graph(str(cache_file))

        # 3. Convert to Searcher and optimize entry vertices and prefetch values
        t_km = time.time()
        self.searcher = graph.to_searcher(
            normalize_query=(self.metric == "cosine"),
            optimize=True,
            threads=self.threads,
        )
        self.graph = self.searcher.graph
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
    Quantized DEG (DEG-QG): Graph search with quantized vectors (INT8, UINT4, EVP, etc.) and FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        threads: int = 8,
        quant: str = "Int8",
        query_metric: str = "auto",
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _VIBE_METRIC_KIND:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_VIBE_METRIC_KIND.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.threads = int(threads)
        quant_key = (quant or "Int8").lower().strip()
        if quant_key not in _QUANT_PREFIXES:
            raise ValueError(f"Unsupported quant '{quant}'. Choose from: {list(_QUANT_PREFIXES.values())}")
        quant_prefix = _QUANT_PREFIXES[quant_key]

        metric_kind = _VIBE_METRIC_KIND[self.metric]
        qmetric = (query_metric or "auto").strip()
        if qmetric.lower() == "auto":
            quant_metric_kind = "InnerProduct" if quant_prefix == "EVP" else metric_kind
        elif qmetric.lower() == "ip":
            quant_metric_kind = "InnerProduct"
        elif qmetric.lower() == "l2":
            quant_metric_kind = "L2"
        else:
            quant_metric_kind = qmetric

        self.rerank_size_factor = 1.0
        self.eps_or_ef = 0.0

        self.build_metric = _get_metric("FP32", metric_kind)
        self.rerank_metric = _get_metric("FP16", metric_kind)
        self.quant_metric = _get_metric(quant_prefix, quant_metric_kind)
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]

        self.graph = None
        self.searcher = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds DEG graph, quantizes vectors automatically, and prepares C++ searcher with FP16 reranking."""
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
            loaded_graph = deglib.load_readonly_graph(str(cache_file))
        else:
            # 1. FLAS 1D Pre-sorting
            if presort_file.exists():
                print(f"Loading cached presort from {presort_file}...", flush=True)
                sorted_indices = np.load(str(presort_file))
            else:
                print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
                sorted_indices = deglib.presort(
                    X,
                    metric=self.build_metric,
                    threads=self.threads,
                    show_progress=True,
                )
                print(f"Saving presort to cache {presort_file}...", flush=True)
                np.save(str(presort_file), sorted_indices)

            # 2. Build graph in FP32
            print(
                f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...",
                flush=True,
            )
            loaded_graph = deglib.build(
                X,
                presort=sorted_indices,
                edges_per_vertex=self.k,
                metric=self.build_metric,
                seed=7,
                optimization_target=self.opt_enum,
                optimize=False,
                threads=self.threads,
                show_progress=True,
            )

            print(f"Saving graph to cache {cache_file}...", flush=True)
            loaded_graph.save_graph(str(cache_file))

        # 3. Quantize, attach FP16 reranker, and optimize searcher in one call
        t_km = time.time()
        self.searcher = loaded_graph.to_searcher(
            metric=self.quant_metric,
            rerank_metric=self.rerank_metric,
            rerank_data=X,
            rerank_order="label",
            normalize_query=(self.metric == "cosine"),
            optimize=True,
            use_routing=True,
            n_clusters=256,
            threads=self.threads,
        )
        self.graph = self.searcher.graph
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
        if self.eps_or_ef >= 1.0:
            return (
                f"DEG-QG[{self.quant_metric.name}](k={self.k}, opt={self.opt_target}, "
                f"rerank_factor={self.rerank_size_factor}, ef={int(round(self.eps_or_ef))})"
            )
        return (
            f"DEG-QG[{self.quant_metric.name}](k={self.k}, opt={self.opt_target}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.eps_or_ef})"
        )
