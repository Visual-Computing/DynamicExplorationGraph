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


_SUPPORTED_METRICS = ("euclidean", "cosine", "ip", "normalized")


def _resolve_deg_cache_dir(n_vectors: int, dims: int, cache_dir: Path | None = None) -> Path:
    """Resolves dataset cache directory under `<cache_dir>/<dataset_name>/deg`."""
    dataset_name = f"corpus_{n_vectors}_{dims}d"
    try:
        from dataset import VIBE_DATASETS, get_default_cache_dir

        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        for key, meta in VIBE_DATASETS.items():
            if meta.get("dim") == dims and abs(meta.get("size", 0) - n_vectors) < 1000:
                dataset_name = key
                break
    except ImportError:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "deg_datasets"

    deg_dir = cache_dir / dataset_name / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)
    return deg_dir


def _resolve_graph_cache_path(
    deg_dir: Path,
    dims: int,
    metric: str,
    k: int,
    opt_target: str,
    extend_k: int | None = None,
    extend_eps: float = 0.1,
    is_fp16: bool = False,
) -> Path:
    """Resolves cached graph path (fp32 or fp16 build metric)."""
    prefix = "FP16" if is_fp16 else "FP32"
    build_metric_str = f"{prefix}_L2" if metric == "euclidean" else f"{prefix}_InnerProduct"
    ext_k = extend_k if extend_k is not None else k * 2

    preferred = deg_dir / f"{dims}D_{build_metric_str}_K{k}_AddK{ext_k}Eps{extend_eps:.1f}_{opt_target}_FLAS.deg"
    if preferred.exists():
        return preferred

    alt = deg_dir / f"{dims}D_{build_metric_str}_K{k}_{opt_target}_FLAS.deg"
    if alt.exists():
        return alt

    matches = list(deg_dir.glob(f"{dims}D_{build_metric_str}_K{k}_AddK*_{opt_target}_FLAS.deg"))
    return matches[0] if matches else preferred


def _load_or_build_base_graph(
    X: np.ndarray,
    metric: str,
    k: int,
    opt_target: str,
    threads: int,
    cache_dir: Path | None,
) -> deglib.DynamicExplorationGraph:
    """Prepares vectors and builds or loads the cached base graph."""
    orig_n, orig_dims = X.shape
    deg_dir = _resolve_deg_cache_dir(orig_n, orig_dims, cache_dir)
    graph_dims = orig_dims

    cache_file = _resolve_graph_cache_path(
        deg_dir, graph_dims, metric, k, opt_target, is_fp16=(X.dtype == np.float16)
    )
    if cache_file.exists():
        print(f"Loading cached DEG graph from {cache_file}...", flush=True)
        return deglib.load_readonly_graph(str(cache_file))

    presort_file = deg_dir / f"{orig_dims}D_l2_presort.npy"
    if presort_file.exists():
        print(f"Loading cached presort from {presort_file}...", flush=True)
        sorted_indices = np.load(str(presort_file))
    else:
        print(f"Running FLAS 1D Pre-sorting{f' (threads={threads})' if threads > 0 else ''}...", flush=True)
        sorted_indices = deglib.presort(X, metric=metric, threads=threads, show_progress=True)
        print(f"Saving presort to cache {presort_file}...", flush=True)
        np.save(str(presort_file), sorted_indices)

    threads_str = f", threads={threads}" if threads > 0 else ""
    print(f"Building DEG graph (K={k}, Opt={opt_target}{threads_str})...", flush=True)
    graph = deglib.build(
        X,
        presort=sorted_indices,
        edges_per_vertex=k,
        metric=metric,
        optimization_target=opt_target,
        optimize=False,
        threads=threads,
        show_progress=True,
    )
    print(f"Saving graph to cache {cache_file}...", flush=True)
    graph.save_graph(str(cache_file))
    return graph


class _BaseDEG(BaseANN):
    """Shared configuration for DEG and QG VIBE adapters."""

    def __init__(self, metric: str, k: int = 30, opt_target: str = "LowLID", threads: int = 0):
        self.metric = metric.lower().strip()
        if self.metric not in _SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_SUPPORTED_METRICS)}")
        self.k = int(k)
        self.opt_target = opt_target
        self.threads = int(threads)
        self.eps_or_ef = 0.1

    @property
    def _search_param_str(self) -> str:
        return f"ef={int(round(self.eps_or_ef))}" if self.eps_or_ef >= 1.0 else f"eps={self.eps_or_ef}"


class DEG(_BaseDEG):
    """Dynamic Exploration Graph (DEG) using Float32 throughout."""

    def __init__(self, metric: str, k: int = 30, opt_target: str = "LowLID", threads: int = 0):
        super().__init__(metric=metric, k=k, opt_target=opt_target, threads=threads)
        self.searcher = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds or loads the DEG graph and prepares the Searcher (fp16 in, fp16 build)."""
        graph = _load_or_build_base_graph(
            X, self.metric, self.k, self.opt_target, self.threads, cache_dir
        )
        t_km = time.time()
        self.searcher = graph.to_searcher(
            normalize_query=(self.metric == "cosine"),
            optimize=True,
            threads=self.threads,
        )
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float):
        """Sets query-time exploration parameter (>= 1.0: ef, < 1.0: eps)."""
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread via C++ Searcher."""
        return self.searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            threads=1,
            return_distances=False,
            unsorted=True,
        ).indices

    def __str__(self) -> str:
        return f"DEG(k={self.k}, opt={self.opt_target}, {self._search_param_str})"


class QG(_BaseDEG):
    """
    Quantized DEG (DEG-QG): Builds the base graph and FP16 reranker once, then creates
    one quantized Searcher per entry in `quants` sharing the same Reranker.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        threads: int = 8,
        quants: list[str] | str | None = None,
    ):
        super().__init__(metric=metric, k=k, opt_target=opt_target, threads=threads)
        if quants is None:
            default_kind = "L2" if self.metric == "euclidean" else "InnerProduct"
            quant_list = [f"Int8_{default_kind}"]
        elif isinstance(quants, str):
            quant_list = [quants]
        else:
            quant_list = list(quants)
        if not quant_list:
            raise ValueError("quants must contain at least one quantized metric (e.g. 'Uint4_L2')")

        self.quant_metrics = {q: deglib.Metric.resolve(q) for q in quant_list}
        self.rerank_size_factor = 1.0
        self.searchers: dict[str, deglib.Searcher] = {}
        self.active_searcher: deglib.Searcher | None = None

    def fit(self, X: np.ndarray, cache_dir: Path | None = None):
        """Builds DEG graph once, creates one shared FP16 Reranker, and converts into one Searcher per quant."""
        loaded_graph = _load_or_build_base_graph(
            X, self.metric, self.k, self.opt_target, self.threads, cache_dir
        )

        t_rr = time.time()
        reranker = deglib.reranker(loaded_graph, compress="16bit", threads=self.threads)
        print(f"Prepared shared FP16 Reranker in {time.time() - t_rr:.2f}s", flush=True)

        self.searchers.clear()
        for q_name, q_metric in self.quant_metrics.items():
            t_km = time.time()
            self.searchers[q_name] = loaded_graph.to_searcher(
                metric=q_metric,
                reranker=reranker,
                normalize_query=(self.metric == "cosine"),
                threads=self.threads,
            )
            print(f"Created & optimized Searcher[{q_name}] in {time.time() - t_km:.2f}s", flush=True)

    def set_query_arguments(
        self,
        eps_or_ef: float,
        rerank_size_factor: float = 1.0,
        quants: str | None = None,
    ):
        """Sets active quantized searcher, rerank factor, and exploration parameter."""
        if quants is not None:
            if quants not in self.searchers:
                raise ValueError(f"Unknown quant '{quants}'. Available: {list(self.searchers.keys())}")
            self.active_searcher = self.searchers[quants]
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with quantized search and shared FP16 reranking."""
        return self.active_searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        ).indices

    def __str__(self) -> str:
        q_label = self.active_searcher.graph.get_metric().name if self.active_searcher is not None else "unfitted"
        return (
            f"DEG-QG[{q_label}](k={self.k}, opt={self.opt_target}, "
            f"rerank_factor={self.rerank_size_factor}, {self._search_param_str})"
        )
