import gc
from pathlib import Path
import time
import numpy as np
import deglib


# Try relative import when executed inside VIBE framework (vibe/algorithms/deg/module.py)
try:
    from ..base.module import BaseANN
except (ImportError, ValueError):

    class BaseANN:
        pass


# Mapping: (is_l2: bool, dtype_str: str) -> deglib.Metric
_METRIC_MAP = {
    (True, "float32"): deglib.Metric.FP32_L2,
    (True, "float16"): deglib.Metric.FP16_L2,
    (True, "int8"): deglib.Metric.Int8_L2,
    (False, "float32"): deglib.Metric.FP32_InnerProduct,
    (False, "float16"): deglib.Metric.FP16_InnerProduct,
    (False, "int8"): deglib.Metric.Int8_InnerProduct,
}

_OPT_TARGET_MAP = {
    "highlid": deglib.builder.OptimizationTarget.HighLID,
    "lowlid": deglib.builder.OptimizationTarget.LowLID,
    "streamingdata": deglib.builder.OptimizationTarget.StreamingData,
}

_VALID_QUERY_DTYPES = ("float32", "int8")


class DegANN(BaseANN):
    """
    VIBE BaseANN adapter for Dynamic Exploration Graph (DEG).
    Standard parameters:
      - metric: Distance metric ('euclidean', 'cosine', 'ip', 'normalized', 'hamming')
      - k: Degree / edges per vertex (e.g. 16, 24, 30, 40, 48)
      - opt_target: 'HighLID' (for Cosine/IP/Normalized), 'LowLID' (for Euclidean), or 'StreamingData'
      - threads: CPU threads during graph build
      - query_dtype: Dtype used to search graph ('float32', 'int8')
      - prune_non_rng: Optional MRNG edge pruning
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "HighLID",
        prune_non_rng: bool = False,
        threads: int = 1,
        query_dtype: str = "float32",
    ):
        self.metric = metric.lower().strip()
        if self.metric not in ("euclidean", "cosine", "ip", "normalized"):
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: euclidean, cosine, ip, normalized")
        self.is_l2: bool = (self.metric == "euclidean")
        self.needs_normalization: bool = (self.metric == "cosine")

        self.k = int(k)
        self.extend_k = self.k*2
        self.extend_eps = 0.1
        self.opt_target = str(opt_target)
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = int(threads)
        self.query_dtype = query_dtype.lower().strip()
        if self.query_dtype not in _VALID_QUERY_DTYPES:
            raise ValueError(f"Unsupported dtype '{self.query_dtype}'. Choose from: {_VALID_QUERY_DTYPES}")
        self.search_eps: float = 0.1
        self.rerank_size_factor: float = 1.0

        # Distance metric resolution for graph build and query space
        self.base_metric_enum = _METRIC_MAP[(self.is_l2, "float32")]
        self.query_metric_enum = _METRIC_MAP[(self.is_l2, self.query_dtype)]
        self.metric_enum = self.query_metric_enum
        self.opt_enum = self._map_opt_target(self.opt_target)

        self.quantizer = None
        self.graph: deglib.ReadOnlyGraph | None = None
        self.original_features_fp16: np.ndarray | None = None
        self.rerank_space_fp16: deglib.FloatSpace | None = None
        self.searcher = None

    def _map_opt_target(self, opt_target: str) -> deglib.builder.OptimizationTarget:
        target = opt_target.lower().strip()
        if target not in _OPT_TARGET_MAP:
            raise ValueError(f"Unknown OptimizationTarget '{opt_target}'. Choose from: 'StreamingData', 'HighLID', 'LowLID'")
        return _OPT_TARGET_MAP[target]

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Ensures contiguous float32 and applies L2-normalization for Cosine metric."""
        if self.needs_normalization:
            norms = np.linalg.norm(X, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return np.ascontiguousarray(X, dtype=np.float32)

    def _build_graph(self, X_f32: np.ndarray, graph_path: Path | None = None) -> deglib.DynamicExplorationGraph:
        """
        Phase 1: Graph Construction.
        - FLAS 1D pre-sorting is always executed in FP32 with radius_decay=0.9.
        - Graph is constructed as SizeBoundedGraph in FP32 for maximum accuracy and topology quality.
        - Saved to disk if graph_path is specified.
        """
        n_vectors, dims = X_f32.shape

        # 1. FLAS 1D Pre-sorting ALWAYS in float32
        print(f"Running FLAS 1D Pre-sorting in float32: N={n_vectors:,}, dim={dims}, threads={self.threads}...")
        t_start = time.perf_counter()
        sorted_indices = deglib.optimization.presort(
            X_f32,
            metric=self.base_metric_enum,
            radius_decay=0.9,
            threads=self.threads,
            callback="progress",
        )
        print(f"FLAS 1D Pre-sorting completed in {time.perf_counter() - t_start:.3f} s.")

        # 2. Build graph ALWAYS in FP32
        print(f"Constructing DEG graph in float32 (Metric: {self.base_metric_enum.name})...")
        graph_mut = deglib.builder.build_from_data(
            data=X_f32[sorted_indices],
            labels=sorted_indices,
            edges_per_vertex=self.k,
            metric=self.base_metric_enum,
            seed=7,
            optimization_target=self.opt_enum,
            extend_k=self.extend_k,
            thread_count=self.threads,
            callback="progress",
        )

        # Save graph in FP32 format if graph_path is given
        if graph_path:
            p = Path(graph_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            graph_mut.save_graph(str(p))
            print(f"Saved built DEG graph to: {p}")

        return graph_mut

    def _resolve_graph_path(self, n_vectors: int, dims: int) -> Path:
        """Determines default cache directory and graph filename for the corpus."""
        cache_dir = Path("D:/Data/DEG")
        if not cache_dir.exists() and not cache_dir.parent.exists():
            cache_dir = Path.home() / ".cache" / "deg_datasets"

        # Try to match known dataset by size & dim
        dataset_name = f"corpus_{n_vectors}_{dims}d"
        try:
            from dataset_utils import VIBE_DATASETS, get_default_cache_dir

            cache_dir = get_default_cache_dir()
            for key, meta in VIBE_DATASETS.items():
                if meta["dim"] == dims and abs(meta["size"] - n_vectors) < 1000:
                    dataset_name = key
                    break
        except ImportError:
            pass

        build_metric_str = deglib.Metric.FP32_L2.name if self.is_l2 else deglib.Metric.FP32_InnerProduct.name
        deg_dir = cache_dir / dataset_name / "deg"
        deg_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{dims}D_{build_metric_str}_K{self.k}_AddK{self.extend_k}Eps{self.extend_eps:.1f}_{self.opt_target}_FLAS.deg"
        return deg_dir / filename

    def _init_and_calibrate_quantizer(self, X_f32: np.ndarray) -> np.ndarray | None:
        """Calibrates the appropriate quantizer on base vectors and returns the quantized features."""
        if self.query_dtype == "float32":
            self.quantizer = None
            return None
        elif self.query_dtype == "int8":
            self.quantizer = deglib.optimization.make_scalar_quantizer_int8(X_f32)
            return self.quantizer.quantize(X_f32, num_threads=self.threads)
        else:
            raise ValueError(f"Unknown query_dtype: {self.query_dtype}")

    def fit(self, X: np.ndarray):
        """Builds or loads the DEG graph for corpus X, applies optional pruning and quantizes for search."""
        X_f32 = self._prepare_input(X)
        n_vectors, dims = X_f32.shape

        # Automatically resolve graph save/load path
        graph_path = self._resolve_graph_path(n_vectors, dims)

        print(
            f"Index Configuration: K={self.k}, ExtendK={self.extend_k}, ExtendEps={self.extend_eps:.2f}, "
            f"Opt={self.opt_target}, Threads={self.threads}, QueryType={self.query_dtype}, "
            f"PruneNonRNG={self.prune_non_rng}"
        )

        # Always calibrate quantizer on base dataset (even when loading cached graph)
        quantized_features = self._init_and_calibrate_quantizer(X_f32)

        # Store original features in FP16 for fast and memory-efficient reranking if quantized
        if self.query_dtype != "float32":
            self.original_features_fp16 = deglib.distances.floats_to_fp16(X_f32)
            self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=_METRIC_MAP[(self.is_l2, "float16")])

        # 1. Obtain base graph in FP32 (either by loading cached file or by building)
        if graph_path and graph_path.is_file():
            load_fn = deglib.load_mutable_graph if self.prune_non_rng else deglib.load_readonly_graph
            print(f"Loading cached DEG graph from: {graph_path}")
            graph = load_fn(str(graph_path))
        else:
            print(f"No cached graph found at {graph_path}. Building from scratch...")
            graph = self._build_graph(X_f32, graph_path=graph_path)

        # 2. Optional MRNG edge pruning on SizeBoundedGraph (MutableGraph)
        if self.prune_non_rng:
            print("Pruning non-RNG edges on SizeBoundedGraph...")
            removed = deglib.optimization.prune_non_rng_edges(graph, num_threads=self.threads)
            print(f"Pruned {removed:,} non-RNG edges.")

        # 3. Finalize ReadOnlyGraph with query_dtype features
        if quantized_features is not None:
            print(f"Finalizing ReadOnlyGraph with {self.query_dtype.upper()} (Metric: {self.query_metric_enum.name})...")
            target_space = deglib.FloatSpace.create(dim=dims, metric=self.query_metric_enum)
            self.graph = graph.to_readonly(target_space, quantized_features)
        else:
            self.graph = graph if not graph.is_mutable() else graph.to_readonly()

        # Initialize high-performance C++ Searcher (Zero-overhead query loop)
        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            quantizer=self.quantizer,
            refine_space=self.rerank_space_fp16 if self.query_dtype != "float32" else None,
            refine_data=self.original_features_fp16 if self.query_dtype != "float32" else None,
            search_eps=self.search_eps,
            rerank_factor=self.rerank_size_factor,
        )

    def set_query_arguments(self, search_eps: float, rerank_size_factor: float = 1.0):
        """Sets query-time search_eps and rerank scaling factor."""
        self.search_eps = float(search_eps)
        self.rerank_size_factor = float(rerank_size_factor)
        if self.searcher is not None:
            self.searcher.set_query_arguments(self.search_eps, self.rerank_size_factor)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with optional FP16 reranking directly in C++."""
        if self.searcher is None:
            raise RuntimeError("Index not fitted. Call fit(X) first.")
        if self.needs_normalization:
            v = v / np.linalg.norm(v)
        return self.searcher.search(np.ascontiguousarray(v, dtype=np.float32), n, return_distances=False, unsorted=True)

    def __str__(self) -> str:
        return (
            f"DEG(k={self.k}, extend_k={self.extend_k}, extend_eps={self.extend_eps}, "
            f"opt={self.opt_target}, "
            f"threads={self.threads}, query_dtype={self.query_dtype}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.search_eps})"
        )
