from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

import deglib.analysis
from dataset import ivecs_read


def compute_graph_quality(
    graph,
    base_top: np.ndarray,
    sample_size: Optional[int] = None,
    num_threads: int = 0,
) -> Dict[str, float]:
    """
    Computes graph quality comparing each vertex's outgoing edges against its true nearest neighbors.

    For each vertex v with d outgoing edges (where d = len(neighbors(v))):
    We check how many of its d edges are among the true top-d nearest neighbors in base_top.
    Quality = total_hits / total_edges (across all evaluated vertices).

    :param graph: Search graph (DynamicExplorationGraph or ReadOnlyGraph)
    :param base_top: 2D array of shape (N, K) with nearest neighbor indices for each vertex
    :param sample_size: If specified, evaluates only on sample_size vertices (e.g. 50000 for fast eval)
    :param num_threads: Number of worker threads (default: 0 = all CPU cores)
    :return: Dictionary containing precision/quality metrics
    """
    sample = sample_size if (sample_size is not None and sample_size > 0) else 0
    quality = deglib.analysis.calc_graph_quality(
        graph,
        base_top,
        sample_size=sample,
        num_threads=num_threads,
    )

    return {
        "graph_quality": quality,
    }


def analyze_graph(
    graph,
    base_top: Optional[Union[np.ndarray, str, Path]] = None,
    dataset_key: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    is_half: Optional[bool] = None,
    log_output: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a search graph by delegating to the native C++ implementation in deglib.analysis
    and optionally computing graph quality metrics if base_top ground truth is available.

    :param graph: The graph to analyze (DynamicExplorationGraph or ReadOnlyGraph)
    :param base_top: Optional base_top array or path to base_top{100,1000}.ivecs
    :param dataset_key: Optional dataset key to auto-detect base_top file in cache_dir
    :param cache_dir: Cache directory for auto-detecting base_top
    :param sample_size: Max number of vertices to evaluate quality on (None = all)
    :param is_half: Explicitly specify if graph is a half-dataset graph (None = auto-detect by vertex count)
    :param log_output: Whether to print formatted statistics to console (default: True)
    :returns: Dictionary with all computed graph statistics and quality metrics
    """
    stats = deglib.analysis.analyze_graph(graph)

    # Resolve base_top if not provided directly
    base_top_arr = None
    if base_top is not None:
        if isinstance(base_top, (str, Path)):
            base_top_path = Path(base_top)
            if base_top_path.is_file():
                base_top_arr = ivecs_read(base_top_path)
        elif isinstance(base_top, np.ndarray):
            base_top_arr = base_top
    elif dataset_key and cache_dir:
        from dataset import DATASET_METADATA, resolve_dataset_key, ensure_dataset
        key = resolve_dataset_key(dataset_key)
        base_count = 0
        search_dirs = []

        if key in DATASET_METADATA:
            meta = DATASET_METADATA[key]
            d_dir = ensure_dataset(key, Path(cache_dir))
            folder_name = meta["folder"]
            base_count = meta.get("base_count", 0)
            search_dirs.append(d_dir)
        else:
            folder_name = key
            cand_folder = Path(cache_dir) / key
            if cand_folder.is_dir():
                search_dirs.append(cand_folder)

        if is_half is None:
            is_half_graph = base_count > 0 and abs(graph.size() - base_count // 2) <= 1
        else:
            is_half_graph = is_half

        if is_half_graph:
            cand_patterns = [
                f"{folder_name}_base_half_top1000.ivecs",
                f"{key}_base_half_top1000.ivecs",
                "base_half_top1000.ivecs",
                "*base_half_top1000.ivecs",
            ]
        else:
            cand_patterns = [
                f"{folder_name}_base_top1000.ivecs",
                f"{key}_base_top1000.ivecs",
                "base_top1000.ivecs",
                "*base_top1000.ivecs",
            ]

        for s_dir in search_dirs:
            for pat in cand_patterns:
                matches = list(s_dir.rglob(pat))
                if matches:
                    base_top_arr = ivecs_read(matches[0])
                    break
            if base_top_arr is not None:
                break

    if base_top_arr is not None:
        if base_top_arr.shape[0] != graph.size():
            print(
                f"Warning: ground truth has {base_top_arr.shape[0]} rows but graph has "
                f"{graph.size()} vertices — skipping graph quality (wrong/mismatched base_top)."
            )
        else:
            quality_stats = compute_graph_quality(graph, base_top_arr, sample_size=sample_size)
            stats.update(quality_stats)

    if log_output:
        print_graph_stats(stats)

    return stats


def print_graph_stats(stats: Dict[str, Any]):
    """Prints graph statistics formatted like the native DEG C++ benchmark."""
    print("Graph Statistics:")
    print(f"  Vertices: {stats.get('vertex_count', 0)}")
    print(f"  Total edges: {stats.get('edge_count', 0)}")
    print(f"  Feature dimensions: {stats.get('feature_dims', 0)}")
    epv = stats.get('edges_per_vertex', 0)
    print(f"  Edges per vertex (k): {epv}")
    print(
        f"  Out-degree: avg={stats.get('avg_out_degree', 0.0):.2f}, "
        f"min={stats.get('min_out_degree', 0)}, max={stats.get('max_out_degree', 0)}"
    )
    print(
        f"  In-degree:  avg={stats.get('avg_in_degree', 0.0):.2f}, "
        f"min={stats.get('min_in_degree', 0)}, max={stats.get('max_in_degree', 0)}, "
        f"source_vertices={stats.get('source_vertices', 0)}"
    )

    if "search_reachability" in stats and stats["search_reachability"] >= 0:
        print(f"  Search Reachability: {stats['search_reachability'] * 100:.2f}%")
    if "exploration_reachability" in stats and stats["exploration_reachability"] >= 0:
        print(f"  Exploration Reachability: {stats['exploration_reachability'] * 100:.2f}%")

    if "memory_bytes" in stats:
        print(f"  Estimated memory: {stats['memory_bytes'] / (1024.0 * 1024.0):.2f} MB")

    # Quality metric
    if "graph_quality" in stats:
        print(f"  Graph Quality: {stats['graph_quality'] * 100:.2f}%")
