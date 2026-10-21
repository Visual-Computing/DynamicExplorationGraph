from typing import Any, Dict, List, Optional, Union
import numpy as np

from deglib_cpp import (
    GraphStats,
    calc_avg_edge_weight as _calc_avg_edge_weight_cpp,
    calc_edge_weight_histogram as _calc_edge_weight_histogram_cpp,
    check_graph_weights as _check_graph_weights_cpp,
    check_graph_regularity as _check_graph_regularity_cpp,
    check_graph_connectivity as _check_graph_connectivity_cpp,
    calc_non_mrng_edges as _calc_non_mrng_edges_cpp,
    calc_search_reachability as _calc_search_reachability_cpp,
    calc_exploration_reach as _calc_exploration_reach_cpp,
    analyze_graph as _analyze_graph_cpp,
    calc_graph_quality as _calc_graph_quality_cpp,
)

__all__ = [
    "GraphStats",
    "analyze_graph",
    "calc_graph_quality",
    "calc_avg_edge_weight",
    "calc_edge_weight_histogram",
    "check_graph_weights",
    "check_graph_regularity",
    "check_graph_connectivity",
    "calc_non_mrng_edges",
    "calc_search_reachability",
    "calc_exploration_reach",
]


def _unwrap(graph: Any) -> Any:
    return getattr(graph, "dynamic_exploration_graph_cpp", graph)


def _prepare_base_top(base_top: Optional[Any]) -> Optional[np.ndarray]:
    if base_top is None:
        return None
    if not isinstance(base_top, np.ndarray):
        return np.asarray(base_top, dtype=np.uint32)
    if base_top.dtype != np.uint32:
        return base_top.view(np.uint32)
    return base_top


def analyze_graph(
    graph: Any,
    base_top: Optional[Any] = None,
    sample_size: int = 0,
    num_threads: int = 0,
) -> GraphStats:
    """Perform comprehensive graph analysis and compute all structural and performance statistics."""
    bt = _prepare_base_top(base_top)
    return _analyze_graph_cpp(_unwrap(graph), bt, sample_size, num_threads)


def calc_graph_quality(
    graph: Any,
    base_top: Any,
    sample_size: int = 0,
    num_threads: int = 0,
) -> float:
    """Calculate ratio of edges matching true nearest neighbor ground truth."""
    bt = _prepare_base_top(base_top)
    return float(_calc_graph_quality_cpp(_unwrap(graph), bt, sample_size, num_threads))


_FIELDS = (
    "vertex_count",
    "edge_count",
    "feature_dims",
    "edges_per_vertex",
    "avg_out_degree",
    "min_out_degree",
    "max_out_degree",
    "avg_in_degree",
    "min_in_degree",
    "max_in_degree",
    "source_vertices",
    "search_reachability",
    "exploration_reachability",
    "memory_bytes",
)


def _stats_str(self: GraphStats) -> str:
    lines = [
        "Graph Statistics:",
        f"  Vertices: {self.vertex_count}",
        f"  Total edges: {self.edge_count}",
        f"  Feature dimensions: {self.feature_dims}",
        f"  Edges per vertex (k): {self.edges_per_vertex}",
        f"  Out-degree: avg={self.avg_out_degree:.2f}, min={self.min_out_degree}, max={self.max_out_degree}",
        f"  In-degree:  avg={self.avg_in_degree:.2f}, min={self.min_in_degree}, max={self.max_in_degree}, source_vertices={self.source_vertices}",
    ]
    if self.graph_quality >= 0:
        lines.append(f"  Graph Quality: {self.graph_quality * 100:.2f}%")
    if self.search_reachability >= 0:
        lines.append(f"  Search Reachability: {self.search_reachability * 100:.2f}%")
    if self.exploration_reachability >= 0:
        lines.append(f"  Exploration Reachability: {self.exploration_reachability * 100:.2f}%")
    lines.append(f"  Estimated memory: {self.memory_bytes / (1024.0 * 1024.0):.2f} MB")
    return "\n".join(lines)


def _stats_keys(self: GraphStats) -> List[str]:
    return [*_FIELDS, "graph_quality"] if self.graph_quality >= 0.0 else list(_FIELDS)


def _stats_getitem(self: GraphStats, key: str) -> Any:
    if key in self.keys():
        return getattr(self, key)
    raise KeyError(key)


def _stats_get(self: GraphStats, key: str, default: Any = None) -> Any:
    return getattr(self, key, default) if key in self.keys() else default


def _stats_contains(self: GraphStats, key: str) -> bool:
    return key in self.keys()


def _stats_to_dict(self: GraphStats) -> Dict[str, Any]:
    return {k: getattr(self, k) for k in self.keys()}


def _stats_repr(self: GraphStats) -> str:
    return (
        f"GraphStats(vertices={self.vertex_count}, edges={self.edge_count}, "
        f"dims={self.feature_dims}, k={self.edges_per_vertex}, quality={self.graph_quality:.4f})"
    )


GraphStats.keys = _stats_keys
GraphStats.__getitem__ = _stats_getitem
GraphStats.get = _stats_get
GraphStats.__contains__ = _stats_contains
GraphStats.__iter__ = lambda self: iter(self.keys())
GraphStats.to_dict = _stats_to_dict
GraphStats.__repr__ = _stats_repr
GraphStats.__str__ = _stats_str


def calc_avg_edge_weight(graph: Any, scale: int = 1) -> float:
    return _calc_avg_edge_weight_cpp(_unwrap(graph), scale)


def calc_edge_weight_histogram(graph: Any, sort: bool, scale: int = 1) -> List[float]:
    return _calc_edge_weight_histogram_cpp(_unwrap(graph), sort, scale)


def check_graph_weights(graph: Any) -> bool:
    return _check_graph_weights_cpp(_unwrap(graph))


def check_graph_regularity(graph: Any, expected_vertices: int, check_back_link: bool = False) -> bool:
    return _check_graph_regularity_cpp(_unwrap(graph), expected_vertices, check_back_link)


def check_graph_connectivity(graph: Any) -> bool:
    return _check_graph_connectivity_cpp(_unwrap(graph))


def calc_non_mrng_edges(graph: Any) -> int:
    return _calc_non_mrng_edges_cpp(_unwrap(graph))


def calc_search_reachability(graph: Any) -> float:
    g = _unwrap(graph)
    return _calc_search_reachability_cpp(g) / max(g.size(), 1)


def calc_exploration_reach(graph: Any) -> float:
    g = _unwrap(graph)
    return _calc_exploration_reach_cpp(g) / max(g.size(), 1)
