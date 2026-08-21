from typing import Dict, Any, List

from deglib_cpp import (
    calc_avg_edge_weight as calc_avg_edge_weight_cpp,
    calc_edge_weight_histogram as calc_edge_weight_histogram_cpp,
    check_graph_weights as check_graph_weights_cpp,
    check_graph_regularity as check_graph_regularity_cpp,
    check_graph_connectivity as check_graph_connectivity_cpp,
    calc_non_rng_edges as calc_non_rng_edges_cpp,
    calc_search_reachability as calc_search_reachability_cpp,
    calc_exploration_reach as calc_exploration_reach_cpp,
    analyze_graph as analyze_graph_cpp,
)
from .graph import DynamicExplorationGraph


def calc_avg_edge_weight(graph: DynamicExplorationGraph, scale: int = 1) -> float:
    """
    Compute the average weight of all edges in the input graph.

    :param graph: The graph for which the average edge weight is calculated.
    :param scale: The scale factor by which edge weights are multiplied (default is 1).
    :return: The average edge weight in the graph after scaling.
    """
    return calc_avg_edge_weight_cpp(graph.dynamic_exploration_graph_cpp, scale)


def calc_edge_weight_histogram(graph: DynamicExplorationGraph, sort: bool, scale: int = 1) -> List[float]:
    """
    Calculate a 10-bin histogram of edge weights for the graph.

    - Collects all non-zero edge weights.
    - Optionally sorts weights before binning.
    - Divides the range into 10 bins.
    - Computes and scales the average weight for each bin.

    :param graph: The graph to calculate the edge weight histogram for.
    :param sort: Whether to sort edge weights before binning.
    :param scale: Scale factor applied to each edge weight.
    :return: A list of 10 float values representing the average weight in each bin.
    """
    return calc_edge_weight_histogram_cpp(graph.dynamic_exploration_graph_cpp, sort, scale)


def check_graph_weights(graph: DynamicExplorationGraph) -> bool:
    """
    Verify that stored edge weights match the actual computed distances between connected vertices.

    :param graph: The graph to verify.
    :return: True if all edge weights match the vertex feature distances, False otherwise.
    """
    return check_graph_weights_cpp(graph.dynamic_exploration_graph_cpp)


def check_graph_regularity(
    graph: DynamicExplorationGraph, expected_vertices: int, check_back_link: bool = False
) -> bool:
    """
    Check structural regularity of the graph, ensuring expected vertex count and edge validity.

    :param graph: The graph to check.
    :param expected_vertices: The expected number of vertices in the graph.
    :param check_back_link: Whether to also verify the existence of reverse links.
    :return: True if the graph is structurally regular and complete, False otherwise.
    """
    return check_graph_regularity_cpp(graph.dynamic_exploration_graph_cpp, expected_vertices, check_back_link)


def check_graph_connectivity(graph: DynamicExplorationGraph) -> bool:
    """
    Check if the graph is fully connected as a single component.

    :param graph: The graph to check connectivity for.
    :return: True if the graph forms a single connected component, False otherwise.
    """
    return check_graph_connectivity_cpp(graph.dynamic_exploration_graph_cpp)


def calc_non_rng_edges(graph: DynamicExplorationGraph) -> int:
    """
    Count the number of edges in the graph that violate the Relative Neighborhood Graph (RNG) rule.

    :param graph: The graph to analyze.
    :return: Total number of non-RNG edges.
    """
    return calc_non_rng_edges_cpp(graph.dynamic_exploration_graph_cpp)


def calc_search_reachability(graph: DynamicExplorationGraph) -> float:
    """
    Compute the search reachability ratio of the graph.

    Measures the fraction of vertices reachable from the graph's entry points via greedy/BFS traversal.

    :param graph: The graph to analyze.
    :return: Ratio between 0.0 and 1.0 representing the reachable fraction of vertices.
    """
    graph_size = graph.size()
    reachable = calc_search_reachability_cpp(graph.dynamic_exploration_graph_cpp)
    return reachable / max(graph_size, 1)


def calc_exploration_reach(graph: DynamicExplorationGraph) -> float:
    """
    Compute the exploration reachability ratio of the graph.

    Measures the average fraction of vertices reachable starting from any arbitrary vertex.

    :param graph: The graph to analyze.
    :return: Average reachability ratio between 0.0 and 1.0 across all vertices.
    """
    graph_size = graph.size()
    avg_reach = calc_exploration_reach_cpp(graph.dynamic_exploration_graph_cpp)
    return avg_reach / max(graph_size, 1)


def analyze_graph(graph: DynamicExplorationGraph) -> Dict[str, Any]:
    """
    Perform a comprehensive graph analysis and compute all structural and performance statistics.

    Returns a dictionary containing:
    - ``vertex_count``: Total number of vertices in the graph.
    - ``edge_count``: Total number of directed edges.
    - ``feature_dims``: Feature vector dimensionality.
    - ``edges_per_vertex``: Average number of outgoing edges per vertex.
    - ``avg_out_degree`` / ``min_out_degree`` / ``max_out_degree``: Out-degree distribution stats.
    - ``avg_in_degree`` / ``min_in_degree`` / ``max_in_degree``: In-degree distribution stats.
    - ``source_vertices``: Number of entry/source vertices.
    - ``search_reachability``: Fraction of vertices reachable from entry points.
    - ``exploration_reachability``: Average fraction of vertices reachable from any vertex.
    - ``memory_bytes``: Estimated graph memory consumption in bytes.

    :param graph: The graph to analyze.
    :return: Dictionary containing all computed graph statistics.
    """
    stats = analyze_graph_cpp(graph.dynamic_exploration_graph_cpp)
    return {
        "vertex_count": stats.vertex_count,
        "edge_count": stats.edge_count,
        "feature_dims": stats.feature_dims,
        "edges_per_vertex": stats.edges_per_vertex,
        "avg_out_degree": stats.avg_out_degree,
        "min_out_degree": stats.min_out_degree,
        "max_out_degree": stats.max_out_degree,
        "avg_in_degree": stats.avg_in_degree,
        "min_in_degree": stats.min_in_degree,
        "max_in_degree": stats.max_in_degree,
        "source_vertices": stats.source_vertices,
        "search_reachability": stats.search_reachability,
        "exploration_reachability": stats.exploration_reachability,
        "memory_bytes": stats.memory_bytes,
    }
