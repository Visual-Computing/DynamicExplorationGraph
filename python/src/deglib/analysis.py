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
    Weights are scaled by the specified scale factor.

    This function uses the C++ version (calc_avg_edge_weight_cpp) in deglib_cpp package as a backend.


    :param graph: The graph for which the average edge weight is calculated.
    :param scale: The scale factor by which edge weights are multiplied (default is 1).

    :returns: The average edge weight in the graph after scaling.
    """
    return calc_avg_edge_weight_cpp(graph.dynamic_exploration_graph_cpp, scale)


def calc_edge_weight_histogram(graph: DynamicExplorationGraph, sort: bool, scale: int = 1) -> List[float]:
    """
    The function calculates a histogram of edge weights for a given graph by:

    - Collecting all non-zero edge weights.
    - Optionally sorting these weights.
    - Dividing the weights into 10 bins.
    - Computing and scaling the average weight for each bin.

    The result is a vector containing the scaled average weights of the edge weights in each bin.

    :param graph: The graph to calculate the average edge weight for
    :param sort: sort edge weights before creating histogram
    :param scale: scale factor for each edge weight

    :returns: A list of 10 float values representing the scaled average weights
    """
    return calc_edge_weight_histogram_cpp(graph.dynamic_exploration_graph_cpp, sort, scale)


def check_graph_weights(graph: DynamicExplorationGraph) -> bool:
    """
    Check if the weights of the graph are still the same to the distance of the vertices

    :param graph: The graph to calculate the average edge weight for

    :returns: True, if the graph are still the same to the distance of the vertices otherwise False
    """
    return check_graph_weights_cpp(graph.dynamic_exploration_graph_cpp)


def check_graph_regularity(
    graph: DynamicExplorationGraph, expected_vertices: int, check_back_link: bool = False
) -> bool:
    """
    TODO: rework documentation
    Is the vertex_index an RNG conform neighbor if it gets connected to target_index?

    Does vertex_index has a neighbor which is connected to the target_index and has a lower weight?
    """
    return check_graph_regularity_cpp(graph.dynamic_exploration_graph_cpp, expected_vertices, check_back_link)


def check_graph_connectivity(graph: DynamicExplorationGraph) -> bool:
    """
    Check if the graph is connected and contains only one graph component.

    :param graph: The graph to check connectivity for
    """
    return check_graph_connectivity_cpp(graph.dynamic_exploration_graph_cpp)


def calc_non_rng_edges(graph: DynamicExplorationGraph) -> int:
    """
    TODO: rework documentation
    """
    return calc_non_rng_edges_cpp(graph.dynamic_exploration_graph_cpp)


def calc_search_reachability(graph: DynamicExplorationGraph) -> float:
    """
    Compute the search reachability ratio of the graph.

    Measures how many vertices can be reached from the graph's entry points via BFS.
    Returns a ratio between 0.0 and 1.0.

    :param graph: The graph to analyze
    :returns: The fraction of vertices reachable from entry points
    """
    graph_size = graph.size()
    reachable = calc_search_reachability_cpp(graph.dynamic_exploration_graph_cpp)
    return reachable / max(graph_size, 1)


def calc_exploration_reach(graph: DynamicExplorationGraph) -> float:
    """
    Compute the exploration reachability ratio of the graph.

    Measures the average number of vertices reachable from any given vertex,
    then normalizes to a ratio between 0.0 and 1.0.

    :param graph: The graph to analyze
    :returns: The average fraction of vertices reachable per vertex
    """
    graph_size = graph.size()
    avg_reach = calc_exploration_reach_cpp(graph.dynamic_exploration_graph_cpp)
    return avg_reach / max(graph_size, 1)


def analyze_graph(graph: DynamicExplorationGraph) -> Dict[str, Any]:
    """
    Analyze a search graph and compute all statistics.

    Computes basic stats (vertex count, edge count, feature dimensions, edges per vertex),
    degree statistics (in/out degree), reachability metrics, and memory estimation.

    :param graph: The graph to analyze
    :returns: A dictionary with all computed statistics
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
