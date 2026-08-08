from typing import List

from deglib_cpp import (calc_avg_edge_weight as calc_avg_edge_weight_cpp,
                       calc_edge_weight_histogram as calc_edge_weight_histogram_cpp,
                       check_graph_weights as check_graph_weights_cpp,
                       check_graph_regularity as check_graph_regularity_cpp,
                       check_graph_connectivity as check_graph_connectivity_cpp,
                       calc_non_rng_edges as calc_non_rng_edges_cpp
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


def check_graph_regularity(graph: DynamicExplorationGraph, expected_vertices: int, check_back_link: bool = False) -> bool:
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
