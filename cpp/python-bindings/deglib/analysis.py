from typing import List

from deglib_cpp import (calc_avg_edge_weight as calc_avg_edge_weight_cpp,
                        calc_edge_weight_histogram as calc_edge_weight_histogram_cpp,
                        check_graph_weights as check_graph_weights_cpp,
                        check_graph_regularity as check_graph_regularity_cpp,
                        check_graph_connectivity as check_graph_connectivity_cpp,
                        calc_non_rng_edges as calc_non_rng_edges_cpp
                        )
from .graph import SizeBoundedGraph


# TODO: replace SizeBoundedGraph with MutableGraph interface
def calc_avg_edge_weight(graph: SizeBoundedGraph, scale: int = 1) -> float:
    return calc_avg_edge_weight_cpp(graph.to_cpp(), scale)


# TODO: replace SizeBoundedGraph with MutableGraph interface
def calc_edge_weight_histogram(graph: SizeBoundedGraph, sort: bool, scale: int = 1) -> List[float]:
    return calc_edge_weight_histogram_cpp(graph.to_cpp(), sort, scale)


# TODO: replace SizeBoundedGraph with MutableGraph interface
def check_graph_weights(graph: SizeBoundedGraph) -> bool:
    return check_graph_weights_cpp(graph.to_cpp())


# TODO: replace SizeBoundedGraph with SearchGraph interface
def check_graph_regularity(graph: SizeBoundedGraph, expected_vertices: int, check_back_link: bool = False) -> bool:
    return check_graph_regularity_cpp(graph.to_cpp(), expected_vertices, check_back_link)


# TODO: replace SizeBoundedGraph with SearchGraph interface
def check_graph_connectivity(graph: SizeBoundedGraph) -> bool:
    return check_graph_connectivity_cpp(graph.to_cpp())


# TODO: replace SizeBoundedGraph with MutableGraph interface
def calc_non_rng_edges(graph: SizeBoundedGraph) -> int:
    return calc_non_rng_edges_cpp(graph.to_cpp())
