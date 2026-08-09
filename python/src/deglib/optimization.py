import deglib_cpp

from .graph import DynamicExplorationGraph


def remove_non_mrng_edges(graph: DynamicExplorationGraph, num_threads: int = 0) -> int:
    """
    Remove all edges which do not satisfy the MRNG condition.

    :param graph: The graph to optimize. Must be mutable.
    :param num_threads: Number of threads to use for parallel processing. If 0, uses hardware concurrency.
    :return: Number of edges removed.
    """
    return deglib_cpp.remove_non_mrng_edges(graph.dynamic_exploration_graph_cpp, num_threads)


def prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, num_threads: int = 0):
    """
    Prune the worst (highest-weight) `prune_worst` neighbors of each vertex
    by replacing them with self-loops.

    :param graph: The graph to optimize. Must be mutable.
    :param prune_worst: Number of worst neighbors to replace with self-loops per vertex.
    :param num_threads: Number of threads to use for parallel processing. If 0, uses hardware concurrency.
    """
    deglib_cpp.prune_worst_edges(graph.dynamic_exploration_graph_cpp, prune_worst, num_threads)


__all__ = ['remove_non_mrng_edges', 'prune_worst_edges']
