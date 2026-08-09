import deglib.analysis


def analyze_graph(graph):
    """
    Analyze a search graph by delegating to the native C++ implementation in deglib.analysis.

    Computes basic stats (vertex count, edge count, feature dimensions, edges per vertex),
    degree statistics (in/out degree), reachability metrics, and memory estimation.

    :param graph: The graph to analyze (DynamicExplorationGraph)
    :returns: A dictionary with all computed statistics
    """
    return deglib.analysis.analyze_graph(graph)
