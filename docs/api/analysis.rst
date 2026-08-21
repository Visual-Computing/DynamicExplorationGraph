Analysis
========

Graph structure inspection, reachability analysis, connectivity, and statistics.

Overview
--------

The analysis module provides diagnostic tools to evaluate Exploration Graph topology,
including edge weight distributions, graph connectivity, RNG compliance, and reachability.

Full Graph Summary
------------------

.. autofunction:: deglib.analysis.analyze_graph

Reachability & Connectivity
---------------------------

.. autofunction:: deglib.analysis.calc_search_reachability

.. autofunction:: deglib.analysis.calc_exploration_reach

.. autofunction:: deglib.analysis.check_graph_connectivity

.. autofunction:: deglib.analysis.check_graph_regularity

Edge & Weight Diagnostics
-------------------------

.. autofunction:: deglib.analysis.calc_avg_edge_weight

.. autofunction:: deglib.analysis.calc_edge_weight_histogram

.. autofunction:: deglib.analysis.calc_non_rng_edges

.. autofunction:: deglib.analysis.check_graph_weights

Example Usage
-------------

.. code-block:: python

   import deglib

   # Load an existing graph
   graph = deglib.graph.load_readonly_graph("my_graph.deg")

   # Obtain full structural diagnostics
   stats = deglib.analysis.analyze_graph(graph)
   print(f"Vertex Count: {stats['vertex_count']}")
   print(f"Average Out-Degree: {stats['avg_out_degree']:.2f}")
   print(f"Search Reachability: {stats['search_reachability']:.2%}")
   print(f"Exploration Reachability: {stats['exploration_reachability']:.2%}")

