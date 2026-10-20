Optimization
============

Graph edge pruning and index pre-sorting (FLAS).

For feature transformations (MIPS) and quantization (INT8, UINT8, EVP), see :doc:`transformation`.

Graph Edge Pruning
------------------

Graph pruning optimizes search performance and graph quality by removing suboptimal or redundant edges:

- **MRNG Pruning**: Removes edges that violate the Monotonic Relative Neighborhood Graph property, reducing search hops without losing reachability.
- **Worst-Edge Pruning**: Replaces the highest-weight edges of each vertex with self-loops to constrain vertex degree.

.. autofunction:: deglib.optimization.prune_non_mrng_edges

.. autofunction:: deglib.optimization.prune_worst_edges

Index Pre-Sorting (FLAS)
------------------------

Fast Linear Alignment Scheme (FLAS) computes a 1D ordering of high-dimensional vectors to enhance spatial locality, resulting in faster graph construction and cache-efficient traversal.

.. autofunction:: deglib.optimization.presort

Example Usage
-------------

.. code-block:: python

   import deglib
   import numpy as np

   # Generate sample data
   data = np.random.randn(1000, 64).astype(np.float32)

   # Pre-sort vectors using FLAS for improved locality
   sorted_indices = deglib.optimization.presort(data)
   sorted_data = data[sorted_indices]

   # Build graph from pre-sorted data
   graph = deglib.builder.build_from_data(sorted_data, metric=deglib.Metric.FP32_L2)

   # Optimize graph edges using MRNG pruning
   edges_removed = deglib.optimization.prune_non_mrng_edges(graph)
   print(f"Pruned {edges_removed} redundant edges.")


