Optimization
============

Graph edge pruning, index pre-sorting (FLAS), MIPS transformations, and quantization.

Graph Edge Pruning
------------------

Graph pruning optimizes search performance and graph quality by removing suboptimal or redundant edges:

- **RNG Pruning**: Removes edges that violate the Relative Neighborhood Graph property, reducing search hops without losing reachability.
- **Worst-Edge Pruning**: Replaces the highest-weight edges of each vertex with self-loops to constrain vertex degree.

.. autofunction:: deglib.optimization.prune_non_rng_edges

.. autofunction:: deglib.optimization.prune_worst_edges

Index Pre-Sorting (FLAS)
------------------------

Fast Linear Alignment Scheme (FLAS) computes a 1D ordering of high-dimensional vectors to enhance spatial locality, resulting in faster graph construction and cache-efficient traversal.

.. autofunction:: deglib.optimization.presort

MIPS to L2 Transformations
--------------------------

Maximum Inner Product Search (MIPS) can be mapped to Euclidean (L2) distance search by augmenting vectors by one extra dimension.

.. autofunction:: deglib.optimization.mips_l2_transform

.. autofunction:: deglib.optimization.mips_l2_transform_query

Quantization
------------

Quantize floating point vectors into compact byte-packed representations.

.. autofunction:: deglib.optimization.quantize_batch

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
   graph = deglib.builder.build_from_data(sorted_data)

   # Optimize graph edges using RNG pruning
   edges_removed = deglib.optimization.prune_non_rng_edges(graph)
   print(f"Pruned {edges_removed} redundant edges.")

