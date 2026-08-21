Quickstart
==========

This quickstart guide demonstrates how to build an Exploration Graph, save and load it, and perform nearest neighbor search in under two minutes.

Core Workflow
-------------

A typical `deglib` workflow consists of four simple steps:

1. **Prepare Data**: Feature vectors represented as a 2D float32 NumPy array.
2. **Build Index**: Construct an Exploration Graph using :func:`deglib.builder.build_from_data`.
3. **Query the Index**: Find the `k` nearest neighbors with :meth:`~deglib.graph.DynamicExplorationGraph.search`.
4. **Save & Reload**: Persist the search graph to disk for production reuse.

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import deglib

   # 1. Generate sample feature vectors (10,000 vectors of 128 dimensions)
   num_samples, dim = 10_000, 128
   data = np.random.randn(num_samples, dim).astype(np.float32)

   # 2. Build the Exploration Graph
   graph = deglib.builder.build_from_data(
       data,
       k=30,      # target number of edges per vertex
       eps=0.1,   # exploration parameter during construction
   )
   print(f"Graph built with {graph.size()} vertices.")

   # 3. Query the index for the 10 nearest neighbors
   query = np.random.randn(dim).astype(np.float32)
   indices, distances = graph.search(query, k=10, eps=0.1)

   print("Top-10 nearest neighbor IDs:", indices)
   print("Corresponding distances:", distances)

   # 4. Save and reload the graph
   graph.save_graph("my_graph.deg")

   # Load read-only graph for fast serving
   loaded_graph = deglib.graph.load_readonly_graph("my_graph.deg")
   loaded_indices, _ = loaded_graph.search(query, k=10, eps=0.1)

Batch Querying
--------------

You can search multiple query vectors in parallel by passing a 2D NumPy array:

.. code-block:: python

   # Search for 100 queries simultaneously using available CPU threads
   queries = np.random.randn(100, dim).astype(np.float32)
   indices, distances = graph.search(queries, k=10, eps=0.1, threads=0)

   # Result shapes: (100, 10)
   print("Batch result shape:", indices.shape)

Understanding the `eps` Parameter
---------------------------------

The `eps` (epsilon) parameter controls the exploration margin during graph traversal:

- **Higher `eps`** (e.g. `0.2`): Examines more candidate nodes, yielding higher recall at slightly lower QPS.
- **Lower `eps`** (e.g. `0.0` to `0.05`): Faster search speed with a small trade-off in accuracy.

Next Steps
----------

- Learn about advanced building strategies and incremental loading in :doc:`building_graphs`.
- Explore query filtering and two-stage re-ranking in :doc:`searching`.
- Check the complete :doc:`/api/index` for detailed parameter specifications.

