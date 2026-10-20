Searching
=========

This guide covers approximate nearest neighbor search, search parameters, label filtering, exploratory graph walks, neighbor retrieval, and candidate re-ranking.

Vector Search (k-NN)
--------------------

Querying the graph finds the `k` closest feature vectors to an unindexed query vector using :meth:`~deglib.graph.DynamicExplorationGraph.search`:

.. code-block:: python

   import numpy as np
   import deglib

   # Assume graph is already built (dimension 128)
   dim = 128
   query = np.random.randn(dim).astype(np.float32)

   # Single query search using relative margin (eps_or_ef < 1.0)
   indices, distances = graph.search(query, k=10, eps_or_ef=0.1)

   print("Top-10 nearest neighbor IDs:", indices)
   print("Distances:", distances)

Batch Search
^^^^^^^^^^^^

Pass a 2D NumPy array of shape ``(N_queries, dim)`` to search multiple queries in parallel with OpenMP multi-threading:

.. code-block:: python

   queries = np.random.randn(100, dim).astype(np.float32)

   # Search 100 queries in parallel using 4 CPU worker threads
   indices, distances = graph.search(queries, k=10, eps_or_ef=0.1, threads=4)

   # Result shapes: (100, 10)
   print("Batch result indices shape:", indices.shape)
   print("Batch result distances shape:", distances.shape)

---

Search Parameter: `eps_or_ef`
-----------------------------

The exploration behavior is controlled by the unified parameter ``eps_or_ef``:

- **Relative Margin Mode (``eps_or_ef < 1.0``)**:
  Interpreted as the relative distance margin :math:`\epsilon` (epsilon). Traversal continues exploring neighbors as long as candidates are within :math:`(1 + \epsilon) \cdot d_{\text{best}}`.
  
  - **Small values** (e.g. ``0.0`` to ``0.05``): Maximum query throughput (QPS) with low latency.
  - **Higher values** (e.g. ``0.1`` to ``0.3``): Higher recall rate by evaluating more candidate nodes.

- **Beam Width Mode (``eps_or_ef >= 1.0``)**:
  Interpreted as the fixed candidate beam width :math:`ef` (similar to HNSW). Traversal maintains a bounded priority queue of candidate vertices.

  - **Examples**: ``eps_or_ef=32``, ``64``, or ``128`` for predictable exploration depth and high recall.

.. code-block:: python

   # Relative margin exploration (DEG default)
   ids_eps, dists_eps = graph.search(query, k=10, eps_or_ef=0.1)

   # Fixed candidate pool exploration (HNSW-style)
   ids_ef, dists_ef = graph.search(query, k=10, eps_or_ef=64)

---

Filtering
---------

Search can be constrained to a specific subset of valid vertex IDs using :class:`~deglib.search.Filter`:

.. code-block:: python

   from deglib.search import Filter

   # Only allow these specific external labels in search results
   allowed_ids = np.array([5, 12, 42, 108, 500], dtype=np.int32)
   search_filter = Filter(allowed_ids)

   indices, distances = graph.search(query, k=5, eps_or_ef=0.1, filter_labels=search_filter)

---

Exploration (Graph Walk)
------------------------

Instead of providing an unindexed feature vector, you can explore the graph topology starting directly from existing vertex labels using :meth:`~deglib.graph.DynamicExplorationGraph.explore`.

Single Vertex Exploration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Explore the graph starting from vertex ID 42
   neighbor_ids, distances = graph.explore(
       entry_external_label=42,
       k=10,
       eps_or_ef=0.1,
       include_entry=False,  # exclude the start vertex itself
   )

   print("Exploration results from vertex 42:", neighbor_ids)

Batch Exploration
^^^^^^^^^^^^^^^^^

Pass a 1D array or list of entry labels to explore multiple neighborhoods concurrently:

.. code-block:: python

   entry_ids = np.array([42, 100, 250, 777], dtype=np.uint32)

   # Returns 2D arrays of shape (4, 10)
   batch_neighbors, batch_dists = graph.explore(
       entry_ids,
       k=10,
       eps_or_ef=0.1,
       threads=4,
   )

---

Neighbor Retrieval & Topology Inspection
----------------------------------------

Directly access the raw outgoing edges of any indexed vertex:

.. code-block:: python

   # Single vertex: returns a Python list of neighbor external labels
   neighbors = graph.get_neighbors(42)
   print(f"Vertex 42 has {len(neighbors)} neighbors: {neighbors}")

   # Batch lookup: returns a 2D NumPy array of shape (N, edges_per_vertex)
   query_labels = np.array([10, 20, 30], dtype=np.uint32)
   batch_edges = graph.get_neighbors_batch(query_labels, threads=2)
   print("Batch neighbor table shape:", batch_edges.shape)

Path Reachability
^^^^^^^^^^^^^^^^^

Check whether a directed path exists between two vertices in the graph:

.. code-block:: python

   can_reach = graph.has_path(from_label=42, to_label=100)
   print(f"Path from 42 to 100 exists: {can_reach}")

---

Re-Ranking
----------

In two-stage retrieval pipelines, a fast preliminary search provides a pool of candidate IDs that can be re-ranked with exact distances using :func:`deglib.search.rerank`:

.. code-block:: python

   import deglib

   space = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.FP32_L2)
   candidate_pool = np.random.randint(0, graph.size(), size=(10, 50), dtype=np.uint32)
   dataset = np.random.randn(graph.size(), dim).astype(np.float32)

   # Re-rank top 10 from the 50 candidates per query
   top_indices, top_distances = deglib.search.rerank(
       space=space,
       queries=queries[:10],
       candidate_indices=candidate_pool,
       base_vectors=dataset,
       k_top=10,
       return_distances=True,
   )
