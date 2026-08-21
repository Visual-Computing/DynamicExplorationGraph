Searching
=========

This guide covers approximate nearest neighbor search, search parameters, label filtering, exploratory search, and candidate re-ranking.

Vector Search (k-NN)
--------------------

Querying the graph finds the `k` closest feature vectors to a query vector:

.. code-block:: python

   import numpy as np
   import deglib

   # Assume graph is already built (dimension 128)
   dim = 128
   query = np.random.randn(dim).astype(np.float32)

   # Single query search
   indices, distances = graph.search(query, k=10, eps=0.1)

   print("Top-10 nearest neighbor IDs:", indices)
   print("Distances:", distances)

Batch Search
^^^^^^^^^^^^

Pass a 2D NumPy array of shape ``(N_queries, dim)`` to search multiple queries in parallel:

.. code-block:: python

   queries = np.random.randn(100, dim).astype(np.float32)
   indices, distances = graph.search(queries, k=10, eps=0.1, threads=0)

   # Result shapes: (100, 10)
   print("Batch result indices shape:", indices.shape)

---

Search Parameter: `eps`
-----------------------

The primary parameter controlling the search trade-off is ``eps`` (epsilon):

- **Small `eps`** (e.g. ``0.0`` to ``0.05``): Maximum query throughput (QPS) with low latency.
- **Higher `eps`** (e.g. ``0.1`` to ``0.3``): Higher recall rate by evaluating more candidate nodes during graph traversal.

---

Filtering
---------

Search can be constrained to a specific subset of valid vertex IDs using :class:`~deglib.search.Filter`:

.. code-block:: python

   from deglib.search import Filter

   # Only allow these specific external labels in search results
   allowed_ids = np.array([5, 12, 42, 108, 500], dtype=np.int32)
   search_filter = Filter(allowed_ids)

   indices, distances = graph.search(query, k=5, eps=0.1, filter_labels=search_filter)

---

Exploration (Graph Walk)
------------------------

Instead of providing a new feature vector, you can explore the neighborhood starting from an existing vertex label using :meth:`~deglib.graph.DynamicExplorationGraph.explore`:

.. code-block:: python

   # Explore the graph starting from vertex ID 42
   neighbor_ids, distances = graph.explore(
       entry_external_label=42,
       k=10,
       eps=0.1,
       include_entry=False,  # exclude the start vertex itself
   )

   print("Exploration results from vertex 42:", neighbor_ids)

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

