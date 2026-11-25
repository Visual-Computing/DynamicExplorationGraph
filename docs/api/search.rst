Search & Re-Ranking
=====================

Query search filtering, high-performance execution, and re-ranking routines.

Overview
--------

``deglib`` provides tools to constrain nearest-neighbor exploration (via :class:`~deglib.search.Filter`),
to execute exact re-ranking across candidate index pools (via :func:`~deglib.search.rerank`),
and to run high-throughput, end-to-end query pipelines (via :class:`~deglib.search.Searcher`).

Searcher
--------

.. autoclass:: deglib.search.SearchResults
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: deglib.search.Searcher
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Filter
------

.. autoclass:: deglib.search.Filter
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Re-Ranking
----------

.. autoclass:: deglib.search.Reranker
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autofunction:: deglib.search.rerank

Example Usage
-------------

High-Performance Searcher with Reranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import deglib
   import numpy as np

   dim = 128
   data = np.random.randn(10000, dim).astype(np.float32)
   query = np.random.randn(dim).astype(np.float32)

   # Build base graph
   graph = deglib.build(data, metric="l2")

   # Extract FP16 reranker from the FP32 graph and create compressed 8-bit Searcher
   reranker = deglib.reranker(graph, compress="16bit")
   searcher = deglib.searcher(
       graph,
       compress="8bit",
       reranker=reranker,
   )

   # Single query using relative margin (eps_or_ef < 1.0)
   res = searcher.search(query, k=10, eps_or_ef=0.1, rerank_factor=1.5, return_distances=True)
   indices, distances = res.indices, res.distances

   # Single query using fixed pool exploration (eps_or_ef >= 1.0, HNSW-style)
   res_ef = searcher.search(query, k=10, eps_or_ef=128, rerank_factor=1.5, return_distances=True)
   indices_ef, distances_ef = res_ef.indices, res_ef.distances

   # Multithreaded batch query (supports eps_or_ef < 1.0 or >= 1.0)
   batch_queries = np.random.randn(100, dim).astype(np.float32)
   batch_res = searcher.search(
       batch_queries, k=10, eps_or_ef=128, rerank_factor=1.5, threads=8, return_distances=True
   )
   batch_indices, batch_distances = batch_res.indices, batch_res.distances

Filtered Search
^^^^^^^^^^^^^^^

.. code-block:: python

   import deglib
   import numpy as np

   # Create sample data and build a graph
   data = np.random.randn(1000, 64).astype(np.float32)
   graph = deglib.build(data)

   # Restrict search results to a specific subset of labels
   valid_labels = np.array([0, 5, 12, 42, 99], dtype=np.int32)
   search_filter = deglib.search.Filter(valid_labels)

   # Perform nearest-neighbor query with filter applied
   query = np.random.randn(64).astype(np.float32)
   res = graph.search(query, k=5, eps_or_ef=0.1, filter_labels=search_filter)
   print("Filtered nearest neighbor indices:", res.indices)
   print("Distances:", res.distances)

Two-Stage Re-Ranking
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import deglib
   import numpy as np

   dim = 128
   num_queries = 10
   num_candidates = 50

   queries = np.random.randn(num_queries, dim).astype(np.float32)
   candidate_pool = np.random.randint(0, 10000, size=(num_queries, num_candidates), dtype=np.uint32)
   dataset = np.random.randn(10000, dim).astype(np.float32)

   # Exact re-ranking of top 10 best candidates per query
   top_indices, top_distances = deglib.search.rerank(
       metric=deglib.Metric.FP32_L2,
       queries=queries,
       candidate_indices=candidate_pool,
       base_vectors=dataset,
       k_top=10,
       return_distances=True,
   )

