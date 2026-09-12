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

.. autoclass:: deglib.search.Searcher
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autofunction:: deglib.search.create_searcher

Filter
------

.. autoclass:: deglib.search.Filter
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Re-Ranking
----------

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
   graph = deglib.builder.build_from_data(data, metric=deglib.Metric.FP32_L2)

   # Quantize to INT8 for compact storage and ultra-fast exploration
   quantizer = deglib.optimization.ScalarQuantizerInt8()
   quantizer.fit(data)
   int8_data = quantizer.quantize(data)
   ro_graph = graph.to_readonly(deglib.FloatSpace.create(dim, deglib.Metric.Int8_L2), int8_data)

   # Deploy with Searcher (query is automatically quantized, top candidates reranked on FP32 data)
   searcher = deglib.create_searcher(
       graph=ro_graph,
       quantizer=quantizer,
       refine_space=deglib.FloatSpace.create(dim, deglib.Metric.FP32_L2),
       refine_data=data,
   )

   # Single query using relative margin (eps_or_ef < 1.0)
   indices, distances = searcher.search(query, k=10, eps_or_ef=0.1, rerank_factor=1.5, return_distances=True)

   # Single query using fixed pool exploration (eps_or_ef >= 1.0, HNSW-style)
   indices_ef, distances_ef = searcher.search(query, k=10, eps_or_ef=128, rerank_factor=1.5, return_distances=True)

   # Multithreaded batch query (supports eps_or_ef < 1.0 or >= 1.0)
   batch_queries = np.random.randn(100, dim).astype(np.float32)
   batch_indices, batch_distances = searcher.search(
       batch_queries, k=10, eps_or_ef=128, rerank_factor=1.5, threads=8, return_distances=True
   )

Filtered Search
^^^^^^^^^^^^^^^

.. code-block:: python

   import deglib
   import numpy as np

   # Create sample data and build a graph
   data = np.random.randn(1000, 64).astype(np.float32)
   graph = deglib.builder.build_from_data(data)

   # Restrict search results to a specific subset of labels
   valid_labels = np.array([0, 5, 12, 42, 99], dtype=np.int32)
   search_filter = deglib.search.Filter(valid_labels)

   # Perform nearest-neighbor query with filter applied
   query = np.random.randn(64).astype(np.float32)
   indices, distances = graph.search(query, k=5, eps=0.1, filter_labels=search_filter)
   print("Filtered nearest neighbor indices:", indices)
   print("Distances:", distances)

Two-Stage Re-Ranking
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import deglib
   import numpy as np

   dim = 128
   num_queries = 10
   num_candidates = 50

   space = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.FP32_L2)
   queries = np.random.randn(num_queries, dim).astype(np.float32)
   candidate_pool = np.random.randint(0, 10000, size=(num_queries, num_candidates), dtype=np.uint32)
   dataset = np.random.randn(10000, dim).astype(np.float32)

   # Exact re-ranking of top 10 best candidates per query
   top_indices, top_distances = deglib.search.rerank(
       space=space,
       queries=queries,
       candidate_indices=candidate_pool,
       base_vectors=dataset,
       k_top=10,
       return_distances=True,
   )

