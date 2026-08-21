Search & Re-Ranking
=====================

Query search filtering and re-ranking routines.

Overview
--------

``deglib`` provides tools to constrain nearest-neighbor exploration (via :class:`~deglib.search.Filter`)
and to execute exact re-ranking across candidate index pools (via :func:`~deglib.search.rerank`).

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

