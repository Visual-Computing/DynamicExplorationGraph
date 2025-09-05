Searching
=========

TODO

Multiple queries at once
************************

It is also possible to search for multiple queries at once:

.. code-block:: python

    N_QUERIES = 10
    query = np.random.random(N_QUERIES, DIMS).astype(np.float32)

    result_indices, dists = graph.search(query, eps=0.1, k=16)

In this case `result_indices` will have shape `(10, 16)` (16 indices for 10 queries).


Parameters
**********

TODO

Filter
******

TODO