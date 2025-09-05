Searching
=========

The main functionality of deglib is to search for similar feature vectors.

.. code-block:: python

    import numpy as np
    import deglib

    # build the graph ...

    # generate query
    query = np.random.random(DIMS).astype(np.float32)

    # search query
    indices, distances = graph.search(query, eps=0.1, k=16)

    print(indices)    # data[result_indices] will show the 16 closest datapoints to "query"
    print(distances)  # numpy array with 16 distances to the results

This will give the 16 indices of the datapoints closest to the query as well as their distances to the query.

Multiple queries at once
************************

It is also possible to search for multiple queries at once:

.. code-block:: python

    N_QUERIES = 10
    query = np.random.random(N_QUERIES, DIMS).astype(np.float32)

    indices, distances = graph.search(query, eps=0.1, k=16)

In this case `indices` and `distances` will have shape `(10, 16)` (16 indices for 10 queries).

Parameters
**********

There are a lot of parameters to control the search speed and recall. See :ref:`search_graph` for more details.

Filters
*******

Sometimes you want to exclude some samples of your dataset from search. This can be done using filters:

.. code-block:: python

    valid_labels = np.array([0, 3, 34, 52, 108])
    query = np.random.random(dims).astype(np.float32)

    results, distances = graph.search(query, filter_labels=valid_labels, eps=0.01, k=4)

In this case results will only include values in `valid_labels`.
