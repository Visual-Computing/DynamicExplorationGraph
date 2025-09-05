Basic Usage
===========

This code generates a random dataset of 10,000 samples, each with 256 dimensions. 

A graph is then built to efficiently search for similar feature vectors within this dataset.

Using a random query, the code searches for the `k` most similar features, where `k` is set to 16.

The output is a NumPy array with shape  `(1, k)` containing `k` indices from the dataset and a corresponding array with the distance to the query for each result.

.. code-block:: python

    import numpy as np
    import deglib

    N_SAMPLES, DIMS = 10_000, 256

    # generate dataset
    data = np.random.random((N_SAMPLES, DIMS)).astype(np.float32)

    # build index
    graph = deglib.builder.build_from_data(data)

    # generate query
    query = np.random.random(DIMS).astype(np.float32)

    # search query
    result_indices, dists = graph.search(query, eps=0.1, k=16)

    print(result_indices)  # data[result_indices] will show the 16 closest datapoints to "query"
    print(dists)           # numpy array with 16 distances to the results

Multi Query
***********

It is also possible to search for multiple queries at once:

.. code-block:: python

    N_QUERIES = 10
    query = np.random.random(N_QUERIES, DIMS).astype(np.float32)

    result_indices, dists = graph.search(query, eps=0.1, k=16)

In this case `result_indices` will have shape `(10, 16)` (16 indices for 10 queries).

More Options
************
There are far more options to build a graph and to search for results.
Look at the documentation for building graphs and the search documentation.
