Quickstart
==========

How to use deglib
*****************

Using deglib follows the following procedure:

* Load your feature vector database
    * The following example code generates a random dataset of 10,000 samples, each with 256 dimensions.
* Building the graph
* A query is submitted. Feature vectors similar to this query need to be found.
* Using the graph to find similar feature vectors in the database

The following code shows an example of how this can be implemented:

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
    indices, distances = graph.search(query, eps=0.1, k=16)

    print(indices)    # data[result_indices] will show the 16 closest datapoints to "query"
    print(distances)  # numpy array with 16 distances to the results

The output is a NumPy array with shape  `(1, k)` containing `k` indices from the dataset and a corresponding array with the distance to the query for each result.

More Options
************
There are far more options to build a graph and to search for results.
Look at the documentation for building graphs and the search documentation.
