Basic Usage
===========

.. code-block:: python

    import numpy as np
    import deglib

    samples, dims = 10_000, 256

    # build index
    data = np.random.random((samples, dims)).astype(np.float32)
    index = deglib.builder.build_from_data(data)

    # search query
    query = np.random.random(dims).astype(np.float32)
    result_indices, dists = index.search(query, eps=0.1, k=16)

    print(result_indices)  # data[result_indices] will show the 16 closest datapoints to "query"
    print(dists)           # numpy array with 16 distances to the results
