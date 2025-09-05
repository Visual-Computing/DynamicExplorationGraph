Building Graphs
===============

This tutorial shows different methods of building graphs.

Using the shorthand
*******************

The easiest way to build a search graph is to use the function `deglib.builder.build_from_data()`.

.. code-block:: python

    import numpy as np
    import deglib

    data = np.random.random((10_000, 256)).astype(np.float32)
    graph = deglib.builder.build_from_data(data)

This will instantiate a `SizeBoundedGraph` containing 10_000 random feature vectors from `data`.

There are many different parameters to control the building process. See :ref:`build_from_data` for more information.

Metrics
*******

Deglib implements two different metrics:

- The `L2` distance or euclidean distance.
- The `InnerProduct` distance.

The L2 metric is also available for uint8 features as `L2_Uint8`.

Using builder object
********************

The above method is easy to use, but has drawbacks.
For example it does not allow to load the data step by step, which leads to more memory usage.

If we iterate over batches of data, we can add them to the graph like this:

.. code-block:: python

    graph = deglib.graph.SizeBoundedGraph.create_empty(N_SAMPLES, DIM, 16, deglib.Metric.L2)
    builder = deglib.builder.EvenRegularGraphBuilder(graph)

    counter = 0
    for batch in enumerate(batches):
        labels = np.arange(counter, batch.shape[0])
        builder.add_entry(labels, batch)
        counter += batch.shape[0]

    builder.build()

    graph.remove_non_mrng_edges()  # optional. This can help improve performance

    # now you can use the graph


Progress callback
*****************

If you want to get more feedback on the progress you can supply a callback function to `builder.build(callback)`.
The callback takes a `BuilderStatus` as only argument.

A `BuilderStatus` has the following attributes:

* step (int) - number of graph manipulation steps
* added (int) - number of added vertices
* deleted (int) - number of deleted vertices
* improved (int) - number of successful improvements
* tries (int) - number of improvement tries
