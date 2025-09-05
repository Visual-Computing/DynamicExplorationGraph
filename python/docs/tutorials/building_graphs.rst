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

LID.Unknown vs LID.High or LID.Low
**********************************
The crEG paper introduces an additional parameter, *LIDType*, to determine whether a dataset exhibits high complexity and Local Intrinsic Dimensionality (LID) or if it is relatively low. In contrast, the DEG paper presents a new algorithm that does not rely on this information. Consequently, DEG defaults to *LID.Unknown*. However, if the LID is known, utilizing it can be beneficial, as multi-threaded graph construction is only possible with these parameters.

Metrics
*******

Deglib implements two different metrics:

- The `L2` distance or euclidean distance.
- The `InnerProduct` distance.

The L2 metric is also available for uint8 features as `L2_Uint8`.

Internal Index vs External Label
********************************

There are two kinds of indices used in a graph: `internal_index` and `external_label`. Both are integers and specify
a vertex in a graph.

Internal Indices are dense, which means that every `internal_index < len(graph)` can be used.
For example: If you add 100 vertices and remove the vertex with internal_index 42, the last vertex in the graph will
be moved to index 42.

In contrast, external label is a user defined identifier for each added vertex
(see `builder.add_entry(external_label, feature_vector)`). Adding or Removing vertices to the graph will keep the
connection between external labels and associated feature vector.

When you create the external labels by starting with `0` and increasing it for each entry by `1` and don't remove
elements from the graph, external labels and internal indices are equal.

.. code-block:: python

    # as long as no elements are removed
    # external labels and internal indices are equal
    for i, vec in enumerate(data):
        builder.add_entry(i, vec)

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
