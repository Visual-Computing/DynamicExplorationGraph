FAQ & Caveats
=============

What problem does deglib solve?
*******************************

The Dynamic Exploration Graph library (deglib) solves the approximate nearest neighbor (ANN) search problem.
This involves finding data points that are most similar or "closest" to a given query point in a high-dimensional space, without having to check every single data point.

Memory Management
*****************

A graph object owns the feature vectors. If the graph is freed, feature vectors are not longer valid.

.. code-block:: python

    feature_vector = graph.get_feature_vector(42)
    del graph
    print(feature_vector)

This will crash as `feature_vector` is holding a reference to memory that is owned by `graph`. This can lead to undefined behaviour (most likely segmentation fault).
Be careful to keep objects in memory that are referenced. If you need it use the `copy=True` option:

.. code-block:: python

    feature_vector = graph.get_feature_vector(10, copy=True)
    del graph
    print(feature_vector)  # no problem

Copying feature vectors will be slower.

Limitations
***********

- The python wrapper at the moment only supports `float32` and `uint8` feature vectors. For `uint8` vectors only `L2` Distance is supported.
- Threaded building is not supported for `lid=LID.Unknown`. Use `LID.High` or `LID.Low` instead.
