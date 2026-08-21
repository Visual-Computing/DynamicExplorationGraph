Memory Management
=================

This page outlines important details regarding memory lifetime and zero-copy semantics in Python.

Lifetime of Feature Vectors
---------------------------

Fetching feature vectors from a graph via :meth:`~deglib.graph.DynamicExplorationGraph.get_feature_vector` returns a NumPy array that directly references the internal memory owned by the graph without copying:

.. code-block:: python

   import deglib

   # Zero-copy view into graph memory
   feature_vector = graph.get_feature_vector(42)

   # If the graph object is deleted or garbage collected, the view becomes invalid!
   del graph
   # print(feature_vector)  # Accessing this causes undefined behavior / crash

If feature vectors need to outlive the graph instance, pass ``copy=True``:

.. code-block:: python

   # Explicitly creates an independent copy
   feature_vector = graph.get_feature_vector(42, copy=True)
   del graph
   print(feature_vector)  # Safe

.. note::
   Copying vectors incurs memory allocation and copy overhead. Use the default ``copy=False`` in performance-critical loops where vectors are consumed immediately.

