Building Graphs
===============

This guide explains how to construct, optimize, save, and load Exploration Graphs in Python.

Graph Types
-----------

All search graphs in `deglib` are instances of :class:`~deglib.graph.DynamicExplorationGraph`. The underlying representation determines whether the graph can be modified, how memory is allocated, and how efficiently queries are executed.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Mode
     - Creation / Loading
     - Purpose & Behavior
   * - **Read-Only**
     - :func:`~deglib.graph.load_readonly_graph`, :meth:`~deglib.graph.DynamicExplorationGraph.to_readonly`
     - Immutable and stripped of all mutation structures. Provides minimal memory consumption and maximum query throughput. Cannot be saved again.
   * - **Fixed-Capacity Mutable**
     - :func:`~deglib.graph.create_empty`, :func:`~deglib.graph.load_mutable_graph`
     - Mutable graph with a pre-allocated capacity. Ideal when the total number of vectors is known beforehand (e.g. batch indexing fixed datasets).
   * - **Dynamic Mutable**
     - :func:`~deglib.graph.create_dynamic_empty`, :func:`~deglib.graph.load_dynamic_graph`
     - Mutable graph that dynamically grows memory in chunks (e.g. 1024 vectors per chunk) as new data arrives. Ideal for open-ended streaming data.

---

Supported Data Types & Metrics
------------------------------

Before building a graph, choose a distance metric compatible with your feature vector type:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Metric Enum
     - NumPy Dtype
     - Description
   * - ``Metric.FP32_L2``
     - ``np.float32``
     - Euclidean (L2) distance for 32-bit floating point vectors.
   * - ``Metric.FP32_InnerProduct``
     - ``np.float32``
     - Inner product (dot product) distance for 32-bit float vectors.
   * - ``Metric.Uint8_L2``
     - ``np.uint8``
     - Euclidean (L2) distance for 8-bit unsigned integer vectors.
   * - ``Metric.FP16_InnerProduct``
     - ``np.uint16``
     - Inner product distance for 16-bit half-precision floating point vectors.
   * - ``Metric.EVP_InnerProduct``
     - ``np.uint8``
     - Inner product distance for quantized byte-packed (EVP) vectors.


---

Optimization Target
-------------------

The :class:`~deglib.builder.OptimizationTarget` defines how new vertices are connected and whether parallel CPU worker threads can be utilized.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Target
     - Multi-Threading
     - Best Used For
   * - ``OptimizationTarget.LowLID`` (Default)
     - Yes (Multi-Threaded)
     - **Recommended for almost all standard datasets**. Enables parallel multi-core graph construction with high insertion throughput.
   * - ``OptimizationTarget.HighLID``
     - Yes (Multi-Threaded)
     - Datasets with high Local Intrinsic Dimensionality (LID > 15), applying specialized exploration paths during multi-threaded insertion.
   * - ``OptimizationTarget.StreamingData``
     - No (Single-Threaded)
     - Streaming scenarios with continuous deletions or sliding-window buffers.

---

Builder Parameters
------------------

When creating a :class:`~deglib.builder.GraphBuilder` or calling :func:`~deglib.builder.build_from_data`, parameters are divided into **Graph Extension** (adding vertices) and **Graph Improvement / Deletion** (edge swaps and neighborhood repair):

Extension Parameters (Used by LowLID & HighLID)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``extend_k``
     - ``edges_per_vertex * 2``
     - Candidate search pool size when finding entry connections for new vertices. **Must be larger than ``edges_per_vertex``**.
   * - ``extend_eps``
     - ``0.1``
     - Exploration margin during vertex insertion. Higher values improve graph quality at the cost of build time.


Improvement & Deletion Parameters (Used after Deletions or during Continuous Refinement)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``improve_k``
     - ``edges_per_vertex``
     - Neighborhood candidate pool size evaluated during 2-hop edge swaps.
   * - ``improve_eps``
     - ``0.001``
     - Exploration parameter used during improvement attempts.
   * - ``max_path_length``
     - ``5``
     - Maximum number of consecutive edge swaps in a single improvement sequence.
   * - ``swap_tries``
     - ``0``
     - Number of improvement attempts executed per build step.
   * - ``additional_swap_tries``
     - ``0``
     - Extra improvement attempts executed immediately after a successful edge swap.

.. note::

   - **LowLID / HighLID**: Primary construction uses the multi-threaded extension parameters (``extend_k``, ``extend_eps``). Improvement parameters are used if vertex deletions occur or explicit continuous edge swaps are desired.
   - **StreamingData**: Uses both extension and continuous improvement/swap parameters simultaneously to repair graph regularity after continuous insertions and deletions.



---

Quick Build
-----------

To build an index from a 2D NumPy array in a single call:

.. code-block:: python

   import numpy as np
   import deglib

   data = np.random.randn(10_000, 128).astype(np.float32)

   # Build graph using all available CPU threads (LowLID default)
   graph = deglib.builder.build_from_data(
       data,
       k=30,                     # Number of edges per vertex
       eps=0.1,                  # Exploration margin during build
       threads=0,                # 0 = use all available CPU cores
       callback="progress",      # Print progress bar to stdout
   )

---

Dynamic Modification
--------------------

For streaming or incremental updates where vertices are added or removed over time, use :class:`~deglib.builder.GraphBuilder`. Calling :meth:`~deglib.builder.GraphBuilder.build` processes queued entries and returns a status object:


.. code-block:: python

   import numpy as np
   import deglib
   from deglib.builder import GraphBuilder, OptimizationTarget

   dim = 128
   space = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.FP32_L2)
   graph = deglib.graph.create_dynamic_empty(space, edges_per_vertex=30)

   # LowLID enables multi-threaded graph extension
   builder = GraphBuilder(
       graph,
       optimization_target=OptimizationTarget.LowLID,
       seed=42,
   )

   # Queue entries to add
   labels = np.array([101, 102, 103], dtype=np.uint32)
   features = np.random.randn(3, dim).astype(np.float32)
   builder.add_entry(labels, features)

   # Queue an entry to remove
   builder.remove_entry(50)

   # Execute build step and inspect results
   status = builder.build()

   print("Build step:", status.step)
   print("Vertices added in this step:", status.added)
   print("Vertices deleted in this step:", status.deleted)
   print("Added vertex IDs:", status.get_added_external_labels())
   print("Deleted vertex IDs:", status.get_deleted_external_labels())

---

Saving & Loading
----------------

Any mutable graph (Fixed-Capacity or Dynamic) can be saved to disk with :meth:`~deglib.graph.DynamicExplorationGraph.save_graph`.


.. code-block:: python

   # Save the mutable graph to a file
   graph.save_graph("my_graph.deg")

When loading a saved graph file, you can choose any of the three graph modes depending on how you plan to use it:

.. code-block:: python

   # 1. Load as Read-Only (recommended for query serving)
   readonly_graph = deglib.graph.load_readonly_graph("my_graph.deg")

   # 2. Load as Dynamic Mutable (to continue adding/removing vectors dynamically)
   dynamic_graph = deglib.graph.load_dynamic_graph("my_graph.deg")

   # 3. Load as Fixed-Capacity Mutable (to continue modifying with a fixed capacity)
   mutable_graph = deglib.graph.load_mutable_graph("my_graph.deg", capacity=20_000)





