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
     - Immutable and stripped of all mutation structures. Provides minimal memory consumption and maximum query throughput. Can be saved to disk via :meth:`~deglib.graph.DynamicExplorationGraph.save_graph`.
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
   * - ``Metric.FP16_L2``
     - ``np.uint16``
     - Euclidean (L2) distance for 16-bit half-precision floating point vectors.
   * - ``Metric.FP16_InnerProduct``
     - ``np.uint16``
     - Inner product distance for 16-bit half-precision floating point vectors.
   * - ``Metric.Int8_L2``
     - ``np.int8``
     - Euclidean (L2) distance for 8-bit signed integer vectors.
   * - ``Metric.Int8_InnerProduct``
     - ``np.int8``
     - Inner product distance for 8-bit signed integer vectors.
   * - ``Metric.Uint8_L2``
     - ``np.uint8``
     - Euclidean (L2) distance for 8-bit unsigned integer vectors.
   * - ``Metric.Uint8_InnerProduct``
     - ``np.uint8``
     - Inner product distance for 8-bit unsigned integer vectors.
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
     - **Recommended for almost all standard datasets** (LID :math:`\le` 15). 
   * - ``OptimizationTarget.HighLID``
     - Yes (Multi-Threaded)
     - Datasets with high Local Intrinsic Dimensionality (LID > 15).
   * - ``OptimizationTarget.StreamingData``
     - No (Single-Threaded)
     - LID-agnostic. Works reliably across all datasets (whether low or high LID) and shifting distributions, but indexes slower.

.. tip::

   **Rule of thumb:**
   - Use **``LowLID``** (or **``HighLID``** for datasets with LID > 15) to get full multi-threaded indexing speed.
   - Use **``StreamingData``** if you are unsure about the data characteristics or want a distribution-agnostic fallback. All three targets support dynamic additions and deletions.

Builder Parameters
------------------

When creating a :class:`~deglib.builder.GraphBuilder` or calling :func:`~deglib.builder.build_from_data`, construction behavior is configured via **Extension Parameters** (finding entry neighbors for newly inserted vertices) and **Improvement Parameters** (2-hop edge swaps to repair distortion or graph regularity):

Extension Parameters (Graph Extension)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by all optimization targets (``LowLID``, ``HighLID``, and ``StreamingData``) during vertex insertion.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``extend_k``
     - ``edges_per_vertex * 2``
     - Number of candidate neighbors evaluated during graph extension search. If set to 0 or :math:`\le` ``edges_per_vertex``, automatically set to ``edges_per_vertex * 2``.
   * - ``extend_eps``
     - ``0.1``
     - Exploration distance margin used during neighbor search when inserting new vertices. Higher values evaluate more candidate paths, improving initial edge quality at the cost of insertion speed.


Improvement Parameters (Edge Swapping & Neighborhood Optimization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls 2-hop edge swapping attempts (executed during vertex deletion repair in ``reduceGraph()`` and optionally during build steps when ``improve_tries > 0``):

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``improve_k``
     - ``edges_per_vertex``
     - Search candidate pool size evaluated during 2-hop edge swaps. Defaults to ``edges_per_vertex`` when set to ``0``.
   * - ``improve_eps``
     - ``0.001``
     - Exploration distance margin during improvement searches.
   * - ``max_path_length``
     - ``5``
     - Maximum number of consecutive edge swaps evaluated in a single improvement attempt.
   * - ``improve_tries``
     - ``0``
     - Number of random 2-hop improvement attempts executed per build step. Defaults to ``0`` (disabled during batch insertion to maximize build throughput). Kantenreparatur beim Löschen (``reduceGraph``) nutzt ``improve_k`` automatisch.

.. note::

   - **LowLID vs. HighLID**: Both run multi-threaded in parallel batches. ``LowLID`` computes distortion :math:`(d_1 + d_2) - d_{\text{removed}}` to pick the edge whose replacement minimizes metric stretch, while ``HighLID`` picks the neighbor's worst edge.
   - **StreamingData**: Processes additions sequentially one by one, avoiding thread synchronizations when additions and deletions are interleaved.
   - **Improvement during Batch Build**: For initial batch construction, keeping ``improve_tries = 0`` is recommended as ``extendGraphKnownLID`` already places edges near-optimally. In dynamic streaming scenarios, setting ``improve_tries > 0`` allows background self-healing of the graph topology.



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
       edges_per_vertex=30,      # Number of edges per vertex
       extend_eps=0.1,           # Exploration margin during build
       thread_count=0,           # 0 = use all available CPU cores
       show_progress=True,       # Print progress bar to stdout
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
   print("Added vertex IDs in this step:", status.step_added_ids)
   print("Deleted vertex IDs in this step:", status.step_deleted_ids)


---

Saving & Loading
----------------

Any graph (whether Read-Only, Fixed-Capacity Mutable, or Dynamic Mutable) can be saved to disk with :meth:`~deglib.graph.DynamicExplorationGraph.save_graph`.


.. code-block:: python

   # Save the graph to a file
   graph.save_graph("my_graph.deg")

When loading a saved graph file, you can choose any of the three graph modes depending on how you plan to use it:

.. code-block:: python

   # 1. Load as Read-Only (recommended for query serving)
   readonly_graph = deglib.graph.load_readonly_graph("my_graph.deg")

   # 2. Load as Dynamic Mutable (to continue adding/removing vectors dynamically)
   dynamic_graph = deglib.graph.load_dynamic_graph("my_graph.deg")

   # 3. Load as Fixed-Capacity Mutable (to continue modifying with a fixed capacity)
   mutable_graph = deglib.graph.load_mutable_graph("my_graph.deg", capacity=20_000)





