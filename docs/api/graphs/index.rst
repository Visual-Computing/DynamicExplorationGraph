Overview Graphs
===============

The following graph classes are defined:

* SearchGraph: An abstract interface for all graph classes.
* MutableGraph: An abstract base class, that can change by adding/removing vertices.
* SizeBoundedGraph: An implementation of the `MutableGraph`-class.
* ReadOnlyGraph: Once a `MutableGraph` is built, you can instantiate a `ReadOnlyGraph` from it.

.. toctree::
   :maxdepth: 2
   :caption: Graph classes:

   search_graph
   mutable_graph
   read_only_graph
   size_bounded_graph
