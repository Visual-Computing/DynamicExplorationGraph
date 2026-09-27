# 2D Graph Explorer

Builds a synthetic 2D point cloud, builds a Dynamic Exploration Graph over it and opens a window where the
drawing coordinates are the feature vectors, so every line on screen is a real edge the search can traverse.

## Start

```bash
cd examples/graph_2d
uv sync
uv run main.py
```

`uv sync` compiles the local `deglib` bindings from `../../python`, so a C++ compiler and CMake must be on your
`PATH`. After changing headers under `cpp/`, refresh with `uv sync --reinstall-package deglib`.

## The Search

The example runs its own ε-search in Python (`pysearch.py`), a faithful port of the library's
`searchImpl`: a min-heap pops the closest candidate, each unvisited neighbour is measured once, kept for
expansion inside the exploration radius and recorded as an answer when it improves on the worst kept
distance, the radius tightening by `1 + eps` (or `1 - eps` when it runs below zero, as an inner-product
search's does). It is held to the C++ search by a parity test — same answer set, same distances, on the
same entry — and the library's own search is kept only as that reference.

The port exists because the C++ search cannot do two things the viewer needs. It returns the top-`k`
answers but throws away the walk that reached them, so the port records a predecessor for every vertex it
queues and reconstructs, per answer, the chain of edges the search itself followed — the route the viewer
draws. And it only ever runs over a built DEG, whereas the reference graphs are read from an edge list the
library never sees; the port takes any adjacency list, so the search runs over the graph on screen whether
that is the DEG or one of the reference graphs. The panel's `route` line reports the run: hops to the top
answer, distances computed, vertices checked.

## Options

| Argument | Default | Description |
|---|---|---|
| `--preset` | `uniform` | Distribution (default `uniform`): `blobs`, `moons`, `circles`, `spiral`, `grid`, `uniform`. |
| `--vertices` | `200` | Number of points, 10 to 1 200. |
| `--k` | `4` | Edges per vertex; even and at least 4. |
| `--metric` | `L2` | Distance the graph is built on; `L2` or `IP`. |
| `--search-metric` | build metric | Distance the traversal ranks neighbours by; `L2` or `IP`. |
| `--mips` | `False` | Lift the data into the MIPS→L2 spherical space: a third dimension the graph and searches run on, while the plot stays 2D. |
| `--seed` | `7` | Seed for the point cloud and the graph build. |
| `--output` | `None` | Write the graph to this image file. |
| `--no-show` | `False` | Headless run: print statistics and write `graph_2d.png`. |

Bad input exits with code 2 and a one-line message.

```bash
uv run main.py --preset spiral --vertices 600   # another distribution and size
uv run main.py --metric IP                      # build in inner-product space
uv run main.py --search-metric IP                # walk an L2-built graph by inner product
uv run main.py --mips                            # build on the spherical MIPS→L2 lift, plot stays 2D
uv run main.py --no-show --output deg.png       # headless render, no GUI
```

## Controls
The top strip is a row of native Tk widgets, left to right: a `distribution` dropdown, a `vertices` spinbox
(10 to 1 200, step 100), a `graph` dropdown, a `k` spinbox (edges per vertex), a `mips transform` dropdown, a
`build metric` dropdown, a `search metric` dropdown, a `top k` spinbox (1 to 8, default 1), and an `eps`
spinbox (the exploration factor, 0.0 to 10.0, step 0.05). Every dropdown chooses on selection: opening it
and walking the entries with the arrow keys (or the mouse) applies the value the cursor stands on exactly
as a click would, and closing the popup keeps that choice. `build metric` decides which neighbours the graph
stores, `search metric` decides how the traversal ranks them. The `top k` field sets how many results the
search reports (the panel lists them one per line as `#1 …`, `#2 …`, with the `hit` marker on whichever line
names the best match, so it marks containment rather than rank one); it is also the size of the result heap
the search keeps, so it bounds the exploration radius — a wider `top_k` holds the radius loose for longer and
reaches further into the graph. The `k` field is enabled only while the `graph` selector shows `deg`,
`knng` or `nsw` and is greyed out otherwise; its bounds depend on the graph — under `knng` and `nsw` it
accepts any integer 1 to 16, under `deg` it is even-only 4 to 16, an odd value snapping up to the next even.
The field value is used for the knng and the NSW; the DEG always builds with the snapped even value. The
`graph` dropdown also offers `knng` (a directed k-nearest-neighbour graph over the current feature space:
each vertex links its `k` nearest and can only walk the neighbours it chose itself) and `nsw` (a navigable
small-world graph, built incrementally and undirected — each vertex links its `k` nearest among the vertices
already present, so a later vertex can raise an earlier one's degree past `k` and the maximum degree is
unbounded), both available under every metric, and `nsg` (the NSG's monotonic relative neighbourhood graph
over the current feature space: the directed greedy graph each vertex walks in increasing distance, keeping
the nearest free neighbour — built by the viewer like the knng and the NSW, so it is offered under every
metric, but unlike the reference graphs it is never scored against the DEG and reads no `k`), and `none`
(draws no edges at all).
The directed knng and NSG are drawn with arrowheads rather than plain strokes: every link ends in a head at the
vertex it points to, sized in screen pixels so it stays square and fixed through a zoom, which is what makes
the directed statistics below readable. A `save` button sits to the left of the `help` button, both packed at the right end
of the strip: `save` writes the graph — the plot alone, its colour key, text panel and controls all cropped
clear — to a PNG file through a save dialog at 150 dpi, and `help`
opens a small window of its own listing every binding — it stays out of the plot, since the drawing is the
reason the window is open.

The text panel names the graph actually on screen and states its shape: under the heading it lists the drawn
graph's `degree`, its `sources`/`sinks` and its two `reach` numbers — the fraction a breadth-first walk from
the start node reaches, and the mean fraction reachable from every vertex. These read off the graph the
search would walk and are phrased by its kind: a directed graph (the knng or the NSG) splits `degree` into its out- and
in-span and counts `sources` and `sinks` apart, while an undirected graph has one degree per vertex, so it
shows a single span and names the `isolated` vertices instead. The numbers are recomputed only when the graph
itself moves, never on a hover. The line under the heading counts the drawn graph's edges and reports how long
it took to construct — the reference graphs time their build in the comparison, while the `knng` and `nsw`
are timed as the viewer builds them on first selection, so picking a graph from the dropdown shows its
construction cost in milliseconds.



The colour key under the plot is itself a control: clicking an entry withdraws that reference graph's colour
and the edges it claimed fall through to the next graph holding them, which is how the nesting gets peeled
apart one layer at a time. `DEG only` names the remainder and is deliberately not a switch — the
`theory colours` checkbox owns that flag on its own.

| Action | Key |
|---|---|
| Select a vertex (left-click) / query the clicked empty space (left-click) / set the traversal start node (right-click) / run the query (double-click) | — |
| Switch one reference graph's colour on or off | click its key entry |
| Release the selection or the start node | `Esc` |
| Next graph | `v` |
| New sample from the same distribution | `r` |
| Overlays: coordinate cross, edges, vertex ids, theory colours, query markers | `g` / `e` / `i` / `c` / `q` |
| Zoom at the cursor / pan by dragging / restore the framing | scroll / drag / `0` |
| Control sheet | `h` or `?` |

A query takes the selected vertex as the search vector. The Python ε-search answers, scored against the
query's best match — the true nearest neighbour under the search metric, the vertex itself excluded — and
carries the route it took to its top answer in the same run. The star marks the best match, the X marks the
answer the search returned, and the path is the search's own recorded route from the start node to the X. A
green X on the star means the search recovered the true nearest neighbour; a red X away from the star means
it answered elsewhere, and a dashed line between them measures the miss. A search that returns nothing
leaves neither X nor path on screen. Because the search runs over whichever graph is on screen, the route is
drawn for a reference graph exactly as it is for the DEG.

Clicking empty space instead of a vertex makes the clicked coordinates themselves the query — a query that
is not a node, drawn as an orange cross. It is searched as itself: nothing is excluded from the ground truth,
because a free point is never a vertex to skip, so the answer is the best match over the whole cloud. `Esc`
releases the free query as it releases a vertex selection.

## Two Distances

Keeping the two metrics apart walks the stored edges by a comparator the build never optimised for. The
edges on screen do not move; only the route the descent takes and the answer it is scored against do, which
is what makes the difference between the two measurable instead of theoretical. An L2 graph walked by inner
product leaves its target far more often, and the panel reports the miss against the inner product's own
optimum, so it is the metrics disagreeing rather than the search failing.

Re-tagging a built graph is cheap here for a specific reason: a read-only graph stores no edge weights, it
recomputes every distance from its own feature space while the traversal runs. Swapping that space therefore
keeps the adjacency exactly as drawn and changes only which neighbour the greedy step picks next — about
0.1 ms at 1200 vertices, and no rebuild. Changing what the graph *stores* is the other dropdown's job and
does rebuild.

## MIPS Transform

The `mips transform` dropdown (and `--mips`) lifts every point onto a sphere by appending a third coordinate,
`x' = [x, √(M² − ‖x‖²)]` with `M` the largest norm in the cloud, so an inner-product search over the original
cloud becomes an L2 search over the lifted one. The graph build, the searches, the ground truth and the
reference graphs all run on those 3D features, while the plot keeps drawing the original 2D cloud — the
picture never changes shape, only the space the graph is navigated in. A query is answered from the
original vector zero-padded to `[x, y, 0]` (not the vertex's own lifted coordinate), so the search is a
true MIPS query: its nearest lifted neighbour is the original that maximises the inner product.

Delaunay drops out of the reference graphs under the transform. It triangulates the two drawing coordinates,
so on the lifted features it would silently report the Delaunay graph of the projection, and a genuine 3D
triangulation is degenerate because the transform places every vector at the same norm. The Gabriel, RNG,
MRNG and MST graphs are read from the distance matrix alone and stay valid in three dimensions, so they
remain.

## Tests

```bash
uv run pytest
```
