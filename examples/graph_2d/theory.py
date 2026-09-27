"""
Theoretical proximity graphs over the same point cloud, compared edge by edge with the built DEG.

The five references are the classical graphs the DEG is usually judged against:

* **Delaunay (DG)** — every edge of the Delaunay triangulation; the superset of all the others.
* **Gabriel (GG)** — `(i, j)` survives when the circle having it as diameter is empty.
* **RNG** — relative neighbourhood graph: `(i, j)` survives when its lune is empty.
* **MRNG** — the minimal monotonic relative neighbourhood graph: the directed NSG graph (Definition 5)
  symmetrised and reduced by dropping every edge whose endpoints are already joined by a chain of strictly
  shorter edges; undirected, and a subgraph of the symmetrised NSG graph it is built from.
* **MST** — the minimum spanning tree of the dissimilarity.

Under Euclidean distance they nest as `MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay`: the diametral disc lies inside
the lune, so an empty lune forces an empty disc, and every step down the chain demands a larger empty
region around the edge. The MRNG does not sit inside that chain, since a candidate blocked by an earlier
edge may still lie in the lune of the one kept.

`compare` takes the metric the DEG was built with and reads the references from the same dissimilarity,
so that a share of 70 % means the same thing whichever metric produced the graph. deglib scores inner
product as the distance `1 - ⟨x, y⟩` (`distance/fp32_ip.h`) and normalises nothing, so that is the matrix
an inner-product graph is compared against. Only graphs decided by *comparisons* survive that change of
matrix, and those are the ones reported: RNG, MRNG and MST. Gabriel squares the distances and Delaunay
needs a triangulation of the plane, and neither exists for a dissimilarity that is negative for half the
pairs — on the example's clouds `1 - ⟨x, y⟩` spans roughly −24 … +24. An inner-product report therefore
carries three references instead of five.

The definitions are the classical ones, and each graph is read from its own definition rather than derived
from the triangulation: under concyclic degeneracies a particular triangulation can omit an edge the
definition demands, which is exactly how a triangulation-built RNG loses one on the spiral cloud. The
loops are shaped the way they are because the naive triple loop over all triples of points does not
survive a thousand vertices:

* The lune condition is a `min` over the third point, so one row-wide `maximum` decides every
 candidate of a node at once; the relation is symmetric, so only `j > i` is ever looked at.
* The Gabriel condition is the same sweep with a sum in place of the `maximum`.
* An accepted MRNG edge blocks exactly the candidates farther away than it, so the walk can skip to
 the next free candidate instead of scanning every one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from deglib.distances import Metric
from matplotlib.tri import Triangulation

__all__ = [
    "GraphOverlap",
    "TheoryReport",
    "compare",
    "delaunay_edges",
    "dissimilarities",
    "gabriel_edges",
    "graph_stats",
    "mrng_edges",
    "mst_edges",
    "nsg_edges",
    "rng_edges",
]


def _keys(pairs: np.ndarray, num_points: int) -> np.ndarray:
    """Unique undirected edge keys `i * n + j` with `i <= j`, sorted."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(pairs.min(axis=1) * num_points + pairs.max(axis=1))


def dissimilarities(points: np.ndarray, metric: Metric = Metric.FP32_L2) -> np.ndarray:
    """
    The matrix the reference graphs are read from, matching what the DEG's own search minimises.

    Plain Euclidean distance under L2; under inner product `1 - ⟨x, y⟩`, which is exactly deglib's score
    (`distance/fp32_ip.h`) — with no normalisation, because the library applies none either.

    The diagonal is set to infinity rather than left at whatever the metric produces for it. Inner product
    scores a point against itself as `1 - ‖x‖²`, which is not the smallest entry of its row, while the
    sweeps quantify over a *third* point: an endpoint must never block its own edge. Infinity makes those
    two terms vacuous under either metric and changes nothing under L2, where the diagonal is already zero.
    """
    if metric == Metric.FP32_InnerProduct:
        scores = 1.0 - points @ points.T
    else:
        scores = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    out = np.ascontiguousarray(scores, dtype=np.float32)
    np.fill_diagonal(out, np.inf)
    return out


def delaunay_edges(points: np.ndarray) -> np.ndarray:
    """The Delaunay graph: every edge of the Delaunay triangulation, as `[e, 2]` index pairs."""
    triangles = Triangulation(points[:, 0], points[:, 1]).triangles
    if triangles.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    sides = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return sides.astype(np.int32)


def rng_edges(distances: np.ndarray) -> np.ndarray:
    """
    Relative neighbourhood graph: `(i, j)` is kept when no point `k` is strictly closer to both
    endpoints than they are to each other.
    """
    num_points = distances.shape[0]
    kept: list[tuple[int, int]] = []
    for i in range(num_points - 1):
        row = distances[i]
        lune = np.maximum(row, distances[i + 1 :])
        survivors = np.flatnonzero(lune.min(axis=1) >= row[i + 1 :])
        kept.extend((i, i + 1 + int(j)) for j in survivors)
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


def gabriel_edges(distances: np.ndarray) -> np.ndarray:
    """
    Gabriel graph: `(i, j)` is kept when the circle having `(i, j)` as its diameter is empty.

    A point `k` lies inside that circle exactly when the angle at `k` is obtuse, and by the law of
    cosines that means `d(i,k)² + d(j,k)² < d(i,j)²`. So the sweep that decides the RNG decides this
    one too, with a sum where the RNG takes a `maximum`. The endpoints contribute `d(i,j)²`
    themselves, which is not below the threshold, so an edge never blocks itself.
    """
    num_points = distances.shape[0]
    squared = distances**2
    kept: list[tuple[int, int]] = []
    for i in range(num_points - 1):
        row = squared[i]
        disc = row + squared[i + 1 :]
        survivors = np.flatnonzero(disc.min(axis=1) >= row[i + 1 :])
        kept.extend((i, i + 1 + int(j)) for j in survivors)
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


def nsg_edges(distances: np.ndarray) -> np.ndarray:
    """
    The NSG's monotonic relative neighbourhood graph, directed. A node walks its candidates in increasing
    distance and keeps the nearest free one; that edge then blocks every candidate farther away than
    it, which is the lune condition of NSG Definition 5 read from the accepted edge instead of from
    the candidate.
    """
    num_points = distances.shape[0]
    order = np.argsort(distances, axis=1)
    kept: list[tuple[int, int]] = []
    for i in range(num_points):
        row = distances[i]
        blocked = np.zeros(num_points, dtype=bool)
        blocked[i] = True
        while True:
            free = order[i][~blocked[order[i]]]
            if free.size == 0:
                break
            j = int(free[0])
            kept.append((i, j))
            blocked[j] = True
            blocked |= (row[j] < row) & (distances[j] < row)
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


def _monotone_reachable(adjacency: list[set[int]], start: int, target: int, to_target: list[float]) -> bool:
    """
    Whether `target` is reachable from `start` along a path that strictly decreases the distance to it,
    never using the direct edge between the two.

    Each hop moves to a neighbour strictly nearer the target, so the walk follows the DAG of decreasing
    distance-to-target rather than any path. The edge under test is skipped, so the answer is whether some
    *other* monotone chain already connects the endpoints — the test that makes that edge redundant.
    """
    seen = bytearray(len(adjacency))
    seen[start] = 1
    stack = [start]
    while stack:
        node = stack.pop()
        gap = to_target[node]
        for neighbour in adjacency[node]:
            if (node == start and neighbour == target) or (node == target and neighbour == start):
                continue
            if not seen[neighbour] and to_target[neighbour] < gap:
                if neighbour == target:
                    return True
                seen[neighbour] = 1
                stack.append(neighbour)
    return False


def _undirected_mrng(directed: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    The minimal undirected MRNG, by top-down transitive reduction of a symmetrised directed MRNG.

    The directed MRNG is already a monotone search network: it carries a monotone path between every pair.
    Symmetrising it, `E_0 = E_dir ∪ E_dirᵀ`, keeps those paths and bounds the minimal graph from above, but
    it over-adds — an edge `u -> v` kept for `u`'s routing can be redundant from `v`'s side, inflating
    `v`'s degree. So prune top-down: walk the edges from longest to shortest, and drop one only when, with
    it removed, `u` still has a strictly-decreasing-distance chain to `v` *and* `v` has one to `u`; if
    either direction has no such chain the edge is indispensable and stays. Longest-first makes the result
    minimal — a kept edge lacked an alternative chain in at least one direction when it was examined, and
    later steps only remove edges, never add one, so that chain never appears afterwards. The output is
    undirected pairs, each stored with the smaller index first.
    """
    num_points = distances.shape[0]
    if directed.shape[0] == 0 or num_points < 2:
        return np.zeros((0, 2), dtype=np.int32)
    edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in directed}
    adjacency: list[set[int]] = [set() for _ in range(num_points)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    columns = distances.tolist()
    for u, v in sorted(edges, key=lambda edge: distances[edge[0], edge[1]], reverse=True):
        if _monotone_reachable(adjacency, u, v, columns[v]) and _monotone_reachable(adjacency, v, u, columns[u]):
            adjacency[u].discard(v)
            adjacency[v].discard(u)
    kept = [(u, v) for u, v in edges if v in adjacency[u]]
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.int32)


def mrng_edges(distances: np.ndarray) -> np.ndarray:
    """
    The minimal undirected MRNG: the directed NSG graph reduced by `_undirected_mrng`.
    """
    return _undirected_mrng(nsg_edges(distances), distances)


def mst_edges(distances: np.ndarray) -> np.ndarray:
    """
    Minimum spanning tree of the dissimilarity, by Prim's algorithm over the matrix.

    Prim carries only the cheapest known link into the growing tree, so one pass per vertex builds the
    whole tree. That reads the tree from its own definition instead of from a triangulation that, under
    degeneracy, might not contain the edge the definition picks.
    """
    num_points = distances.shape[0]
    if num_points < 2:
        return np.zeros((0, 2), dtype=np.int32)

    best = np.ascontiguousarray(distances[0], dtype=np.float64)
    origin = np.zeros(num_points, dtype=np.int32)
    inside = np.zeros(num_points, dtype=bool)
    inside[0] = True
    best[0] = np.inf
    kept: list[tuple[int, int]] = []
    for _ in range(num_points - 1):
        v = int(np.argmin(best))
        kept.append((int(origin[v]), v))
        inside[v] = True
        best[v] = np.inf
        link = distances[v]
        cheaper = (link < best) & ~inside
        best[cheaper] = link[cheaper]
        origin[cheaper] = v
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


@dataclass(frozen=True)
class GraphOverlap:
    """One theoretical graph and the part of the DEG it shares."""

    name: str
    edges: int
    shared: int
    keys: np.ndarray = field(compare=False, repr=False)
    #: Wall-clock seconds the graph took to build over the dissimilarity, so a viewer can report the
    #: construction cost of the graph it draws.
    seconds: float = field(default=0.0, compare=False, repr=False)

    def share_of(self, deg_edges: int) -> float:
        """Fraction of the DEG's edges this graph also contains."""
        return self.shared / deg_edges if deg_edges else 0.0


@dataclass(frozen=True)
class TheoryReport:
    """DEG versus the reference graphs built over the same dissimilarity it was searched with."""

    deg_edges: int
    seconds: float = field(compare=False)
    graphs: tuple[GraphOverlap, ...]
    deg_keys: np.ndarray = field(compare=False, repr=False)

    @property
    def names(self) -> tuple[str, ...]:
        """The reference graphs held in this report, in reporting order."""
        return tuple(item.name for item in self.graphs)

    def edges_of(self, name: str, num_points: int) -> np.ndarray:
        """
        The reference graph `name` as `[e, 2]` int32 index pairs, in the shape the DEG uses.

        The graph is rebuilt from its deduplicated `i < j` keys. The report keeps edges as `i * n + j`
        keys because that is what the set operations compare, but a caller that wants to draw one of the
        graphs needs pairs.
        """
        item = self.graph(name)
        row, column = np.divmod(item.keys, num_points)
        return np.column_stack((row, column)).astype(np.int32).reshape(-1, 2)

    def graph(self, name: str) -> GraphOverlap:
        """The overlap entry of one reference graph, by name."""
        if (item := self._lookup(name)) is not None:
            return item
        raise KeyError(f"{name} is not part of this report, which holds {list(self.names)}")

    def _lookup(self, name: str) -> GraphOverlap | None:
        return next((item for item in self.graphs if item.name == name), None)

    def membership(self, last_wins: tuple[str, ...]) -> np.ndarray:
        """
        One code per DEG edge telling which reference graph contains it: its position in `graphs`, or
        -1 when none does. `last_wins` names the graphs in increasing precedence, so that one decides
        an edge belonging to several of them — the override order the reference project uses when it
        paints the overlaps, where the last colour written is the one that shows. A name the metric
        does not admit is skipped, since the report simply does not carry it.
        """
        codes = np.full(self.deg_keys.shape[0], -1, dtype=np.int8)
        for name in last_wins:
            if (item := self._lookup(name)) is None:
                continue  # the metric does not admit this graph; see the module docstring
            codes[np.isin(self.deg_keys, item.keys)] = self.graphs.index(item)
        return codes


def compare(points: np.ndarray, deg_edges: np.ndarray, metric: Metric = Metric.FP32_L2) -> TheoryReport:
    """
    Builds the reference graphs under `metric` and counts, for each, how many of the DEG's edges it holds.

    The MRNG is built from the NSG graph: the directed greedy graph is symmetrised and transitively
    reduced, and only the resulting undirected graph is compared against the DEG. This is the
    slow part of the example — the distance matrix is quadratic in memory and the lune and disc sweeps
    are quadratic on top of it — which is why the vertex count is capped.

    Delaunay and Gabriel are statements about the plane — a triangulation, a diametral disc — so they are
    only built over a Euclidean matrix; an inner-product report carries the graphs its dissimilarity
    can decide. Delaunay is gated on the plane twice over: it triangulates the two drawing coordinates, so
    on a lifted feature space it would silently report the Delaunay graph of the projection, and a genuine
    3D triangulation is degenerate because the MIPS transform places every vector at the same norm. The
    Gabriel, RNG, MRNG and MST graphs are read from the distance matrix alone and stay valid in any
    dimension, so they survive the lift.
    """
    num_points = int(points.shape[0])
    deg = _keys(deg_edges, num_points)

    started = time.perf_counter()
    distances = dissimilarities(points, metric)
    # The directed NSG graph is built once and reused: the undirected MRNG is its transitive reduction, so
    # recomputing it there would double the cost. Its build time is charged to the MRNG.
    nsg_began = time.perf_counter()
    nsg = nsg_edges(distances)
    nsg_seconds = time.perf_counter() - nsg_began
    builders: list[tuple[str, Callable[[], np.ndarray], float]] = [
        ("rng", lambda: rng_edges(distances), 0.0),
        ("mst", lambda: mst_edges(distances), 0.0),
        ("mrng", lambda: _undirected_mrng(nsg, distances), nsg_seconds),
    ]
    if metric == Metric.FP32_L2:
        builders = [("gabriel", lambda: gabriel_edges(distances), 0.0), *builders]
        if points.shape[1] == 2:
            builders = [("delaunay", lambda: delaunay_edges(points), 0.0), *builders]
    graphs: list[GraphOverlap] = []
    for name, build, preset_seconds in builders:
        began = time.perf_counter()
        raw = build()
        seconds = preset_seconds + (time.perf_counter() - began)
        keys = _keys(raw, num_points)
        graphs.append(
            GraphOverlap(
                name,
                edges=int(keys.size),
                shared=int(np.intersect1d(deg, keys).size),
                keys=keys,
                seconds=seconds,
            )
        )
    return TheoryReport(int(deg.size), time.perf_counter() - started, tuple(graphs), deg)


def graph_stats(edges: np.ndarray, num_points: int, directed: bool, entry: int | None) -> dict:
    """
    Degree and reachability statistics of a graph, mirroring the semantics of `deglib/analysis.h`.

    Degrees are taken over *every* vertex — isolated ones included — so an empty graph reports zero
    rather than crashing on an empty `min`. An undirected edge counts in both directions, so its in- and
    out-degree are the same number; a directed edge `(a, b)` adds one to `a`'s out- and `b`'s in-degree.
    `sources` counts vertices nothing points into (in-degree 0), `sinks` those that point nowhere
    (out-degree 0).

    The two reachabilities are the analysis header's, normalised to `0..1`:

    * `search_reachability` is `calc_search_reachability` — the fraction of vertices a BFS from the entry
      can reach, following edge direction when the graph is directed and both ways when it is not. A
      missing or out-of-range entry reaches nothing, so it scores `0.0`; the entry itself counts as
      reachable.
    * `explore_reachability` is `calc_exploration_reach` — the mean, over *every* start vertex, of the
      fraction reachable from it (the start included).

    A BFS from every vertex is too slow for the example's clouds (a thousand-plus vertices, sixteen
    edges each), so the all-pairs count runs on the strongly-connected condensation: Kosaraju's algorithm
    (iterative, so a long chain cannot exhaust the recursion) labels the components in topological order,
    and a single reverse-topological sweep propagates each component's reachable set as a Python int used
    as a bitset — bit `v` set means vertex `v` is reachable. A vertex's reach count is the popcount of its
    component's bitset; the mean over vertices divided by `num_points` is the exploration reach. The
    search reach is a plain BFS with an int-bitset frontier, cheap for the single start it needs.
    """
    n = int(num_points)
    if n <= 0:
        return {
            "min_out": 0,
            "max_out": 0,
            "min_in": 0,
            "max_in": 0,
            "sources": 0,
            "sinks": 0,
            "search_reachability": 0.0,
            "explore_reachability": 0.0,
        }

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out_deg = np.zeros(n, dtype=np.int64)
    in_deg = np.zeros(n, dtype=np.int64)
    out_adj: list[list[int]] = [[] for _ in range(n)]
    in_adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        a, b = int(a), int(b)
        out_adj[a].append(b)
        in_adj[b].append(a)
        out_deg[a] += 1
        in_deg[b] += 1
        if not directed:
            out_adj[b].append(a)
            in_adj[a].append(b)
            out_deg[b] += 1
            in_deg[a] += 1

    # Search reachability: one BFS from the entry, its frontier carried as an int bitset.
    if entry is None or not 0 <= int(entry) < n:
        search_reachability = 0.0
    else:
        start = int(entry)
        visited = 1 << start
        frontier = visited
        while frontier:
            nxt = 0
            remaining = frontier
            while remaining:
                low = remaining & -remaining
                for w in out_adj[low.bit_length() - 1]:
                    bit = 1 << w
                    if not visited & bit:
                        visited |= bit
                        nxt |= bit
                remaining ^= low
            frontier = nxt
        search_reachability = visited.bit_count() / n

    # Explore reachability: Kosaraju SCC (iterative) then a reverse-topological bitset DP over the
    # condensation. Components are labelled in topological order, so a successor always carries a higher
    # id than its predecessor and a descending sweep visits every component after all of its successors.
    visited = 0
    order: list[int] = []
    for start in range(n):
        if visited >> start & 1:
            continue
        visited |= 1 << start
        stack = [(start, 0)]
        while stack:
            v, index = stack[-1]
            if index < len(out_adj[v]):
                w = out_adj[v][index]
                stack[-1] = (v, index + 1)
                if not visited >> w & 1:
                    visited |= 1 << w
                    stack.append((w, 0))
            else:
                order.append(v)
                stack.pop()

    component = [-1] * n
    self_mask: list[int] = []
    for start in reversed(order):
        if component[start] != -1:
            continue
        cid = len(self_mask)
        mask = 0
        stack = [start]
        component[start] = cid
        while stack:
            v = stack.pop()
            mask |= 1 << v
            for w in in_adj[v]:
                if component[w] == -1:
                    component[w] = cid
                    stack.append(w)
        self_mask.append(mask)

    successors: list[set[int]] = [set() for _ in self_mask]
    for v in range(n):
        cv = component[v]
        for w in out_adj[v]:
            cw = component[w]
            if cv != cw:
                successors[cv].add(cw)

    reachable: list[int] = [0] * len(self_mask)
    for cid in range(len(self_mask) - 1, -1, -1):
        bits = self_mask[cid]
        for succ in successors[cid]:
            bits |= reachable[succ]
        reachable[cid] = bits

    total = sum(reachable[component[v]].bit_count() for v in range(n))
    return {
        "min_out": int(out_deg.min()),
        "max_out": int(out_deg.max()),
        "min_in": int(in_deg.min()),
        "max_in": int(in_deg.max()),
        "sources": int((in_deg == 0).sum()),
        "sinks": int((out_deg == 0).sum()),
        "search_reachability": search_reachability,
        "explore_reachability": total / n / n,
    }
