"""DEG construction, adjacency extraction and search navigation on 2D points."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from deglib import DynamicExplorationGraph
from deglib.builder import OptimizationTarget, build_from_data
from deglib.distances import FloatSpace, Metric
from deglib.optimization import mips_l2_transform_query
from pysearch import epsilon_search
from theory import TheoryReport, compare, dissimilarities

__all__ = [
    "DEFAULT_K",
    "METRIC_BY_LABEL",
    "METRIC_LABELS",
    "MIN_K",
    "QUERY_EPS",
    "GraphModel",
    "QueryResult",
    "build_model",
    "describe",
    "knng_edges",
    "nsw_edges",
    "query_graph",
]

#: Outgoing edges per vertex. Must be even and at least `MIN_K`; the example defaults to k=4.
DEFAULT_K = 4
MIN_K = 4
DEFAULT_EXTEND_K = 16
DEFAULT_EXTEND_EPS = 0.1
DEFAULT_IMPROVE_TRIES = 3
#: Exploration factor shared by the graph traversal and the library search it is compared against.
QUERY_EPS = 0.1
#: The metrics the example offers, in the order its selector lists them.
METRIC_ORDER = (Metric.FP32_L2, Metric.FP32_InnerProduct)
#: Short names for them, as they appear in the panel and the CLI, and the reverse lookup the CLI parses.
METRIC_LABELS = {Metric.FP32_L2: "L2", Metric.FP32_InnerProduct: "IP"}
METRIC_BY_LABEL = {label: metric for metric, label in METRIC_LABELS.items()}


@dataclass(frozen=True)
class GraphModel:
    """Immutable snapshot of a built DEG: features, adjacency and the unique undirected edge list."""

    points: np.ndarray  # [n, d] float32 feature vectors the graph is built, searched and compared on
    plot_points: np.ndarray  # [n, 2] drawing coordinates; the same array as `points` when no transform is active
    groups: np.ndarray  # [n] int8 synthetic cluster id, for coloring only
    adjacency: tuple[np.ndarray, ...]  # [n] sorted neighbor index arrays (undirected view)
    edges: np.ndarray  # [e, 2] int32 index pairs with i < j, ascending by (i, j)
    build_seconds: float
    k: int
    metric: Metric  # the metric the graph was built and searched with
    graph: DynamicExplorationGraph = field(repr=False, compare=False)
    theory: TheoryReport | None = field(default=None, repr=False, compare=False)
    mips: bool = False  # True when the features are the MIPS→L2 spherical lift of `plot_points`

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def degrees(self) -> np.ndarray:
        """Vertex degree — the DEG adjacency is symmetric, so this is exactly `k` unless a vertex cannot be saturated."""
        return np.fromiter((a.size for a in self.adjacency), dtype=np.int32, count=len(self.adjacency))

    def edge_lengths(self) -> np.ndarray:
        if self.edges.size == 0:
            return np.zeros(0, dtype=np.float32)
        return np.linalg.norm(self.points[self.edges[:, 0]] - self.points[self.edges[:, 1]], axis=1)

    def adjacency_of(self, name: str, feature_edges: np.ndarray | None = None) -> list[np.ndarray]:
        """
        The neighbour lists the search runs over for one view: the DEG's own, the knng's, the NSW's,
        or a reference graph's.

        The DEG keeps its adjacency as built. The knng, the NSW and the NSG are not stored on the model —
        they are read off the current feature space — so their edge list is passed in by the viewer that
        cached it: the knng and the NSG keep their direction, because a vertex can only walk the neighbours
        it chose itself, while the NSW collapses to an undirected adjacency. A reference
        graph is undirected and collapses the same way. Unlike
        the DEG these lists are not regular — a spanning tree vertex may hold one neighbour, a Delaunay
        one many — which is why the search takes a list of variable-length arrays rather than a fixed
        degree.
        """
        if name == "deg":
            return list(self.adjacency)
        if name in ("knng", "nsw", "nsg"):
            if feature_edges is None:
                raise ValueError(f"the {name} view needs the edge list the viewer built it from")
            return self._collapse(feature_edges) if name == "nsw" else self._forward(feature_edges)
        if self.theory is None:
            raise ValueError(f"the model carries no reference graphs, so {name!r} has no adjacency")
        edges = self.theory.edges_of(name, self.num_points)
        return self._collapse(edges)

    def _collapse(self, edges: np.ndarray) -> list[np.ndarray]:
        """Collapses an undirected `[e, 2]` edge list into sorted per-vertex neighbour arrays."""
        neighbors: list[list[int]] = [[] for _ in range(self.num_points)]
        for i, j in edges:
            neighbors[int(i)].append(int(j))
            neighbors[int(j)].append(int(i))
        return [np.asarray(sorted(row), dtype=np.int32) for row in neighbors]

    def _forward(self, edges: np.ndarray) -> list[np.ndarray]:
        """Collapses a directed `[e, 2]` edge list into sorted per-vertex arrays of forward neighbours."""
        neighbors: list[list[int]] = [[] for _ in range(self.num_points)]
        for i, j in edges:
            neighbors[int(i)].append(int(j))
        return [np.asarray(sorted(row), dtype=np.int32) for row in neighbors]

    def traversal(self, metric: Metric) -> DynamicExplorationGraph:
        """
        The graph to traverse when the search should rank neighbours by `metric` rather than the one
        the build used.

        A read-only graph stores no edge weights: it recomputes every distance from its own feature
        space while the traversal runs. Re-tagging that space therefore keeps the exact adjacency the
        drawing shows and changes only which neighbour the greedy step picks next — which is the one
        thing a separate search metric asks. The copy costs about 0.1 ms at 1200 vertices, so it is
        made on the spot instead of cached on the model.
        """
        if metric is self.metric:
            return self.graph
        return self.graph.to_readonly(FloatSpace.create(self.points.shape[1], metric))

    def best_match(self, query: np.ndarray, metric: Metric | None = None, *, exclude: int | None = None) -> int:
        """
        Brute-force ground truth: the vertex `metric` ranks highest for `query`, the build metric by
        default and the search metric once the two are deliberately pulled apart. `exclude` masks one
        index — the query's own vertex — out of the ranking, so the answer is the best match among the
        *other* vertices.

        Under L2 that is the closest vertex, so a query taken from the cloud matches itself until
        `exclude` removes it. Under inner product it is the vertex with the largest dot product —
        generally a *longer* vector in the query's direction, so the query vertex is not the answer
        and matching it would be wrong.
        """
        if (self.metric if metric is None else metric) is Metric.FP32_InnerProduct:
            scores = self.points @ query
            if exclude is not None:
                scores[int(exclude)] = -np.inf
            return int(np.argmax(scores))
        delta = self.points - query
        squared = np.einsum("ij,ij->i", delta, delta)
        if exclude is not None:
            squared[int(exclude)] = np.inf
        return int(np.argmin(squared))

    def query_vector(self, plot_xy: np.ndarray) -> np.ndarray:
        """
        The feature-space vector a plot-space point is searched with.

        Under a MIPS model the features are the spherical lift, so the query is the original 2D vector
        zero-padded to `[x, y, 0]` — the standard MIPS→L2 query — never a lifted coordinate: searching
        the lifted database from that padded query is the true inner-product maximisation over the
        originals. An untransformed model searches the plot point itself. This is the one padding rule
        every query — a vertex's coordinates or a free click — passes through.
        """
        point = np.asarray(plot_xy, dtype=np.float32).reshape(1, -1)
        return mips_l2_transform_query(point)[0] if self.mips else point[0]

    def farthest_from_centroid(self) -> int:
        """
        The vertex most remote from the cloud's centre — the viewer's default start node.

        A traversal wants the pessimistic entry: a vertex far from everything, so a descent from it
        has ground to cover and a stall is worth looking at. The centroid keeps that choice the same
        for every viewer and every rebuild, unlike the query-relative farthest point a search picks
        for itself when no start node was given.
        """
        delta = self.points - self.points.mean(axis=0)
        return int(np.argmax(np.einsum("ij,ij->i", delta, delta)))

    def search(
        self, query: np.ndarray, eps: float, k: int = 1, metric: Metric | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The library's own graph search, returning (indices, distances).

        The example no longer searches through this — `query_graph` runs the Python `epsilon_search`, which
        records the route and runs on reference graphs too. This facade stays only as the parity reference
        the tests hold the Python search against: same answers, same distances, on the same graph.
        """
        indices, distances = self.traversal(self.metric if metric is None else metric).search(
            np.ascontiguousarray(query, dtype=np.float32), eps, k
        )
        return np.asarray(indices), np.asarray(distances)


@dataclass(frozen=True)
class QueryResult:
    """
    One ε-search run against a query point, measured against brute force.

    The query is either a vertex — scored against its best match among the *other* vertices — or a
    free plot-space coordinate, scored against the best match over the whole cloud. `target` names the
    vertex and is `None` for a free query; `query_xy` carries the plot coordinates either way, so the
    panel can name the query without re-deriving them.

    The search answers and carries its own route in the same run: `path` is the trackback the search
    recorded to its top answer `deg_indices[0]`, so the drawn route is the walk the search genuinely
    followed rather than a second traversal aimed at it afterwards. Because that route is reconstructed
    from the search's own predecessor chain, it is present whenever there is an answer to reach — a
    search that returns nothing leaves `path` empty, since there is no answer to walk to. The three
    counters describe the run: distances computed, vertices expanded, vertices checked.
    """

    target: int | None
    query_xy: tuple[float, float]
    point: np.ndarray
    path: np.ndarray
    eps: float
    metric: Metric
    exact: int
    deg_indices: np.ndarray  # the top-`top_k` non-self search results, in the search's own ranking order
    deg_distances: np.ndarray  # the distance the search ranked each of those results by, same order
    distance_computations: int  # distances the search measured over the whole run
    expanded: int  # vertices whose neighbour list the search scanned
    visited: int  # vertices the search ever marked checked

    @property
    def hops(self) -> int:
        return max(int(self.path.size) - 1, 0)

    @property
    def entry(self) -> int | None:
        return int(self.path[0]) if self.path.size else None

    @property
    def walked(self) -> bool:
        """The traversal arrived at the search's top answer — the vertex the path was aimed at."""
        return self.path.size > 0 and self.deg_indices.size > 0 and int(self.path[-1]) == int(self.deg_indices[0])

    @property
    def deg_hit(self) -> bool:
        return bool(self.exact in self.deg_indices)


def _nearest_neighbours(points: np.ndarray, k: int, metric: Metric) -> np.ndarray:
    """
    The `[n, m]` matrix of each vertex's `m = min(k, n - 1)` nearest neighbour indices under `metric`.

    The knng reads its links from this one computation. The neighbour set is read from the same
    dissimilarity the reference graphs use (`theory.dissimilarities`, whose infinite diagonal keeps a
    vertex out of its own neighbour set), so an inner-product cloud links by inner product and an L2
    cloud by Euclidean distance. The indices are distinct by construction, so a vertex's out-degree is
    exactly `m`.
    """
    points = np.asarray(points)
    num_points = int(points.shape[0])
    neighbours = min(int(k), max(num_points - 1, 0))
    if num_points < 2 or neighbours <= 0:
        return np.zeros((num_points, 0), dtype=np.int64)
    distances = dissimilarities(points, metric)
    return np.argsort(distances, axis=1)[:, :neighbours].astype(np.int64)


def knng_edges(points: np.ndarray, k: int, metric: Metric = Metric.FP32_L2) -> np.ndarray:
    """
    The directed k-nearest-neighbour graph over `points`, as a directed `[e, 2]` int32 edge list.

    Each vertex points at its `k` nearest neighbours under `metric`, read from the shared neighbour
    computation, and the link is kept in the direction it was chosen — the edge list holds `(v, neighbour)`
    for every selection and is never symmetrized. So the out-degree is exactly `k` per vertex (fewer only
    when `k` exceeds the number of other vertices), while an in-degree can exceed `k` when a vertex is a
    popular neighbour. A vertex can only walk the neighbours it chose itself.
    """
    points = np.asarray(points)
    num_points = int(points.shape[0])
    nearest = _nearest_neighbours(points, k, metric)
    neighbours = int(nearest.shape[1])
    if neighbours <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    sources = np.repeat(np.arange(num_points, dtype=np.int32), neighbours)
    targets = nearest.reshape(-1).astype(np.int32)
    return np.column_stack((sources, targets)).astype(np.int32).reshape(-1, 2)


def nsw_edges(points: np.ndarray, k: int, metric: Metric = Metric.FP32_L2) -> np.ndarray:
    """
    The navigable small world over `points`, built incrementally as an undirected `[e, 2]` int32 edge list.

    Vertices are inserted in index order, and each new vertex links — undirected — to its `k` nearest among
    the vertices already in the graph, the same dissimilarity the reference graphs read. Because a later
    vertex can add a link to an earlier one, an earlier vertex ends up with more than `k` neighbours: the
    degree is the count of vertices that chose it plus the `k` it chose itself, so the maximum degree is
    unbounded. The first vertex has no earlier vertex to link and the first few hold fewer than `k` links,
    simply because too few vertices precede them.
    """
    points = np.asarray(points)
    num_points = int(points.shape[0])
    if num_points < 2:
        return np.zeros((0, 2), dtype=np.int32)

    distances = dissimilarities(points, metric)
    kept: list[tuple[int, int]] = []
    for i in range(1, num_points):
        earlier = distances[i, :i]
        chosen = min(int(k), i)
        nearest = np.argpartition(earlier, chosen - 1)[:chosen]
        nearest = nearest[np.argsort(earlier[nearest], kind="stable")]
        kept.extend((i, int(j)) for j in nearest)
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


def build_model(
    points: np.ndarray,
    groups: np.ndarray,
    k: int = DEFAULT_K,
    metric: Metric = Metric.FP32_L2,
    *,
    extend_k: int = DEFAULT_EXTEND_K,
    extend_eps: float = DEFAULT_EXTEND_EPS,
    improve_tries: int = DEFAULT_IMPROVE_TRIES,
    threads: int = 1,
    seed: int = 7,
    theory: bool = True,
    plot_points: np.ndarray | None = None,
    mips: bool = False,
) -> GraphModel:
    """
    Builds a DEG with `k` edges per vertex over `points` and extracts its topology.

    With `metric` the graph is built and searched under that metric; the example offers L2 and inner
    product. With `theory` the reference graphs are compared against the result too, and they are read
    from the same dissimilarity the search minimised — so an inner-product model is judged against the
    graphs an inner product admits, not against Euclidean discs. That comparison is the quadratic part
    of the example, so a caller that only wants the graph can pass `theory=False`.

    `points` are the feature vectors the graph lives in; `plot_points` are the coordinates the viewer
    draws. They are the same array unless a transform has lifted the features into a space the picture
    cannot show — the MIPS→L2 spherical transform, which appends a third dimension. Leaving
    `plot_points` unset keeps the two identical, which is the case for every untransformed cloud.

    `mips` records that the features are that spherical lift. It changes nothing about the build; it
    only tells `query_graph` to answer a query with the original vector zero-padded into the lifted
    space, which is what makes the search a true MIPS query rather than one run from a vertex's own
    extra coordinate.

    `threads` defaults to 1 so the build is reproducible: the parallel extension does not guarantee the
    order vertices are processed in, so a multi-threaded build reaches a different graph on every run.
    A single thread fixes that order and makes the same seed and cloud rebuild the same edges.
    """
    if k < MIN_K or k % 2:
        raise ValueError(f"edges_per_vertex must be an even number of at least {MIN_K}, got {k}")

    started = time.perf_counter()
    graph = build_from_data(
        points,
        edges_per_vertex=k,
        metric=metric,
        optimization_target=OptimizationTarget.LowLID,
        extend_k=extend_k,
        extend_eps=extend_eps,
        improve_tries=improve_tries,
        thread_count=threads,
        seed=seed,
    )
    build_seconds = time.perf_counter() - started

    adjacency: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    for label in range(points.shape[0]):
        neighbors = np.asarray(sorted(graph.get_neighbors(label)), dtype=np.int32)
        adjacency.append(neighbors)
        pairs.extend((label, int(j)) for j in neighbors if label < j)
    edges = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)

    return GraphModel(
        points=points,
        plot_points=points if plot_points is None else plot_points,
        groups=groups,
        adjacency=tuple(adjacency),
        edges=edges,
        build_seconds=build_seconds,
        k=k,
        metric=metric,
        mips=mips,
        graph=graph,
        theory=compare(points, edges, metric) if theory else None,
    )


def query_graph(
    model: GraphModel,
    target: int | np.ndarray,
    entry: int | None = None,
    eps: float = QUERY_EPS,
    metric: Metric | None = None,
    *,
    top_k: int = 1,
    view: str = "deg",
    feature_edges: np.ndarray | None = None,
) -> QueryResult:
    """
    Runs the ε-search for a query over the current view's graph, measured against brute force.

    `view` names the graph the search walks: the built DEG, the knng or the NSW built from the current
    feature space (whose edge list the caller passes as `feature_edges`), or one of the reference graphs the theory
    report holds. The search is the Python `epsilon_search`, a faithful port of the library's ε-search that
    additionally records the route it took to each answer — so a reference graph, which the library never
    sees, is searched and drawn exactly as the DEG is.

    `target` is either a vertex or a free 2D plot-space coordinate. A vertex supplies the query vector
    and the answer the search is scored against is its best match under `metric` — the true nearest
    neighbour with the vertex itself excluded — so a hit is a real retrieval rather than a vertex
    matching itself. The search runs over a database that still holds `target`, so a vertex query asks it
    for `top_k + 1` answers, drops the query vertex from them and keeps the first `top_k`; a free
    coordinate is not a vertex, so nothing is excluded — the search is asked for `top_k` answers and keeps
    them all. Either way the answers travel as `deg_indices` and `deg_distances` in the search's own
    ranking order, and `deg_hit` says whether the best match is among them. A search that returns nothing
    is an empty result list, not an error — and with no answer there is no route to record either.

    The route is the search's own, not a second traversal aimed at it: `path` is the trackback the search
    recorded to its top answer `deg_indices[0]`, reconstructed from the predecessor it stored for every
    vertex it queued. It is therefore present whenever there is an answer to reach, and `walked` reads true
    on every non-empty result — the search reached the vertex it answered with. The star (`exact`) and the
    X (`deg_indices[0]`) tell two different stories when the search misses: the first is the truth, the
    second is what the search actually returned, and the path is the road it took to the second.

    Without an explicit entry the search starts at the vertex farthest from the query point — the
    pessimistic entry point — which makes the descent travel far enough to be worth looking at. `eps`
    widens the exploration radius the search bounds its frontier by; at 0.0 it degenerates into a greedy
    descent that tightens its radius the moment the answer set fills.

    Under a MIPS model the query is not the point's own transformed vector — that carries the extra
    coordinate `√(M² − ‖x‖²)` the spherical lift appended — but the original 2D vector zero-padded to
    `[x, y, 0]`, the standard MIPS→L2 query. Searching the lifted database from that padded query is
    the true inner-product maximisation over the originals, so both the search and the ground truth
    answer `argmax ⟨q, x⟩` rather than a nearest neighbour in the inflated space.

    `metric` is the comparator both the search and the ground truth use, the build metric unless the
    caller pulls the two apart. Setting it apart walks the stored edges by a comparator the build never
    optimised for, and judges the answer against that comparator's own optimum — so a search that lands
    elsewhere is the two metrics disagreeing, not the graph failing.
    """
    search = model.metric if metric is None else metric
    if isinstance(target, np.ndarray):
        vertex = None
        plot_xy = np.asarray(target, dtype=np.float32).reshape(2)
    else:
        vertex = int(target)
        if entry is not None and int(entry) == vertex:
            raise ValueError("the entry vertex must differ from the target vertex")
        plot_xy = model.plot_points[vertex]
    point = model.query_vector(plot_xy)
    query_xy = (float(plot_xy[0]), float(plot_xy[1]))
    if entry is None:
        delta = model.points - point
        entry = int(np.argmax(np.einsum("ij,ij->i", delta, delta)))

    exact = model.best_match(point, search, exclude=vertex)
    outcome = epsilon_search(
        model.adjacency_of(view, feature_edges), model.points, point, entry, eps, top_k + (vertex is not None), search
    )
    answers = [(int(i), float(d)) for i, d in zip(outcome.indices, outcome.distances)]
    if vertex is not None:
        answers = [answer for answer in answers if answer[0] != vertex]
    answers = answers[:top_k]
    # The route is the search's own recorded trackback to its top answer — the vertex it answered with.
    # No answer, no route.
    if answers:
        rank = int(np.flatnonzero(outcome.indices == answers[0][0])[0])
        path = outcome.paths[rank]
    else:
        path = np.zeros(0, dtype=np.int32)
    return QueryResult(
        target=vertex,
        query_xy=query_xy,
        point=point,
        path=path,
        eps=float(eps),
        metric=search,
        exact=exact,
        deg_indices=np.asarray([i for i, _ in answers], dtype=np.int32),
        deg_distances=np.asarray([d for _, d in answers], dtype=np.float32),
        distance_computations=outcome.distance_computations,
        expanded=outcome.expanded,
        visited=outcome.visited,
    )


def describe(model: GraphModel) -> list[str]:
    """
    Graph statistics, aligned for both the console and the UI panel.

    The first line is the headline — the panel shows it directly under the heading naming the drawn
    graph, so the order is part of
    the contract and the block has to stay narrow enough to fit beside the plot. `model.edges` is the
    deduplicated undirected list, so the headline count is exactly ``num_points * k / 2``.

    A model built with its theory report closes the block on the `∩` lines: how many of the DEG's own
    edges each reference graph also contains, and what share of the DEG that is.
    """
    lengths = model.edge_lengths()
    degrees = model.degrees()
    build = f"build          {model.build_seconds * 1000.0:.1f} ms"
    if lengths.size == 0:
        return [
            f"{model.num_points} vertices · k={model.k} · {METRIC_LABELS[model.metric]} · no edges",
            "degree         —",
            "edge length    —",
            build,
        ]
    lines = [
        f"{model.num_points} vertices · k={model.k} · {METRIC_LABELS[model.metric]} · {model.edges.shape[0]} edges",
        f"degree         min {degrees.min()} / mean {degrees.mean():.2f} / max {degrees.max()}",
        f"edge length    mean {lengths.mean():.3f} / max {lengths.max():.3f}",
        build,
    ]
    if model.theory is not None:
        lines += [
            f"∩ {item.name:<9s}{item.edges:6d} edges{item.shared:7d} shared{item.share_of(model.theory.deg_edges):7.1%}"
            for item in model.theory.graphs
        ]
    return lines
