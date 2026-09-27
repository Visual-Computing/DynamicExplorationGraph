"""
A faithful Python port of the library's ε-search, recording the route to every answer.

The C++ search (`graph/internal_graph.h::searchImpl`) returns the top-`k` answers but discards the walk
that reached them, and it only ever runs over a built DEG — the theoretical reference graphs are read from
an edge list the library never sees. This module reimplements the same greedy best-first traversal in pure
numpy so the example can (a) draw the route the search actually took and (b) search a reference graph on
screen exactly as it searches the DEG.

The traversal is the one the library runs, mirrored line for line: a min-heap of unchecked candidates pops
the closest vertex, each of its unvisited neighbours is measured once, kept for expansion if it falls
inside the exploration radius and recorded as an answer if it improves on the worst kept distance. The
exploration radius is the worst kept answer scaled by `1 + eps` (or `1 - eps` when that distance is
negative, which is how an inner-product search widens a radius that runs below zero). The one addition is a
predecessor recorded for every vertex the search pushes — the same trackback the C++ `hasPathImpl` keeps —
so each answer carries the chain of edges the search itself followed to reach it.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from deglib.distances import Metric

__all__ = ["SearchOutcome", "epsilon_search"]


@dataclass(frozen=True)
class SearchOutcome:
    """
    The answers a search returned, the route it took to each, and what it cost to get there.

    `indices` and `distances` are the result set ranked best-first — the same top-`k` the library's search
    answers with. `paths` runs parallel to them: one entry-to-answer vertex chain per result, the search's
    own recorded route, so a drawn path is the walk the search genuinely took rather than a second
    traversal aimed at it after the fact. The three counters describe the run: how many distances were
    computed, how many vertices had their neighbourhood expanded, and how many vertices were ever checked.
    """

    indices: np.ndarray  # [r] int32 result vertex ids, ranked best-first
    distances: np.ndarray  # [r] float32 the distance each was ranked by, same order
    paths: tuple[np.ndarray, ...]  # [r] int32 entry-to-result chains, parallel to `indices`
    distance_computations: int
    expanded: int  # vertices whose neighbour list was scanned
    visited: int  # vertices ever marked checked


def _distance(points: np.ndarray, query: np.ndarray, index: int, metric: Metric) -> float:
    """The one score the search ranks by: squared Euclidean under L2, `1 - dot` under inner product."""
    if metric is Metric.FP32_InnerProduct:
        return 1.0 - float(points[index] @ query)
    delta = points[index] - query
    return float(delta @ delta)


def epsilon_search(
    adjacency: Sequence[np.ndarray],
    points: np.ndarray,
    query: np.ndarray,
    entry: int,
    eps: float,
    k: int,
    metric: Metric,
) -> SearchOutcome:
    """
    Greedy best-first ε-search over `adjacency`, recording a trackback path per answer.

    `adjacency` is one neighbour-id array per vertex, of any degree — a reference graph is not regular the
    way a built DEG is. `points` are the feature vectors the search measures, `query` the vector it searches
    for, and `metric` the comparator it ranks by. The search starts at the single `entry` vertex, which is
    marked checked, queued for expansion and — exactly as the library's `include_entry` does — recorded as
    an answer and its own predecessor.

    The loop is the library's `searchImpl`: pop the closest candidate and stop the moment it lies beyond the
    exploration radius; mark and collect its unchecked neighbours; measure each one, queue it for expansion
    when it falls inside the exploration radius and record it as an answer when it beats the worst kept
    distance, tightening the radius and the exploration radius whenever the answer set overflows `k`. The
    single deviation from `searchImpl` is the predecessor recorded for every queued neighbour — the vertex
    currently being expanded — which is what lets each answer be walked back to the entry.
    """
    num_vertices = len(adjacency)
    k = min(num_vertices, int(k))

    checked = np.zeros(num_vertices, dtype=bool)
    predecessor = np.full(num_vertices, -1, dtype=np.int64)
    next_vertices: list[tuple[float, int]] = []  # min-heap of (distance, vertex), closest popped first
    results: list[tuple[float, int]] = []  # max-heap of (-distance, vertex), the worst answer on top

    # The entry is checked, queued for expansion and kept as an answer — its own predecessor, so the
    # trackback walk stops the moment it returns to the start.
    entry_distance = _distance(points, query, entry, metric)
    distance_computations = 1
    visited = 1
    checked[entry] = True
    predecessor[entry] = entry
    heapq.heappush(next_vertices, (entry_distance, entry))
    heapq.heappush(results, (-entry_distance, entry))

    radius = float("inf")
    exploration_radius = radius
    expanded = 0

    while next_vertices:
        distance, vertex = heapq.heappop(next_vertices)
        if distance > exploration_radius:
            break
        expanded += 1

        neighbors = adjacency[vertex]
        unchecked = neighbors[~checked[neighbors]]
        checked[unchecked] = True
        visited += int(unchecked.size)

        for neighbor in unchecked:
            neighbor_index = int(neighbor)
            neighbor_distance = _distance(points, query, neighbor_index, metric)
            distance_computations += 1

            if neighbor_distance <= exploration_radius:
                heapq.heappush(next_vertices, (neighbor_distance, neighbor_index))
                predecessor[neighbor_index] = vertex

                if neighbor_distance < radius:
                    heapq.heappush(results, (-neighbor_distance, neighbor_index))
                    if len(results) > k:
                        heapq.heappop(results)
                        radius = -results[0][0]
                        exploration_radius = radius * (1.0 - eps) if radius < 0.0 else radius * (1.0 + eps)

    ranked = sorted((-negated, index) for negated, index in results)
    indices = np.fromiter((index for _, index in ranked), dtype=np.int32, count=len(ranked))
    distances = np.fromiter((distance for distance, _ in ranked), dtype=np.float32, count=len(ranked))
    paths = tuple(_reconstruct(predecessor, int(index)) for index in indices)

    return SearchOutcome(
        indices=indices,
        distances=distances,
        paths=paths,
        distance_computations=distance_computations,
        expanded=expanded,
        visited=visited,
    )


def _reconstruct(predecessor: np.ndarray, target: int) -> np.ndarray:
    """Walks the predecessor chain from an answer back to the entry, then hands it back entry-first."""
    path = [target]
    vertex = target
    while predecessor[vertex] != vertex:
        vertex = int(predecessor[vertex])
        path.append(vertex)
    path.reverse()
    return np.asarray(path, dtype=np.int32)
