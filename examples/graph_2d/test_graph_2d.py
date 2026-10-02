"""Unit tests for the 2D graph explorer: data, graph topology, navigation and the UI wiring."""

from __future__ import annotations

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")

import tkinter
from dataclasses import replace
from functools import partial
from itertools import pairwise
from tkinter import ttk

import numpy as np
import pytest
from dataset import MAX_POINTS, MIN_POINTS, PRESETS, make_points
from deg_graph import DEFAULT_K, MIN_K, build_model, describe, knng_edges, nsw_edges, query_graph
from deglib.distances import Metric
from main import build_scene
from matplotlib.backend_bases import KeyEvent, MouseEvent
from pysearch import epsilon_search
from theory import (
    GraphOverlap,
    _monotone_reachable,
    compare,
    delaunay_edges,
    dissimilarities,
    gabriel_edges,
    graph_stats,
    mrng_edges,
    mst_edges,
    nsg_edges,
    rng_edges,
)
from viewer import (
    BG_DEFAULT,
    BG_LIGHT,
    DEG_VIEW,
    EDGE_COLOR,
    EXACT_COLOR,
    EXACT_FILL,
    MARKER_EDGE,
    FLAG_START,
    HOVER_EDGE_COLOR,
    K_LIMIT,
    KNNG_COLOR,
    KNNG_VIEW,
    NEIGHBOR_LABEL,
    NONE_VIEW,
    NSG_VIEW,
    NSW_COLOR,
    NSW_VIEW,
    OVERLAP_COLORS,
    OVERLAP_ORDER,
    PANEL_STYLES,
    PANEL_WIDTH,
    PICK_RADIUS_PX,
    TERMINAL_COLOR,
    TOP_K_MAX,
    GraphViewer,
    _fan_out,
    arrowhead_triangles,
    edge_colours,
    edge_segments,
    legend_entries,
    render_static,
    vertex_colors,
)


@pytest.fixture(scope="module")
def model():
    points, groups = make_points("blobs", 400, seed=3)
    return build_model(points, groups, k=DEFAULT_K, seed=3, threads=1)


# --------------------------------------------------------------------------- synthetic data


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_yield_finite_float32_points(preset: str) -> None:
    points, groups = make_points(preset, 500, seed=11)

    assert points.shape == (500, 2)
    assert points.dtype == np.float32
    assert groups.shape == (500,) and groups.dtype == np.int8
    assert np.isfinite(points).all()
    assert groups.min() >= 0


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_are_reproducible(preset: str) -> None:
    first, _ = make_points(preset, 300, seed=5)
    again, _ = make_points(preset, 300, seed=5)
    other, _ = make_points(preset, 300, seed=6)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_preset_spans_a_usable_range() -> None:
    points, _ = make_points("blobs", MAX_POINTS, seed=2)
    span = points.max(axis=0) - points.min(axis=0)
    assert (span > 1.0).all()


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_straddle_the_origin(preset: str) -> None:
    points, _ = make_points(preset, 400, seed=4)
    low, high = points.min(axis=0), points.max(axis=0)

    assert (low < 0.0).all() and (high > 0.0).all(), "both signs on both axes"
    np.testing.assert_allclose((low + high) / 2.0, 0.0, atol=1e-4, err_msg="the bounding box centres on zero")


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        make_points("hypercube", 100)


def test_tiny_point_count_is_rejected() -> None:
    with pytest.raises(ValueError, match=f"between {MIN_POINTS} and {MAX_POINTS}"):
        make_points("blobs", MIN_POINTS - 2)


def test_oversized_point_count_is_rejected() -> None:
    with pytest.raises(ValueError, match=f"between {MIN_POINTS} and {MAX_POINTS}"):
        make_points("blobs", MAX_POINTS + 100)


# --------------------------------------------------------------------------- graph topology


def test_model_reports_the_requested_degree(model) -> None:
    assert model.k == DEFAULT_K
    assert model.num_points == 400
    assert model.build_seconds > 0.0
    assert model.graph.get_edges_per_vertex() == DEFAULT_K


def test_edges_are_unique_undirected_pairs(model) -> None:
    edges = model.edges
    assert edges.shape[1] == 2
    assert (edges[:, 0] < edges[:, 1]).all()
    assert len({tuple(edge) for edge in edges}) == edges.shape[0]


def test_adjacency_is_symmetric_and_matches_the_edge_list(model) -> None:
    for vertex, neighbors in enumerate(model.adjacency):
        assert neighbors.size > 0, "every vertex must stay reachable"
        assert np.all(np.diff(neighbors) > 0), "neighbor lists must be sorted and free of duplicates"
        for neighbor in neighbors:
            assert vertex in model.adjacency[int(neighbor)]

    pairs = {tuple(edge) for edge in model.edges}
    expanded = {
        (min(vertex, int(neighbor)), max(vertex, int(neighbor)))
        for vertex, neighbors in enumerate(model.adjacency)
        for neighbor in neighbors
    }
    assert pairs == expanded


def test_edges_connect_near_points(model) -> None:
    drawn = model.edge_lengths()
    delta = model.points[:, None, :] - model.points[None, :, :]
    random_pairs = np.einsum("ijk,ijk->ij", delta, delta).reshape(-1)
    reference = np.quantile(random_pairs[random_pairs > 0], 0.5)

    assert drawn.mean() < np.sqrt(reference) * 0.5, "the graph should be a proximity graph, not random"


def test_odd_degree_is_rejected() -> None:
    points, groups = make_points("uniform", 64, seed=1)
    with pytest.raises(ValueError, match="even"):
        build_model(points, groups, k=5)


def test_degree_below_the_minimum_is_rejected() -> None:
    points, groups = make_points("uniform", 64, seed=1)

    with pytest.raises(ValueError, match=f"at least {MIN_K}"):
        build_model(points, groups, k=MIN_K - 2)


# --------------------------------------------------------------------------- navigation


def _adjacency_edges(adjacency) -> set[tuple[int, int]]:
    """The undirected edge set an adjacency list describes, as `(min, max)` index pairs."""
    return {
        (min(vertex, int(neighbor)), max(vertex, int(neighbor)))
        for vertex, neighbors in enumerate(adjacency)
        for neighbor in neighbors
    }


@pytest.mark.parametrize("metric", [Metric.FP32_L2, Metric.FP32_InnerProduct])
@pytest.mark.parametrize("eps", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("top_k", [1, 5, 10])
def test_epsilon_search_matches_the_library_search(model, metric, eps, top_k) -> None:
    """
    The Python port answers with the very set the C++ search returns, and the same distances.

    The library search starts at the graph's own entry vertex, so the port is seeded with the same one —
    a different entry would explore a different graph and the sets could not be expected to agree. The
    result is compared as a set and the distances per vertex, so a tie the two implementations break in
    opposite orders cannot turn a faithful port into a failed test.
    """
    entry = int(model.graph.get_entry_vertex_indices()[0])
    query = model.points[42]

    outcome = epsilon_search(list(model.adjacency), model.points, query, entry, eps, top_k, metric)
    library_indices, library_distances = model.search(query, eps, top_k, metric)

    ours = dict(zip(outcome.indices.tolist(), outcome.distances.tolist()))
    theirs = dict(zip((int(i) for i in library_indices), (float(d) for d in library_distances)))
    assert ours.keys() == theirs.keys(), "the same answer set, whatever order each ranked it in"
    for index, distance in ours.items():
        assert distance == pytest.approx(theirs[index], rel=1e-4), "the same distance for every answer"


def test_epsilon_search_matches_the_library_search_under_the_mips_lift() -> None:
    """The port runs on the lifted 3D features from a zero-padded query exactly as the library does."""
    model = build_scene("blobs", 200, DEFAULT_K, 3, Metric.FP32_L2, mips=True, threads=1)
    entry = int(model.graph.get_entry_vertex_indices()[0])
    query = model.query_vector(model.plot_points[42])

    outcome = epsilon_search(list(model.adjacency), model.points, query, entry, 0.1, 5, model.metric)
    library_indices, _ = model.search(query, 0.1, 5, model.metric)

    assert set(outcome.indices.tolist()) == {int(i) for i in library_indices}


def test_epsilon_search_records_a_real_route_to_every_answer(model) -> None:
    """Each answer carries the chain of real edges the search followed to reach it, entry-first."""
    entry = int(model.graph.get_entry_vertex_indices()[0])
    outcome = epsilon_search(list(model.adjacency), model.points, model.points[42], entry, 0.1, 5, Metric.FP32_L2)
    edges = _adjacency_edges(model.adjacency)

    assert outcome.indices.size == len(outcome.paths), "one route per answer"
    for index, path in zip(outcome.indices, outcome.paths):
        assert int(path[0]) == entry, "the route starts at the entry"
        assert int(path[-1]) == int(index), "and ends on the answer it was recorded for"
        for a, b in pairwise(path):
            assert (min(a, b), max(a, b)) in edges, "every consecutive pair is a real edge"


def test_epsilon_search_stats_are_consistent(model) -> None:
    """Every checked vertex is measured once, and only checked vertices are ever expanded."""
    entry = int(model.graph.get_entry_vertex_indices()[0])
    outcome = epsilon_search(list(model.adjacency), model.points, model.points[42], entry, 0.5, 8, Metric.FP32_L2)

    assert outcome.distance_computations >= outcome.visited, "each checked vertex is measured at least once"
    assert outcome.expanded <= outcome.visited, "a vertex is expanded only after it was checked"


def test_query_graph_searches_a_reference_graph_and_walks_its_edges(model) -> None:
    """The same search runs over a reference graph's adjacency, recording a route along its real edges."""
    result = query_graph(model, 42, 7, eps=0.5, view="rng")
    edges = _adjacency_edges(model.adjacency_of("rng"))

    assert result.deg_indices.size == 1, "the search answers over the reference graph"
    assert int(result.path[0]) == 7, "the route starts at the seeded entry"
    assert int(result.path[-1]) == int(result.deg_indices[0]), "and ends on the answer it returned"
    for a, b in pairwise(result.path):
        assert (int(min(a, b)), int(max(a, b))) in edges, "every step is an edge of the reference graph"


def test_query_graph_traverses_to_the_searchs_top_answer(model) -> None:
    target, entry = 7, 211

    result = query_graph(model, target, entry, eps=0.5)

    delta = model.points - model.points[target]
    squared = np.einsum("ij,ij->i", delta, delta)
    squared[target] = np.inf
    neighbour = int(np.argmin(squared))

    assert model.best_match(model.points[target]) == target, "without `exclude` a vertex is still its own match"
    assert model.best_match(model.points[target], exclude=target) == neighbour, (
        "excluding the diagonal the answer is the closest other vertex"
    )
    assert result.target == target
    assert result.point.shape == (2,)
    assert result.eps == pytest.approx(0.5)
    assert result.exact == neighbour, "the query is scored against the closest other vertex, not itself"
    assert result.path[0] == entry, "the traversal must start at the requested seed"
    assert result.deg_indices.size == 1 and int(result.deg_indices[0]) != target, (
        "the search drops the query vertex and reports its best non-self answer"
    )
    assert result.walked, "a wide enough exploration factor reaches the search's answer"
    assert int(result.path[-1]) == int(result.deg_indices[0]), "the walk ends on the vertex the search answered"
    assert result.deg_hit and int(result.deg_indices[0]) == result.exact, (
        "on a well-connected graph the search recovers the best match, so here the answer and the truth coincide"
    )


def test_query_graph_defaults_to_the_farthest_entry(model) -> None:
    result = query_graph(model, 7, eps=0.5)

    assert result.entry is not None and result.entry != 7, "without a seed the traversal starts away from the target"
    assert result.walked and result.hops >= 1


def test_query_graph_answers_the_nearest_other_vertex(model) -> None:
    """The selected vertex is the query; the answer is its nearest neighbour, never the vertex itself."""
    target = 42

    result = query_graph(model, target, eps=0.5)

    delta = model.points - model.points[target]
    squared = np.einsum("ij,ij->i", delta, delta)
    squared[target] = np.inf

    assert result.exact != target, "the search runs against the best match, not the selected vertex"
    assert result.exact == int(np.argmin(squared)), "the answer is the brute-force nearest neighbour of the query"


def test_query_graph_rejects_a_walk_onto_itself(model) -> None:
    with pytest.raises(ValueError, match="must differ"):
        query_graph(model, 7, 7)


def test_top_k_widens_the_search_without_ever_reporting_the_query_vertex(model) -> None:
    """A vertex query asks for one answer more than it keeps, so dropping the self-hit costs it no result."""
    single = query_graph(model, 7, 211, eps=0.5)
    wide = query_graph(model, 7, 211, eps=0.5, top_k=5)

    assert single.deg_indices.size == 1, "the default reports the search's single best answer"
    assert wide.deg_indices.size == 5, "the widened search keeps five non-self results"
    assert 7 not in wide.deg_indices, "the query vertex is dropped, never reported as an answer"
    assert np.all(np.diff(wide.deg_distances) >= 0), "the results stay in the search's own ranking order"
    np.testing.assert_array_equal(wide.deg_indices[:1], single.deg_indices, "widening only adds to the tail")


def test_top_k_sets_the_answer_count_and_the_route_always_reaches_the_answer(model) -> None:
    """
    `top_k` is the answer count; the route is the search's own, so it always reaches what it answered.

    The search records the route to its top answer as it runs, so widening `top_k` changes only how many
    answers come back — the walk never stalls before the answer the search itself returned. Because `k`
    also bounds the exploration radius, the search's top answer can move between breadths, but whichever
    answer it returns, the recorded route ends on it.
    """
    narrow = query_graph(model, 123, 211, eps=0.1, top_k=1)
    wide = query_graph(model, 123, 211, eps=0.1, top_k=8)

    assert narrow.deg_indices.size == 1 and wide.deg_indices.size == 8, "top_k is the answer count"
    assert narrow.walked and wide.walked, "the search reaches the answer it returned at either breadth"
    assert int(narrow.path[-1]) == int(narrow.deg_indices[0]), "the narrow route ends on its answer"
    assert int(wide.path[-1]) == int(wide.deg_indices[0]), "and so does the wide one"


def test_deg_hit_is_containment_in_the_results_and_not_rank_one(model) -> None:
    """
    The hit marker asks whether the best match is among the results, not whether it leads them.

    The library returns its result list ranked by distance, and the best match is the global optimum,
    so a real search can never place it below rank one — the fixture cannot stage rank two without
    faking the list. The containment rule is therefore pinned on the result object itself.
    """
    base = query_graph(model, 7, 211, eps=0.5)
    distances = np.asarray([0.1, 0.2], dtype=np.float32)

    second = replace(base, deg_indices=np.asarray([999, base.exact], dtype=np.int32), deg_distances=distances)
    absent = replace(base, deg_indices=np.asarray([999, 998], dtype=np.int32), deg_distances=distances)
    empty = replace(base, deg_indices=np.zeros(0, dtype=np.int32), deg_distances=np.zeros(0, dtype=np.float32))

    assert second.deg_hit, "the best match at rank two is still a hit"
    assert not absent.deg_hit, "a result list without the best match is a miss"
    assert not empty.deg_hit, "no results, no hit — and no crash"


def test_a_free_query_with_top_k_keeps_every_answer(model) -> None:
    """A free point is not a vertex, so there is no self-hit to drop: `top_k` answers in, `top_k` out."""
    spot = model.plot_points[7] + np.float32(0.017)

    result = query_graph(model, spot, eps=0.5, top_k=3)

    assert result.target is None, "a free query names no vertex"
    assert result.deg_indices.size == 3, "nothing is dropped, so all three answers are kept"
    assert np.all(np.diff(result.deg_distances) >= 0), "in the search's own ranking order"


def test_describe_covers_the_built_graph(model) -> None:
    lines = describe(model)
    text = "\n".join(lines)

    assert "vertices" in text and "build" in text and "4" in text
    overlap = [line for line in lines if line.startswith("∩")]
    assert [line.split()[1] for line in overlap] == [
        "knng",
        "nsw",
        "delaunay",
        "gabriel",
        "rng",
        "mst",
        "mrng",
        "nsg",
    ], "one line per reference, in the selector's order"
    for line, item in zip(overlap, model.theory.graphs):
        assert f"{item.edges:6d}" in line and f"{item.shared:7d}" in line, "the panel quotes the report"


# --------------------------------------------------------------------------- rendering and UI


def test_colors_and_segments_follow_the_model(model) -> None:
    colors = vertex_colors(model)

    assert colors.shape == (model.num_points, 4)
    assert np.unique(colors, axis=0).shape[0] > 1, "the generating clusters must be distinguishable"
    assert edge_segments(model).shape == (model.edges.shape[0], 2, 2)


def test_static_render_writes_an_image(model, tmp_path) -> None:
    target = render_static(model, tmp_path / "nested" / "graph.png")

    assert target.exists()
    assert target.stat().st_size > 10_000


VIEWER_POINTS = 240  # above the vertex 211 the traversal tests seed as their start node
#: The build the shared window opens with, and the state every test is wound back to.
BUILD_START = ("blobs", VIEWER_POINTS, DEFAULT_K, 3, Metric.FP32_L2)


@pytest.fixture(scope="module")
def window():
    # One window for the whole module: constructing the figure and its widgets costs around half a second,
    # which dwarfs the graph build. Single-threaded, because the parallel DEG build is not bit-reproducible
    # and these tests assert concrete traversal results.
    instance = GraphViewer(partial(build_scene, threads=1), *BUILD_START)
    # Every state change ends in `draw_idle`, and on the Agg backend that renders the whole 1500 × 850
    # window synchronously — thirteen of them made up 88 % of the slowest UI test. No test observes an
    # idle render: the ones that measure pixels call `canvas.draw()` themselves. On the Tk backend the
    # real `draw_idle` only schedules one, so this makes the harness behave like the shipped window.
    instance._fig.canvas.draw_idle = lambda *_args, **_keywords: None
    yield instance
    instance.close()


@pytest.fixture()
def viewer(window: GraphViewer) -> GraphViewer:
    """
    The shared window wound back to its opening state, so every test still starts from the first frame.

    The graph is only rebuilt when a test moved one of the build parameters. Most tests touch nothing but
    the selection, the query and the framing, and for those a redraw of the graph already on screen is
    enough — a rebuild would recompute a distance matrix that no test in that group ever looks at.
    """
    stale = (window.preset, window.num_points, window.k, window.seed, window.metric) != BUILD_START
    window.preset, window.num_points, window.k, window.seed, window.metric = BUILD_START
    window.selected = window.entry = window.hovered = None
    window.free_query = None
    window.query = window.status = None
    window._home = None
    window._drag = None
    window.view = DEG_VIEW
    window.search_metric = BUILD_START[4]
    window.top_k = 1
    # The colour key's switches are the user's, not the report's, so the rewind has to put them back:
    # every graph the current report holds, on, and the same set marked as seen.
    window.overlays = window._held_graphs()
    window._held_seen = set(window.overlays)
    # The numeric fields are Tk-only widgets; off Tk the rewind has only to put the viewer's own state
    # back (done above), and the next `_rebuild` drops the cached knng edges so no stale list survives.
    window._knng_edges = None
    for position, wanted in enumerate(FLAG_START):
        checks = window._checks1 if position < 3 else window._checks2
        local = position if position < 3 else position - 3
        if checks.get_status()[local] != wanted:
            window._toggle_flag(position)  # through the viewer's own path, not the widget's internals
    if stale:
        window._rebuild()  # re-defaults the start node on the rebuilt model
    else:
        # A start node always exists; the released one re-defaults exactly as `_rebuild` would.
        window.entry = window.model.farthest_from_centroid()
        window.redraw()
    return window


def _key(viewer: GraphViewer, key: str) -> KeyEvent:
    """A keyboard event aimed at the viewer's canvas."""
    return KeyEvent("key_press_event", viewer._fig.canvas, key)


def _click(viewer: GraphViewer, ax) -> None:
    """Press and release straight through the canvas registry, exactly as a mouse click arrives."""
    viewer._fig.canvas.draw()
    box = ax.get_window_extent()
    x, y = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    for kind, step in (("button_press_event", 1), ("button_release_event", 2)):
        event = MouseEvent(kind, viewer._fig.canvas, x, y, button=1, step=step)
        viewer._fig.canvas.callbacks.process(kind, event)


def _click_vertex(viewer: GraphViewer, index: int, button: int = 1, dblclick: bool = False) -> None:
    """Clicks directly on a vertex, as the mouse would; `dblclick` marks the second press."""
    viewer._fig.canvas.draw()
    x, y = viewer._ax.transData.transform(viewer.model.points[index])
    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=button, dblclick=dblclick))
    viewer._on_release(MouseEvent("button_release_event", viewer._fig.canvas, x, y, button=button))


def _empty_spot(viewer: GraphViewer) -> tuple[float, float]:
    """
    A display point inside the axes that no vertex is within the pick radius of.

    A grid over the axes is scored by its distance to the nearest vertex and the farthest point taken,
    so the click lands on genuinely empty space on any deterministic cloud — a hand-picked corner could
    sit beside a blob on one preset and not another.
    """
    viewer._fig.canvas.draw()
    box = viewer._ax.get_window_extent()
    grid = np.stack(
        np.meshgrid(
            np.linspace(box.x0 + 20.0, box.x1 - 20.0, 9),
            np.linspace(box.y0 + 20.0, box.y1 - 20.0, 9),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    display = viewer._ax.transData.transform(viewer.model.plot_points)
    nearest = np.linalg.norm(grid[:, None, :] - display[None, :, :], axis=2).min(axis=1)
    assert nearest.max() > PICK_RADIUS_PX, "the cloud must leave a pickable gap somewhere in the axes"
    x, y = grid[int(np.argmax(nearest))]
    return float(x), float(y)


def _click_empty(viewer: GraphViewer, at: tuple[float, float], dblclick: bool = True) -> None:
    """Double-clicks a display point far from every vertex, placing and running a free-space query."""
    x, y = at
    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=1, dblclick=dblclick))
    viewer._on_release(MouseEvent("button_release_event", viewer._fig.canvas, x, y, button=1))


def _zoom_in(viewer: GraphViewer, notches: int = 3, at: tuple[float, float] = (0.0, 0.0)) -> None:
    """Scrolls in on a data point, which is what a user does before panning."""
    viewer._fig.canvas.draw()
    centre = viewer._ax.transData.transform(at)
    for _ in range(notches):
        viewer._on_scroll(MouseEvent("scroll_event", viewer._fig.canvas, centre[0], centre[1], step=1))


def _drag(viewer: GraphViewer, start: tuple[float, float], stop: tuple[float, float]) -> None:
    """Presses, moves and releases through the canvas registry, exactly as a drag arrives."""
    canvas = viewer._fig.canvas
    canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, *start, button=1))
    canvas.callbacks.process("motion_notify_event", MouseEvent("motion_notify_event", canvas, *stop))
    canvas.callbacks.process("button_release_event", MouseEvent("button_release_event", canvas, *stop, button=1))


def _click_key(viewer: GraphViewer, entry: int) -> None:
    """Clicks the centre of one colour-key entry, the way the mouse would reach it."""
    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()
    box = (
        viewer._legend.legend_handles[entry]
        .get_window_extent(renderer)
        .union([viewer._legend.get_texts()[entry].get_window_extent(renderer)])
    )
    event = MouseEvent(
        "button_press_event", viewer._fig.canvas, (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0, button=1
    )
    assert event.inaxes is viewer._legend_ax, "the click has to land in the key's own strip"
    viewer._on_click(event)


def _check_mark_visible(viewer: GraphViewer, index: int) -> bool:
    """
    Whether one check mark would paint. The check is an 'x' marker, drawn by its edge colour, so it is
    hidden only when that point's face *and* edge colour are both 'none'; a pinned edge colour — the
    bug — leaves the stroke on screen however the state reads.
    """
    checks = viewer._checks1 if index < 3 else viewer._checks2
    local = index if index < 3 else index - 3
    face = checks._buttons.get_facecolor()
    edge = checks._buttons.get_edgecolor()
    none = matplotlib.colors.to_rgba("none")
    face = face[local] if len(face) > local else face[0]
    edge = edge[local] if len(edge) > local else edge[0]
    return not (matplotlib.colors.same_color(face, none) and matplotlib.colors.same_color(edge, none))


def _click_check(viewer: GraphViewer, index: int, *, on_label: bool) -> None:
    """
    Clicks one checkbox — its frame, or its nudged text label — through the canvas registry, so the
    real `CheckButtons` handler runs exactly as it does under a mouse rather than only the viewer's.
    """
    viewer._fig.canvas.draw()
    if on_label:
        box = (
            (viewer._checks1 if index < 3 else viewer._checks2)
            .labels[index if index < 3 else index - 3]
            .get_window_extent(viewer._fig.canvas.get_renderer())
        )
        x, y = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    else:
        checks = viewer._checks1 if index < 3 else viewer._checks2
        ax = viewer._check_ax1 if index < 3 else viewer._check_ax2
        x, y = ax.transAxes.transform(checks._buttons.get_offsets()[index if index < 3 else index - 3])
    canvas = viewer._fig.canvas
    canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, x, y, button=1))


def test_viewer_starts_with_the_requested_graph(viewer: GraphViewer) -> None:
    assert viewer.model.num_points == VIEWER_POINTS
    assert viewer.model.k == DEFAULT_K
    assert viewer._edges.get_segments().__len__() == viewer.model.edges.shape[0]


def test_viewer_selecting_a_vertex_draws_only_its_ring(viewer: GraphViewer) -> None:
    """A single-clicked target gets the blue ring and no edge highlight — the edges stay plain."""
    viewer.selected = 7
    viewer.redraw()

    assert viewer._ring.get_offsets().__len__() == 1, "the target ring is drawn"
    assert viewer._hover_focus.get_segments().__len__() == 0, "selecting lights no edges"
    assert "TARGET   7" in viewer._panel_text.get_text()

    viewer.selected = None
    viewer.redraw()
    assert viewer._ring.get_offsets().__len__() == 0, "the ring goes with the target"


def test_viewer_hovering_a_vertex_lights_its_edges_green(viewer: GraphViewer) -> None:
    """Hovering a vertex lights its edges in the green hover colour, reading off the drawn graph."""
    viewer.hovered = 7
    viewer._paint_hover_edges()

    segments = viewer._hover_focus.get_segments()
    assert len(segments) == viewer.model.adjacency[7].size, "the hovered vertex's edges light up"
    assert matplotlib.colors.to_hex(viewer._hover_focus.get_color()) == matplotlib.colors.to_hex(HOVER_EDGE_COLOR)

    viewer.hovered = None
    viewer._paint_hover_edges()
    assert viewer._hover_focus.get_segments().__len__() == 0, "un-hovering clears the green edges"


def test_viewer_overlay_flags_toggle_rendering(viewer: GraphViewer) -> None:
    viewer.show_ids = True
    viewer.redraw()
    assert len(viewer._labels) == viewer.model.num_points

    viewer.show_edges = False
    viewer.redraw()
    assert not viewer._edges.get_visible()


def test_viewer_rebuilds_on_preset_and_size_change(viewer: GraphViewer) -> None:
    viewer._on_preset("moons")
    assert viewer.model.num_points == VIEWER_POINTS
    assert "moons" in viewer._ax.get_title(loc="left")

    viewer._on_vertices(700)
    assert viewer.model.num_points == 700

    viewer._on_k(8)
    assert viewer.model.k == 8
    assert viewer.model.graph.get_edges_per_vertex() == 8


def test_a_rebuild_keeps_the_search_setup(viewer: GraphViewer) -> None:
    """
    Degree, metric, transform and vertex count rebuild the graph but not the search: the target, the
    start node and the open query survive, and the open query re-runs so the panel answers on the
    graph that is now drawn — a fresh result object, the same target.
    """
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    viewer.run_query()

    for change in (
        lambda: viewer._on_k(8),
        lambda: viewer._on_vertices(400),
        lambda: viewer._on_metric("IP"),
        lambda: viewer._on_mips("on"),
    ):
        before = viewer.query
        change()
        assert (viewer.selected, viewer.entry) == (7, 211), "the focus stays through the rebuild"
        assert viewer.query is not None and viewer.query is not before, "the open query re-ran on the new graph"
        assert viewer.query.target == 7, "and answers the same target"

    viewer._on_mips("off")  # the shared window must not carry the transform into the next test


def test_a_preset_change_starts_the_search_over(viewer: GraphViewer) -> None:
    """
    A new distribution is a different cloud, so it is the one rebuild that starts from scratch: the
    target, the free-space query and the open query are all released, and the start node re-defaults
    to the farthest vertex of the new cloud.
    """
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    viewer.run_query()

    viewer._on_preset("moons")

    assert (viewer.selected, viewer.free_query, viewer.query) == (None, None, None)
    assert viewer.entry == viewer.model.farthest_from_centroid(), "the start node re-defaults on the new cloud"

    _click_empty(viewer, _empty_spot(viewer))
    viewer.run_query()
    assert viewer.query is not None

    viewer._on_preset("spiral")

    assert (viewer.free_query, viewer.query) == (None, None), "a free query goes with the distribution too"


def test_shrinking_past_the_selection_releases_it_and_keeps_the_start_node(viewer: GraphViewer) -> None:
    """
    A target past the shrunken cloud's end is released rather than crashing the repaint, and its query
    goes with it; a start node still inside the cloud keeps its place.
    """
    viewer.select(200)
    viewer.set_entry(42)
    viewer._on_eps(0.5)
    viewer.run_query()
    assert viewer.query is not None

    viewer._on_vertices(100)

    assert viewer.selected is None, "a target outside the new cloud is dropped, not crashed on"
    assert viewer.query is None, "the query went with its target"
    assert viewer.entry == 42, "a start node still inside the cloud keeps its place"


def test_a_free_query_survives_a_rebuild_and_its_open_query_re_runs(viewer: GraphViewer) -> None:
    """The clicked coordinate is not a vertex index, so a rebuild neither drops nor re-picks it."""
    _click_empty(viewer, _empty_spot(viewer))
    viewer._on_eps(0.5)
    first = viewer.run_query()
    spot = viewer.free_query.copy()

    viewer._on_k(8)

    np.testing.assert_array_equal(viewer.free_query, spot, "the clicked coordinate is kept")
    assert viewer.query is not None and viewer.query is not first, "the open query re-ran on the new graph"
    assert viewer.query.target is None
    np.testing.assert_array_equal(viewer.query.point, spot, "and answers the very coordinate")


def test_viewer_reseed_changes_the_point_cloud(viewer: GraphViewer) -> None:
    before = viewer.model.points.copy()
    viewer.reseed()

    assert viewer.seed != 3
    assert not np.array_equal(before, viewer.model.points)


def test_viewer_query_walks_from_the_entry_to_the_selection(viewer: GraphViewer) -> None:
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)

    result = viewer.run_query()

    assert result is not None and viewer.query is result
    assert result.target == 7 and result.walked
    assert result.path[0] == 211, "the traversal must start at the seeded entry"
    assert viewer._path.get_segments().__len__() == result.hops
    assert viewer._entry_mark.get_offsets().__len__() == 1
    assert "START" in viewer._panel_text.get_text()
    assert "QUERY    L2  eps=0.50" in viewer._panel_text.get_text(), "the panel names the comparator"

    viewer.query = None
    viewer.redraw()
    assert viewer._path.get_segments().__len__() == 0


def test_the_route_is_recorded_and_a_wider_top_k_widens_the_answers(viewer: GraphViewer) -> None:
    """
    The search's recorded route is drawn, and widening `top_k` widens the answer list.

    There is no stalled walk to rescue any more: the route is the search's own trackback to its top
    answer, so it ends on that answer at every breadth. Pinned on the window's own cloud — entry 42 to
    target 7 — the narrow search returns one answer and the route ends on it; the wide search returns
    eight and the route still ends on the top one.
    """
    viewer.select(7)
    viewer.set_entry(42)
    viewer._on_eps(0.5)

    viewer.top_k = 1
    narrow = viewer.run_query()
    assert narrow is not None and narrow.walked, "the search reaches the answer it returned"
    assert int(narrow.path[-1]) == int(narrow.deg_indices[0]), "the route ends on that answer"
    assert viewer._path.get_segments().__len__() == narrow.hops, "and every hop is drawn"

    viewer.top_k = 8
    wide = viewer.run_query()
    assert wide is not None and wide.deg_indices.size == 8, "widening widens the answer list"
    assert wide.walked and int(wide.path[-1]) == int(wide.deg_indices[0]), "the route still ends on the top answer"


def test_the_panel_reports_the_route_within_the_column(viewer: GraphViewer) -> None:
    """The route line names the walk's length and its cost, and stays inside the panel column."""
    viewer.select(7)
    viewer.set_entry(42)
    viewer._on_eps(0.5)
    result = viewer.run_query()

    body = viewer._panel_text.get_text()

    assert f"  route    {result.hops} hops · {result.distance_computations} dist · {result.expanded} checked" in body
    assert all(len(line) <= PANEL_WIDTH for line in body.split("\n")), "the route line stays in the column"


def test_a_search_miss_marks_the_answer_in_red_and_connects_it_to_the_star(viewer: GraphViewer) -> None:
    """
    When the search answers elsewhere, the X marks its answer in red and a dashed line runs to the star.

    Pinned on the window's own cloud: vertex 6 at the opening eps and breadth has best match 43, but the
    search returns 112 — a genuine miss. The traversal is aimed at the answer rather than the truth and
    walks all the way to 112, so the X lands there in red, the star stays on 43, and the connector
    measures the distance between the two.
    """
    viewer.select(6)

    result = viewer.run_query()

    assert result is not None and result.walked, "the traversal reaches the search's answer"
    assert result.exact == 43 and int(result.deg_indices[0]) == 112, "the search misses: answer 112, truth 43"
    assert not result.deg_hit, "the best match is not among the answers"
    np.testing.assert_allclose(
        viewer._terminal_mark.get_facecolor()[0], matplotlib.colors.to_rgba(TERMINAL_COLOR), atol=1e-6
    )
    np.testing.assert_allclose(viewer._terminal_mark.get_offsets()[0], viewer.model.plot_points[112], atol=1e-6)
    np.testing.assert_allclose(viewer._exact_mark.get_offsets()[0], viewer.model.plot_points[43], atol=1e-6)
    assert viewer._connector.get_visible(), "the dashed line measures the search's miss"
    np.testing.assert_allclose(viewer._connector.get_xdata(), viewer.model.plot_points[[112, 43], 0], atol=1e-6)
    np.testing.assert_allclose(viewer._connector.get_ydata(), viewer.model.plot_points[[112, 43], 1], atol=1e-6)
    body = viewer._panel_text.get_text()
    assert f"  route    {result.hops} hops" in body, "the panel reports the route the search recorded"
    assert "  search   #1 112" in body, "and the search block names the answer it returned"


def test_a_correct_answer_marks_the_star_in_green_and_hides_the_connector(viewer: GraphViewer) -> None:
    """When the search's top answer *is* the best match, the X lands green on the star and nothing connects."""
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)

    result = viewer.run_query()

    assert result is not None and result.walked
    assert int(result.deg_indices[0]) == result.exact == 199, "the search recovers the best match"
    np.testing.assert_allclose(
        viewer._terminal_mark.get_facecolor()[0], matplotlib.colors.to_rgba(EXACT_FILL), atol=1e-6
    )
    np.testing.assert_allclose(
        viewer._terminal_mark.get_edgecolor()[0], matplotlib.colors.to_rgba(MARKER_EDGE), atol=1e-6
    )
    np.testing.assert_allclose(viewer._terminal_mark.get_offsets()[0], viewer._exact_mark.get_offsets()[0], atol=1e-6)
    assert not viewer._connector.get_visible(), "a correct answer leaves no miss to measure"


def test_a_search_without_an_answer_leaves_no_x_and_no_path(viewer: GraphViewer) -> None:
    """
    A search that returned nothing has no answer to aim the traversal at: the X, the path and the
    connector all go, and only the star of the ground truth stays. The library answers a reachable
    cloud every time, so the empty result list is staged on the result object — the very state
    `query_graph` carries when the search comes back empty.
    """
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    result = viewer.run_query()
    viewer.query = replace(
        result,
        path=np.zeros(0, dtype=np.int32),
        deg_indices=np.zeros(0, dtype=np.int32),
        deg_distances=np.zeros(0, dtype=np.float32),
    )

    viewer.redraw()

    assert not viewer.query.walked, "nothing to walk to, nothing walked"
    assert viewer._terminal_mark.get_offsets().__len__() == 0, "no X without an answer"
    assert viewer._path.get_segments().__len__() == 0, "and no path"
    assert not viewer._connector.get_visible(), "nor a connector"
    assert viewer._exact_mark.get_offsets().__len__() == 1, "the ground truth keeps its star"


def test_viewer_reports_a_query_without_a_target(viewer: GraphViewer) -> None:
    assert viewer.run_query() is None

    assert viewer.query is None
    assert "no target" in viewer._panel_text.get_text()


def test_viewer_reports_a_walk_onto_itself(viewer: GraphViewer) -> None:
    viewer.select(7)
    viewer.set_entry(7)

    assert viewer.run_query() is None
    assert "must differ" in viewer._panel_text.get_text()


def test_viewer_sets_the_start_node_by_right_click(viewer: GraphViewer) -> None:
    entry, target = 211, 7
    viewer._on_eps(0.5)

    _click_vertex(viewer, entry, button=3)
    assert viewer.entry == entry, "right-click makes the vertex the traversal start node"
    assert viewer._entry_mark.get_offsets().__len__() == 1
    assert "START" in viewer._panel_text.get_text()

    _click_vertex(viewer, target)
    assert viewer.selected == target, "a left-click selects without moving the start node"
    assert viewer.entry == entry

    result = viewer.run_query()
    assert result is not None, "a start node and a target make a valid query"
    assert result.walked, "at eps 0.5 the traversal reaches the search's answer"
    assert (result.entry, result.target, result.hops) == (entry, target, result.path.size - 1)
    assert int(result.path[-1]) == int(result.deg_indices[0]), "the walk ends on the vertex the search answered"

    empty = viewer._ax.transAxes.transform((0.01, 0.01))
    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, empty[0], empty[1], button=3))
    assert viewer.entry == viewer.model.farthest_from_centroid(), (
        "right-clicking empty space re-defaults the start node — one always exists"
    )


def test_viewer_double_click_runs_the_query(viewer: GraphViewer) -> None:
    viewer._on_eps(0.5)
    _click_vertex(viewer, 211, button=3)

    _click_vertex(viewer, 7)
    assert viewer.selected == 7, "a single click selects"
    assert viewer.query is None, "and a single click must not search"

    _click_vertex(viewer, 7, dblclick=True)
    assert viewer.query is not None, "a double-click runs the query at once"
    assert viewer.free_query is not None, "the double-click sets the orange query cross on the vertex"
    assert viewer.selected is None, "the double-click queries the point, releasing the vertex selection"
    np.testing.assert_array_equal(viewer.free_query, viewer.model.plot_points[7], "the cross sits on the vertex")
    assert (viewer.query.entry) == 211
    assert viewer._path.get_segments().__len__() == viewer.query.hops


def test_a_free_query_draws_a_dashed_line_to_the_star(viewer: GraphViewer) -> None:
    """The free query's cross is joined by a dashed line running to the star — the ground-truth best match."""
    viewer.set_free_query(np.array([0.5, 0.5], dtype=np.float32))
    viewer.run_query()
    viewer.redraw()
    star = viewer.query.exact

    assert viewer._free_line.get_visible(), "the dashed tail is drawn for an open free query"
    np.testing.assert_allclose(viewer._free_line.get_xdata(), [viewer.free_query[0], viewer.model.plot_points[star, 0]], atol=1e-6)
    np.testing.assert_allclose(viewer._free_line.get_ydata(), [viewer.free_query[1], viewer.model.plot_points[star, 1]], atol=1e-6)

    viewer.set_free_query(None)
    viewer.redraw()

    assert not viewer._free_line.get_visible(), "releasing the free query hides the tail"


def test_clicking_empty_space_places_a_free_query(viewer: GraphViewer) -> None:
    """A click with no vertex under the cursor queries the clicked coordinates, not the nearest vertex."""
    viewer.select(42)

    _click_empty(viewer, _empty_spot(viewer))

    assert viewer.free_query is not None, "empty space becomes the query point"
    assert viewer.selected is None, "placing a free query releases the vertex selection"
    assert viewer._free_mark.get_offsets().__len__() == 1, "the cross marks the clicked coordinate"

    _click_vertex(viewer, 42)

    assert viewer.selected == 42
    assert viewer.free_query is not None, "a vertex click leaves the open query in place"
    assert viewer._free_mark.get_offsets().__len__() == 1, "and its cross stays with the query it marks"

    _click_vertex(viewer, 42)
    assert viewer.selected is None, "clicking the selected vertex again deselects it in 2D"


def test_the_none_view_hides_the_free_query_cross(viewer: GraphViewer) -> None:
    """A query marker on a screen without a graph points at a search that cannot run — it hides with it."""
    _click_empty(viewer, _empty_spot(viewer))
    assert viewer._free_mark.get_offsets().__len__() == 1

    viewer._on_view(NONE_VIEW)
    assert viewer._free_mark.get_offsets().__len__() == 0, "the cross hides with the graph"

    viewer._on_view(DEG_VIEW)
    assert viewer._free_mark.get_offsets().__len__() == 1, "and returns with it — the point stays chosen"


def test_arrow_keys_in_an_open_dropdown_choose_without_a_click() -> None:
    """The popdown wiring applies the entry the cursor stands on, and the value stays applied."""
    root = tkinter.Tk()
    try:
        variable = tkinter.StringVar(root, value="a")
        combo = ttk.Combobox(root, textvariable=variable, values=["a", "b", "c"], state="readonly")
        chosen: list[str] = []
        GraphViewer._wire_dropdown(None, combo, chosen.append)
        combo.pack()
        root.update()
        combo.event_generate("<Button-1>", x=5, y=5)  # the opening press also schedules the wiring
        root.update()
        listbox = f"{combo}.popdown.f.l"
        combo.tk.call("focus", listbox)
        combo.tk.call("event", "generate", listbox, "<KeyPress>", "-keysym", "Down")
        combo.tk.call("event", "generate", listbox, "<KeyRelease>", "-keysym", "Down")
        root.update()
        assert chosen == ["b"], "the arrow-key selection is applied at once, without a click"
        assert combo.get() == "b", "and the dropdown shows it, so closing the popup keeps the choice"
    finally:
        root.destroy()


def test_a_free_query_is_searched_as_itself(viewer: GraphViewer) -> None:
    """The clicked coordinate is the query vector; the answer is the best match over the whole cloud."""
    _click_empty(viewer, _empty_spot(viewer))
    viewer._on_eps(0.5)

    result = viewer.run_query()

    assert result is not None and result.target is None, "a free query names no vertex"
    np.testing.assert_array_equal(result.point, viewer.free_query, "the query vector is the clicked coordinate")
    np.testing.assert_array_equal(result.query_xy, viewer.free_query, "the plot coordinates travel with the result")
    delta = viewer.model.points - result.point
    squared = np.einsum("ij,ij->i", delta, delta)
    assert result.exact == int(np.argmin(squared)), "a free point is not a vertex, so nothing is excluded"
    assert result.walked and int(result.path[-1]) == int(result.deg_indices[0]), (
        "the walk ends on the vertex the search answered"
    )


def test_double_click_on_empty_space_runs_the_free_query(viewer: GraphViewer) -> None:
    viewer._on_eps(0.5)
    spot = _empty_spot(viewer)

    _click_empty(viewer, spot, dblclick=False)
    assert viewer.free_query is None, "a single click on empty space places nothing"

    _click_empty(viewer, spot, dblclick=True)
    assert viewer.free_query is not None, "a double-click places the query"
    assert viewer.query is not None and viewer.query.target is None, "and runs it at once"


def test_escape_clears_the_free_query(viewer: GraphViewer) -> None:
    _click_empty(viewer, _empty_spot(viewer))
    assert viewer.free_query is not None

    viewer._on_key(_key(viewer, "escape"))

    assert viewer.free_query is None, "Esc releases the free query as it does the selection"
    assert viewer._free_mark.get_offsets().__len__() == 0, "and its cross with it"


def test_the_panel_names_the_free_query_point(viewer: GraphViewer) -> None:
    _click_empty(viewer, _empty_spot(viewer))

    body = viewer._panel_text.get_text()
    x, y = viewer.free_query

    assert f"QUERYPT  ({x:.3f}, {y:.3f})" in body, "the panel names the clicked coordinate"
    assert "TARGET" not in body, "a free point has no vertex block and no neighbours to list"
    assert all(len(line) <= PANEL_WIDTH for line in body.split("\n")), "the line stays inside the column"


def test_viewer_buttons_react_to_clicks(viewer: GraphViewer) -> None:
    before = viewer.model.points.copy()
    _click(viewer, viewer._reseed_ax)
    assert not np.array_equal(before, viewer.model.points), "the reseed button must rebuild the cloud"

    viewer.select(7)
    _click(viewer, viewer._check_ax1)
    assert viewer.selected == 7, "a click on a widget must not read as empty space"


def test_the_k_field_snaps_odd_to_even_for_the_deg(viewer: GraphViewer) -> None:
    """
    The k field bounds follow the graph: the DEG takes even degrees only, the knng any integer.

    An odd value beside the DEG snaps up to the next even and clamps into 4..16; beside the knng the
    field keeps the raw value while the DEG still builds at the snapped even degree; switching back to
    the DEG snaps the field to that even value. Garbage reverts to the stored value.
    """
    viewer._commit_k("5")
    assert viewer.k == 6 and viewer.model.k == 6, "an odd degree beside the DEG snaps up to the next even"

    viewer._commit_k("3")
    assert viewer.k == MIN_K and viewer.model.k == MIN_K, "a value below the minimum clamps up to it"

    viewer._commit_k("99")
    assert viewer.k == K_LIMIT and viewer.model.k == K_LIMIT, "a value past the cap clamps to it"

    viewer._commit_k("half-typed")
    assert viewer.k == K_LIMIT, "garbage leaves the stored value where it was"

    viewer._on_view(KNNG_VIEW)
    viewer._commit_k("5")
    assert viewer.k == 5, "beside the knng the field keeps the raw integer"
    assert viewer.model.k == 6, "while the DEG still builds at the snapped even degree"

    viewer._on_view(DEG_VIEW)
    assert viewer.k == 6, "switching back to the DEG snaps the field to the even value"


def test_the_k_field_follows_the_view_it_lands_on(viewer: GraphViewer) -> None:
    """The field greys for graphs without a k and wakes for the ones that read it — chosen or previewed."""

    class _SpinStub:
        state = "normal"
        increment = 1

        def configure(self, state: str, increment: int = 1) -> None:
            self.state = state
            self.increment = increment

    viewer._k_spin = _SpinStub()
    try:
        viewer._on_view("mst")
        assert viewer._k_spin.state == "disabled", "the mst has no k to edit"

        viewer._on_view(KNNG_VIEW)
        assert viewer._k_spin.state == "normal", "the knng reads the field, so it must be editable"

        viewer._on_view(NSW_VIEW)
        assert viewer._k_spin.state == "normal", "the NSW reads the field too"

        viewer._on_view("rng")
        assert viewer._k_spin.state == "disabled", "the rng reads no k either"
    finally:
        viewer._k_spin = None


def test_the_k_field_steps_by_two_only_for_the_deg(viewer: GraphViewer) -> None:
    """The DEG builds at an even degree, so its k arrows step by 2; the knng and NSW read k raw and step by 1."""

    class _SpinStub:
        state = "normal"
        increment = 1

        def configure(self, state: str, increment: int = 1) -> None:
            self.state = state
            self.increment = increment

    viewer._k_spin = _SpinStub()
    try:
        viewer._on_view(DEG_VIEW)
        assert viewer._k_spin.increment == 2, "the DEG snaps k to even, so a step of 1 would leave the down arrow inert"

        viewer._on_view(KNNG_VIEW)
        assert viewer._k_spin.increment == 1, "the knng reads the raw neighbour count, so it steps by 1"

        viewer._on_view(NSW_VIEW)
        assert viewer._k_spin.increment == 1, "the NSW reads the raw k too"
    finally:
        viewer._k_spin = None


def test_viewer_picks_the_vertex_under_the_cursor(viewer: GraphViewer) -> None:
    viewer._fig.canvas.draw()
    target = 42
    x, y = viewer._ax.transData.transform(viewer.model.points[target])

    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, x, y))
    assert viewer.hovered == target
    assert "HOVER    42" in viewer._panel_text.get_text()

    _click_vertex(viewer, target)
    assert viewer.selected == target, "a released left-click selects"

    away = viewer._ax.transData.transform([-50.0, -50.0])
    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, away[0], away[1]))
    assert viewer.hovered is None


def test_the_panel_heading_names_the_graph_on_screen(viewer: GraphViewer) -> None:
    """The heading follows the view — a reference graph never wears the DEG's numbers as its own."""
    assert viewer._panel_text.get_text().startswith("DEG"), "the DEG view opens the column with its name"

    viewer._on_view("mst")
    panel = viewer._panel_text.get_text()
    assert panel.startswith("mst"), "the reference graph heads the column with its own name"
    assert "of DEG" in panel, "and states its share of the DEG below it"

    viewer._on_view(KNNG_VIEW)
    assert viewer._panel_text.get_text().startswith("knng"), "the knng heads with its own name"

    viewer._on_view(NONE_VIEW)
    assert viewer._panel_text.get_text().startswith("no graph"), "the empty view says so in the heading"


def test_the_mrng_view_is_offered_undirected_and_timed(viewer: GraphViewer) -> None:
    """The undirected MRNG is a reference graph: offered under every metric, reads no k, states its DEG share."""
    assert "mrng" in viewer.views, "the undirected MRNG is offered as a reference graph"

    class _SpinStub:
        state = "normal"
        increment = 1

        def configure(self, state: str, increment: int = 1) -> None:
            self.state = state
            self.increment = increment

    viewer._k_spin = _SpinStub()
    try:
        viewer._on_view("mrng")
        assert viewer._k_spin.state == "disabled", "the MRNG reads no k, so the field greys out"
    finally:
        viewer._k_spin = None

    panel = viewer._panel_text.get_text()
    assert panel.startswith("mrng"), "the column heads with the MRNG's name"
    assert "of DEG" in panel and "ms" in panel, "the sub-line states the DEG share and the build time"

    drawn = {tuple(sorted((int(u), int(v)))) for u, v in viewer._display_edges()}
    library = {
        tuple(sorted((int(u), int(v)))) for u, v in mrng_edges(dissimilarities(viewer.model.points, viewer.metric))
    }
    symmetrised = {
        tuple(sorted((int(u), int(v)))) for u, v in nsg_edges(dissimilarities(viewer.model.points, viewer.metric))
    }
    assert drawn == library, "the view draws exactly the library's reduced MRNG"
    assert drawn <= symmetrised, "and that graph is a subgraph of the symmetrised NSG graph"


def test_the_nsg_view_is_offered_directed_and_scored_in_the_report(viewer: GraphViewer) -> None:
    """
    The NSG is a viewer-built view like the knng and the NSW: offered under every metric, drawn directed,
    reads no k. It carries a report entry, so the panel scores it against the DEG — yet it stays out of
    the colour key, so its own view states its edge count and time, never a DEG share.
    """
    assert NSG_VIEW in viewer.views, "the NSG is offered in the selector"
    assert NSG_VIEW in viewer.model.theory.names, "the NSG is scored against the DEG in the report"

    class _SpinStub:
        state = "normal"
        increment = 1

        def configure(self, state: str, increment: int = 1) -> None:
            self.state = state
            self.increment = increment

    viewer._k_spin = _SpinStub()
    try:
        viewer._on_view(NSG_VIEW)
        assert viewer._k_spin.state == "disabled", "the NSG reads no k, so the field greys out"
    finally:
        viewer._k_spin = None

    assert viewer._view_directed(), "the NSG is drawn directed"
    panel = viewer._panel_text.get_text()
    assert panel.startswith("nsg"), "the column heads with the NSG's name"
    assert "of DEG" not in panel, "the NSG view states its own edge count and time, not a DEG share"

    drawn = {(int(u), int(v)) for u, v in viewer._display_edges()}
    library = {(int(u), int(v)) for u, v in nsg_edges(dissimilarities(viewer.model.points, viewer.metric))}
    assert drawn == library, "the view draws exactly the library's directed NSG edges"


def test_a_reference_view_reports_its_construction_time(viewer: GraphViewer) -> None:
    """Selecting a reference graph from the dropdown shows how long it took to build, beside its edge count."""
    viewer._on_view("mst")

    panel = viewer._panel_text.get_text()

    assert "ms" in panel, "the reference sub-line reports the build time the comparison measured"


def test_a_free_query_keeps_its_line_while_hovering(viewer: GraphViewer) -> None:
    """The query point is the user's choice; a hover reads out another vertex beside it, not instead."""
    viewer._fig.canvas.draw()
    _click_empty(viewer, _empty_spot(viewer))
    assert "QUERYPT" in viewer._panel_text.get_text()

    x, y = viewer._ax.transData.transform(viewer.model.points[42])
    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, x, y))
    panel = viewer._panel_text.get_text()
    assert "HOVER    42" in panel and "QUERYPT" in panel, "both lines stand while the mouse reads out"


def test_the_panel_styles_its_blocks(viewer: GraphViewer) -> None:
    """The heading is bold and larger, block labels carry their own colour — hierarchy by type."""
    viewer.select(42)
    viewer.run_query()
    artists = viewer._panel_text._artists
    assert artists[0].get_fontweight() == "bold", "the heading is bold"
    assert artists[0].get_fontsize() > artists[-1].get_fontsize(), "and larger than the body"
    label = next(a for a in artists if a.get_text().startswith("QUERY"))
    assert label.get_color() == PANEL_STYLES["label"]["color"], "block labels are coloured"


def test_the_panel_grades_the_search_on_a_truth_line(viewer: GraphViewer) -> None:
    """The line under the results names the brute-force best match and the verdict the colour shows."""
    viewer.select(42)
    result = viewer.run_query()
    verdict = "hit" if result.deg_hit else "missed"
    assert f"  truth    {result.exact} · {verdict}" in viewer._panel_text.get_text()


def test_the_hover_closes_the_column_without_shifting_it(viewer: GraphViewer) -> None:
    """Everything the user set keeps its line; the hover — the only block that comes and goes — sits last."""
    viewer._fig.canvas.draw()
    _click_empty(viewer, _empty_spot(viewer))
    viewer.run_query()
    before = viewer._panel_text.get_text().split("\n")

    x, y = viewer._ax.transData.transform(viewer.model.points[42])
    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, x, y))
    after = viewer._panel_text.get_text().split("\n")

    assert after[: len(before)] == before, "no block above the hover moves when it appears"
    assert any(line.startswith("HOVER") for line in after[len(before) :]), "the hover reads out at the end"


def test_viewer_zooms_towards_the_cursor(viewer: GraphViewer) -> None:
    viewer._fig.canvas.draw()
    target = 42
    x, y = viewer._ax.transData.transform(viewer.model.points[target])
    home_span = viewer._home[0][1] - viewer._home[0][0]

    viewer._on_scroll(MouseEvent("scroll_event", viewer._fig.canvas, x, y, step=1))
    viewer._fig.canvas.draw()

    span = viewer._ax.get_xlim()[1] - viewer._ax.get_xlim()[0]
    assert span < home_span, "scrolling up zooms in"
    landed = viewer._ax.transData.transform(viewer.model.points[target])
    assert np.allclose(landed, (x, y), atol=1.0), "the vertex under the cursor must stay put"


def test_viewer_never_zooms_out_past_the_initial_framing(viewer: GraphViewer) -> None:
    viewer._fig.canvas.draw()
    home = viewer._home
    center = viewer._ax.transData.transform(viewer.model.points.mean(axis=0))

    for _ in range(6):
        viewer._on_scroll(MouseEvent("scroll_event", viewer._fig.canvas, center[0], center[1], step=-1))

    assert viewer._ax.get_xlim() == pytest.approx(home[0]), "zooming out stops at the initial framing"

    viewer._on_scroll(MouseEvent("scroll_event", viewer._fig.canvas, center[0], center[1], step=1))
    narrowed = viewer._ax.get_xlim()[1] - viewer._ax.get_xlim()[0]
    assert narrowed < home[0][1] - home[0][0], "scrolling in narrows the view"

    viewer._on_key(_key(viewer, "0"))
    assert viewer._ax.get_ylim() == pytest.approx(home[1]), "0 restores the initial framing"


def test_dragging_pans_the_zoomed_view_with_the_cursor(viewer: GraphViewer) -> None:
    _zoom_in(viewer)
    before = viewer._ax.get_xlim()
    centre = viewer._ax.transData.transform([0.0, 0.0])

    _drag(viewer, centre, (centre[0] + 90.0, centre[1]))

    after = viewer._ax.get_xlim()
    assert after[1] - after[0] == pytest.approx(before[1] - before[0]), "a pan slides the framing, never resizes it"
    assert after[0] < before[0], "dragging right pulls the window left so the cloud follows the cursor"
    assert viewer._drag is None, "releasing ends the drag"


def test_a_vertex_travels_the_distance_the_cursor_moved(viewer: GraphViewer) -> None:
    _zoom_in(viewer)
    centre = viewer._ax.transData.transform([0.0, 0.0])
    before = viewer._ax.transData.transform(viewer.model.points[42])

    _drag(viewer, centre, (centre[0] + 60.0, centre[1] + 30.0))
    viewer._fig.canvas.draw()

    after = viewer._ax.transData.transform(viewer.model.points[42])
    assert np.allclose(after - before, (60.0, 30.0), atol=1.0), "the cloud must keep up with the cursor"


def test_dragging_at_the_initial_framing_does_not_move_the_cloud(viewer: GraphViewer) -> None:
    viewer._fig.canvas.draw()
    before = viewer._ax.get_xlim()
    centre = viewer._ax.transData.transform([0.0, 0.0])

    _drag(viewer, centre, (centre[0] + 200.0, centre[1]))

    assert viewer._ax.get_xlim() == pytest.approx(before), "there is nothing to pan before the view is zoomed in"


def test_dragging_cannot_leave_the_initial_box(viewer: GraphViewer) -> None:
    _zoom_in(viewer)
    home = viewer._home
    span = viewer._ax.get_xlim()[1] - viewer._ax.get_xlim()[0]
    centre = viewer._ax.transData.transform([0.0, 0.0])

    _drag(viewer, centre, (centre[0] + 4000.0, centre[1] + 4000.0))

    x0, x1 = viewer._ax.get_xlim()
    y0, y1 = viewer._ax.get_ylim()
    assert x0 >= home[0][0] - 1e-9 and x1 <= home[0][1] + 1e-9, "the view never slides out of the cloud's box"
    assert y0 >= home[1][0] - 1e-9 and y1 <= home[1][1] + 1e-9
    assert x1 - x0 == pytest.approx(span), "nor does a pan that hits the wall change the zoom"


def test_a_released_click_does_not_leave_the_viewer_panning(viewer: GraphViewer) -> None:
    _zoom_in(viewer, at=tuple(viewer.model.points[42]))
    framing = viewer._ax.get_xlim()

    _click_vertex(viewer, 42)
    centre = viewer._ax.transData.transform([0.0, 0.0])
    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, centre[0] + 120.0, centre[1]))

    assert viewer.selected == 42, "the click still selects its vertex"
    assert viewer._ax.get_xlim() == pytest.approx(framing), "and leaves the framing where it was"


def test_the_eps_field_re_runs_the_open_query_only(viewer: GraphViewer) -> None:
    """The eps field widens the search's radius and re-runs the open query; it never rebuilds the graph."""
    viewer.select(7)
    built = viewer.model
    first = viewer.run_query()

    viewer._commit_eps("0.6")

    assert viewer.path_eps == pytest.approx(0.6)
    assert viewer.model is built, "the eps field never rebuilds the graph"
    assert viewer.query is not first, "committing a new factor re-runs the query"
    assert viewer.query.eps == pytest.approx(0.6)
    assert "eps=0.60" in viewer._panel_text.get_text()

    viewer._commit_eps("5")
    assert viewer.path_eps == pytest.approx(5.0), "a factor above the old cap is now accepted"

    viewer._commit_eps("50")
    assert viewer.path_eps == pytest.approx(10.0), "a value past the field's cap is clamped to it"

    viewer._commit_eps("nope")
    assert viewer.path_eps == pytest.approx(10.0), "garbage leaves the stored factor where it was"


def test_the_top_k_field_re_runs_the_open_query(viewer: GraphViewer) -> None:
    """Widening the result list changes only the search, so the open query re-runs on the same graph."""
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    first = viewer.run_query()

    assert first.deg_indices.size == 1, "the window opens asking the search for one answer"

    viewer._on_top_k(5)

    assert viewer.top_k == 5
    assert viewer.query is not first, "widening the result list re-runs the search"
    assert viewer.query.deg_indices.size == 5, "and the open query comes back with the wider list"
    body = viewer._panel_text.get_text()
    assert "  search   #1" in body, "the first result keeps the block's label"
    assert "           #5" in body, "and the panel lists every rank behind blank padding"

    before = viewer.query
    viewer._on_top_k(5)
    assert viewer.query is before, "the same value twice is a no-op"


def test_the_committed_top_k_is_clamped_and_garbage_reverts(viewer: GraphViewer) -> None:
    """The field is free text between the arrows, so a committed value is clamped and garbage reverts."""
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    viewer.run_query()

    viewer._commit_top_k("99")
    assert viewer.top_k == TOP_K_MAX, "a request past the field's cap is clamped to it"

    viewer._commit_top_k("half-typed")
    assert viewer.top_k == TOP_K_MAX, "garbage leaves the stored value where it was"

    viewer._commit_top_k("3")
    assert viewer.top_k == 3 and viewer.query.deg_indices.size == 3, "a clean value re-runs the search"


def test_the_panel_says_no_result_when_the_search_returns_nothing(viewer: GraphViewer) -> None:
    """An empty result list is a state the panel has to name, not a line it has to crash on."""
    viewer.select(7)
    viewer.set_entry(211)
    viewer._on_eps(0.5)
    result = viewer.run_query()
    viewer.query = replace(result, deg_indices=np.zeros(0, dtype=np.int32), deg_distances=np.zeros(0, dtype=np.float32))

    viewer.redraw()

    assert "  search   no result" in viewer._panel_text.get_text()
    assert viewer.query.deg_hit is False, "and the hit marker reads the empty list as a miss"


# --------------------------------------------------------------------------- theoretical graphs


def test_rng_on_a_line_keeps_only_the_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = {tuple(sorted(edge)) for edge in rng_edges(distances)}

    assert edges == {(0, 1), (1, 2), (2, 3)}, "a point between two others must break their edge"


def test_nsg_on_a_line_keeps_only_the_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = {tuple(sorted(edge)) for edge in nsg_edges(distances)}

    assert edges == {(0, 1), (1, 2), (2, 3)}


def test_mrng_on_a_line_keeps_only_the_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = {tuple(sorted(edge)) for edge in mrng_edges(distances)}

    assert edges == {(0, 1), (1, 2), (2, 3)}, (
        "a chain edge has no monotone bypass in either direction, so none is dropped"
    )


def test_mrng_is_monotone_minimal_and_a_subgraph_of_the_symmetrised_nsg() -> None:
    """The undirected MRNG keeps only symmetrised NSG edges, and no kept edge is redundant for monotone routing."""
    points, _ = make_points("uniform", 300, seed=7)
    distances = dissimilarities(points, Metric.FP32_L2)

    symmetrised = {tuple(sorted((int(u), int(v)))) for u, v in nsg_edges(distances)}
    mrng = {tuple(sorted((int(u), int(v)))) for u, v in mrng_edges(distances)}

    assert mrng <= symmetrised, "every MRNG edge is a symmetrised NSG edge"

    columns = distances.tolist()
    adjacency: list[set[int]] = [set() for _ in range(points.shape[0])]
    for u, v in mrng:
        adjacency[u].add(v)
        adjacency[v].add(u)
    for u, v in mrng:
        adjacency[u].discard(v)
        adjacency[v].discard(u)
        assert not (
            _monotone_reachable(adjacency, u, v, columns[v]) and _monotone_reachable(adjacency, v, u, columns[u])
        ), "a kept edge must lack an alternative monotone path in at least one direction, else it was removable"
        adjacency[u].add(v)
        adjacency[v].add(u)


def test_gabriel_on_a_line_keeps_only_the_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = {tuple(sorted(edge)) for edge in gabriel_edges(distances)}

    assert edges == {(0, 1), (1, 2), (2, 3)}, "the disc over neighbours covers the point between them"


def test_mst_on_a_line_takes_the_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = {tuple(sorted(edge)) for edge in mst_edges(distances)}

    assert edges == {(0, 1), (1, 2), (2, 3)}, "the lightest way to span a line is the line itself"


def test_mst_spans_every_vertex_without_a_triangulation() -> None:
    points, _ = make_points("blobs", 120, seed=9)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    edges = mst_edges(distances)

    assert edges.shape == (119, 2), "a tree on n vertices has n-1 edges"
    assert {int(i) for i in edges[:, 0]} | {int(i) for i in edges[:, 1]} == set(range(120)), "every vertex is reached"


def test_delaunay_spans_every_point() -> None:
    points, _ = make_points("blobs", 120, seed=9)

    edges = delaunay_edges(points)

    assert {int(i) for i in edges[:, 0]} | {int(i) for i in edges[:, 1]} == set(range(120))


def test_compare_counts_the_deg_edges_each_graph_shares() -> None:
    scene = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)

    report = compare(scene.points, scene.edges)

    undirected = {tuple(sorted(edge)) for edge in scene.edges.reshape(-1, 2)}
    assert report.deg_edges == len(undirected)
    assert tuple(item.name for item in report.graphs) == (
        "knng",
        "nsw",
        "delaunay",
        "gabriel",
        "rng",
        "mst",
        "mrng",
        "nsg",
    )
    assert all(0 < item.shared <= min(report.deg_edges, item.edges) for item in report.graphs)

    by_name = {item.name: item for item in report.graphs}
    assert by_name["mst"].edges < by_name["rng"].edges < by_name["gabriel"].edges < by_name["delaunay"].edges
    assert by_name["mst"].shared <= by_name["rng"].shared <= by_name["gabriel"].shared <= by_name["delaunay"].shared
    assert by_name["mst"].edges == scene.points.shape[0] - 1, "a spanning tree on n vertices has n-1 edges"
    assert report == scene.theory, "the model carries the very same comparison"


def test_compare_records_a_construction_time_for_each_graph() -> None:
    scene = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)

    report = compare(scene.points, scene.edges)

    assert all(item.seconds > 0.0 for item in report.graphs), "every reference graph reports the time it took to build"


def test_the_reference_graphs_nest_as_theory_demands() -> None:
    scene = build_scene("blobs", 200, DEFAULT_K, 3, Metric.FP32_L2, threads=1)
    keys = {name: set(scene.theory.graph(name).keys.tolist()) for name in ("delaunay", "gabriel", "rng", "mst")}

    assert keys["mst"] <= keys["rng"] <= keys["gabriel"] <= keys["delaunay"], "MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay"


def test_share_of_handles_an_empty_deg() -> None:
    empty = GraphOverlap("rng", 10, 4, np.zeros(0, dtype=np.int64))

    assert empty.share_of(0) == 0.0
    assert empty.share_of(8) == pytest.approx(0.5)


def test_membership_points_at_the_graph_holding_each_edge() -> None:
    scene = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)

    codes = scene.theory.membership(OVERLAP_ORDER)

    assert codes.shape == (scene.edges.shape[0],), "one code per DEG edge, in the edge list's own order"
    for code in np.unique(codes):
        if code < 0:
            continue
        keys = scene.theory.deg_keys[codes == code]
        assert np.isin(keys, scene.theory.graphs[int(code)].keys).all(), "the code names a graph holding it"


def test_membership_lets_the_last_named_graph_win() -> None:
    theory = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1).theory
    codes = theory.membership(OVERLAP_ORDER)
    index = {item.name: position for position, item in enumerate(theory.graphs)}

    tree = np.intersect1d(theory.deg_keys, theory.graph("mst").keys, assume_unique=True)
    for name in ("delaunay", "rng", "gabriel"):
        tree = np.intersect1d(tree, theory.graph(name).keys, assume_unique=True)
    assert tree.size > 0, "the chain must share DEG edges for this to say anything"
    assert (codes[np.searchsorted(theory.deg_keys, tree)] == index["mst"]).all(), "the sparsest graph paints last"

    pair = np.intersect1d(theory.deg_keys, theory.graph("gabriel").keys, assume_unique=True)
    pair = np.intersect1d(pair, theory.graph("delaunay").keys, assume_unique=True)
    for later in ("rng", "mrng", "mst"):
        pair = np.setdiff1d(pair, theory.graph(later).keys)
    assert pair.size > 0, "the Gabriel graph must hold DEG edges outside the later-named graphs"
    assert (codes[np.searchsorted(theory.deg_keys, pair)] == index["gabriel"]).all(), (
        "the Gabriel graph outranks every graph named before it"
    )


def test_edge_colours_follow_the_membership() -> None:
    scene = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)

    colours = edge_colours(scene)

    neutral = matplotlib.colors.to_rgba(EDGE_COLOR)
    assert colours.shape == (scene.edges.shape[0], 4), "one colour per edge"
    assert (colours == neutral).any(), "edges no reference graph shares stay neutral"
    assert not (colours == neutral).all(), "the shared ones must stand out"


def test_the_inner_product_matrix_is_the_score_the_library_minimises() -> None:
    """The references have to be read from the number the greedy search itself ranks candidates by."""
    points, _ = make_points("blobs", 60, seed=5)
    points = points.astype(np.float32)

    matrix = dissimilarities(points, Metric.FP32_InnerProduct)

    off_diagonal = ~np.eye(points.shape[0], dtype=bool)
    np.testing.assert_allclose(matrix[off_diagonal], (1.0 - points @ points.T)[off_diagonal], rtol=1e-6, atol=1e-6)
    assert np.isinf(np.diag(matrix)).all(), "a point is never the third point of its own edge"


def test_the_euclidean_graphs_are_unmoved_by_the_infinite_diagonal() -> None:
    """
    The sweeps now read a diagonal of infinity instead of zero, which is what makes them correct for a
    dissimilarity that does not score a point against itself as zero. Under Euclidean distance the two
    matrices have to yield the identical graph, or the change would have moved every number the example
    reports without anyone noticing.
    """
    for preset in PRESETS:
        points, _ = make_points(preset, 120, seed=5)
        points = points.astype(np.float32)
        zeroed = dissimilarities(points, Metric.FP32_L2)
        np.fill_diagonal(zeroed, 0.0)  # the matrix as it was before the diagonal was neutralised
        assert zeroed.diagonal().max() == 0.0
        for build in (rng_edges, gabriel_edges, nsg_edges, mst_edges):
            assert np.array_equal(build(zeroed), build(dissimilarities(points, Metric.FP32_L2))), (
                f"{build.__name__} moved on {preset}"
            )


def test_the_reference_graphs_follow_the_metric_and_not_just_the_label() -> None:
    """
    Switching the metric has to change the reference graphs themselves. Re-reading the same Euclidean
    matrix under both labels would leave every number in the panel untouched and still look plausible.
    """
    points, _ = make_points("blobs", 120, seed=5)
    points = points.astype(np.float32)

    under_l2 = rng_edges(dissimilarities(points, Metric.FP32_L2))
    under_ip = rng_edges(dissimilarities(points, Metric.FP32_InnerProduct))

    assert not np.array_equal(under_l2, under_ip), "the lune is swept over a different matrix"


def test_an_inner_product_report_carries_only_the_graphs_it_can_decide() -> None:
    """
    Gabriel squares the distances and Delaunay triangulates the plane. An inner product gives neither a
    non-negative distance nor an embedding, so the report has to stop at the three graphs its own
    dissimilarity decides — keeping a Euclidean Delaunay beside an inner-product graph would put two
    different geometries in one column of numbers.
    """
    euclidean = build_scene("blobs", 120, DEFAULT_K, 5, Metric.FP32_L2, threads=1).theory
    inner = build_scene("blobs", 120, DEFAULT_K, 5, Metric.FP32_InnerProduct, threads=1).theory

    assert euclidean.names == ("knng", "nsw", "delaunay", "gabriel", "rng", "mst", "mrng", "nsg")
    assert inner.names == ("knng", "nsw", "rng", "mst", "mrng", "nsg")


def test_the_colour_key_names_only_what_the_report_holds() -> None:
    """A swatch for a graph the metric never builds would promise a colour no edge can ever take."""
    inner = build_scene("blobs", 120, DEFAULT_K, 5, Metric.FP32_InnerProduct, threads=1).theory

    assert [name for name, _ in legend_entries(inner)] == ["DEG only", "rng", "mrng", "mst"]
    codes = inner.membership(OVERLAP_ORDER)  # the full override order, two names of which it lacks
    assert codes.shape == (inner.deg_edges,)
    assert codes.min() >= -1 and codes.max() < len(inner.graphs)


def test_viewer_refuses_a_graph_the_current_metric_does_not_build(viewer: GraphViewer) -> None:
    viewer._on_metric("IP")
    assert "delaunay" not in viewer.views

    viewer._on_view("delaunay")

    assert viewer.view == DEG_VIEW, "the choice has to be refused, not drawn"
    assert "needs L2" in viewer.status
    assert len(viewer._edges.get_segments()) == viewer.model.edges.shape[0], "the DEG stays on screen"

    viewer._on_view("rng")
    assert viewer.view == "rng", "a graph the metric does build stays chooseable"


# --------------------------------------------------------------------------- panel layout


def test_the_vertices_field_clamps_reverts_and_rebuilds(viewer: GraphViewer) -> None:
    """
    The vertices spinbox commits a number: clamped into the cloud's bounds, garbage answered by the
    stored value, and every accepted change rebuilds the graph at the new size.
    """
    viewer._commit_vertices("1")
    assert viewer.num_points == MIN_POINTS and viewer.model.num_points == MIN_POINTS, "a tiny value clamps up"

    viewer._commit_vertices("300")
    assert viewer.num_points == 300 and viewer.model.num_points == 300, "a clean value rebuilds"

    viewer._commit_vertices("99999")
    assert viewer.num_points == MAX_POINTS and viewer.model.num_points == MAX_POINTS, "a huge value clamps down"

    viewer._commit_vertices("not-a-number")
    assert viewer.num_points == MAX_POINTS, "garbage leaves the stored value where it was"


def test_viewer_overlay_toggles_share_two_rows(viewer: GraphViewer) -> None:
    labels = viewer._checks1.labels + viewer._checks2.labels

    assert [label.get_text() for label in labels] == [
        "coords",
        "edges",
        "vertex ids",
        "edge colors",
        "query",
        "light bg",
    ]
    row1_y = [y for _, y in (l.get_position() for l in viewer._checks1.labels)]
    row2_y = [y for _, y in (l.get_position() for l in viewer._checks2.labels)]
    assert all(y == pytest.approx(row1_y[0]) for y in row1_y), "row 1 shares a y"
    assert all(y == pytest.approx(row2_y[0]) for y in row2_y), "row 2 shares a y"
    assert [x for x, _ in (l.get_position() for l in viewer._checks1.labels)] == sorted(
        x for x, _ in (l.get_position() for l in viewer._checks1.labels)
    ), "row 1 runs left to right"


def test_clicking_a_checkbox_frame_or_label_flips_it_exactly_once(viewer: GraphViewer) -> None:
    """
    The reported bug: every box always carried a mark whatever its state. Styling the check through
    `set_check_props` pinned the marker's edge colour to one broadcast value, and since the 'x' is drawn
    by its edge the stroke never blanked. Clicking the frame and the nudged label must each flip the
    state exactly once, drive the viewer flag, and repaint the check mark — on shows it, off hides it.
    """
    flags = ("show_coords", "show_edges", "show_ids", "show_colours", "show_query")
    for on_label in (False, True):
        for index, flag in enumerate(flags):
            start = (viewer._checks1 if index < 3 else viewer._checks2).get_status()[index if index < 3 else index - 3]
            assert _check_mark_visible(viewer, index) is start, "the mark agrees with the state to begin"

            _click_check(viewer, index, on_label=on_label)

            assert (viewer._checks1 if index < 3 else viewer._checks2).get_status()[
                index if index < 3 else index - 3
            ] is not start, "the click flips the state once"
            assert getattr(viewer, flag) is (not start), "and the viewer flag follows it"
            assert _check_mark_visible(viewer, index) is (not start), "the mark repaints to the new state"

            _click_check(viewer, index, on_label=on_label)

            assert (viewer._checks1 if index < 3 else viewer._checks2).get_status()[
                index if index < 3 else index - 3
            ] is start, "a second click flips it back"
            assert getattr(viewer, flag) is start, "and the flag returns with it"
            assert _check_mark_visible(viewer, index) is start, "and so does the mark"


def _coordinate_frame_is_on(viewer: GraphViewer) -> bool:
    """Whether the whole reference frame shows: the cross through the origin, the tick labels on both axes and the spines."""
    return (
        bool(viewer._cross)
        and all(line.get_visible() for line in viewer._cross)
        and bool(viewer._ax.get_xticklabels())
        and bool(viewer._ax.get_yticklabels())
        and all(spine.get_visible() for spine in viewer._ax.spines.values())
    )


def test_the_coords_toggle_switches_the_coordinate_system_both_ways(viewer: GraphViewer) -> None:
    """
    The `coords` overlay is one switch for the whole reference frame: the cross through the origin, the
    tick labels and the spines leave together and return together, whether the `g` key or the checkbox
    moves it.
    """
    assert viewer.show_coords is True, "the coordinate system greets the user on"
    viewer._fig.canvas.draw()
    assert _coordinate_frame_is_on(viewer), "on start the cross, the ticks and the spines all show"

    viewer._on_key(_key(viewer, "g"))
    viewer._fig.canvas.draw()
    assert viewer.show_coords is False
    assert not any(line.get_visible() for line in viewer._cross), "the cross is off"
    assert not viewer._ax.get_xticklabels() and not viewer._ax.get_yticklabels(), "the tick labels are off"
    assert not any(spine.get_visible() for spine in viewer._ax.spines.values()), "the spines are off"

    viewer._on_key(_key(viewer, "g"))
    viewer._fig.canvas.draw()
    assert viewer.show_coords is True
    assert _coordinate_frame_is_on(viewer), "the key brings the whole frame back"

    # The checkbox drives the same state as the key, through the real `CheckButtons` handler.
    _click_check(viewer, 0, on_label=False)
    viewer._fig.canvas.draw()
    assert viewer.show_coords is False and not _coordinate_frame_is_on(viewer), "the checkbox turns it off too"

    _click_check(viewer, 0, on_label=False)
    viewer._fig.canvas.draw()
    assert viewer.show_coords is True and _coordinate_frame_is_on(viewer), "and back on"


def test_show_coords_survives_a_rebuild(viewer: GraphViewer) -> None:
    """
    The coordinate system is the user's overlay, not the graph's: a rebuild changes the graph, and the
    frame the user switched off stays off across it — the cross a fresh framing would otherwise leave
    showing is re-hidden, so the flag is not silently reset by a new cloud, degree or metric.
    """
    viewer._on_key(_key(viewer, "g"))
    assert viewer.show_coords is False

    viewer._on_metric("IP")  # a real rebuild: a new metric, a new report, a fresh framing
    viewer._fig.canvas.draw()

    assert viewer.show_coords is False, "the flag is untouched by the rebuild"
    assert not any(line.get_visible() for line in viewer._cross), "the cross stays off across it"
    assert not viewer._ax.get_xticklabels(), "and so do the tick labels"
    assert not any(spine.get_visible() for spine in viewer._ax.spines.values()), "and the spines"


def test_the_light_bg_checkbox_switches_the_2d_background(viewer: GraphViewer) -> None:
    """The bottom-right 'light bg' checkbox paints the 2D plot #EFF0F1 and back to white."""
    assert viewer.show_light_bg is False
    assert matplotlib.colors.to_hex(viewer._ax.get_facecolor()).lower() == BG_DEFAULT.lower()

    viewer._toggle_flag(5)

    assert viewer.show_light_bg is True, "the flag is on"
    assert matplotlib.colors.to_hex(viewer._ax.get_facecolor()).lower() == BG_LIGHT.lower(), "the plot turns light grey"

    viewer._toggle_flag(5)

    assert viewer.show_light_bg is False
    assert matplotlib.colors.to_hex(viewer._ax.get_facecolor()).lower() == BG_DEFAULT.lower(), "and back to white"


def test_the_query_overlay_switches_the_traversal_markers(viewer: GraphViewer) -> None:
    """
    The `query` overlay hides the traversal's own marks — start node, route, target ring and the answer
    and ground-truth marks — as one switch, and brings them all back, without dropping the underlying
    selection or query state.
    """
    viewer.select(42)
    viewer.set_entry(7)
    viewer.run_query()
    viewer._fig.canvas.draw()

    assert viewer._entry_mark.get_offsets().shape[0] == 1, "the start node is marked"
    assert len(viewer._path.get_segments()) > 0, "the route is drawn"
    assert viewer._ring.get_offsets().shape[0] == 1, "the target ring is drawn"
    assert viewer._exact_mark.get_offsets().shape[0] == 1, "the ground-truth star is drawn"

    viewer._toggle_flag(4)

    assert viewer.show_query is False, "the key drives the flag"
    assert viewer._entry_mark.get_offsets().shape[0] == 0, "the start node is hidden"
    assert len(viewer._path.get_segments()) == 0, "the route is hidden"
    assert viewer._ring.get_offsets().shape[0] == 0, "the target ring is hidden"
    assert viewer._terminal_mark.get_offsets().shape[0] == 0, "the answer X is hidden"
    assert viewer._exact_mark.get_offsets().shape[0] == 0, "the star is hidden"
    assert not viewer._connector.get_visible(), "the miss connector is hidden"
    assert viewer.selected == 42 and viewer.query is not None, "the selection and query state survive the hide"

    viewer._toggle_flag(4)

    assert viewer._entry_mark.get_offsets().shape[0] == 1, "the start node returns with the switch"
    assert len(viewer._path.get_segments()) > 0, "and the route with it"


def test_viewer_colours_edges_by_their_reference_graph(viewer: GraphViewer) -> None:
    colours = viewer._edges.get_colors()

    assert colours.shape == (viewer.model.edges.shape[0], 4), "one colour per edge"
    assert len(np.unique(colours, axis=0)) > 1, "the shared edges stand out from the rest"

    viewer._on_key(_key(viewer, "c"))

    assert viewer.show_colours is False
    assert viewer._edges.get_colors().shape[0] == 1, "without the overlay every edge is neutral"
    assert tuple(viewer._edges.get_colors()[0]) == tuple(matplotlib.colors.to_rgba(EDGE_COLOR))


def test_viewer_reports_the_theory_overlap_on_every_build(viewer: GraphViewer) -> None:
    text = viewer._panel_text.get_text()

    assert viewer.model.theory is not None, "the comparison arrives with the model"

    viewer.reseed()

    assert viewer.model.theory is not None, "a fresh cloud is compared again"
    names = tuple(item.name for item in viewer.model.theory.graphs)
    assert len(names) == 8, "every graph the selector offers is listed, the knng and NSW among them"
    assert all(f"∩ {name}" in viewer._panel_text.get_text() for name in names), "the reseeded panel lists them all"
    assert all(f"∩ {name}" in text for name in names), "and so did the panel before the reseed"
    assert all(item.shared > 0 for item in viewer.model.theory.graphs)


def test_viewer_shows_a_colour_key_while_the_overlay_is_on(viewer: GraphViewer) -> None:
    assert viewer._legend.get_visible(), "the overlay starts on, so the key does too"
    expected = legend_entries(viewer.model.theory)
    assert [text.get_text() for text in viewer._legend.get_texts()] == [name for name, _ in expected]

    drawn = [line.get_color() for line in viewer._legend.get_lines()]
    assert drawn == [color for _, color in expected], "the key swatches are the colours that paint"

    viewer._on_key(_key(viewer, "c"))
    assert not viewer._legend.get_visible(), "hiding the overlay hides its key"


def test_colour_key_owns_the_strip_under_the_plot(viewer: GraphViewer) -> None:
    """The key has the band beneath the plot to itself now that the control line is gone from it."""
    viewer.select(42)
    viewer.set_entry(7)
    viewer.run_query()
    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()

    key_box = viewer._legend.get_window_extent(renderer)
    strip = viewer._legend_ax.get_window_extent(renderer)
    axes = viewer._ax.get_window_extent(renderer)

    assert key_box.y1 <= axes.y0 + 1.0, "the key sits below the plot, never over it"
    assert key_box.y0 >= 0.0, "and stays on the canvas"
    assert key_box.x0 >= strip.x0 - 1.0 and key_box.x1 <= strip.x1 + 1.0, "inside its own strip"
    assert axes.y0 - key_box.y1 >= 20.0, "far enough below the plot to click without grazing its edge"


def test_the_colour_key_switches_one_reference_graph_off(viewer: GraphViewer) -> None:
    """
    The key is the colouring's switch: clicking an entry takes that graph out of the overlay and its
    edges fall through to the next graph in the override order holding them, which is what lets the
    nesting be peeled apart one layer at a time instead of read as one flat wash of colour.
    """
    neutral = matplotlib.colors.to_rgba(EDGE_COLOR)
    position = viewer._legend_names().index("delaunay")
    before = edge_colours(viewer.model, viewer.overlays)

    _click_key(viewer, position)

    after = edge_colours(viewer.model, viewer.overlays)
    assert not np.array_equal(before, after), "the drawing loses that graph's colour"
    assert (after == neutral).all(axis=1).sum() > (before == neutral).all(axis=1).sum(), (
        "the edges it alone claimed go neutral"
    )

    _click_key(viewer, position)

    assert "delaunay" in viewer.overlays, "the same entry switches it back"
    np.testing.assert_array_equal(edge_colours(viewer.model, viewer.overlays), before)


def test_a_switched_off_entry_reads_as_switched_off(viewer: GraphViewer) -> None:
    """A key that switches has to show what is switched, or it is a control with no face."""
    position = viewer._legend_names().index("rng")
    handle = viewer._legend.legend_handles[position]
    text = viewer._legend.get_texts()[position]
    assert text.get_color() == "#2a2e35"

    _click_key(viewer, position)

    assert text.get_color() == "#b0b5bd", "and the label fades with it"

    _click_key(viewer, position)
    assert handle.get_alpha() == 1.0 and text.get_color() == "#2a2e35"


def test_the_neutral_entry_is_not_a_second_switch_for_the_overlay(viewer: GraphViewer) -> None:
    """`DEG only` names the remainder; the checkbox already owns the one flag that hides it."""
    colours_before = viewer.show_colours

    _click_key(viewer, 0)

    assert viewer.show_colours is colours_before, "clicking it changes nothing"


def test_an_overlay_switch_survives_a_rebuild(viewer: GraphViewer) -> None:
    """
    The colour key is the user's switch, not the report's: a graph hidden before a metric round-trip
    comes back hidden, because a rebuild changes the graph, not the key.
    """
    _click_key(viewer, viewer._legend_names().index("rng"))
    assert "rng" not in viewer.overlays

    viewer._on_metric("IP")

    assert viewer.overlays == {"mst", "mrng", "nsg", "knng", "nsw"}, (
        "the graphs that were on stay on, the hidden one stays off"
    )
    assert viewer._legend_names() == ("DEG only", "rng", "mrng", "mst"), "and the key names exactly those"

    viewer._on_metric("L2")

    assert "rng" not in viewer.overlays, "the graph is still hidden when the metric brings it back"
    assert viewer.overlays == {"delaunay", "gabriel", "mst", "mrng", "nsg", "knng", "nsw"}


def test_a_graph_new_to_the_report_defaults_on(viewer: GraphViewer) -> None:
    """
    A graph the user has never seen has no switch to remember, so it arrives switched on; a graph they
    hid keeps its switch. Driven through `_adopt_views` directly, which is where the rule lives.
    """
    viewer.overlays = {"delaunay", "gabriel", "mst"}  # rng hidden; the MRNG never seen yet
    viewer._held_seen = {"delaunay", "gabriel", "rng", "mst"}  # the last report never carried the MRNG

    viewer._adopt_views()

    assert "mrng" in viewer.overlays, "a graph appearing for the first time paints"
    assert "rng" not in viewer.overlays, "a graph the user hid stays hidden"


def test_the_key_table_is_the_only_binding_table(viewer: GraphViewer) -> None:
    """
    The control sheet and the key handler read one table, so a key the sheet lists cannot be dead — the
    one thing a help window gets wrong without anyone noticing.
    """
    assert set(viewer._bindings) == {"r", "v", "g", "e", "i", "c", "q", "3", "0", "escape", "h"}
    assert all(callable(action) for _, _, action in viewer._bindings.values())
    assert all(label and meaning for label, meaning, _ in viewer._bindings.values())


def test_every_key_reaches_the_action_beside_it_in_the_table(viewer: GraphViewer) -> None:
    """A key has to run the action sitting next to it in the table, not a branch that went stale."""
    for name, flag in (
        ("g", "show_coords"),
        ("e", "show_edges"),
        ("i", "show_ids"),
        ("c", "show_colours"),
        ("q", "show_query"),
    ):
        before = getattr(viewer, flag)
        viewer._on_key(_key(viewer, name))
        assert getattr(viewer, flag) is not before, f"{name} has to drive the {flag} flag"
        viewer._on_key(_key(viewer, name))
        assert getattr(viewer, flag) is before, "and drive it back, so the window keeps its start state"

    viewer.select(42)
    viewer.set_entry(7)
    viewer._on_key(_key(viewer, "escape"))
    assert viewer.selected is None, "Esc releases the target"
    assert viewer.entry == viewer.model.farthest_from_centroid(), "and re-defaults the start node"


def test_the_control_sheet_stays_out_of_the_figure(viewer: GraphViewer) -> None:
    """
    The sheet is a window of its own. On a backend that is not Tk's there is nothing to open, and
    nothing is painted over the drawing either — the plot never carries the keys.
    """
    assert viewer._help_window is None
    assert not any(t.get_text().startswith("left click") for t in viewer._ax.texts)

    viewer._toggle_help()
    assert viewer._help_window is None, "a non-Tk canvas has no window to open, and must not raise"


def test_the_help_control_is_a_tk_button_not_a_canvas_axes(viewer: GraphViewer) -> None:
    """
    The help button lives in the Tk selector strip, level with the dropdowns, so the figure carries
    no button of its own. Off Tk there is no strip at all, and the key that opens the sheet — the
    only way to reach help on a headless canvas — must stay a safe no-op rather than a crash.
    """
    assert not hasattr(viewer, "_help_ax"), "the matplotlib help button is gone from the figure"
    assert "h" in viewer._bindings, "the key table still owns the help key"
    assert viewer._bindings["h"][2] == viewer._toggle_help, "and the key runs the sheet's own toggle"

    viewer._on_key(_key(viewer, "h"))
    assert viewer._toggle_help() is None, "a non-Tk canvas has no window to open, and must not raise"
    assert viewer._help_window is None


def test_panel_text_stays_inside_its_area(viewer: GraphViewer) -> None:
    viewer.select(42)
    viewer.set_entry(7)
    viewer.run_query()
    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()

    text = viewer._panel_text.get_window_extent(renderer)
    panel = viewer._panel.get_window_extent(renderer)
    below = viewer._check_ax1.get_window_extent(renderer)

    assert "∩ delaunay" in viewer._panel_text.get_text(), "the fullest panel is the one that must fit"
    assert text.y1 <= panel.y1 + 1.0, "the panel starts inside its area"
    assert text.y0 >= below.y1, "and never runs into the widgets below it"


def test_panel_fits_at_the_widget_caps(viewer: GraphViewer) -> None:
    """The fullest panel is the one at every cap: vertex count, degree, query, theory and top-k filled."""
    viewer.num_points = MAX_POINTS
    viewer.k = K_LIMIT
    viewer.top_k = TOP_K_MAX  # the widest search the field can ask for lists one panel line per result
    viewer._rebuild()
    viewer.select(42)
    viewer.set_entry(7)
    viewer.run_query()
    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()

    text = viewer._panel_text.get_window_extent(renderer)
    panel = viewer._panel.get_window_extent(renderer)
    below = viewer._check_ax1.get_window_extent(renderer)

    body = viewer._panel_text.get_text()
    assert f"{MAX_POINTS} vertices · k={K_LIMIT}" in body, "the heading names the DEG's own shape"
    assert "  search   #1" in body and "           #8" in body, "the fullest panel lists all eight results"
    assert all(len(line) <= PANEL_WIDTH for line in body.split("\n")), "no line may leave the column"
    assert text.y0 >= below.y1, "the fullest panel must not reach the graph selector"
    assert text.x1 <= panel.x1 + 1.0, "nor spill out of its column"


def test_fan_out_lists_neighbours_within_the_column() -> None:
    assert _fan_out(np.array([], dtype=np.int64)) == [f"{NEIGHBOR_LABEL}—"], "an isolated vertex gets a line too"
    assert _fan_out(np.array([7])) == [f"{NEIGHBOR_LABEL}7"], "one neighbour needs one line"

    wide = _fan_out(np.arange(1000, 1016))
    assert len(wide) == 2, "the panel has room for two indented lines"
    assert all(len(line) <= PANEL_WIDTH for line in wide), "four-digit ids must not leave the column"
    assert wide[1].startswith(" " * len(NEIGHBOR_LABEL)), "a continuation line keeps the list's indent"
    assert wide[1].endswith("…"), "the ids that no longer fit are marked, not silently dropped"

    short = _fan_out(np.arange(8))
    assert len(short) == 1, "ids short enough to share a row are not split"

    balanced = _fan_out(np.arange(100, 112))
    assert len(balanced) == 2, "twelve three-digit ids need two rows"
    assert "…" not in balanced[1], "nothing is lost when everything fits"
    assert abs(len(balanced[0]) - len(balanced[1])) <= 2, "the rows are balanced, not greedy"
    listed = {int(t) for line in balanced for t in line[len(NEIGHBOR_LABEL) :].replace(",", " ").split()}
    assert listed == set(range(100, 112)), "the split keeps every neighbour"


# --------------------------------------------------------------------------- one graph or another


def test_edges_of_decodes_the_keys_back_to_pairs() -> None:
    """The report keeps keys for set algebra; drawing needs the vertex pairs back."""
    scene = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)
    theory = scene.theory

    assert theory.names == tuple(overlap.name for overlap in theory.graphs)

    for name in theory.names:
        overlap = theory.graph(name)
        pairs = theory.edges_of(name, scene.num_points)

        assert pairs.shape == (overlap.edges, 2), f"{name} must decode to one row per edge"
        assert (pairs[:, 0] < pairs[:, 1]).all(), f"{name} must keep the i < j order of the DEG's own edges"
        np.testing.assert_array_equal(
            pairs[:, 0] * scene.num_points + pairs[:, 1],
            np.sort(overlap.keys),
            f"{name} must re-encode to exactly the keys it was stored under",
        )


def test_the_inner_product_ground_truth_prefers_the_longer_vector() -> None:
    """What counts as the right answer follows the metric, so the ground truth has to as well."""
    points, groups = make_points("blobs", 200, seed=3)
    common = {"k": MIN_K, "seed": 3, "threads": 1, "theory": False}
    l2 = build_model(points, groups, metric=Metric.FP32_L2, **common)
    ip = build_model(points, groups, metric=Metric.FP32_InnerProduct, **common)

    assert l2.best_match(points[42]) == 42, "under L2 a vertex is its own nearest neighbour"

    rival = ip.best_match(points[42])
    assert rival != 42, "under inner product a longer vector in the same direction wins instead"
    assert points[42] @ points[rival] > points[42] @ points[42]


def test_the_headline_names_the_metric_it_built_with() -> None:
    points, groups = make_points("blobs", 100, seed=3)
    model = build_model(points, groups, k=MIN_K, seed=3, threads=1, metric=Metric.FP32_InnerProduct, theory=False)

    assert "· IP ·" in describe(model)[0], "the panel must not present an inner-product graph as L2"


def test_viewer_draws_the_graph_the_selector_picks(viewer: GraphViewer) -> None:
    assert viewer.views == (
        DEG_VIEW,
        KNNG_VIEW,
        NSW_VIEW,
        "delaunay",
        "gabriel",
        "rng",
        "mst",
        "mrng",
        NSG_VIEW,
        NONE_VIEW,
    ), "the selector lists every graph in GRAPH_ORDER's order, the knng, NSW and NSG among them"

    viewer._fig.canvas.draw()
    assert len(viewer._edges.get_segments()) == viewer.model.edges.shape[0], "the DEG view draws the graph itself"
    assert viewer._legend.get_visible(), "and has to explain what its edge colours mean"

    for name in (n for n in OVERLAP_ORDER if n in viewer.model.theory.names):
        viewer._on_view(name)
        viewer._fig.canvas.draw()

        assert len(viewer._edges.get_segments()) == viewer.model.theory.graph(name).edges, (
            f"{name} must draw its own edge set, not the DEG's"
        )
        colours = viewer._edges.get_colors()
        assert len(np.unique(colours, axis=0)) == 1, f"{name} is a single graph, so it gets a single colour"
        np.testing.assert_allclose(colours[0], matplotlib.colors.to_rgba(OVERLAP_COLORS[name]), atol=1e-6)
        assert not viewer._legend.get_visible(), "the DEG colour key explains nothing about a reference graph"


def test_viewer_searches_the_reference_graph_on_screen(viewer: GraphViewer) -> None:
    """
    A reference graph is searched and drawn just like the DEG — the whole point of the Python search.

    Switching the view replays the open query on the new graph, and the recorded route follows the
    reference graph's real edges, which the library's search could never do because it only ever
    sees a built DEG.
    """
    viewer.select(42)
    viewer.set_entry(7)
    opened = viewer.run_query()
    assert opened is not None, "the DEG is searchable"

    viewer._on_view("mst")
    assert viewer.query is not None, "switching graphs replays the open query on the new edges"
    assert viewer.query.target == opened.target, "and the replayed query keeps its target"

    result = viewer.run_query()
    assert result is not None, "the reference graph is searched on screen"
    assert result.path.size >= 1, "and it records a route"
    edges = _adjacency_edges(viewer.model.adjacency_of("mst"))
    for a, b in pairwise(result.path):
        assert (int(min(a, b)), int(max(a, b))) in edges, "every step is an edge of the drawn tree"
    assert viewer._path.get_segments().__len__() == result.hops, "the route is drawn"


def test_viewer_searches_the_nsg_view(viewer: GraphViewer) -> None:
    """The NSG view is searchable like the knng and the NSW: the route follows its directed edges."""
    viewer.select(42)
    viewer.set_entry(7)
    viewer._on_view(NSG_VIEW)

    result = viewer.run_query()

    assert result is not None, "the NSG view is searched on screen, not refused for missing a report entry"
    forward = {(int(u), int(v)) for u, v in viewer._nsg_for()}
    for a, b in pairwise(result.path):
        assert (int(a), int(b)) in forward, "every step is a forward edge of the drawn NSG"


def test_the_start_node_starts_set_and_the_marker_is_drawn(viewer: GraphViewer) -> None:
    """The viewer opens with a start node — the vertex farthest from the cloud's centre — drawn."""
    delta = viewer.model.points - viewer.model.points.mean(axis=0)
    expected = int(np.argmax(np.einsum("ij,ij->i", delta, delta)))
    assert viewer.entry == expected, "the default start node is the vertex farthest from the centroid"
    assert viewer._entry_mark.get_offsets().__len__() == 1, "its marker is drawn from the first frame"
    assert "START" in viewer._panel_text.get_text(), "and the panel names it from the first frame"


def test_the_start_node_survives_rebuilds_and_redefaults_when_lost(viewer: GraphViewer) -> None:
    viewer.set_entry(42)
    viewer._on_k(6)
    assert viewer.entry == 42, "a start node inside the new cloud keeps its place through a rebuild"
    viewer._on_vertices(MIN_POINTS)
    assert viewer.entry == viewer.model.farthest_from_centroid(), "one past the shrunken end re-defaults"


def test_the_empty_view_hides_the_start_node_and_switching_back_replays(viewer: GraphViewer) -> None:
    """The start node is drawn in every graph but the empty one, where no walk could start anyway."""
    viewer.select(42)
    viewer.run_query()

    viewer._on_view(NONE_VIEW)
    assert viewer._entry_mark.get_offsets().__len__() == 0, "the empty view hides the start marker"
    assert viewer.query is None and viewer.status == "no graph — pick one"

    viewer._on_view(DEG_VIEW)
    assert viewer.query is not None, "coming back replays the query on the DEG"
    assert viewer._entry_mark.get_offsets().__len__() == 1, "and the start marker is drawn again"


def test_the_view_selector_draws_what_it_names_and_refuses_the_rest(viewer: GraphViewer) -> None:
    """Choosing a graph draws it at once; a name the report does not hold is refused, not drawn."""
    viewer._on_view("rng")
    assert viewer.view == "rng", "a chosen graph is drawn at once"

    viewer._on_view("no-such-graph")
    assert viewer.view == "rng", "a name the report does not hold is refused"
    assert "no-such-graph" in viewer.status, "and the status line says so"

    viewer._on_view("rng")
    assert viewer.view == "rng", "re-choosing the drawn graph changes nothing"


def test_the_hover_highlight_follows_the_drawn_graph(viewer: GraphViewer) -> None:
    viewer.hovered = 42
    viewer._paint_hover_edges()

    assert len(viewer._hover_focus.get_segments()) == viewer.model.k, "the DEG view lights all k edges of the vertex"

    viewer._on_view("mst")

    tree = viewer.model.theory.edges_of("mst", viewer.model.num_points)
    degree = int(np.count_nonzero((tree[:, 0] == 42) | (tree[:, 1] == 42)))
    assert degree < viewer.model.k, "a spanning tree cannot reach the degree of a k-regular graph"
    assert len(viewer._hover_focus.get_segments()) == degree, "the tree view lights only the edges the tree really has"


def test_the_view_key_walks_the_selector_and_wraps(viewer: GraphViewer) -> None:
    seen = [viewer.view]
    for _ in viewer.views:
        viewer._on_key(_key(viewer, "v"))
        seen.append(viewer.view)

    assert seen == [*viewer.views, DEG_VIEW], "v must step through every graph and return to the DEG"


def test_viewer_rebuilds_under_the_other_metric(viewer: GraphViewer) -> None:
    assert viewer.metric is Metric.FP32_L2
    assert "· L2 ·" in viewer._panel_text.get_text()

    viewer._on_metric("IP")

    assert viewer.metric is Metric.FP32_InnerProduct
    assert viewer.model.metric is Metric.FP32_InnerProduct
    assert "· IP ·" in viewer._panel_text.get_text(), "the panel must name the metric it was rebuilt with"
    assert viewer.model.edges.shape[0] == viewer.model.num_points * DEFAULT_K // 2, "still the same k-regular graph"


def test_viewer_groups_the_controls_below_the_panel(viewer: GraphViewer) -> None:
    """The numeric settings moved to the strip, so only the overlays and the reseed button remain below."""
    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()
    panel = viewer._panel.get_window_extent(renderer)

    rows = (
        ("overlay", viewer._check_ax1),
        ("overlay2", viewer._check_ax2),
        ("reseed", viewer._reseed_ax),
    )
    boxes = [(name, ax.get_window_extent(renderer)) for name, ax in rows]

    for name, box in boxes:
        assert box.y1 <= panel.y0 + 1.0, f"the {name} controls must sit below the panel"
        assert box.y0 > 0.0, f"the {name} controls must stay on the figure"
        assert box.x0 == pytest.approx(panel.x0), f"the {name} controls must share the panel column"

    for (upper, first), (lower, second) in pairwise(boxes):
        assert second.y1 <= first.y0 + 1.0, f"the {upper} controls must end above the {lower} ones"


def test_viewer_selectors_stay_drivable_off_tk(viewer: GraphViewer) -> None:
    """The dropdowns are Tk widgets, so off Tk the viewer must hold no Tk state and still switch.

    The handlers take the same labels the menu entries send — an index here is the bug that shipped once.
    """
    assert viewer._view_var is None and viewer._preset_var is None and viewer._metric_var is None
    assert not hasattr(viewer, "_selector_strip"), "no Tk strip may be claimed on a headless canvas"

    viewer._on_metric("IP")
    viewer._on_preset("spiral")

    assert viewer.metric is Metric.FP32_InnerProduct
    assert viewer.preset == "spiral"


# --------------------------------------------------------------------------- search metric


def test_a_search_metric_reranks_the_same_edges_without_rebuilding(model) -> None:
    """
    The other comparator has to be a view over the built graph rather than a second build.

    A read-only graph stores no edge weights, so re-tagging its feature space leaves the adjacency
    exactly as drawn and changes only the order the traversal reads its neighbours in. Should this
    ever re-rank the topology, the picture on screen stops being the graph that was searched.
    """
    other = model.traversal(Metric.FP32_InnerProduct)

    assert other is not model.graph
    assert other.get_feature_space().metric() is Metric.FP32_InnerProduct
    assert [sorted(other.get_neighbors(vertex)) for vertex in range(model.num_points)] == [
        list(neighbors) for neighbors in model.adjacency
    ], "the view must answer with the very edges the model carries"


def test_the_build_metric_needs_no_view(model) -> None:
    assert model.traversal(model.metric) is model.graph, "searching with the build metric copies nothing"


def test_the_search_metric_is_judged_by_its_own_ground_truth(model) -> None:
    """Asking for inner product moves both the descent and the answer it is scored against."""
    same = query_graph(model, 7, 211, eps=0.5)
    apart = query_graph(model, 7, 211, eps=0.5, metric=Metric.FP32_InnerProduct)

    assert same.metric is Metric.FP32_L2, "a query given no metric uses the one the graph was built with"
    assert apart.metric is Metric.FP32_InnerProduct
    assert apart.exact == model.best_match(model.points[7], Metric.FP32_InnerProduct, exclude=7)
    assert apart.exact != same.exact, "the two metrics must not agree on what the right answer is"


def test_the_other_comparator_sends_the_descent_elsewhere(model) -> None:
    l2 = query_graph(model, 7, 211, eps=0.5)
    ip = query_graph(model, 7, 211, eps=0.5, metric=Metric.FP32_InnerProduct)

    assert not np.array_equal(l2.path, ip.path), "another metric over the same edges takes another route"
    steps = list(zip(ip.path[:-1], ip.path[1:], strict=True))
    assert all(j in model.adjacency[i] for i, j in steps), "and every step is still an edge of the built graph"


def test_the_search_metric_selectors_leave_the_graph_alone(viewer: GraphViewer) -> None:
    """Switching the comparator must not rebuild: the cloud, its edges and the seed all stay put."""
    viewer.select(7)
    viewer.set_entry(211)
    before = viewer.run_query()
    built = viewer.model

    viewer._on_search_metric("IP")

    assert viewer.model is built, "the graph on screen is still the one that was built"
    assert viewer.search_metric is Metric.FP32_InnerProduct
    assert viewer.query is not None and viewer.query.metric is Metric.FP32_InnerProduct, "the query re-ran"
    assert not np.array_equal(viewer.query.path, before.path), "and came back by a different route"
    assert "QUERY    IP" in viewer._panel_text.get_text(), "the panel says which comparator answered"


def test_the_two_metric_selectors_are_independent(viewer: GraphViewer) -> None:
    """One dropdown rebuilds the graph, the other only re-ranks it; neither may move the other."""
    viewer._on_search_metric("IP")
    viewer._on_search_metric("IP")  # the same choice twice must not re-run anything
    viewer._on_metric("IP")

    assert viewer.metric is Metric.FP32_InnerProduct and viewer.search_metric is Metric.FP32_InnerProduct

    viewer._on_metric("L2")
    assert viewer.search_metric is Metric.FP32_InnerProduct, "rebuilding for L2 keeps the inner-product search"
    assert viewer.query is None, "no query was open before the rebuild, and a rebuild invents none"

    viewer.select(7)
    viewer.set_entry(211)
    result = viewer.run_query()

    assert viewer.model.metric is Metric.FP32_L2
    assert result is not None and result.metric is Metric.FP32_InnerProduct, "an L2 graph walked by IP"


def test_the_search_metric_survives_a_new_sample(viewer: GraphViewer) -> None:
    """The comparator is how the graph is read, not a property of the cloud, so a reseed keeps it."""
    viewer._on_search_metric("IP")
    viewer.reseed()

    assert viewer.search_metric is Metric.FP32_InnerProduct
    assert viewer.model.metric is Metric.FP32_L2


# --------------------------------------------------------------------------- MIPS→L2 transform


@pytest.fixture(scope="module")
def mips_window():
    """A second shared window, opened with the spherical transform on, mirroring `window`."""
    instance = GraphViewer(partial(build_scene, threads=1), *BUILD_START, mips=True)
    instance._fig.canvas.draw_idle = lambda *_args, **_keywords: None
    yield instance
    instance.close()


@pytest.fixture()
def mips_viewer(mips_window: GraphViewer) -> GraphViewer:
    """The shared transformed window wound back to its opening state, so each test starts from frame one."""
    stale = (
        mips_window.preset,
        mips_window.num_points,
        mips_window.k,
        mips_window.seed,
        mips_window.metric,
        mips_window.mips,
    ) != (*BUILD_START, True)
    mips_window.preset, mips_window.num_points, mips_window.k, mips_window.seed, mips_window.metric = BUILD_START
    mips_window.mips = True
    mips_window.selected = mips_window.entry = mips_window.hovered = None
    mips_window.free_query = None
    mips_window.query = mips_window.status = None
    mips_window._home = None
    mips_window._drag = None
    mips_window.view = DEG_VIEW
    mips_window.search_metric = BUILD_START[4]
    mips_window.top_k = 1
    mips_window.overlays = mips_window._held_graphs()
    mips_window._held_seen = set(mips_window.overlays)
    mips_window._knng_edges = None
    for position, wanted in enumerate(FLAG_START):
        checks = mips_window._checks1 if position < 3 else mips_window._checks2
        local = position if position < 3 else position - 3
        if checks.get_status()[local] != wanted:
            mips_window._toggle_flag(position)
    if mips_window.show_3d:
        mips_window.show_3d = False
        mips_window._set_3d_visible(False)
    mips_window._cube_3d.reset_limits()
    if stale:
        mips_window._rebuild()
    else:
        mips_window.redraw()
    return mips_window


def test_the_mips_transform_lifts_the_features_and_keeps_the_plot_2d() -> None:
    """The graph runs on the spherical transform while the drawing keeps the original cloud."""
    lifted = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, mips=True, threads=1)
    plain = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, threads=1)

    assert lifted.points.shape == (150, 3), "the transform appends a third coordinate to the features"
    assert lifted.plot_points.shape == (150, 2), "the drawing stays in the original plane"
    norms = np.linalg.norm(lifted.points, axis=1)
    assert np.allclose(norms, norms[0]), "the transform places every vector on one sphere"
    np.testing.assert_array_equal(lifted.plot_points, plain.points, "the same seed projects to the same cloud")


def test_the_mips_query_is_the_original_vector_zero_padded() -> None:
    """
    Under MIPS the search runs from the original vector padded with a zero third coordinate, not the
    vertex's own lifted coordinate — so the search and its ground truth answer the inner-product
    maximisation over the originals rather than a nearest neighbour in the inflated space.
    """
    model = build_scene("blobs", 200, DEFAULT_K, 3, Metric.FP32_L2, mips=True, threads=1)
    target = 42

    result = query_graph(model, target, eps=0.5)

    assert result.point.shape == (3,), "the query lives in the lifted space"
    assert result.point[2] == 0.0, "the padding coordinate is zero, not the vertex's own √(M²−‖x‖²)"
    np.testing.assert_allclose(result.point[:2], model.plot_points[target], rtol=1e-6)

    scores = model.plot_points @ model.plot_points[target]
    scores[target] = -np.inf
    assert result.exact == int(np.argmax(scores)), "the answer maximises the original 2D inner product"


def test_the_transform_drops_delaunay_but_keeps_the_distance_graphs() -> None:
    """
    Delaunay triangulates the two drawing coordinates, so on a lifted cloud it would report the graph of
    the projection and a true 3D triangulation is degenerate on co-spherical points — the report drops it.
    The distance-matrix graphs are dimension-agnostic and stay.
    """
    names = build_scene("blobs", 150, DEFAULT_K, 3, Metric.FP32_L2, mips=True, threads=1).theory.names

    assert "delaunay" not in names, "a 3D Delaunay of co-spherical points is not the planar graph"
    assert {"gabriel", "rng", "mst", "mrng"} <= set(names), "the distance-matrix graphs stay valid in 3D"


def test_the_transformed_viewer_picks_on_the_plot_and_searches_the_features(mips_viewer: GraphViewer) -> None:
    """Picking reads the 2D drawing, but the search and its answer live in the transformed feature space."""
    viewer = mips_viewer
    viewer._fig.canvas.draw()
    target = 42
    x, y = viewer._ax.transData.transform(viewer.model.plot_points[target])

    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=1))
    viewer._on_release(MouseEvent("button_release_event", viewer._fig.canvas, x, y, button=1))
    assert viewer.selected == target, "a click on a drawn vertex selects it, though the features are 3D"

    viewer._on_eps(0.5)
    result = viewer.run_query()
    assert result is not None and result.point.shape == (3,), "the query is answered in the lifted feature space"
    assert result.point[2] == 0.0, "the query is the original vector zero-padded, not the vertex's own lift"

    px, py = viewer.model.plot_points[target]
    panel = viewer._panel_text.get_text()
    assert f"TARGET   {target}  ({px:.3f}, {py:.3f})" in panel, "the panel reports the 2D drawing coordinates"


def test_the_mips_free_query_is_zero_padded(mips_viewer: GraphViewer) -> None:
    """A free click under the transform is searched as the original coordinate padded with a zero."""
    viewer = mips_viewer
    _click_empty(viewer, _empty_spot(viewer))
    viewer._on_eps(0.5)

    result = viewer.run_query()

    assert result is not None and result.target is None, "a free query names no vertex"
    assert result.point.shape == (3,), "the query lives in the lifted space"
    assert result.point[2] == 0.0, "the padding coordinate is zero, not a lifted one"
    np.testing.assert_allclose(
        result.point[:2], viewer.free_query, rtol=1e-6, err_msg="the plot coordinate is the query"
    )


def test_the_mips_dropdown_rebuilds_the_features_and_reverts(mips_viewer: GraphViewer) -> None:
    """Toggling the transform rebuilds the graph in the other space while the projected cloud stays put."""
    viewer = mips_viewer
    assert viewer.model.points.shape[1] == 3, "the window opens transformed"
    cloud = viewer.model.plot_points.copy()

    viewer._on_mips("off")
    assert viewer.mips is False and viewer.model.points.shape[1] == 2, "off drops back to the 2D features"

    viewer._on_mips("on")
    assert viewer.mips is True and viewer.model.points.shape[1] == 3, "on lifts them again"
    assert (viewer.preset, viewer.seed) == (BUILD_START[0], BUILD_START[3]), "the same preset and seed are kept"
    np.testing.assert_array_equal(viewer.model.plot_points, cloud, "the projected cloud never moves")


def test_the_mips_dropdown_ignores_the_choice_it_already_has(mips_viewer: GraphViewer) -> None:
    """Re-choosing the current setting must not rebuild, mirroring the preset dropdown's no-op guard."""
    viewer = mips_viewer
    built = viewer.model

    viewer._on_mips("on")  # the window already opens on

    assert viewer.model is built, "the same choice twice must not rebuild the graph"


# --------------------------------------------------------------------------- the knng and the empty view


def _brute_force_knng(points: np.ndarray, k: int, metric: Metric) -> set[tuple[int, int]]:
    """The directed k-nearest-neighbour edge set (each vertex → its k nearest), as ordered `(i, j)` pairs."""
    distances = dissimilarities(points, metric)
    nearest = np.argsort(distances, axis=1)[:, :k]
    return {(i, int(j)) for i in range(points.shape[0]) for j in nearest[i]}


def test_knng_edges_is_the_directed_k_nearest_graph() -> None:
    points, _ = make_points("blobs", 60, seed=8)
    k = 4

    edges = knng_edges(points, k, Metric.FP32_L2)

    assert edges.dtype == np.int32 and edges.shape[1] == 2
    # Out-degree is exactly k for every vertex: the directed list keeps every selection, never symmetrized.
    assert edges.shape[0] == points.shape[0] * k, "every vertex contributes exactly k outgoing links"
    out_degree = np.bincount(edges[:, 0], minlength=points.shape[0])
    assert (out_degree == k).all(), "each vertex selects exactly its k nearest"
    assert {(int(a), int(b)) for a, b in edges} == _brute_force_knng(points, k, Metric.FP32_L2)

    # The knng is directed: a popular neighbour is chosen by many yet does not choose them all back, so
    # some edge has no reverse — a vertex can only walk the neighbours it chose itself.
    directed = {(int(a), int(b)) for a, b in edges}
    assert any((b, a) not in directed for a, b in directed), "the directed knng is not symmetric"


def test_knng_edges_caps_at_the_other_vertices_and_handles_a_tiny_cloud() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], dtype=np.float32)

    wide = knng_edges(points, 99, Metric.FP32_L2)
    complete = {(i, j) for i in range(6) for j in range(6) if i != j}
    assert {(int(a), int(b)) for a, b in wide} == complete, "a k past the cloud links every ordered pair"

    assert knng_edges(np.empty((0, 2), dtype=np.float32), 4).shape == (0, 2)
    assert knng_edges(points[:1], 4).shape == (0, 2), "a single vertex has no neighbours to link"


def test_knng_edges_follows_the_metric() -> None:
    points, _ = make_points("blobs", 60, seed=8)

    under_l2 = knng_edges(points, 4, Metric.FP32_L2)
    under_ip = knng_edges(points, 4, Metric.FP32_InnerProduct)

    assert not np.array_equal(under_l2, under_ip), "the nearest neighbours are read from a different matrix"


# --------------------------------------------------------------------------- the NSW and graph statistics


def _brute_force_nsw(points: np.ndarray, k: int, metric: Metric) -> set[tuple[int, int]]:
    """The incremental undirected NSW edge set: each vertex links its k nearest among the earlier vertices."""
    distances = dissimilarities(points, metric)
    pairs: set[tuple[int, int]] = set()
    for i in range(1, points.shape[0]):
        earlier = distances[i, :i]
        nearest = np.argsort(earlier, kind="stable")[: min(k, i)]
        pairs.update((i, int(j)) for j in nearest)  # (later, earlier), the orientation nsw_edges emits
    return pairs


def test_nsw_edges_is_the_incremental_undirected_graph() -> None:
    points, _ = make_points("blobs", 60, seed=8)
    k = 4

    edges = nsw_edges(points, k, Metric.FP32_L2)

    assert edges.dtype == np.int32 and edges.shape[1] == 2
    assert (edges[:, 0] > edges[:, 1]).all(), "each edge joins a vertex to an earlier one"
    expected = sum(min(k, i) for i in range(1, points.shape[0]))
    assert edges.shape[0] == expected, "each vertex links its k nearest among the vertices already present"
    assert {(int(a), int(b)) for a, b in edges} == _brute_force_nsw(points, k, Metric.FP32_L2)

    # The incremental build lets a popular early vertex be chosen by many later ones, so its undirected
    # degree runs past k — the unbounded maximum degree the NSW is known for.
    degree = np.bincount(edges[:, 0], minlength=points.shape[0]) + np.bincount(edges[:, 1], minlength=points.shape[0])
    assert degree.max() > k, "a later vertex can push an earlier one past k neighbours"


def test_nsw_edges_caps_at_the_other_vertices_and_handles_a_tiny_cloud() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    wide = nsw_edges(points, 99, Metric.FP32_L2)
    complete = {(i, j) for i in range(4) for j in range(i)}  # (later, earlier) for every pair
    assert {(int(a), int(b)) for a, b in wide} == complete, "a k past the cloud links every earlier pair"

    assert nsw_edges(np.empty((0, 2), dtype=np.float32), 4).shape == (0, 2)
    assert nsw_edges(points[:1], 4).shape == (0, 2), "a single vertex has no neighbours to link"


def test_graph_stats_on_a_directed_path() -> None:
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    stats = graph_stats(edges, 3, directed=True, entry=0)

    assert stats["min_out"] == 0 and stats["max_out"] == 1, "the last vertex has no outgoing edge"
    assert stats["min_in"] == 0 and stats["max_in"] == 1, "the first vertex has no incoming edge"
    assert stats["sources"] == 1 and stats["sinks"] == 1, "one source at the head, one sink at the tail"
    assert stats["search_reachability"] == 1.0, "from the head the whole path is reachable"
    assert graph_stats(edges, 3, directed=True, entry=2)["search_reachability"] == pytest.approx(1 / 3)
    assert stats["explore_reachability"] == pytest.approx(2 / 3), "(3 + 2 + 1) / 3 / 3"


def test_graph_stats_on_an_undirected_triangle() -> None:
    edges = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int32)

    stats = graph_stats(edges, 3, directed=False, entry=0)

    assert stats["min_out"] == stats["max_out"] == 2, "every vertex has degree two"
    assert stats["min_in"] == stats["max_in"] == 2, "undirected in- and out-degree are the same"
    assert stats["sources"] == 0 and stats["sinks"] == 0, "no isolated endpoint in a triangle"
    assert stats["search_reachability"] == 1.0 and stats["explore_reachability"] == 1.0


def test_graph_stats_on_an_empty_graph() -> None:
    stats = graph_stats(np.zeros((0, 2), dtype=np.int32), 4, directed=True, entry=0)

    assert stats["min_out"] == stats["max_out"] == stats["min_in"] == stats["max_in"] == 0, "no edges, no degrees"
    assert stats["sources"] == 4 and stats["sinks"] == 4, "every vertex is both a source and a sink"
    assert stats["search_reachability"] == pytest.approx(1 / 4), "the entry reaches only itself"
    assert stats["explore_reachability"] == pytest.approx(1 / 4), "each vertex reaches only itself"
    assert graph_stats(np.zeros((0, 2), dtype=np.int32), 4, directed=True, entry=None)["search_reachability"] == 0.0


def test_graph_stats_counts_an_isolated_vertex() -> None:
    edges = np.asarray([[0, 1]], dtype=np.int32)

    stats = graph_stats(edges, 3, directed=True, entry=0)

    assert stats["min_out"] == 0 and stats["min_in"] == 0, "the lone vertex drives both minima to zero"
    assert stats["sources"] == 2 and stats["sinks"] == 2, "the isolated vertex is a source and a sink"
    assert stats["search_reachability"] == pytest.approx(2 / 3), "from 0 only 0 and 1 are reachable"
    assert stats["explore_reachability"] == pytest.approx(4 / 9), "(2 + 1 + 1) / 3 / 3"


def test_graph_stats_are_consistent_on_the_knng_and_the_nsw() -> None:
    points, _ = make_points("blobs", 60, seed=8)
    k = 4

    knng = graph_stats(knng_edges(points, k), points.shape[0], directed=False, entry=0)
    assert knng["min_in"] == knng["min_out"], "an undirected graph has matching in- and out-degree"
    assert knng["max_in"] == knng["max_out"], "and matching maxima too"

    # A star forces one vertex to be the nearest neighbour of many, so its in-degree climbs past k.
    star = np.vstack(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.array(
                [[1.0, 0.0], [0.309, 0.951], [-0.809, 0.588], [-0.809, -0.588], [0.309, -0.951]], dtype=np.float32
            ),
        ]
    )
    nsw = graph_stats(nsw_edges(star, 2), star.shape[0], directed=True, entry=0)
    assert nsw["max_in"] > 2, "the hub is chosen by every leaf, so its in-degree exceeds k"


def test_the_none_view_draws_no_edges_and_refuses_a_query(viewer: GraphViewer) -> None:
    viewer._on_view(NONE_VIEW)
    viewer._fig.canvas.draw()

    assert len(viewer._edges.get_segments()) == 0, "the empty view draws no edges at all"
    assert not viewer._legend.get_visible(), "and has no colour key to explain them"
    assert "none" in viewer._ax.get_title(loc="left"), "the title names the empty view"

    viewer.select(7)
    assert viewer.run_query() is None, "there is no graph to search"
    assert "no graph" in viewer.status
    assert "no graph" in viewer._panel_text.get_text(), "the panel says so too"


def test_the_knng_view_draws_its_edges_and_searches_them(viewer: GraphViewer) -> None:
    """The knng is drawn in its own colour, directed, and searched over exactly the edges the view shows."""
    viewer._on_view(KNNG_VIEW)
    viewer._fig.canvas.draw()

    edges = viewer._knng_for()
    assert edges.shape[0] == viewer.model.num_points * viewer.k, "every vertex keeps exactly k outgoing links"
    assert len(viewer._edges.get_segments()) == edges.shape[0], "every knng edge is drawn"
    np.testing.assert_allclose(
        viewer._edges.get_colors()[0],
        matplotlib.colors.to_rgba(KNNG_COLOR),
        atol=1e-6,
        err_msg="the knng wears its own colour",
    )
    assert len(viewer._arrowheads.get_paths()) == edges.shape[0], "the directed knng caps every link with a head"
    assert viewer._arrowheads.get_visible(), "and the heads are drawn"

    viewer.select(42)
    viewer.set_entry(7)
    result = viewer.run_query()

    assert result is not None and result.path.size >= 1, "the knng is searchable on screen"
    forward = {
        (vertex, int(neighbor))
        for vertex, neighbors in enumerate(viewer.model.adjacency_of(KNNG_VIEW, edges))
        for neighbor in neighbors
    }
    for a, b in pairwise(result.path):
        assert (int(a), int(b)) in forward, "every step follows a link the source vertex chose itself"


def test_the_nsw_view_draws_its_edges_and_searches_them(viewer: GraphViewer) -> None:
    """The NSW is undirected: plain strokes, no heads, and the search walks it in both directions."""
    viewer._on_view(NSW_VIEW)
    viewer._fig.canvas.draw()

    edges = viewer._nsw_for()
    num_points = viewer.model.num_points
    expected = sum(min(viewer.k, i) for i in range(1, num_points))
    assert edges.shape[0] == expected, "each vertex links its k nearest among the earlier vertices"
    assert len(viewer._edges.get_segments()) == edges.shape[0], "every undirected link is drawn"
    np.testing.assert_allclose(
        viewer._edges.get_colors()[0],
        matplotlib.colors.to_rgba(NSW_COLOR),
        atol=1e-6,
        err_msg="the NSW wears its own colour",
    )
    assert len(viewer._arrowheads.get_paths()) == 0, "the undirected NSW draws no arrowheads"
    assert not viewer._arrowheads.get_visible(), "and its head collection stays hidden"

    viewer.select(42)
    viewer.set_entry(7)
    result = viewer.run_query()

    assert result is not None and result.path.size >= 1, "the NSW is searchable on screen"
    drawn = _adjacency_edges(viewer.model.adjacency_of(NSW_VIEW, edges))
    for a, b in pairwise(result.path):
        assert (int(min(a, b)), int(max(a, b))) in drawn, "every step is an edge of the drawn NSW"


def test_an_undirected_view_draws_plain_strokes_without_heads(viewer: GraphViewer) -> None:
    """Only the directed graphs cap edges with arrowheads; the undirected ones stay plain strokes."""
    viewer._fig.canvas.draw()
    assert len(viewer._edges.get_segments()) > 0, "the DEG draws edges"
    assert len(viewer._arrowheads.get_paths()) == 0, "the undirected DEG draws no heads"
    assert not viewer._arrowheads.get_visible(), "and its head collection stays hidden"

    viewer._on_view("rng")
    assert len(viewer._arrowheads.get_paths()) == 0, "the undirected rng draws no heads either"

    viewer._on_view(KNNG_VIEW)
    assert len(viewer._arrowheads.get_paths()) > 0, "the directed knng caps its links with heads"

    viewer._on_view(NSG_VIEW)
    assert len(viewer._arrowheads.get_paths()) > 0, "the directed nsg caps its links with heads"

    viewer._on_view("mrng")
    assert len(viewer._arrowheads.get_paths()) == 0, "the undirected mrng draws no heads"


def test_arrowhead_triangles_stand_square_at_the_tip_in_display_size() -> None:
    """A head is a filled triangle meeting at the edge tip, its base back from the tip and symmetric."""
    endpoints = np.array([[[0.0, 0.0], [10.0, 0.0]]])
    (triangle,) = arrowhead_triangles(endpoints, matplotlib.transforms.Affine2D(), length=4.0, width=3.0)

    assert triangle.shape == (3, 2), "one edge yields one three-vertex head"
    np.testing.assert_allclose(triangle[0], [10.0, 0.0], err_msg="the first vertex is the tip")
    base = triangle[1:]
    assert np.all(base[:, 0] < 10.0), "the base sits back from the tip"
    assert base[0, 1] == pytest.approx(-base[1, 1]), "the base is symmetric about the edge"


def test_the_panel_reports_the_drawn_graphs_stats(viewer: GraphViewer) -> None:
    """The panel states the on-screen graph's degree, source/sink and reachability numbers."""
    viewer._on_view(KNNG_VIEW)  # directed
    viewer.set_entry(0)
    text = viewer._panel_text.get_text()

    assert "degree   out" in text and "in " in text, "the degree line reports the drawn graph's in- and out-degree"
    assert "sources" in text and "sinks" in text, "and its source and sink counts"
    assert "reach    search" in text and "explore" in text, "and its two reachabilities"


def test_an_undirected_graph_reports_one_degree_and_isolated_count(viewer: GraphViewer) -> None:
    """An undirected graph has in==out, so the panel drops the split and names the isolated vertices."""
    viewer._on_view(NSW_VIEW)  # undirected
    text = viewer._panel_text.get_text()

    assert "degree   " in text and "out " not in text, "no in/out split on an undirected graph"
    assert "isolated" in text and "sources" not in text, "isolated replaces the source/sink pair"


def test_the_panel_stats_are_cached_until_the_graph_moves(viewer: GraphViewer) -> None:
    """A hover rewrites the panel but never recomputes the statistics — only a graph change does."""
    viewer._on_view(NSW_VIEW)
    viewer.set_entry(0)

    first = viewer._view_stats()
    viewer.hovered = 5
    viewer._panel_text.set_text(viewer._panel_lines())

    assert viewer._view_stats() is first, "a hover reuses the cached numbers"

    viewer.set_entry(1)
    assert viewer._view_stats() is not first, "a new start node reaches a different graph and recomputes"


def test_save_figure_writes_a_png_headless(viewer: GraphViewer, tmp_path) -> None:
    """With a path given, `save_figure` writes the file directly — no dialog, no Tk, no event loop."""
    target = viewer.save_figure(tmp_path / "shot.png")

    assert target is not None and target.exists()
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "the file carries the PNG signature"


def test_save_figure_crops_to_the_graph_headless(viewer: GraphViewer, tmp_path) -> None:
    """
    The saved picture is the plot rectangle alone: the white axes and the graph it holds, cropped clear
    of the title, the tick labels, the colour key, the text panel and every control that shares the
    window. So the PNG's size is the main axes' *drawn* rectangle at 150 dpi — the aspect-adjusted
    extent, not the allocated box — and it is strictly smaller than the crop the axes' tight box (which
    folds in the title and tick labels) would have produced.
    """
    target = viewer.save_figure(tmp_path / "graph.png")
    assert target is not None

    image = matplotlib.image.imread(target)
    height, width = image.shape[:2]

    full_width = viewer._fig.get_size_inches()[0] * 150
    full_height = viewer._fig.get_size_inches()[1] * 150
    assert width < full_width * 0.8, "the crop drops the panel column, so it is far narrower than the window"
    assert height < full_height, "and shorter than the window too"

    viewer._fig.canvas.draw()
    renderer = viewer._fig.canvas.get_renderer()

    # The axes' drawn rectangle — the aspect-adjusted position, not the allocated `[0.045, 0.1, 0.675,
    # 0.84]` — is the contract: the saved PNG is that rectangle rendered at 150 dpi.
    extent = viewer._ax.get_window_extent(renderer)
    expected_width = extent.width / renderer.dpi * 150
    expected_height = extent.height / renderer.dpi * 150
    assert width == pytest.approx(expected_width, abs=2.0), "the width is the plot's drawn rectangle"
    assert height == pytest.approx(expected_height, abs=2.0), "and so is the height"

    # The tight box adds the title above and the tick labels to the left and below, so a crop to it is
    # larger on every decorated side. The saved picture must be strictly smaller — the decorations stay out.
    tight = viewer._ax.get_tightbbox(renderer)
    tight_width = tight.width / renderer.dpi * 150
    tight_height = tight.height / renderer.dpi * 150
    assert width < tight_width, "the tick labels are not part of the saved picture"
    assert height < tight_height, "nor is the title"
