"""Interactive matplotlib UI for the 2D DEG: rebuild, inspect vertices, trace searches."""

from __future__ import annotations

import time
import tkinter
from tkinter import filedialog, ttk
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataset import MAX_POINTS, MIN_POINTS, PRESETS
from deg_graph import (
    METRIC_BY_LABEL,
    METRIC_LABELS,
    METRIC_ORDER,
    MIN_K,
    QUERY_EPS,
    GraphModel,
    QueryResult,
    describe,
    knng_edges,
    nsw_edges,
    query_graph,
)
from deglib.distances import Metric
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons
from theory import TheoryReport, dissimilarities, graph_stats, nsg_edges

__all__ = [
    "GraphViewer",
    "KNNG_VIEW",
    "NONE_VIEW",
    "NSG_VIEW",
    "NSW_VIEW",
    "arrowhead_triangles",
    "edge_colours",
    "edge_segments",
    "legend_entries",
    "render_static",
    "vertex_colors",
]

GroupColors = matplotlib.colormaps["tab10"]

SELECT_COLOR = "#d81b60"
HOVER_COLOR = "#1a73e8"
EDGE_COLOR = "#b3b9c2"
#: The knng's own colour: a teal no reference graph and no DEG overlay wears, so a knng edge is never
#: mistaken for a Delaunay, Gabriel, RNG, MST or MRNG one.
KNNG_COLOR = "#00897b"
#: The NSW's own colour: a brown no other graph wears, so its links read apart from the knng's teal ones
#: at a glance.
NSW_COLOR = "#795548"
#: The NSG's own colour: a red no other graph wears, so its directed links read apart from the knng's
#: teal and the NSW's brown ones.
NSG_COLOR = "#c62828"
#: The length and half-width of a directed edge's arrowhead in display pixels. Fixed in pixels, not data
#: units, so the head keeps its size through a zoom and stands square at the tip despite the axes'
#: non-equal aspect. The head is a filled triangle, wide enough to read as an arrow at a glance.
ARROWHEAD_PX = 14.0
ARROWHEAD_WIDTH_PX = 10.0
ENTRY_COLOR = "#5f6368"
TERMINAL_COLOR = "#d81b60"
EXACT_COLOR = "#188038"
#: The free-space query cross. The overlay's orange: no vertex marker wears it, so the cross cannot be
#: misread as one of the traversal's own marks.
FREE_QUERY_COLOR = "#e8710a"
#: Reference-graph colours: blue Delaunay and green RNG as in the 2d-graph reference
#: project, extended by orange Gabriel, violet MST and a crimson MRNG. These are exactly the graphs the
#: overlay paints and the colour key lists. The NSG, knng and NSW are scored in the panel but never
#: coloured — they are drawn in their own view colour and carry no entry here.
OVERLAP_COLORS = {
    "delaunay": "blue",
    "gabriel": "#e8710a",
    "rng": "green",
    "mrng": "#c2185b",
    "mst": "#7b1fa2",
}
#: Precedence where an edge belongs to several reference graphs, lowest first. It runs up the nesting
#: chain, so whichever graph holds the fewest edges has the last word and the rarest structure shows. The
#: MRNG sits inside the RNG and around the MST, so it is placed between them.
OVERLAP_ORDER = ("delaunay", "gabriel", "rng", "mrng", "mst")

NO_POINTS = np.empty((0, 2))
PICK_RADIUS_PX = 14.0
ID_LIMIT = 600
#: The bounds of the k field. The knng accepts any integer in this range; the DEG narrows it to the even
#: values `MIN_K` to `K_LIMIT`, snapping an odd field value up to the next even.
K_MIN = 1
K_LIMIT = 16
#: The bounds of the top-k field: how many results the search reports and the panel lists.
TOP_K_MIN = 1
TOP_K_MAX = 8
#: The bounds of the eps field: the exploration factor the search widens its radius by.
EPS_MIN = 0.0
EPS_MAX = 10.0
#: Fraction of the visible span added or removed per scroll notch.
ZOOM_FACTOR = 0.2
#: How far the view may shrink relative to the initial framing.
MIN_ZOOM = 0.02
FLAG_LABELS = ("coords", "edges", "vertex ids", "theory colours", "query")
#: Which overlays are on when the window opens, in the order of `FLAG_LABELS`. The key bindings and the
#: checkboxes both move this state, so a test that reuses one window across cases needs it to know what
#: "as it started" means. The coordinate system leads and starts on, so the coordinate cross greets the
#: user and is switched off only for a clean screenshot.
FLAG_START = (True, True, False, True, True)
#: The view drawing the built graph; every other view draws one reference graph from the report, the
#: knng, or nothing at all.
DEG_VIEW = "deg"
#: The directed k-nearest-neighbour graph over the current feature space: each vertex keeps a link to its
#: k nearest in the direction it chose, so it can only walk the neighbours it selected. Built by the
#: viewer rather than the library, so it is offered under every metric and dimension — it reads only the
#: distance matrix.
KNNG_VIEW = "knng"
#: The navigable small-world graph over the current feature space, built incrementally and undirected:
#: each vertex links its k nearest among the vertices already present, so a later vertex can raise an
#: earlier one's degree past k and the maximum degree is unbounded. Built by the viewer like the knng.
NSW_VIEW = "nsw"
#: The NSG's monotonic relative neighbourhood graph over the current feature space, built by the viewer
#: like the knng and the NSW and offered under every metric: the directed greedy graph each vertex walks
#: in increasing distance, keeping the nearest free neighbour. It is drawn directed and is scored against
#: the DEG in the panel's report, but it stays out of the colour key — it paints no edge and earns no swatch.
NSG_VIEW = "nsg"
#: The empty view: no edges are drawn and a query on it names the missing graph rather than searching.
NONE_VIEW = "none"

#: Every entry the graph selector offers, in the order the dropdown lists them: the built graph first,
#: then the knng and the NSW, then the reference graphs — the undirected MRNG among
#: them — then the NSG, then the empty view last. A metric admits only part of the reference graphs — inner product
#: builds neither Delaunay nor Gabriel — so the menu lists only the graphs the current report holds, while
#: the DEG, the knng, the NSW, the NSG and the empty view are always available.
GRAPH_ORDER = (DEG_VIEW, KNNG_VIEW, NSW_VIEW, "delaunay", "gabriel", "rng", "mst", "mrng", NSG_VIEW, NONE_VIEW)
#: The pointer gestures, which no key table can carry. The help window prints these above the keys it
#: reads from its own binding table, so the sheet has one source for everything it claims.
MOUSE_HELP = (
    ("left click", "pick a target vertex, or click empty space to query those coordinates"),
    ("right click", "set the traversal start node"),
    ("double click", "run the path query"),
    ("drag", "pan the zoomed view"),
    ("scroll", "zoom towards the cursor"),
)
#: Extra space, in figure pixels, opened between each checkbox frame and its label. Matplotlib's
#: layout places a label 5.5 pt from the frame's centre — barely past the frame's own edge — which
#: reads as text fused to the box.
CHECK_LABEL_GAP_PX = 6

#: The panel column holds this many monospace characters; a longer line spills out of the column
#: into the figure's edge. Measured: the column is 353 px wide at 7 px per character.
PANEL_WIDTH = 50
#: The label a neighbour list starts behind, and that continuation lines repeat as blank padding.
NEIGHBOR_LABEL = "  neighbors  "

#: The panel's line styles by name. The column is no longer one monospace block: the drawn graph is
#: named in a heading, its shape summarised beneath, block labels stand out in bold blue, the DEG's
#: own statistics fade to grey, and a query's verdict is coloured — so the eye finds the section it
#: wants without reading every line.
PANEL_STYLES: dict[str, dict] = {
    "head": {"fontsize": 11, "fontweight": "bold", "color": "#202124", "family": "DejaVu Sans"},
    "sub": {"fontsize": 8.5, "color": "#5f6368", "family": "DejaVu Sans"},
    "label": {"fontsize": 8.5, "fontweight": "bold", "color": "#1a73e8", "family": "DejaVu Sans Mono"},
    "body": {"fontsize": 8.5, "color": "#202124", "family": "DejaVu Sans Mono"},
    "dim": {"fontsize": 8, "color": "#5f6368", "family": "DejaVu Sans Mono"},
    "good": {"fontsize": 8.5, "fontweight": "bold", "color": "#188038", "family": "DejaVu Sans Mono"},
    "bad": {"fontsize": 8.5, "fontweight": "bold", "color": "#d81b60", "family": "DejaVu Sans Mono"},
    "warn": {"fontsize": 8.5, "fontweight": "bold", "color": "#e8710a", "family": "DejaVu Sans Mono"},
}
#: Vertical distance between panel lines, in axes fraction — sized for the tallest style with room.
PANEL_LINE_PITCH = 0.026


class _PanelText:
    """
    The right-hand column as a stack of individually styled lines.

    One `Text` artist cannot vary its font per line, so the panel keeps one artist per line and is
    rewritten wholesale whenever the state changes — the column is at most thirty lines, and the
    repaint that follows costs what a single text update cost before. `get_text` and
    `get_window_extent` keep the shape of a `Text`, so the panel still reads as one block.
    """

    def __init__(self, ax: Axes) -> None:
        self._ax = ax
        self._lines: list[tuple[str, str]] = []
        self._artists: list[Text] = []

    def set_text(self, lines: str | list[tuple[str, str]]) -> None:
        """Draws the column from `(text, style)` pairs; a plain string becomes one body line each."""
        self._lines = [(line, "body") for line in lines.split("\n")] if isinstance(lines, str) else list(lines)
        for artist in self._artists:
            artist.remove()
        self._artists = [
            self._ax.text(
                0.0,
                1.0 - position * PANEL_LINE_PITCH,
                text,
                transform=self._ax.transAxes,
                ha="left",
                va="top",
                **PANEL_STYLES[style],
            )
            for position, (text, style) in enumerate(self._lines)
        ]

    def get_text(self) -> str:
        return "\n".join(text for text, _ in self._lines)

    def get_window_extent(self, renderer) -> matplotlib.transforms.Bbox:
        boxes = [artist.get_window_extent(renderer) for artist in self._artists if artist.get_text()]
        return (
            matplotlib.transforms.Bbox.union(boxes) if boxes else matplotlib.transforms.Bbox([[0.0, 0.0], [0.0, 0.0]])
        )


#: The label the first search result starts behind, and that the remaining ranks repeat as blank
#: padding, so a widened result list reads as one block.
SEARCH_LABEL = "  search   "


class _Drag(NamedTuple):
    """The framing captured at button press, with the scale that turns later pixel deltas into data shifts."""

    x: float
    y: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    px_per_unit_x: float
    px_per_unit_y: float


def _capture_drag(ax: Axes, event: MouseEvent) -> _Drag | None:
    """Records what a drag has to move; `None` once the cursor has left the figure."""
    if event.x is None or event.y is None:
        return None
    origin, unit = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    return _Drag(event.x, event.y, ax.get_xlim(), ax.get_ylim(), unit[0] - origin[0], unit[1] - origin[1])


def _slide(lo: float, hi: float, home: tuple[float, float]) -> tuple[float, float]:
    """Keeps a window of the given width inside the initial framing."""
    if lo < home[0]:
        lo, hi = home[0], home[0] + (hi - lo)
    if hi > home[1]:
        hi, lo = home[1], home[1] - (hi - lo)
    return lo, hi


def _fan_out(neighbors: np.ndarray) -> list[str]:
    """
    Lays a vertex's neighbours over at most two panel lines. The panel has room for a header plus two
    indented lines per block, so a vertex at the degree cap lists what fits and marks the rest.
    """
    ids = [str(int(vertex)) for vertex in neighbors]
    if not ids:
        return [f"{NEIGHBOR_LABEL}—"]

    # Two characters of the column are held back for the ellipsis, and the row is sized off the widest
    # id, so every row fits and not merely the longest of them.
    per_row = max(1, (PANEL_WIDTH - len(NEIGHBOR_LABEL)) // (max(map(len, ids)) + 2))
    if len(ids) <= per_row:
        rows = [", ".join(ids)]
    else:
        half = min(-(-len(ids) // 2), per_row)
        rows = [", ".join(ids[:half]), ", ".join(ids[half : 2 * half])]
        if len(ids) > 2 * half:
            rows[1] += " …"

    pad = " " * len(NEIGHBOR_LABEL)
    return [f"{NEIGHBOR_LABEL}{rows[0]}", *(f"{pad}{row}" for row in rows[1:])]


BuildScene = Callable[[str, int, int, int, Metric, bool], GraphModel]


def edge_segments(model: GraphModel) -> np.ndarray:
    """One `[e, 2, 2]` segment array covering every undirected edge, in the drawing coordinates."""
    return model.plot_points[model.edges]


def arrowhead_triangles(endpoints: np.ndarray, transform, length: float, width: float) -> list[np.ndarray]:
    """
    The filled triangular heads that turn directed `[e, 2, 2]` edge segments into arrows, as data vertices.

    Every edge yields one triangle meeting at its tip, so the head reads as a solid arrow rather than a
    thin fork lost among the strokes. The geometry is built in display coordinates and mapped back through
    the inverse data transform, because the axes keep no equal aspect: a head measured in data units would
    lean with the x/y scale and stretch on a zoom, while one measured in pixels stands square at the tip
    and holds its size as the view scales.
    """
    if endpoints.shape[0] == 0:
        return []
    disp = transform.transform(endpoints.reshape(-1, 2)).reshape(-1, 2, 2)
    tail, tip = disp[:, 0], disp[:, 1]
    run = tip - tail
    unit = run / np.maximum(np.hypot(run[:, 0], run[:, 1]), 1e-9)[:, None]
    perp = np.stack((-unit[:, 1], unit[:, 0]), axis=1)
    base = tip - unit * length
    left = base + perp * (width * 0.5)
    right = base - perp * (width * 0.5)
    triangles = np.stack([tip, left, right], axis=1)
    data = transform.inverted().transform(triangles.reshape(-1, 2)).reshape(-1, 3, 2)
    return list(data)


def edge_colours(model: GraphModel, overlays: Iterable[str] = OVERLAP_ORDER) -> np.ndarray:
    """
    RGBA per DEG edge: the colour of the reference graph that contains it, neutral where none does.

    `overlays` names the graphs allowed to paint, lowest precedence first. The colour key narrows it when
    an entry is switched off, and a graph left out simply stops claiming its edges, which then show the
    colour of the next graph holding them. `TheoryReport.membership` answers in the order of `model.edges`
    and codes by position in `report.graphs`, so the palette below indexes it either way.
    """
    if model.theory is None:
        return matplotlib.colors.to_rgba_array(EDGE_COLOR)
    # One palette slot per report graph, so a membership code (its position in `report.graphs`) indexes
    # the palette directly. Graphs outside `OVERLAP_ORDER` — the NSG, knng and NSW — are never painted,
    # so their slot takes the neutral colour and is never read; they need no entry in `OVERLAP_COLORS`.
    palette = [EDGE_COLOR, *[OVERLAP_COLORS.get(item.name, EDGE_COLOR) for item in model.theory.graphs]]
    allowed = set(overlays)
    order = tuple(name for name in OVERLAP_ORDER if name in allowed)
    return matplotlib.colors.to_rgba_array(palette)[model.theory.membership(order) + 1]


def legend_entries(report: TheoryReport | None) -> tuple[tuple[str, str], ...]:
    """
    The colour key of the overlay: the neutral remainder first, then the reference graphs in the same
    override order that paints the edges. Only the graphs the report actually carries are named — an
    inner-product report holds neither Delaunay nor Gabriel, so a swatch for them would promise a colour
    no edge ever takes.
    """
    held = report.names if report is not None else ()
    return (("DEG only", EDGE_COLOR),) + tuple((name, OVERLAP_COLORS[name]) for name in OVERLAP_ORDER if name in held)


def vertex_colors(model: GraphModel) -> np.ndarray:
    """RGBA per vertex, taken from the cluster that generated the point."""
    palette = matplotlib.colors.to_rgba_array(GroupColors.colors)
    return palette[np.asarray(model.groups, dtype=int) % palette.shape[0]]


def _apply_limits(ax: matplotlib.axes.Axes, points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Frames the point cloud with a margin and returns the resulting home view, ((xmin, xmax), (ymin, ymax))."""
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-3) * 0.05
    home = ((float(low[0] - span[0]), float(high[0] + span[0])), (float(low[1] - span[1]), float(high[1] + span[1])))
    ax.set_xlim(home[0])
    ax.set_ylim(home[1])
    return home


class GraphViewer:
    """
    Live canvas over the graph.

    Left: the embedding with the DEG edges. Right: controls (distribution, size, k, eps, overlays) and a
    text panel reporting graph statistics, the focused vertex and the last traversal.

    Clicking drives the query directly: left-click selects a vertex or — on empty space — places the
    query at the clicked coordinates, right-click makes a vertex the start node of the search, and a
    double-click runs the query immediately from that start node. The query supplies the search vector;
    the Python ε-search answers and records the route it took to its top answer in the same run. It runs
    over whichever graph is on screen — the DEG or a reference graph — so the route is drawn for the
    reference graphs too, which the library's own search cannot traverse. The star marks the query's best
    match — for a vertex the true nearest neighbour under the search metric, the vertex itself excluded;
    for a free point the best match over the whole cloud — and the X marks the search's top answer, green
    when the two coincide and red when the search answered elsewhere. The path is the search's own route
    from the start node to the X. The eps field widens the search's exploration radius — at 0.0 it is a
    pure greedy descent.
    """

    def __init__(
        self,
        build: BuildScene,
        preset: str,
        num_points: int,
        k: int,
        seed: int = 7,
        metric: Metric = Metric.FP32_L2,
        search_metric: Metric | None = None,
        mips: bool = False,
    ) -> None:
        self._build = build
        self.preset = preset
        self.num_points = num_points
        #: The k field's value. It is the knng's neighbour count directly, but the DEG builds with the
        #: even value `self._deg_k()` snaps it to, so the two are kept apart.
        self.k = k
        self.seed = seed
        self.metric = metric
        #: The comparator the traversal and its ground truth use, kept apart from the build metric.
        self.search_metric = metric if search_metric is None else search_metric
        #: Whether the cloud is lifted into the MIPS→L2 spherical space before the graph is built.
        self.mips = mips
        self._rng = np.random.default_rng(seed)
        #: The knng edge list the viewer built for the current points, k and metric; rebuilt lazily and
        #: dropped whenever a rebuild, a k change or a metric change makes it stale.
        self._knng_edges: np.ndarray | None = None
        #: Wall-clock seconds the cached knng took to build, shown in the panel beside its edge count.
        self._knng_seconds = 0.0
        #: The NSW edge list the viewer built for the current points, k and metric; dropped the same way.
        self._nsw_edges: np.ndarray | None = None
        #: Wall-clock seconds the cached NSW took to build.
        self._nsw_seconds = 0.0
        #: The NSG edge list the viewer built for the current points and metric; dropped the same way.
        self._nsg_edges: np.ndarray | None = None
        #: Wall-clock seconds the cached NSG took to build.
        self._nsg_seconds = 0.0
        #: `(key, stats)` for the graph on screen, recomputed only when the key moves — the panel redraws
        #: on every hover, while a graph's statistics change only with the graph itself.
        self._stats_cache: tuple[tuple, dict] | None = None

        self.selected: int | None = None
        #: A query that is not a vertex: plot-space `[x, y]` clicked on empty space, mutually exclusive
        #: with `selected` — one query at a time, whichever the last click placed.
        self.free_query: np.ndarray | None = None
        self.entry: int | None = None
        self.hovered: int | None = None
        self.path_eps = QUERY_EPS
        #: How many results the search reports and the panel lists, set by the top-k field.
        self.top_k = TOP_K_MIN
        self._home: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._drag: _Drag | None = None
        self.query: QueryResult | None = None
        self.status: str | None = None
        self.show_coords, self.show_edges, self.show_ids, self.show_colours, self.show_query = FLAG_START
        self.view = DEG_VIEW

        self._fig: Figure = plt.figure("DEG 2D Explorer", figsize=(18.0, 8.5))
        self._fig.patch.set_facecolor("#f2f4f7")
        self._ax = self._fig.add_axes([0.045, 0.1, 0.675, 0.84])
        self._ax.set_facecolor("#ffffff")
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelsize=8)
        # The coordinate system is styled once here — a faint cross through the origin (the x = 0 and
        # y = 0 axes) sitting below the edges — and the `coords` overlay only flips its visibility
        # together with the tick labels and the spines, so the whole reference frame turns on and off as
        # one without restyling it each time.
        self._cross = (
            self._ax.axvline(0.0, color="#9aa0a6", linewidth=0.8, zorder=0),
            self._ax.axhline(0.0, color="#9aa0a6", linewidth=0.8, zorder=0),
        )
        self._ax.format_coord = self._format_coord

        # The panel column now runs down to just above the overlay checkboxes: the numeric settings that
        # used to sit on sliders below it moved into the top strip, so the freed right column is given to
        # the text. At the widget caps — the widest vertex ids, the widest degree and a top-k list of
        # eight — its text is twenty-seven lines tall, and the column has to hold that without touching
        # the checkbox row below it.
        self._panel = self._fig.add_axes([0.755, 0.135, 0.235, 0.83])
        self._panel.axis("off")
        self._panel_text = _PanelText(self._panel)

        # The key runs under the plot rather than inside the panel column: with a query filled the panel
        # text reaches almost to the bottom of its area, leaving no band the key could sit in.
        self._legend_ax = self._fig.add_axes([0.045, 0.028, 0.675, 0.036])
        self._legend_ax.axis("off")

        # The selectors and numeric fields are native Tk widgets in the strip above the canvas, so this
        # column carries only the overlay checkboxes and the reseed button, stacked below the panel.
        self._check_ax = self._fig.add_axes([0.755, 0.06, 0.235, 0.055])
        self._check_ax.axis("off")
        self._reseed_ax = self._fig.add_axes([0.755, 0.01, 0.11, 0.033])
        self._edges = self._ax.add_collection(LineCollection([], colors=EDGE_COLOR, linewidths=0.55, zorder=1))
        #: The arrowheads of a directed graph (the knng and the NSG). A directed edge is drawn as a line
        #: ending in a filled triangular head rather than a plain stroke, so the direction the search may
        #: walk is visible; the heads are kept in their own collection because their size is fixed in
        #: display pixels and so must be re-derived from the live axis transform on every zoom, unlike the
        #: plain strokes. Their colour follows the graph on screen, set in `redraw`.
        self._arrowheads = self._ax.add_collection(
            PolyCollection([], facecolors=EDGE_COLOR, edgecolors="none", zorder=1)
        )
        self._path = self._ax.add_collection(LineCollection([], linewidths=2.6, zorder=3))
        self._focus = self._ax.add_collection(LineCollection([], colors=SELECT_COLOR, linewidths=1.4, zorder=4))
        #: The heads of the focused vertex's edges on a directed graph, in the focus colour, so a
        #: highlighted link keeps the direction the plain shafts would otherwise hide under the highlight.
        self._focus_arrows = self._ax.add_collection(
            PolyCollection([], facecolors=SELECT_COLOR, edgecolors="none", zorder=4)
        )
        self._vertices = self._ax.scatter([], [], s=16, edgecolors="#2a2e35", linewidths=0.35, zorder=2)
        self._ring = self._ax.scatter(
            [], [], s=200, facecolors="none", edgecolors=SELECT_COLOR, linewidths=1.6, zorder=5
        )
        self._hover_ring = self._ax.scatter(
            [], [], s=120, facecolors="none", edgecolors=HOVER_COLOR, linewidths=1.1, zorder=5
        )
        self._entry_mark = self._ax.scatter(
            [], [], s=170, marker="s", facecolors="white", edgecolors=ENTRY_COLOR, linewidths=1.4, zorder=6
        )
        self._terminal_mark = self._ax.scatter([], [], s=170, marker="X", color=TERMINAL_COLOR, zorder=6)
        self._exact_mark = self._ax.scatter(
            [], [], s=260, marker="*", facecolors="none", edgecolors=EXACT_COLOR, linewidths=1.6, zorder=6
        )
        self._free_mark = self._ax.scatter([], [], s=170, marker="+", color=FREE_QUERY_COLOR, linewidths=1.6, zorder=6)
        # The query's dashed tail. On an arrived walk it runs from the X to the star, measuring how far
        # the search's answer sat from the truth; on a stalled walk it runs from the stop point to the
        # answer the walk never reached. It sits below the path so a reached walk paints over it, and
        # is shown only while a query is open and one of those two gaps exists.
        self._connector = Line2D([], [], linestyle="--", color=TERMINAL_COLOR, linewidth=1.4, zorder=2)
        self._connector.set_visible(False)
        self._ax.add_line(self._connector)
        self._labels: list[Text] = []

        # The model comes first: the graph selector lists one entry per reference graph the report
        # holds, so its labels are not known until the first build is through. The DEG is built at the
        # even degree the k field snaps to, not the raw field value.
        self.model = self._build(preset, num_points, self._deg_k(), seed, metric, mips)
        # The start node exists from the first frame: the vertex farthest from the cloud's centre,
        # so the entry marker is drawn and a query can run before a single right-click.
        self.entry = self.model.farthest_from_centroid()
        # Every reference graph paints until the colour key switches it off. The key is styled from this
        # set, and it is styled below, so it has to exist before the first legend is drawn.
        self.overlays = self._held_graphs()
        #: The graphs the last report held, which is what tells a returning graph from a new one when a
        #: rebuild decides whether an overlay keeps its switch or defaults on.
        self._held_seen = set(self.overlays)
        self._paint_legend()

        # The check and frame styling goes through the constructor, not the post-construction setters:
        # `set_check_props({"color": ...})` routes through `Collection.set_color`, which pins the check
        # marker's edge colour to one broadcast value, while the per-point visibility is driven by face
        # colour alone. The 'x' check is drawn by its edge, so a pinned edge leaves the mark on screen
        # whatever the state. The constructor instead folds `color` into face colour and leaves edge
        # colour at scatter's per-point 'face' default, so hiding a check hides its stroke too.
        self._checks = CheckButtons(
            self._check_ax,
            list(FLAG_LABELS),
            list(FLAG_START),
            layout="horizontal",
            check_props={"color": "#1a73e8", "linewidth": 2.2, "s": 85},
            frame_props={"edgecolor": "#80868b", "facecolor": "#ffffff", "linewidth": 1.2, "s": 85},
        )
        # The layout algorithm leaves barely a pixel between a frame's edge and its label; nudge
        # every label a few figure pixels right, scaled to the row's own width, to open a gap.
        row_px = self._fig.get_size_inches()[0] * self._fig.dpi * self._check_ax.get_position().width
        for lbl in self._checks.labels:
            lbl.set_fontsize(9)
            lbl.set_color("#202124")
            x, y = lbl.get_position()
            lbl.set_position((x + CHECK_LABEL_GAP_PX / row_px, y))
        self._checks.on_clicked(self._on_check)
        # The canvas holds widget callbacks weakly, so the button must stay referenced here.
        self._reseed_button = Button(self._reseed_ax, "reseed", color="#ffffff", hovercolor="#e8f0fe")
        self._reseed_button.label.set_color("#202124")
        self._reseed_button.label.set_fontsize(8.5)
        self._reseed_button.label.set_fontweight("bold")
        for spine in self._reseed_ax.spines.values():
            spine.set_edgecolor("#80868b")
            spine.set_linewidth(1.0)
        self._reseed_button.on_clicked(lambda _event: self.reseed())

        self._view_var: tkinter.StringVar | None = None
        self._preset_var: tkinter.StringVar | None = None
        self._metric_var: tkinter.StringVar | None = None
        self._search_var: tkinter.StringVar | None = None
        self._top_k_var: tkinter.StringVar | None = None
        self._num_var: tkinter.StringVar | None = None
        self._k_var: tkinter.StringVar | None = None
        self._eps_var: tkinter.StringVar | None = None
        self._view_menu: ttk.Combobox | None = None
        self._search_menu: ttk.Combobox | None = None
        self._top_k_spin: ttk.Spinbox | None = None
        self._k_spin: ttk.Spinbox | None = None
        self._build_selectors()
        self._adopt_views()

        # One table feeds both the key dispatch and the help window, so a binding cannot be added without
        # the window learning about it, and the window cannot advertise a key that does nothing.
        self._bindings: dict[str, tuple[str, str, Callable[[], None]]] = {
            "r": ("r", "new sample from the same distribution", self.reseed),
            "v": ("v", "next graph", self._cycle_view),
            "g": ("g", "show or hide the coordinate system", lambda: self._toggle_flag(0)),
            "e": ("e", "show or hide the edges", lambda: self._toggle_flag(1)),
            "i": ("i", "show or hide the vertex ids", lambda: self._toggle_flag(2)),
            "c": ("c", "colour the edges by their reference graph", lambda: self._toggle_flag(3)),
            "q": ("q", "show or hide the query markers — start node, path and target", lambda: self._toggle_flag(4)),
            "0": ("0", "restore the initial framing", self._reset_view),
            "escape": ("Esc", "release the selection; the start node re-defaults", self._clear_focus),
            "h": ("h", "open and close the control sheet", self._toggle_help),
        }
        self._help_window: tkinter.Toplevel | None = None
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._apply_coords()
        self.redraw()

    # ---------------------------------------------------------------- selectors

    def _build_selectors(self) -> None:
        """
        Builds the top strip: every dropdown and numeric field the settings need, in one native row.

        matplotlib removed its own ``OptionMenu``, but this window is Tk, so the toolkit's real widget
        is right there: a third of the height of a radio grid, and it behaves like every other dropdown
        on the machine. The numeric settings — vertex count, degree, result count, exploration factor —
        are spinboxes rather than the sliders that used to sit below the canvas, so the whole control
        surface is one strip and the freed right column goes to the text panel. A canvas that is not
        Tk's — the headless backend the tests run on — gets no strip at all, which is what keeps the
        viewer importable without a display. The help and save buttons ride the same strip, packed to
        the right, so they sit level with the fields instead of floating over the figure where no
        matplotlib axis could keep them aligned.
        """
        canvas = self._tk_canvas()
        if canvas is None:
            return

        strip = tkinter.Frame(canvas.master, bg="#f2f4f7")
        strip.pack(side="top", fill="x", padx=10, pady=(6, 0), before=canvas)
        # Tk keeps widgets through the interpreter, but the Python proxies own their command callbacks,
        # so the menus have to stay referenced for as long as the window is open.
        self._selector_strip = strip
        # The strip reads left to right in the order a setting takes effect: which cloud, how many
        # points, which graph, how connected it is, whether the cloud is lifted, the two distances,
        # how many answers, and how wide the search reaches. The help and save buttons are packed to
        # the right, save first so it lands to the left of help.
        self._preset_var, self._preset_menu = self._dropdown(
            strip, "distribution", list(PRESETS), self.preset, self._on_preset
        )
        self._num_var, self._num_spin = self._field(
            strip, "vertices", str(self.num_points), MIN_POINTS, MAX_POINTS, 100, self._commit_vertices, width=6
        )
        self._view_var, self._view_menu = self._dropdown(strip, "graph", list(GRAPH_ORDER), self.view, self._on_view)
        self._wire_dropdown(self._preset_menu, self._on_preset)
        self._wire_dropdown(self._view_menu, self._on_view)
        self._k_var, self._k_spin = self._field(strip, "k", str(self.k), K_MIN, K_LIMIT, 1, self._commit_k, width=3)
        self._mips_var, self._mips_menu = self._dropdown(
            strip, "mips transform", ["off", "on"], "on" if self.mips else "off", self._on_mips
        )
        self._metric_var, self._metric_menu = self._dropdown(
            strip,
            "build metric",
            [METRIC_LABELS[choice] for choice in METRIC_ORDER],
            METRIC_LABELS[self.metric],
            self._on_metric,
        )
        self._search_var, self._search_menu = self._dropdown(
            strip,
            "search metric",
            [METRIC_LABELS[choice] for choice in METRIC_ORDER],
            METRIC_LABELS[self.search_metric],
            self._on_search_metric,
        )
        self._wire_dropdown(self._mips_menu, self._on_mips)
        self._wire_dropdown(self._metric_menu, self._on_metric)
        self._wire_dropdown(self._search_menu, self._on_search_metric)
        self._top_k_var, self._top_k_spin = self._field(
            strip, "top k", str(self.top_k), TOP_K_MIN, TOP_K_MAX, 1, self._commit_top_k, width=3
        )
        self._eps_var, self._eps_spin = self._field(
            strip,
            "eps",
            f"{self.path_eps:.2f}",
            EPS_MIN,
            EPS_MAX,
            0.05,
            self._commit_eps,
            width=8,
            format="%.2f",
        )
        self._help_button = ttk.Button(strip, text="help", command=self._toggle_help)
        self._help_button.pack(side="right")
        self._save_button = ttk.Button(strip, text="save", command=self.save_figure)
        self._save_button.pack(side="right")
        self._sync_k_enabled()

    def _field(
        self,
        strip: tkinter.Frame,
        label: str,
        value: str,
        from_: float,
        to: float,
        increment: float,
        commit: Callable[[str], None],
        width: int = 4,
        format: str | None = None,
    ) -> tuple[tkinter.StringVar, ttk.Spinbox]:
        """
        Adds one labelled spinbox field to the strip and hands back the pair the viewer has to keep.

        A numeric setting is a spinbox rather than a dropdown: the arrows commit on their own, and a
        typed value commits on Return or on losing the focus — the same contract every field here uses,
        so a half-typed value never reaches the graph. `commit` parses, clamps and reverts the raw text.
        """
        tkinter.Label(strip, text=label, bg="#f2f4f7", fg="#5f6368", font=("DejaVu Sans", 9, "bold")).pack(
            side="left", padx=(0, 6)
        )
        variable = tkinter.StringVar(strip, value=value)
        spinbox = ttk.Spinbox(
            strip,
            from_=from_,
            to=to,
            increment=increment,
            width=width,
            textvariable=variable,
            command=lambda: commit(variable.get()),
            **({"format": format} if format is not None else {}),
        )
        spinbox.configure(font=("DejaVu Sans", 9), background="#ffffff")
        spinbox.bind("<Return>", lambda _event: commit(variable.get()))
        spinbox.bind("<FocusOut>", lambda _event: commit(variable.get()))
        spinbox.pack(side="left", padx=(0, 18))
        return variable, spinbox

    def _dropdown(
        self,
        strip: tkinter.Frame,
        label: str,
        options: list[str],
        chosen: str,
        choose: Callable[[str], None],
    ) -> tuple[tkinter.StringVar, ttk.Combobox]:
        """
        Adds one labelled dropdown to the strip and hands back the pair the viewer has to keep.

        The combobox is sized to its longest entry instead of a fixed width. Four of them share one
        row above a 1500 px window, and a flat eleven characters each — right for `delaunay`, eleven
        too many for `IP` — asks for 1496 of the 1506 px the strip is given. Sizing to the content
        leaves the row room to spare whatever the toolkit's font metrics turn out to be.
        """
        tkinter.Label(strip, text=label, bg="#f2f4f7", fg="#5f6368", font=("DejaVu Sans", 9, "bold")).pack(
            side="left", padx=(0, 6)
        )
        variable = tkinter.StringVar(strip, value=chosen)
        combobox = ttk.Combobox(
            strip,
            textvariable=variable,
            values=options,
            state="readonly",
            width=max(len(option) for option in options) + 2,
        )
        combobox.configure(font=("DejaVu Sans", 9))
        combobox.bind("<<ComboboxSelected>>", lambda _event: choose(combobox.get()))
        combobox.pack(side="left", padx=(0, 18))
        return variable, combobox

    # ---------------------------------------------------------------- rendering

    def _knng_for(self) -> np.ndarray:
        """
        The knng edge list for the current points, k and metric, built once and cached.

        The cache is dropped by `_rebuild`, which every k-field and metric change runs through, so a
        stale list is never drawn or searched after the cloud, the degree or the distance moves.
        """
        if self._knng_edges is None:
            began = time.perf_counter()
            self._knng_edges = knng_edges(self.model.points, self.k, self.metric)
            self._knng_seconds = time.perf_counter() - began
        return self._knng_edges

    def _nsw_for(self) -> np.ndarray:
        """
        The NSW edge list for the current points, k and metric, built once and cached.

        The same contract as `_knng_for`: dropped by `_rebuild`, so a stale list is never drawn or
        searched after the cloud, the degree or the distance moves.
        """
        if self._nsw_edges is None:
            began = time.perf_counter()
            self._nsw_edges = nsw_edges(self.model.points, self.k, self.metric)
            self._nsw_seconds = time.perf_counter() - began
        return self._nsw_edges

    def _nsg_for(self) -> np.ndarray:
        """
        The NSG edge list for the current points and metric, built once and cached.

        The same contract as `_knng_for` and `_nsw_for`: dropped by `_rebuild`, so a stale list is never
        drawn or searched after the cloud or the distance moves. The greedy walk reads no `k`, so the
        field stays greyed out while this graph is on screen.
        """
        if self._nsg_edges is None:
            began = time.perf_counter()
            self._nsg_edges = nsg_edges(dissimilarities(self.model.points, self.metric))
            self._nsg_seconds = time.perf_counter() - began
        return self._nsg_edges

    def _display_edges(self) -> np.ndarray:
        """The pairs the edge collection draws: the DEG's own, the knng's, the NSW's, the reference graph's, or none."""
        if self.view == DEG_VIEW:
            return self.model.edges
        if self.view == KNNG_VIEW:
            return self._knng_for()
        if self.view == NSW_VIEW:
            return self._nsw_for()
        if self.view == NSG_VIEW:
            return self._nsg_for()
        if self.view == NONE_VIEW or self.model.theory is None:
            return np.zeros((0, 2), dtype=np.int32)
        return self.model.theory.edges_of(self.view, self.model.num_points)

    def redraw(self) -> None:
        """Re-derives every artist from the current model, view, focus and overlay flags."""
        model = self.model
        # Frame the cloud before any head is derived: the heads are measured in display pixels through the
        # axis transform, so the framing they are measured against must already be the one they draw under.
        if self._home is None:
            self._home = _apply_limits(self._ax, model.plot_points)
        edges = self._display_edges()
        endpoints = model.plot_points[edges]
        self._edges.set_segments(endpoints)
        if self.view == DEG_VIEW:
            self._edges.set_colors(edge_colours(model, self.overlays) if self.show_colours else EDGE_COLOR)
        elif self.view == KNNG_VIEW:
            self._edges.set_colors(KNNG_COLOR)
        elif self.view == NSW_VIEW:
            self._edges.set_colors(NSW_COLOR)
        elif self.view == NSG_VIEW:
            self._edges.set_colors(NSG_COLOR)
        else:
            self._edges.set_colors(EDGE_COLOR if self.view == NONE_VIEW else OVERLAP_COLORS[self.view])
        self._edges.set_visible(self.show_edges)
        # A directed graph caps every shaft with a head. The heads are re-derived from the live axis
        # transform on each paint, so they stay square and fixed-size through a zoom, and they drop out
        # with the edges or on any undirected view, which leaves the plain strokes as they were.
        if self._view_directed() and self.show_edges:
            self._arrowheads.set_facecolor(NSG_COLOR if self.view == NSG_VIEW else KNNG_COLOR)
            self._arrowheads.set_paths(
                arrowhead_triangles(endpoints, self._ax.transData, ARROWHEAD_PX, ARROWHEAD_WIDTH_PX)
            )
            self._arrowheads.set_visible(True)
        else:
            self._arrowheads.set_paths([])
            self._arrowheads.set_visible(False)
        self._legend.set_visible(self.view == DEG_VIEW and self.show_colours and model.theory is not None)
        self._style_legend()
        self._vertices.set_offsets(model.plot_points)
        self._vertices.set_facecolor(vertex_colors(model))
        self._paint_focus()
        self._paint_free_query()
        self._paint_query()
        self._paint_labels()
        name = "DEG" if self.view == DEG_VIEW else self.view
        self._ax.set_title(
            f"{name} on {self.preset} · {model.num_points} vertices · {edges.shape[0]} edges", fontsize=11, loc="left"
        )
        self._panel_text.set_text(self._panel_lines())
        self._fig.canvas.draw_idle()

    def _paint_focus(self) -> None:
        # The start node, the selected vertex's ring and its highlighted edges are the traversal's own
        # marks; the `query` overlay switches them off as a set for a clean picture.
        if not self.show_query:
            self._entry_mark.set_offsets(NO_POINTS)
            self._focus.set_segments([])
            self._focus_arrows.set_paths([])
            self._ring.set_offsets(NO_POINTS)
            return
        # The start node is drawn whenever one is set — the empty view is the one exception, since a
        # traversal seed without a graph to walk would mark a start to a journey that cannot happen.
        show_entry = self.entry is not None and self.view != NONE_VIEW
        self._entry_mark.set_offsets(NO_POINTS if not show_entry else self.model.plot_points[[self.entry]])
        if self.selected is None:
            self._focus.set_segments([])
            self._focus_arrows.set_paths([])
            self._ring.set_offsets(NO_POINTS)
            return
        edges = self._display_edges()
        if edges.size:
            # Read off the drawn edge list rather than the DEG adjacency, so the highlight follows
            # whichever graph is on screen.
            touching = (edges[:, 0] == self.selected) | (edges[:, 1] == self.selected)
            focused = self.model.plot_points[edges[touching]]
            self._focus.set_segments(focused)
            if self._view_directed():
                self._focus_arrows.set_paths(
                    arrowhead_triangles(focused, self._ax.transData, ARROWHEAD_PX, ARROWHEAD_WIDTH_PX)
                )
            else:
                self._focus_arrows.set_paths([])
        else:
            self._focus.set_segments([])
            self._focus_arrows.set_paths([])
        self._ring.set_offsets(self.model.plot_points[[self.selected]])

    def _paint_free_query(self) -> None:
        """
        Places the cross that marks a free-space query; hidden while no free query is set.

        The empty view hides it as it hides the start node: a query marker on a screen without a graph
        points at a search that cannot run. The point itself stays chosen — switch back to any graph and
        the cross, and the query it marks, are exactly where they were.
        """
        show = self.free_query is not None and self.view != NONE_VIEW and self.show_query
        self._free_mark.set_offsets(NO_POINTS if not show else self.free_query[None, :])

    def _paint_query(self) -> None:
        query, points = self.query, self.model.plot_points
        if not self.show_query:
            self._path.set_segments([])
            self._path.set_array(None)
            self._terminal_mark.set_offsets(NO_POINTS)
            self._exact_mark.set_offsets(NO_POINTS)
            self._connector.set_visible(False)
            return
        if query is None:
            self._path.set_segments([])
            self._path.set_array(None)
            self._terminal_mark.set_offsets(NO_POINTS)
            self._exact_mark.set_offsets(NO_POINTS)
            self._connector.set_visible(False)
            return
        self._exact_mark.set_offsets(points[[query.exact]])
        if query.deg_indices.size == 0:
            # A search that returned nothing has no answer to aim the traversal at, so there is no X
            # to mark and no walk that could have been taken — only the star of the ground truth stays.
            self._path.set_segments([])
            self._path.set_array(None)
            self._terminal_mark.set_offsets(NO_POINTS)
            self._connector.set_visible(False)
            return
        answer = int(query.deg_indices[0])
        star, target = points[query.exact], points[answer]
        # The path is the search's own recorded route to its top answer, so it is present whenever there is
        # an answer and it ends on that answer — the walk never stalls before what the search returned.
        walk = points[query.path]
        self._path.set_segments(np.stack([walk[:-1], walk[1:]], axis=1))
        if query.hops:
            self._path.set_array(np.linspace(0.15, 0.95, query.hops))
            self._path.set_cmap("cividis")
        else:
            self._path.set_array(None)
        # The walk ends on the search's answer: a green X when that answer is the ground truth — the search
        # answered correctly — and a red one when it is not, with a dashed line from the X to the star
        # measuring the miss the search took.
        self._terminal_mark.set_offsets(points[[answer]])
        self._terminal_mark.set_color(EXACT_COLOR if answer == query.exact else TERMINAL_COLOR)
        self._connector.set_data([target[0], star[0]], [target[1], star[1]])
        self._connector.set_visible(answer != query.exact)

    def _paint_labels(self) -> None:
        for artist in self._labels:
            artist.remove()
        self._labels = []
        if not self.show_ids:
            return
        if self.model.num_points > ID_LIMIT:
            self._labels.append(
                self._ax.text(
                    0.012,
                    0.012,
                    f"vertex ids are hidden above {ID_LIMIT} vertices",
                    transform=self._ax.transAxes,
                    fontsize=8,
                    color="#80868b",
                )
            )
            return
        for index in range(self.model.num_points):
            x, y = self.model.plot_points[index]
            self._labels.append(
                self._ax.text(x, y, str(index), fontsize=6, color="#5f6368", ha="left", va="bottom", zorder=7)
            )

    def _view_directed(self) -> bool:
        """
        Whether the graph on screen is directed. The knng and the NSG are — each keeps the links in the
        direction they were chosen — while the DEG, the NSW and every reference graph, including the
        undirected MRNG, are undirected.

        One source for the fact, read both by the statistics (which count in- and out-degree apart only
        when it holds) and by the renderer (which draws arrowheads only when it does).
        """
        return self.view in (KNNG_VIEW, NSG_VIEW)

    def _view_stats(self) -> dict | None:
        """
        The drawn graph's degree, source/sink and reachability numbers, recomputed only when the graph
        the panel reports actually changed.

        The panel rewrites on every hover, while a graph's statistics move only with the graph itself —
        so the numbers are keyed by everything they read (the model, the view, the degree, the entry and
        the metric) and recomputed on a key miss. The empty view has no graph and answers `None`.
        """
        if self.view == DEG_VIEW:
            edges = self.model.edges
        elif self.view == KNNG_VIEW:
            edges = self._knng_for()
        elif self.view == NSW_VIEW:
            edges = self._nsw_for()
        elif self.view == NSG_VIEW:
            edges = self._nsg_for()
        elif self.view == NONE_VIEW or self.model.theory is None:
            return None
        else:
            edges = self.model.theory.edges_of(self.view, self.model.num_points)
        key = (id(self.model), self.view, self.k, self.entry, self.metric)
        if self._stats_cache is None or self._stats_cache[0] != key:
            self._stats_cache = (
                key,
                graph_stats(edges, self.model.num_points, self._view_directed(), self.entry),
            )
        return self._stats_cache[1]

    def _panel_lines(self) -> list[tuple[str, str]]:
        """
        Builds the panel as styled lines: a heading naming the graph actually on screen, its shape
        summarised beneath, then the blocks in a fixed order — target, free query, start node, query,
        status — and the hover last.

        The order is the contract: everything the user set sits on a stable line, and only the hover —
        the one block that comes and goes with the mouse — closes the column, so no block ever jumps.

        The heading follows the view, so a reference graph never reports the DEG's numbers as its own:
        the DEG carries the full statistics from `describe` in grey, while a reference graph, the knng
        or the empty view states its own edge count and keeps the DEG's headline as context. Block
        labels are bold blue, details monospace, the search's hit line green and the status orange —
        hierarchy by type, so the eye finds the section it wants without reading every line. No line
        may pass `PANEL_WIDTH`, and at the widget caps the column must still stop above the checkboxes.
        """
        stats = describe(self.model)
        if self.view == DEG_VIEW or self.model.theory is None:
            lines: list[tuple[str, str]] = [("DEG", "head"), (stats[0], "sub")]
            lines += [(f"  {line}", "dim") for line in stats[1:]]
        else:
            if self.view == KNNG_VIEW:
                head = "knng"
                sub = (
                    f"{self._knng_for().shape[0]} edges · k={self.k} · {METRIC_LABELS[self.metric]}"
                    f" · {self._knng_seconds * 1000.0:.1f} ms"
                )
            elif self.view == NSW_VIEW:
                head = "nsw"
                sub = (
                    f"{self._nsw_for().shape[0]} edges · k={self.k} · {METRIC_LABELS[self.metric]}"
                    f" · {self._nsw_seconds * 1000.0:.1f} ms"
                )
            elif self.view == NSG_VIEW:
                head = "nsg"
                sub = (
                    f"{self._nsg_for().shape[0]} edges · {METRIC_LABELS[self.metric]}"
                    f" · {self._nsg_seconds * 1000.0:.1f} ms"
                )
            elif self.view == NONE_VIEW:
                head, sub = "no graph", f"{self.model.num_points} vertices · nothing drawn"
            else:
                item = self.model.theory.graph(self.view)
                head = self.view
                sub = (
                    f"{item.edges} edges · ∩ {item.share_of(self.model.theory.deg_edges):.1%} of DEG"
                    f" · {item.seconds * 1000.0:.1f} ms"
                )
            lines = [(head, "head"), (sub, "sub"), (f"  DEG {stats[0]}", "dim")]
        graph = self._view_stats()
        if graph is not None:
            # An undirected graph has one degree per vertex and no source/sink distinction — in- and
            # out-degree are the same number and the sources are exactly the isolated vertices — so it
            # is read as a single span and an isolated count. Only a directed graph earns the split.
            if self._view_directed():
                degree = (
                    f"  degree   out {graph['min_out']}–{graph['max_out']} · in {graph['min_in']}–{graph['max_in']}"
                )
                ends = f"  sources  {graph['sources']} · sinks {graph['sinks']}"
            else:
                degree = f"  degree   {graph['min_out']}–{graph['max_out']}"
                ends = f"  isolated {graph['sources']}"
            # The DEG block already prints its own degree line from `describe`, so there only the
            # reachability and isolated count are added; every other graph gets the degree line too.
            if self.view != DEG_VIEW:
                lines.append((degree, "dim"))
            lines += [
                (ends, "dim"),
                (
                    f"  reach    search {graph['search_reachability']:.1%} · explore {graph['explore_reachability']:.1%}",
                    "dim",
                ),
            ]
        focus = self.selected
        if focus is not None:
            x, y = self.model.plot_points[focus]
            lines += [
                ("", "body"),
                (f"TARGET   {focus}  ({x:.3f}, {y:.3f})", "label"),
                *[(line, "body") for line in _fan_out(self.model.adjacency[focus])],
            ]
        elif self.free_query is None:
            lines += [("", "body"), ("TARGET   none — left-click a vertex", "dim")]
        if self.free_query is not None:
            # A free point has no vertex id and no neighbours, and it keeps its line even while a hover
            # reads out another vertex — the query is the user's choice, the hover is only the mouse.
            x, y = self.free_query
            lines += [("", "body"), (f"QUERYPT  ({x:.3f}, {y:.3f})", "label")]
        if self.entry is not None:
            x, y = self.model.plot_points[self.entry]
            lines += [("", "body"), (f"START    {self.entry}  ({x:.3f}, {y:.3f})", "label")]
        if self.query is not None:
            # The route line reports the search's own run: how many edges its recorded walk took, how many
            # distances it measured, and how many vertices it checked on the way to its top answer.
            start = self.query.entry if self.query.entry is not None else "—"
            lines += [
                ("", "body"),
                (f"QUERY    {METRIC_LABELS[self.query.metric]}  eps={self.query.eps:.2f}  from {start}", "label"),
                (
                    f"  route    {self.query.hops} hops · {self.query.distance_computations} dist"
                    f" · {self.query.expanded} checked",
                    "body",
                ),
                *self._search_lines(self.query),
                # The truth line is the brute-force best match — the vertex the search is graded against,
                # green when its own answers include that vertex and red when the search walked past it.
                (
                    f"  truth    {self.query.exact} · {'hit' if self.query.deg_hit else 'missed'}",
                    "good" if self.query.deg_hit else "bad",
                ),
            ]
        if self.status is not None:
            lines += [("", "body"), (f"STATUS   {self.status}", "warn")]
        if self.hovered is not None and self.selected is None:
            # The hover only reads out whatever the mouse passes, so it closes the column: appearing and
            # vanishing with the mouse can no longer shift the blocks above it by two lines at a time.
            x, y = self.model.plot_points[self.hovered]
            lines += [
                ("", "body"),
                (f"HOVER    {self.hovered}  ({x:.3f}, {y:.3f})", "label"),
                *[(line, "body") for line in _fan_out(self.model.adjacency[self.hovered])],
            ]
        return lines

    def _search_lines(self, query: QueryResult) -> list[tuple[str, str]]:
        """
        The search block: one line per result, in the search's own ranking order.

        The first result sits behind `SEARCH_LABEL` and the rest repeat it as blank padding, so a widened
        list reads as one block. The `hit` marker lands on whichever line names the best match — and that
        line turns green, so a recovered search shows from across the room while the plain results stay
        quiet. An empty result list says so in the warning colour rather than printing nothing.
        """
        if query.deg_indices.size == 0:
            return [(f"{SEARCH_LABEL}no result", "warn")]
        pad = " " * len(SEARCH_LABEL)
        return [
            (
                f"{SEARCH_LABEL if rank == 1 else pad}#{rank} {int(index)} (d={distance:.3f})"
                + (" hit" if int(index) == query.exact else ""),
                "good" if int(index) == query.exact else "body",
            )
            for rank, (index, distance) in enumerate(zip(query.deg_indices, query.deg_distances), start=1)
        ]

    def _format_coord(self, x: float, y: float) -> str:
        text = f"x={x:6.3f}  y={y:6.3f}"
        if self.hovered is not None:
            text += f"  ·  vertex {self.hovered}"
        return text

    # ---------------------------------------------------------------- interaction

    def _nearest(self, event: MouseEvent) -> int | None:
        if event.inaxes is not self._ax or self.model.num_points == 0:
            return None
        display = self._ax.transData.transform(self.model.plot_points) - np.array([event.x, event.y])
        squared = np.einsum("ij,ij->i", display, display)
        index = int(np.argmin(squared))
        return index if squared[index] <= PICK_RADIUS_PX**2 else None

    def _on_click(self, event: MouseEvent) -> None:
        """
        Left-click selects a vertex, or — on empty space — places a free-space query at the clicked
        coordinates and arms a pan; right-click sets the traversal start node; a double-click runs the query.
        """
        if self._on_legend(event):
            return
        # Clicks on the control widgets must not be read as "empty space" — that would place a free
        # query the widget is about to act on.
        if event.button not in (1, 3) or event.inaxes is not self._ax:
            return
        index = self._nearest(event)
        if event.button == 3:
            self.set_entry(index)
            return
        if index is None:
            self.set_free_query(np.array([event.xdata, event.ydata], dtype=np.float32))
        else:
            self.select(index)
        if event.dblclick:
            self.run_query()
            self._drag = None
        else:
            self._drag = _capture_drag(self._ax, event)

    def _on_motion(self, event: MouseEvent) -> None:
        if self._drag is not None:
            self._pan(self._drag, event)
            return
        index = self._nearest(event)
        if index == self.hovered:
            return
        self.hovered = index
        self._hover_ring.set_offsets(NO_POINTS if index is None else self.model.plot_points[[index]])
        self._panel_text.set_text(self._panel_lines())
        self._fig.canvas.draw_idle()

    def _clear_focus(self) -> None:
        """Releases the selection and the free-space query; the start node falls back to its default, since one always exists."""
        self.free_query = None
        self.select(None)
        self.set_entry(None)
        self.redraw()

    def _on_key(self, event: MouseEvent) -> None:
        key = (event.key or "").lower()
        binding = self._bindings.get("h" if key == "?" else key)
        if binding is not None:
            binding[2]()

    def _tk_canvas(self) -> tkinter.Misc | None:
        """The Tk widget this canvas draws into, or `None` on a backend that is not Tk's."""
        get_tk_widget = getattr(self._fig.canvas, "get_tk_widget", None)
        return None if get_tk_widget is None else get_tk_widget()

    def _toggle_help(self, *_) -> None:
        """
        Opens the control sheet in a window of its own.

        It stays out of the figure on purpose: the drawing is the reason the window is open, so covering
        it with a list of keys buries the thing those keys are for. A window of its own also survives a
        resize of the plot and can be pushed aside while the mouse works.
        """
        if self._help_window is not None:
            self._close_help()
            return
        canvas = self._tk_canvas()
        if canvas is None:
            return

        sheet = tkinter.Toplevel(canvas.master, bg="white")
        sheet.title("Controls")
        sheet.resizable(False, False)
        rows = (*MOUSE_HELP, *(binding[:2] for binding in self._bindings.values()))
        for row, (key, meaning) in enumerate(rows):
            tkinter.Label(sheet, text=key, bg="white", fg="#1a73e8", font=("DejaVu Sans Mono", 9)).grid(
                row=row, column=0, sticky="w", padx=(14, 12), pady=1
            )
            tkinter.Label(sheet, text=meaning, bg="white", fg="#2a2e35", font=("DejaVu Sans", 9)).grid(
                row=row, column=1, sticky="w", padx=(0, 16)
            )
        # The window can be closed from its own title bar, and the next key press must not destroy a corpse.
        sheet.protocol("WM_DELETE_WINDOW", self._close_help)
        self._help_window = sheet

    def _close_help(self) -> None:
        window, self._help_window = self._help_window, None
        if window is not None:
            window.destroy()

    def _on_preset(self, label: str) -> None:
        """
        Switches to another distribution, the one change that starts the search over.

        A vertex index names a different point on every cloud, so the target, the start node and any
        free-space query are released before the rebuild; every other knob keeps them where they are.
        """
        if label != self.preset:
            self.preset = label
            self.selected = None
            self.free_query = None
            self.entry = None
            self._rebuild()

    def _on_mips(self, label: str) -> None:
        """
        Lifts the cloud into the MIPS→L2 spherical space or drops it back, which rebuilds the graph.

        The transform changes what the graph is built on — a third coordinate of equal norm — so unlike a
        search-metric switch this is a real rebuild. The projected cloud is regenerated from the same seed,
        so the drawing does not move; only the features the graph and searches run on gain a dimension.
        """
        mips = label == "on"
        if mips != self.mips:
            self.mips = mips
            self._rebuild()

    def _on_view(self, name: str) -> None:
        """
        Switches which graph is drawn and re-runs the open query on it.

        A graph the current metric does not build is refused with a status line instead of drawn. The
        dropdown only lists graphs the report holds, but the `v` key drives the same choice and a switch
        to inner product can leave the graph already on screen without a report entry behind it. Switching
        to the DEG snaps the k field to the even value the DEG builds with, since an odd field value is
        only meaningful to the knng. A target and a start node survive the switch — the search runs over
        any graph now — so the query is replayed on the new one instead of being dropped.
        """
        if name not in self.views:
            self.status = f"{name} needs L2 — not built under {METRIC_LABELS[self.metric]}"
            self._panel_text.set_text(self._panel_lines())
            self._fig.canvas.draw_idle()
            return
        if name == DEG_VIEW and self.k != self._deg_k():
            self.k = self._deg_k()
            if self._k_var is not None:
                self._k_var.set(str(self.k))
        if self._view_var is not None:
            self._view_var.set(name)  # the `v` key changes the view without the dropdown knowing
        # Arrow-key browsing in the open dropdown already applied this graph and the popup's own event
        # then repeats the choice — the redraw must not run twice, but the snap and the sync below are
        # owed to every choice. The k field is synced after the draw, since it follows the view the
        # draw installs, not the one still on screen.
        if name != self.view:
            self._draw_view(name)
        self._sync_k_enabled()

    def _draw_view(self, name: str) -> None:
        """
        Draws one view and replays the open query on it.

        The query survives a graph switch whenever its target and the start node are still there —
        searching the graph on screen is the point — while the empty view has nothing to search and
        answers with the status line naming the missing graph.
        """
        self.view = name
        self.query = None
        self.status = None
        self.redraw()
        target = self.selected if self.selected is not None else self.free_query
        if name == NONE_VIEW:
            self.status = "no graph — pick one"
            self._panel_text.set_text(self._panel_lines())
            self._fig.canvas.draw_idle()
        elif target is not None and self.entry is not None:
            self.run_query()

    def _wire_dropdown(self, combo: ttk.Combobox, choose: Callable[[str], None]) -> None:
        """
        Lets the arrow keys choose from an open dropdown exactly as a click would.

        ttk's popdown is a Tcl-owned listbox that arrives at the first opening with its bindtags
        emptied — not even its own path, its class or `all` — so nothing it receives reaches any
        binding. Wiring therefore waits for the popdown to exist, restores those tags and binds
        `KeyRelease` and `Motion` at the Tcl level: whatever entry the cursor stands on is applied
        through `choose` and the combobox's own value follows it, so closing the popup — by click,
        key or focus loss — keeps the selection instead of reverting it.
        """
        listbox = f"{combo}.popdown.f.l"
        wired = [False]

        def on_browse(*_args) -> None:
            try:
                if not int(combo.tk.call("winfo", "ismapped", listbox)):
                    return
                selected = combo.tk.call(listbox, "curselection")
                if not selected:
                    return
                value = str(combo.tk.call(listbox, "get", selected[0]))
                if value == combo.get():
                    return
                combo.set(value)
                choose(value)
            except tkinter.TclError:
                pass

        def wire() -> None:
            if wired[0]:
                return
            try:
                if not int(combo.tk.call("winfo", "exists", listbox)):
                    return
                widget_class = combo.tk.call("winfo", "class", listbox)
                combo.tk.call("bind", "tags", listbox, (listbox, widget_class, "all"))
                for sequence in ("<KeyRelease>", "<Motion>"):
                    command = f"choose_{sequence.strip('<>')}_{id(combo)}"
                    combo.tk.createcommand(command, on_browse)
                    combo.tk.call("bind", listbox, sequence, command)
                wired[0] = True
            except tkinter.TclError:
                return

        def schedule(*_args) -> None:
            combo.after_idle(wire)

        wire()
        combo.bind("<Button-1>", schedule, add="+")
        combo.bind("<KeyPress>", schedule, add="+")

    def _cycle_view(self) -> None:
        """Moves to the next graph in the selector, so the views can be walked without the mouse."""
        self._on_view(self.views[(self.views.index(self.view) + 1) % len(self.views)])

    def _snap_even(self, value: int) -> int:
        """The even degree the DEG builds with for a field value: clamped to `MIN_K`..`K_LIMIT`, odd snapped up."""
        clamped = min(max(int(value), MIN_K), K_LIMIT)
        return clamped if clamped % 2 == 0 else min(clamped + 1, K_LIMIT)

    def _deg_k(self) -> int:
        """The even degree the DEG is built at, snapped from the k field's current value."""
        return self._snap_even(self.k)

    def _sync_k_enabled(self) -> None:
        """The k field drives the DEG, the knng and the NSW, so it is greyed out for every other view."""
        if self._k_spin is not None:
            active = self.view in (DEG_VIEW, KNNG_VIEW, NSW_VIEW)
            # The DEG builds at an even degree, so its arrows step by 2: a step of 1 lands on an odd value
            # that snaps back up to the same even, which leaves the down arrow inert. The knng and NSW read
            # k raw as a neighbour count, so they keep the finer step of 1.
            self._k_spin.configure(
                state="normal" if active else "disabled",
                increment=2 if self.view == DEG_VIEW else 1,
            )

    def _on_vertices(self, value: float) -> None:
        num_points = int(value)
        if num_points != self.num_points:
            self.num_points = num_points
            self._rebuild()

    def _commit_vertices(self, raw: str) -> None:
        """Reads a committed vertex count: clamped into the cloud's bounds, garbage answered by the stored value."""
        try:
            num_points = min(max(int(raw), MIN_POINTS), MAX_POINTS)
        except ValueError:
            num_points = self.num_points
        if self._num_var is not None:
            self._num_var.set(str(num_points))
        self._on_vertices(num_points)

    def _on_k(self, value: float) -> None:
        """
        Sets the k field's value and rebuilds.

        While the DEG is on screen the field is snapped to the even degree it builds with, so an odd
        value cannot sit in the field beside a graph that never uses it; the knng keeps the raw value,
        which is its neighbour count directly. The DEG is rebuilt at the snapped degree either way.
        """
        value = int(value)
        if self.view == DEG_VIEW:
            value = self._snap_even(value)
            if self._k_var is not None:
                self._k_var.set(str(value))
        if value != self.k:
            self.k = value
            self._rebuild()

    def _commit_k(self, raw: str) -> None:
        """Reads a committed k: clamped to the field's bounds, garbage answered by the stored value."""
        try:
            value = min(max(int(raw), K_MIN), K_LIMIT)
        except ValueError:
            value = self.k
        if self._k_var is not None:
            self._k_var.set(str(value))
        self._on_k(value)

    def _on_metric(self, label: str) -> None:
        """Rebuilds under the other metric; the reference graphs follow it, being read from the same matrix."""
        metric = METRIC_BY_LABEL[label]
        if metric == self.metric:
            return
        self.metric = metric
        self._rebuild()

    def _on_search_metric(self, label: str) -> None:
        """
        Switches the comparator the traversal ranks its neighbours by, which needs no rebuild.

        The graph keeps the edges it was built with — only the greedy step's choice of neighbour
        changes — so an open query is re-run and redrawn while the cloud stays exactly where it was.
        That is the point of keeping the two metrics apart: one graph, two answers.
        """
        metric = METRIC_BY_LABEL[label]
        if metric == self.search_metric:
            return
        self.search_metric = metric
        if self.query is not None:
            self.run_query()

    def _on_eps(self, value: float) -> None:
        """Re-runs the current query at a new exploration factor; the graph itself is unaffected."""
        eps = float(value)
        if eps == self.path_eps:
            return
        self.path_eps = eps
        if self.query is not None:
            self.run_query()

    def _commit_eps(self, raw: str) -> None:
        """Reads a committed exploration factor: clamped to the field's bounds, garbage answered by the stored value."""
        try:
            eps = min(max(float(raw), EPS_MIN), EPS_MAX)
        except ValueError:
            eps = self.path_eps
        if self._eps_var is not None:
            self._eps_var.set(f"{eps:.2f}")
        self._on_eps(eps)

    def _on_top_k(self, value: int) -> None:
        """Re-runs the current query with a wider result list; the graph itself is unaffected."""
        top_k = int(value)
        if top_k == self.top_k:
            return
        self.top_k = top_k
        if self.query is not None:
            self.run_query()

    def _commit_top_k(self, raw: str) -> None:
        """
        Reads a committed top-k value: an integer is clamped into the bounds and stored, garbage is
        answered by putting the stored value back into the field.

        The field is free text between the arrows, so a half-typed value must not search for nothing —
        the same contract the spinbox's own arrows enforce by only ever producing a number.
        """
        try:
            top_k = min(max(int(raw), TOP_K_MIN), TOP_K_MAX)
        except ValueError:
            top_k = self.top_k
        if self._top_k_var is not None:
            self._top_k_var.set(str(top_k))
        self._on_top_k(top_k)

    def _on_check(self, _label: str) -> None:
        self._apply_flags()

    def _apply_flags(self) -> None:
        checked = set(self._checks.get_checked_labels())
        self.show_coords = FLAG_LABELS[0] in checked
        self.show_edges = FLAG_LABELS[1] in checked
        self.show_ids = FLAG_LABELS[2] in checked
        self.show_colours = FLAG_LABELS[3] in checked
        self.show_query = FLAG_LABELS[4] in checked
        self._apply_coords()
        self.redraw()

    def _apply_coords(self) -> None:
        """Turns the coordinate system — the cross through the origin, the tick labels and the spines — on or off as one overlay."""
        for line in self._cross:
            line.set_visible(self.show_coords)
        self._ax.tick_params(labelbottom=self.show_coords, labelleft=self.show_coords)
        for spine in self._ax.spines.values():
            spine.set_visible(self.show_coords)

    def _toggle_flag(self, index: int) -> None:
        self._checks.set_active(index, not self._checks.get_status()[index])
        self._apply_flags()

    # ---------------------------------------------------------------- view

    def _on_scroll(self, event: MouseEvent) -> None:
        """Zooms towards the cursor, never further out than the initial framing."""
        if event.inaxes is not self._ax or not event.step or self._home is None:
            return
        home_x0, home_x1 = self._home[0]
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        scale = 1.0 / (1.0 + ZOOM_FACTOR * event.step)
        if (x1 - x0) * scale >= home_x1 - home_x0:
            self._reset_view()
            return
        if (x1 - x0) * scale <= (home_x1 - home_x0) * MIN_ZOOM:
            return
        x, y = event.xdata, event.ydata
        self._ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        self._ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        self._fig.canvas.draw_idle()

    def _reset_view(self) -> None:
        """Restores the framing derived from the current model."""
        if self._home is None:
            return
        self._ax.set_xlim(self._home[0])
        self._ax.set_ylim(self._home[1])
        self._fig.canvas.draw_idle()

    def _on_release(self, _event: MouseEvent) -> None:
        self._drag = None

    def _pan(self, drag: _Drag, event: MouseEvent) -> None:
        """Shifts the framing so the cloud follows the cursor, never leaving the initial box."""
        if event.x is None or event.y is None or self._home is None:
            return
        x0, x1 = drag.xlim
        y0, y1 = drag.ylim
        # Measured from the press snapshot, not from the previous event, so a framing the backend
        # adjusts to keep the aspect square cannot drift the cloud away over a long drag.
        shifted_x = x0 - (event.x - drag.x) / drag.px_per_unit_x, x1 - (event.x - drag.x) / drag.px_per_unit_x
        shifted_y = y0 - (event.y - drag.y) / drag.px_per_unit_y, y1 - (event.y - drag.y) / drag.px_per_unit_y
        self._ax.set_xlim(*_slide(*shifted_x, self._home[0]))
        self._ax.set_ylim(*_slide(*shifted_y, self._home[1]))
        self._fig.canvas.draw_idle()

    # ---------------------------------------------------------------- commands

    def select(self, index: int | None) -> None:
        """Focuses a vertex — the target of the next path query — which releases any free-space query."""
        if index == self.selected:
            return
        self.selected = index
        self.free_query = None
        self.redraw()

    def set_free_query(self, point: np.ndarray | None) -> None:
        """
        Places the query at a plot-space coordinate that is not a vertex, releasing any selection.

        A free point is searched as itself rather than as some vertex's vector, so the traversal and its
        ground truth run over the whole cloud with nothing excluded — there is no query vertex to skip.
        """
        self.free_query = point
        self.selected = None
        self.redraw()

    def set_entry(self, index: int | None) -> None:
        """Makes a vertex the traversal seed; `None` re-defaults it to the farthest-from-centre vertex."""
        if index is None:
            index = self.model.farthest_from_centroid()
        if index == self.entry:
            return
        self.entry = index
        self.redraw()

    def reseed(self) -> None:
        """Draws a fresh sample from the same distribution."""
        self.seed = int(self._rng.integers(1_000_000))
        self._rebuild()

    def run_query(self) -> QueryResult | None:
        """
        Runs the ε-search for the current query — a selected vertex or a free plot-space coordinate. The
        search is scored against the query's best match and carries its own route to its top answer.

        The search runs over whichever graph is on screen: the DEG, the knng, NSW or NSG built from the
        current feature space, or a reference graph read from the report's edge list. Searching the graph the
        viewer is drawing is the whole point of the Python search — the library's own search only ever
        runs over a built DEG. On the empty view there is no graph to walk, so it names the missing one
        rather than searching.
        """
        if self.view == NONE_VIEW:
            self.query = None
            self.status = "no graph — pick one"
            self._paint_query()
            self._panel_text.set_text(self._panel_lines())
            self._fig.canvas.draw_idle()
            return None
        target = self.selected if self.selected is not None else self.free_query
        if target is None:
            self.query = None
            self.status = "no target — left-click a vertex first"
        else:
            try:
                self.query = query_graph(
                    self.model,
                    target,
                    self.entry,
                    self.path_eps,
                    self.search_metric,
                    top_k=self.top_k,
                    view=self.view,
                    feature_edges=(
                        self._knng_for()
                        if self.view == KNNG_VIEW
                        else self._nsw_for()
                        if self.view == NSW_VIEW
                        else self._nsg_for()
                        if self.view == NSG_VIEW
                        else None
                    ),
                )
            except ValueError as error:
                self.query = None
                self.status = str(error)
            else:
                self.status = None
        self._paint_query()
        self._panel_text.set_text(self._panel_lines())
        self._fig.canvas.draw_idle()
        return self.query

    def _paint_legend(self) -> None:
        """
        Rebuilds the colour key from the report the current model carries. Switching the metric takes two
        reference graphs and their swatches away with it, so the key cannot be built once at startup.
        """
        entries = legend_entries(self.model.theory)
        self._legend = self._legend_ax.legend(
            handles=[Line2D([], [], color=color, linewidth=2.4) for _, color in entries],
            labels=[name for name, _ in entries],
            frameon=False,
            fontsize=8.5,
            loc="center left",
            ncol=len(entries),
            borderpad=0.0,
            columnspacing=2.4,
            handlelength=1.7,
            handletextpad=0.6,
        )
        self._legend.set_visible(False)
        self._style_legend()

    def _held_graphs(self) -> set[str]:
        """The reference graphs the current report carries, all of them switched on."""
        return set(self.model.theory.names) if self.model.theory is not None else set()

    def _legend_names(self) -> tuple[str, ...]:
        """The key's entries in the legend's own order, `DEG only` first."""
        return tuple(name for name, _ in legend_entries(self.model.theory))

    def _style_legend(self) -> None:
        """Dims the entries that are switched off, so the key reads as the switch it now is."""
        names = self._legend_names()
        for position, (handle, text) in enumerate(zip(self._legend.legend_handles, self._legend.get_texts())):
            live = position == 0 or names[position] in self.overlays
            handle.set_alpha(1.0 if live else 0.2)
            text.set_color("#2a2e35" if live else "#b0b5bd")

    def _on_legend(self, event: MouseEvent) -> bool:
        """
        Consumes a click that lands on the colour key, switching that reference graph's colour.

        `DEG only` is deliberately not a switch: the whole overlay already has one in the `theory
        colours` checkbox, and a second control for the same state would only get out of step with it.
        Switching a graph off hands its edges to the next graph in the override order that holds them,
        which is what makes peeling the nesting apart — the tree out of the Gabriel graph out of the
        Delaunay — something the eye can follow.
        """
        if event.inaxes is not self._legend_ax or not self._legend.get_visible():
            return False
        names = self._legend_names()
        renderer = self._fig.canvas.get_renderer()
        legend_bbox = self._legend_ax.get_window_extent(renderer)
        boxes = []
        for handle, text in zip(self._legend.legend_handles, self._legend.get_texts()):
            hb = handle.get_window_extent(renderer)
            tb = text.get_window_extent(renderer)
            # Matplotlib Bbox.union takes a list of bboxes and computes their true bounding envelope
            boxes.append(matplotlib.transforms.Bbox.union([hb, tb]))

        for position, box in enumerate(boxes):
            hit_box = box.padded(10.0)
            hit_box.y0 = legend_bbox.y0
            hit_box.y1 = legend_bbox.y1
            if hit_box.contains(event.x, event.y) and position > 0:
                self.overlays ^= {names[position]}
                self.redraw()
                return True
        return True

    def _adopt_views(self) -> None:
        """
        Takes the view list from the report the model now carries and marks the menu accordingly.

        Inner product admits no Delaunay and no Gabriel, so a metric switch can leave the graph currently
        on screen with nothing to draw. That view falls back to the DEG rather than naming a report entry
        that is not there. A combobox cannot grey entries out, so the dropdown simply lists the graphs
        the new report holds — the ones the metric cannot build are not offered at all.

        The colour key's switches survive the rebuild: a reference graph the user has seen before keeps
        its on/off state, so a metric round-trip does not silently switch a hidden colour back on, while
        a graph appearing in the report for the first time — Delaunay returning after a switch away from
        inner product — has no switch to remember and defaults to on. Hence the new overlay is the held
        graphs kept down to those previously on, widened by those never seen: `held & (on | held - seen)`.
        """
        # The DEG, the knng, the NSW, the NSG and the empty view are offered under every metric; the
        # reference graphs — the undirected MRNG among them — come and go with the report. The knng, the
        # NSW and the NSG are built by the viewer like the DEG, so they are offered even when the report is
        # absent. The menu keeps `GRAPH_ORDER`'s order, listing a graph only when the report carries it (or
        # it is one of the always-available views), so a metric that drops Delaunay and Gabriel drops them
        # from the menu too.
        names = set(self.model.theory.names) if self.model.theory is not None else set()
        available = names | {DEG_VIEW, KNNG_VIEW, NSW_VIEW, NSG_VIEW, NONE_VIEW}
        self.views = tuple(view for view in GRAPH_ORDER if view in available)
        held = self._held_graphs()
        self.overlays = held & (self.overlays | (held - self._held_seen))
        self._held_seen = held
        if self.view not in self.views:
            self.view = DEG_VIEW
            self.k = self._deg_k()
            if self._view_var is not None:
                self._view_var.set(DEG_VIEW)
            if self._k_var is not None:
                self._k_var.set(str(self.k))
        if self._view_menu is not None:
            self._view_menu.configure(values=list(self.views))
        self._sync_k_enabled()

    def _rebuild(self) -> None:
        """
        Builds the current settings afresh while keeping the user's search setup.

        Only a distribution change is a fresh start (`_on_preset`); every other knob — degree, metric,
        transform, vertex count — leaves the target, the start node and an open query where they are.
        A vertex index is clamped against the new cloud, so a selection past its end is released
        rather than crashing the next paint, and an open query whose target survived is re-run so the
        panel answers on the graph that is now drawn. The old query indexes the old model, so it is
        dropped before the repaint and only ever comes back through `run_query`.
        """
        self.model = self._build(self.preset, self.num_points, self._deg_k(), self.seed, self.metric, self.mips)
        # The knng, the NSW and the NSG are read off the feature space, so a fresh cloud, degree or metric
        # makes the cached edge lists stale; drop them and let the next draw rebuild the one it needs.
        self._knng_edges = None
        self._nsw_edges = None
        self._nsg_edges = None
        if self.selected is not None and self.selected >= self.model.num_points:
            self.selected = None
        if self.entry is not None and self.entry >= self.model.num_points:
            self.entry = None
        # The start node is never left empty: a released one (or the fresh start after a preset
        # change) re-defaults to the vertex farthest from the new cloud's centre.
        if self.entry is None:
            self.entry = self.model.farthest_from_centroid()
        requery = self.query is not None and (self.selected is not None or self.free_query is not None)
        self.query = None
        self.hovered = None
        self.status = None
        self._home = None
        self._drag = None
        self._adopt_views()
        self._paint_legend()
        self.redraw()
        if requery:
            self.run_query()

    def save_figure(self, path: str | Path | None = None) -> Path | None:
        """
        Writes the plot rectangle alone — the white axes and the graph it holds — to a PNG at 150 dpi.

        The crop is the main axes' *drawn* rectangle, so the saved picture is the plot and nothing else:
        no title, no tick labels, no colour key, no text panel, no overlay checkboxes, no reseed button
        and none of the figure's grey margin. Because the axes draws with `set_aspect("equal")`, the
        rectangle it actually paints is the ADJUSTED position, not the allocated `[0.045, 0.1, 0.675,
        0.84]`; cropping to the allocated box (or to the axes' tight box, which is measured against it)
        is what left the old export showing decorations on three sides while clipping the plot on the
        fourth. With no path it asks for one through the Tk save dialog — so the button, which only
        exists on a Tk canvas, needs no argument — and a cancelled dialog is a no-op. A caller that
        already knows the path (the headless tests, a script) passes it and gets the file written
        without any dialog, which is why this works on a backend that has no strip and no save button.
        """
        if path is None:
            canvas = self._tk_canvas()
            if canvas is None:
                return None
            path = filedialog.asksaveasfilename(
                parent=canvas.master,
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if not path:
                return None
        # The axes' own drawn rectangle in display pixels. A draw first fixes the layout, so the extent
        # is the aspect-adjusted position the axes really paints rather than the box it was allocated.
        # That extent is measured in the renderer's display pixels, so it turns into inches by dividing
        # by that renderer's dpi (not the figure's nominal dpi), and `bbox_inches` crops the saved file
        # to exactly the plot — dropping the title, the tick labels, the colour key, the panel and the
        # controls, and leaving none of the figure's grey margin around the plot.
        self._fig.canvas.draw()
        renderer = self._fig.canvas.get_renderer()
        extent = self._ax.get_window_extent(renderer)
        crop = matplotlib.transforms.Bbox(
            [
                [extent.x0 / renderer.dpi, extent.y0 / renderer.dpi],
                [extent.x1 / renderer.dpi, extent.y1 / renderer.dpi],
            ]
        )
        self._fig.savefig(path, bbox_inches=crop, dpi=150, facecolor=self._fig.get_facecolor())
        return Path(path)

    def close(self) -> None:
        self._close_help()
        plt.close(self._fig)

    def run(self) -> None:
        plt.show()


def render_static(model: GraphModel, target: str | Path) -> Path:
    """Writes the graph to an image file — the headless path, no widgets and no event loop."""
    target = Path(target)
    fig = plt.figure(figsize=(11.0, 11.0))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.add_collection(LineCollection(edge_segments(model), colors=EDGE_COLOR, linewidths=0.6, zorder=1))
    ax.scatter(
        model.plot_points[:, 0],
        model.plot_points[:, 1],
        s=18,
        c=vertex_colors(model),
        edgecolors="#2a2e35",
        linewidths=0.35,
        zorder=2,
    )
    _apply_limits(ax, model.plot_points)
    ax.set_title(f"DEG · {model.num_points} vertices · {model.edges.shape[0]} edges · k={model.k}", loc="left")
    fig.text(0.03, 0.015, "\n".join(describe(model)), fontsize=9, family="DejaVu Sans Mono")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target
