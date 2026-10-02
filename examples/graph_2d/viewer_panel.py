"""The right-hand text panel for displaying graph statistics, queries, and hover information."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.transforms
from matplotlib.axes import Axes
from matplotlib.text import Text

if TYPE_CHECKING:
    from deg_graph import QueryResult

#: The panel column holds this many monospace characters; a longer line spills out of the column
#: into the figure's edge. Measured: the column is 353 px wide at 7 px per character.
PANEL_WIDTH = 50

#: The label a neighbour list starts behind, and that continuation lines repeat as blank padding.
NEIGHBOR_LABEL = "  neighbors  "

#: The panel's line styles by name.
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

#: The bold 11pt heading is taller than the 8.5pt body, so it advances further to clear the line below it.
PANEL_HEAD_PITCH = 0.036

#: The label the first search result starts behind.
SEARCH_LABEL = "  search   "


class StyledPanel:
    """The right-hand column as a stack of individually styled lines."""

    def __init__(self, ax: Axes) -> None:
        self._ax = ax
        self._lines: list[tuple[str, str]] = []
        self._artists: list[Text] = []

    def set_text(self, lines: str | list[tuple[str, str]]) -> None:
        """Draws the column from `(text, style)` pairs; a plain string becomes one body line each."""
        self._lines = [(line, "body") for line in lines.split("\n")] if isinstance(lines, str) else list(lines)
        for artist in self._artists:
            artist.remove()
        # Lines advance by their own pitch: the bold heading is taller than the body, so it needs a larger
        # step to clear the line below it.
        self._artists = []
        y = 1.0
        for text, style in self._lines:
            self._artists.append(
                self._ax.text(0.0, y, text, transform=self._ax.transAxes, ha="left", va="top", **PANEL_STYLES[style])
            )
            y -= PANEL_HEAD_PITCH if style == "head" else PANEL_LINE_PITCH

    def get_text(self) -> str:
        return "\n".join(text for text, _ in self._lines)

    def get_window_extent(self, renderer) -> matplotlib.transforms.Bbox:
        boxes = [artist.get_window_extent(renderer) for artist in self._artists if artist.get_text()]
        return (
            matplotlib.transforms.Bbox.union(boxes) if boxes else matplotlib.transforms.Bbox([[0.0, 0.0], [0.0, 0.0]])
        )


def format_search_lines(query: QueryResult) -> list[tuple[str, str]]:
    """The search block: one line per result, in the search's own ranking order."""
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
