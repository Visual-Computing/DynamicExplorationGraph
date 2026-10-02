"""Shared fixtures and test helpers for the graph_2d test suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from functools import partial

import pytest
from deg_graph import DEFAULT_K
from deglib.distances import Metric
from main import build_scene
from matplotlib.backend_bases import KeyEvent
from viewer import (
    DEG_VIEW,
    FLAG_START,
    GraphViewer,
)

BUILD_START = ("blobs", 240, DEFAULT_K, 3, Metric.FP32_L2)


@pytest.fixture(scope="module")
def window():
    instance = GraphViewer(partial(build_scene, threads=1), *BUILD_START)
    instance._fig.canvas.draw_idle = lambda *_args, **_keywords: None
    yield instance
    instance.close()


@pytest.fixture()
def viewer(window: GraphViewer) -> GraphViewer:
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
    window.overlays = window._held_graphs()
    window._held_seen = set(window.overlays)
    window._knng_edges = None
    for position, wanted in enumerate(FLAG_START):
        checks = window._checks1 if position < 3 else window._checks2
        local = position if position < 3 else position - 3
        if checks.get_status()[local] != wanted:
            window._toggle_flag(position)
    if window.show_3d:
        window.show_3d = False
        window._set_3d_visible(False)
    window._cube_3d.reset_limits()
    if stale:
        window._rebuild()
    else:
        window.entry = window.model.farthest_from_centroid()
        window.redraw()
    return window


@pytest.fixture(scope="module")
def mips_window():
    instance = GraphViewer(partial(build_scene, threads=1), *BUILD_START, mips=True)
    instance._fig.canvas.draw_idle = lambda *_args, **_keywords: None
    yield instance
    instance.close()


@pytest.fixture()
def mips_viewer(mips_window: GraphViewer) -> GraphViewer:
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


def _key(viewer: GraphViewer, key: str) -> KeyEvent:
    return KeyEvent("key_press_event", viewer._fig.canvas, key)
