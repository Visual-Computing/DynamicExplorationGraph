"""Tests for the 3D cube view, turntable navigation, and 3D projections in the graph viewer."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _key
from matplotlib.backend_bases import MouseEvent
from viewer import PICK_RADIUS_PX, GraphViewer
from viewer_3d import DEFAULT_ZOOM


def test_the_3d_view_is_gated_on_the_mips_transform(viewer: GraphViewer) -> None:
    """Opening the cube while MIPS is off is refused: the 2D plot stays, the cube never opens."""
    viewer._toggle_3d()

    assert viewer.show_3d is False, "the toggle is refused when the transform is off"
    assert viewer._ax3d is None, "the 2D plot is never replaced"
    assert viewer._ax.get_visible() is True and viewer._ax3d is None, "the 2D plot is never replaced"


def test_the_3d_key_is_ignored_without_the_transform(viewer: GraphViewer) -> None:
    """The "3" key is a no-op while MIPS is off, so it cannot open a cube over a 2D feature space."""
    _key(viewer, "3")
    viewer._on_key(_key(viewer, "3"))

    assert viewer.show_3d is False and viewer._ax3d is None


def test_the_3d_view_swaps_the_axes_when_mips_is_on(mips_viewer: GraphViewer) -> None:
    """On the lifted cloud, the overlay hides the 2D plot and paints the cube in its place."""
    viewer = mips_viewer
    viewer._toggle_3d()

    assert viewer.show_3d is True
    assert viewer._ax.get_visible() is False, "the 2D plot is hidden"
    assert viewer._ax3d is not None and viewer._ax3d.get_visible() is True, "the cube is shown"
    assert len(viewer._ax3d.collections) >= 1, "the cube paints its vertices"

    viewer._toggle_3d()
    assert viewer.show_3d is False
    assert viewer._ax.get_visible() is True and viewer._ax3d.get_visible() is False, "the 2D plot returns"


def test_turning_mips_off_drops_the_3d_view(mips_viewer: GraphViewer) -> None:
    """Dropping the transform forces the cube closed and unchecks its overlay."""
    viewer = mips_viewer
    viewer._toggle_3d()
    assert viewer.show_3d is True

    viewer._on_mips("off")

    assert viewer.show_3d is False
    assert viewer._ax.get_visible() is True, "the 2D plot is restored"


def _show_3d(viewer: GraphViewer) -> None:
    """Opens the cube at a tilted angle and draws it, so the lifted z separates the vertices in projection."""
    viewer._toggle_3d()
    viewer._ax3d.view_init(elev=30.0, azim=45.0)
    viewer._fig.canvas.draw()


def _display_3d(viewer: GraphViewer, index: int) -> tuple[float, float]:
    """The display pixel of a vertex as the 3D axes projects it, the same map the picker inverts."""
    return tuple(float(c) for c in viewer._project_3d(viewer.model.points[index : index + 1])[0])


def _on_screen_vertex_3d(viewer: GraphViewer) -> tuple[int, float, float]:
    """A vertex whose projection lands inside the axes window, with its pixel — the cube zooms in, so not every vertex stays on frame."""
    box = viewer._ax3d.get_window_extent()
    display = viewer._project_3d(viewer.model.points)
    inside = np.flatnonzero(
        (display[:, 0] > box.x0 + PICK_RADIUS_PX)
        & (display[:, 0] < box.x1 - PICK_RADIUS_PX)
        & (display[:, 1] > box.y0 + PICK_RADIUS_PX)
        & (display[:, 1] < box.y1 - PICK_RADIUS_PX)
    )
    assert inside.size, "at least one vertex projects inside the axes"
    index = int(inside[0])
    return index, float(display[index, 0]), float(display[index, 1])


def _empty_spot_3d(viewer: GraphViewer) -> tuple[float, float]:
    """A display point inside the cube's axes that no projected vertex is within the pick radius of."""
    box = viewer._ax3d.get_window_extent()
    grid = np.stack(
        np.meshgrid(
            np.linspace(box.x0 + 20.0, box.x1 - 20.0, 9),
            np.linspace(box.y0 + 20.0, box.y1 - 20.0, 9),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    display = viewer._project_3d(viewer.model.points)
    nearest = np.linalg.norm(grid[:, None, :] - display[None, :, :], axis=2).min(axis=1)
    assert nearest.max() > PICK_RADIUS_PX, "the projected cloud must leave a pickable gap in the axes"
    x, y = grid[int(np.argmax(nearest))]
    return float(x), float(y)


def test_the_3d_view_keeps_the_camera_across_a_redraw(mips_viewer: GraphViewer) -> None:
    """A selection repaints the cube without snapping the camera back to top-down, so picking stays true."""
    viewer = mips_viewer
    _show_3d(viewer)
    viewer._ax3d.view_init(elev=30.0, azim=45.0)

    viewer.select(3)

    assert (viewer._ax3d.elev, viewer._ax3d.azim) == (30.0, 45.0), "the redraw must preserve the user's rotation"


def test_the_3d_view_hovers_the_vertex_under_the_cursor(mips_viewer: GraphViewer) -> None:
    """Moving over a vertex in the cube lights the hover ring, mirroring the 2D hover."""
    viewer = mips_viewer
    _show_3d(viewer)
    _index, x, y = _on_screen_vertex_3d(viewer)

    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, x, y))
    picked = viewer.hovered

    assert picked is not None, "the cursor over a lifted vertex hovers it"

    viewer._on_motion(MouseEvent("motion_notify_event", viewer._fig.canvas, *_empty_spot_3d(viewer)))
    assert viewer.hovered is None, "moving off the cloud clears the hover"


def test_the_3d_view_selects_and_sets_entry_by_click(mips_viewer: GraphViewer) -> None:
    """A left-click picks the vertex under the cursor as the target, a right-click as the start node."""
    viewer = mips_viewer
    _show_3d(viewer)
    _index, x, y = _on_screen_vertex_3d(viewer)

    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=1))
    picked = viewer.selected
    assert picked is not None, "left-click on a vertex selects it"
    assert f"TARGET   {picked}" in viewer._panel_text.get_text(), "the panel updates TARGET in 3D"

    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=3))
    assert viewer.entry == picked, "right-click on the same pixel sets that vertex as the entry"
    assert f"START    {picked}" in viewer._panel_text.get_text(), "the panel updates START in 3D"

    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, x, y, button=1))
    assert viewer.selected is None, "clicking the selected vertex again deselects it in 3D"
    assert "TARGET   none" in viewer._panel_text.get_text()

    # Re-select vertex, then click in empty white space to deselect
    viewer.select(picked)
    assert viewer.selected == picked
    ex, ey = _empty_spot_3d(viewer)
    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, ex, ey, button=1))
    viewer._on_release(MouseEvent("button_release_event", viewer._fig.canvas, ex, ey, button=1))
    assert viewer.selected is None, "clicking in empty space deselects the target in 3D"
    assert "TARGET   none" in viewer._panel_text.get_text()

    # Camera rotation over empty space must NOT deselect
    viewer.select(picked)
    assert viewer.selected == picked
    press = MouseEvent("button_press_event", viewer._fig.canvas, ex, ey, button=1)
    press.inaxes = viewer._ax3d
    viewer._fig.canvas.callbacks.process("button_press_event", press)
    motion = MouseEvent("motion_notify_event", viewer._fig.canvas, ex + 10, ey + 10, button=1)
    motion.inaxes = viewer._ax3d
    viewer._fig.canvas.callbacks.process("motion_notify_event", motion)
    release = MouseEvent("button_release_event", viewer._fig.canvas, ex + 10, ey + 10, button=1)
    release.inaxes = viewer._ax3d
    viewer._fig.canvas.callbacks.process("button_release_event", release)
    assert viewer.selected == picked, "dragging to rotate camera does not deselect"


def test_the_3d_view_places_a_free_query_on_the_plane(mips_viewer: GraphViewer) -> None:
    """In the top-down view a click on open plane — far from every vertex, ray crossing z = 0 in bounds — drops a query."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()
    box = viewer._ax3d.get_window_extent()
    display = viewer._project_3d(viewer.model.points)
    placed = None
    for gx in np.linspace(box.x0 + 5.0, box.x1 - 5.0, 41):
        for gy in np.linspace(box.y0 + 5.0, box.y1 - 5.0, 41):
            if np.linalg.norm(display - np.array([gx, gy]), axis=1).min() <= PICK_RADIUS_PX:
                continue
            if viewer._plane_point_from_click(MouseEvent("button_press_event", viewer._fig.canvas, gx, gy)) is not None:
                placed = (gx, gy)
                break
        if placed is not None:
            break
    assert placed is not None, "some open-plane pixel must cross the ground plane inside its bounds"

    viewer._on_click(MouseEvent("button_press_event", viewer._fig.canvas, *placed, button=1, dblclick=True))

    assert viewer.free_query is not None, "the click on open plane places a query"


def test_the_3d_view_supports_coords_and_vertex_ids(mips_viewer: GraphViewer) -> None:
    """The coords and vertex ids overlays must work in the 3D cube view as they do in 2D."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()
    assert viewer._ax3d is not None

    # Coords start on: only the dashed axis lines show, no grid, no panes
    assert viewer.show_coords is True
    assert viewer._ax3d.xaxis.pane.get_visible() is False, "3D panes stay hidden when coords are on"
    dashed_on = sum(1 for ln in viewer._ax3d.lines if ln.get_linestyle() == "--")
    assert dashed_on >= 2, "the dashed axis lines are drawn when coords are on"

    # Toggle coords off
    viewer._toggle_flag(0)
    assert viewer.show_coords is False
    assert sum(1 for ln in viewer._ax3d.lines if ln.get_linestyle() == "--") == 0, "the dashed lines hide when coords are off"

    # Toggle coords back on
    viewer._toggle_flag(0)
    assert viewer.show_coords is True
    assert sum(1 for ln in viewer._ax3d.lines if ln.get_linestyle() == "--") >= 2, "the dashed lines return with coords"

    # Vertex IDs start off
    assert viewer.show_ids is False
    assert len(viewer._ax3d.texts) == 0, "no 3D text labels when vertex ids are off"

    # Toggle vertex IDs on
    viewer._toggle_flag(2)
    assert viewer.show_ids is True
    assert len(viewer._ax3d.texts) == viewer.model.num_points, "each 3D vertex gets an id label"

    # Toggle vertex IDs back off
    viewer._toggle_flag(2)
    assert viewer.show_ids is False
    assert len(viewer._ax3d.texts) == 0, "labels clear when vertex ids are turned off"


def test_the_3d_view_query_overlay_toggles_traversal_and_selection_marks(mips_viewer: GraphViewer) -> None:
    """The query overlay hides the traversal path, selection marks and start node in 3D."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer.select(3)
    viewer.set_entry(7)
    viewer.run_query()
    viewer._fig.canvas.draw()
    assert viewer._ax3d is not None

    # Initially show_query is True
    assert viewer.show_query is True
    initial_collections = len(viewer._ax3d.collections)
    initial_lines = len(viewer._ax3d.lines)

    # Toggle query overlay off
    viewer._toggle_flag(4)
    assert viewer.show_query is False
    assert len(viewer._ax3d.collections) < initial_collections, "query marks and target rings are hidden"
    assert len(viewer._ax3d.lines) < initial_lines, "traversal route and drop lines are hidden"

    # Toggle query overlay back on
    viewer._toggle_flag(4)
    assert viewer.show_query is True
    assert len(viewer._ax3d.collections) == initial_collections
    assert len(viewer._ax3d.lines) == initial_lines


def test_the_3d_view_path_visualization_parity(mips_viewer: GraphViewer) -> None:
    """The 3D view paints the traversal route, entry square, ground-truth star and answer X."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer.select(5)
    viewer.set_entry(10)
    query = viewer.run_query()
    assert query is not None
    viewer._fig.canvas.draw()
    assert viewer._ax3d is not None

    # Traversal lines exist (both 3D feature route and projected 2D ground route)
    assert len(viewer._ax3d.lines) >= 2, "path lines are drawn"
    # Scatter collections hold entry, terminal and star markers
    markers = [c for c in viewer._ax3d.collections if getattr(c, "_sizes", None) is not None]
    assert len(markers) > 0, "path markers are present in 3D"


def test_the_3d_view_keeps_query_path_when_selection_deselected(mips_viewer: GraphViewer) -> None:
    """Selecting another vertex and clicking into white space to deselect does not wipe out the query/path."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer.select(5)
    viewer.set_entry(10)
    query = viewer.run_query()
    assert query is not None
    assert viewer.query is not None

    # Single click another vertex to select it
    viewer.select(15)
    assert viewer.selected == 15
    assert viewer.query is not None, "selecting another vertex keeps the existing query"

    # Set free query on plane
    viewer.set_free_query(np.array([0.2, 0.3], dtype=np.float32))
    assert viewer.free_query is not None
    viewer.select(20)

    # Click in white space / deselect selection
    empty_press = MouseEvent("button_press_event", viewer._fig.canvas, 5.0, 5.0, button=1)
    empty_press.inaxes = None
    viewer._fig.canvas.callbacks.process("button_press_event", empty_press)

    assert viewer.selected is None, "vertex selection is cleared"
    assert viewer.query is not None, "query path remains displayed after deselecting vertex"
    assert len(viewer._ax3d.lines) >= 2, "path lines remain rendered in 3D"

    # Double click into empty white space clears the query
    dbl_click = MouseEvent("button_press_event", viewer._fig.canvas, 5.0, 5.0, button=1, dblclick=True)
    dbl_click.inaxes = None
    viewer._fig.canvas.callbacks.process("button_press_event", dbl_click)
    assert viewer.query is None, "double click in white space removes the query"


def test_the_3d_strip_sets_the_camera_angles(mips_viewer: GraphViewer) -> None:
    """The strip's angle fields drive the cube's camera: a committed elevation and azimuth are applied."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()

    viewer._set_3d_angles(elev=30.0, azim=45.0)

    assert (viewer._ax3d.elev, viewer._ax3d.azim) == (30.0, 45.0), "the typed angles set the camera"


def test_the_3d_panel_clamps_and_wraps_the_angles(mips_viewer: GraphViewer) -> None:
    """Setting the camera angle clamps elevation to [-90, 90] and wraps azimuth to [-180, 180]."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()

    viewer._set_3d_angles(elev=200.0)
    assert viewer._ax3d.elev == 90.0, "elevation clamps to the top"

    viewer._set_3d_angles(azim=200.0)
    assert viewer._ax3d.azim == -160.0, "azimuth wraps past 180"


def test_animate_to_lands_the_camera_on_the_target(mips_viewer: GraphViewer) -> None:
    """The animate glide ends on the target angle, elevation clamped and azimuth wrapped."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._ax3d.view_init(elev=0.0, azim=0.0)

    viewer._cube_3d.animate_to(200.0, 200.0, delay=0.0)

    assert viewer._ax3d.elev == 90.0, "elevation clamps to the top"
    assert viewer._ax3d.azim == -160.0, "azimuth wraps past 180"


def test_the_camera_animates_the_short_way_round(mips_viewer: GraphViewer) -> None:
    """A move from 170° to -170° crosses 180°, not the long sweep back through 0°."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._ax3d.view_init(elev=0.0, azim=170.0)
    frames: list[float] = []
    real = viewer._fig.canvas.draw
    viewer._fig.canvas.draw = lambda *_a, **_k: frames.append(viewer._ax3d.azim)
    try:
        viewer._cube_3d.animate_to(0.0, -170.0, steps=8, delay=0.0)
    finally:
        viewer._fig.canvas.draw = real

    assert viewer._ax3d.azim == -170.0, "the camera lands on the wrapped target"
    mid = frames[1:-1]
    assert any(abs(abs(az) - 180.0) < 30.0 for az in mid), "the camera crosses 180°"
    assert all(abs(az) > 90.0 for az in mid), "the camera never swings through 0°"


def test_animate_to_gif_writes_an_animation(mips_viewer: GraphViewer, tmp_path) -> None:
    """Recording the camera glide writes a GIF file and lands the camera on the target."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._ax3d.view_init(elev=0.0, azim=0.0)
    target = tmp_path / "cube.gif"

    viewer._cube_3d.animate_to_gif(45.0, 30.0, zoom=1.5, path=str(target), steps=6, fps=10)

    assert target.exists() and target.stat().st_size > 0, "the gif file is written"
    assert viewer._ax3d.elev == pytest.approx(45.0), "the camera lands on the target elevation"
    assert viewer._ax3d.azim == pytest.approx(30.0), "the camera lands on the target azimuth"


def test_the_mouse_wheel_zooms_the_cube(mips_viewer: GraphViewer) -> None:
    """Scrolling up over the cube zooms in; scrolling down zooms out."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()
    centre = viewer._ax3d.get_window_extent().get_points().mean(axis=0)
    start = viewer._cube_3d.zoom()

    zoom_in = MouseEvent("scroll_event", viewer._fig.canvas, *centre, step=1)
    zoom_in.inaxes = viewer._ax3d
    viewer._on_scroll(zoom_in)
    zoomed_in = viewer._cube_3d.zoom()

    zoom_out = MouseEvent("scroll_event", viewer._fig.canvas, *centre, step=-1)
    zoom_out.inaxes = viewer._ax3d
    viewer._on_scroll(zoom_out)
    zoomed_out = viewer._cube_3d.zoom()

    assert zoomed_in > start, "scrolling up zooms in"
    assert zoomed_out < zoomed_in, "scrolling down zooms out"


def test_the_fitted_cube_opens_at_the_default_zoom(mips_viewer: GraphViewer) -> None:
    """Right after the cube is shown, the framing reads as the default zoom, centred on the coordinate origin."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()

    assert viewer._cube_3d.zoom() == pytest.approx(DEFAULT_ZOOM, rel=1e-6), "the cube opens at the default zoom"
    lo, hi = viewer._ax3d.get_xlim3d()
    assert (lo + hi) / 2.0 == pytest.approx(0.0, abs=1e-6), "the x view is centred on the origin"
    lo, hi = viewer._ax3d.get_ylim3d()
    assert (lo + hi) / 2.0 == pytest.approx(0.0, abs=1e-6), "the y view is centred on the origin"
    assert viewer._ax3d.get_zlim3d()[0] == pytest.approx(0.0), "the view never dips below the ground plane"


def test_set_zoom_scales_the_cube_and_reads_back(mips_viewer: GraphViewer) -> None:
    """Setting a zoom level drives the x span to the fit span over that zoom, and the read-out agrees."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()
    fit_span = viewer._cube_3d._fit_span

    viewer._cube_3d.set_zoom(2.0)

    span_after = viewer._ax3d.get_xlim3d()[1] - viewer._ax3d.get_xlim3d()[0]
    assert span_after == pytest.approx(fit_span / 2.0, rel=1e-6), "2x zoom drives the span to fit/2"
    assert viewer._cube_3d.zoom() == pytest.approx(2.0, rel=1e-6), "the read-out reports 2x"


def test_set_zoom_clamps_to_the_allowed_range(mips_viewer: GraphViewer) -> None:
    """Zoom is clamped to [0.2, 20] at both ends."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._fig.canvas.draw()

    viewer._cube_3d.set_zoom(100.0)
    assert viewer._cube_3d.zoom() == pytest.approx(20.0, rel=1e-6), "zoom clamps to the top"

    viewer._cube_3d.set_zoom(0.001)
    assert viewer._cube_3d.zoom() == pytest.approx(0.2, rel=1e-6), "zoom clamps to the bottom"


def test_animate_to_lands_on_the_target_zoom(mips_viewer: GraphViewer) -> None:
    """The animate glide ends on the target zoom as well as the target angle."""
    viewer = mips_viewer
    viewer._toggle_3d()
    viewer._ax3d.view_init(elev=0.0, azim=0.0)

    viewer._cube_3d.animate_to(45.0, 30.0, zoom=2.0, delay=0.0)

    assert viewer._cube_3d.zoom() == pytest.approx(2.0, rel=1e-6), "the camera lands on the target zoom"


def test_the_3d_view_shows_no_grid_when_coords_is_on(mips_viewer: GraphViewer) -> None:
    """With the coordinate flag on, the cube draws only the dashed axis lines: no grid, no panes."""
    viewer = mips_viewer
    viewer.show_coords = True
    viewer._toggle_3d()
    viewer._fig.canvas.draw()
    ax = viewer._ax3d

    assert not any(gl.get_visible() for gl in ax.xaxis.get_gridlines()), "the x grid stays hidden"
    assert not any(gl.get_visible() for gl in ax.yaxis.get_gridlines()), "the y grid stays hidden"
    assert not any(gl.get_visible() for gl in ax.zaxis.get_gridlines()), "the z grid stays hidden"
    assert not ax.xaxis.pane.get_visible(), "the panes stay hidden"
    assert all(t.get_text() == "" for t in ax.get_xticklabels()), "no tick numbers on x"
    assert all(t.get_text() == "" for t in ax.get_yticklabels()), "no tick numbers on y"
    assert all(t.get_text() == "" for t in ax.get_zticklabels()), "no tick numbers on z"
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) >= 2, "the dashed axis reference lines are drawn"


def test_the_3d_toggle_is_refused_when_mips_is_off(viewer: GraphViewer) -> None:
    """Clicking the 3D toggle with the transform off reverts the box and never opens the cube."""
    viewer._3d_toggle.set_active(0, True)

    assert viewer.show_3d is False, "the toggle is refused when the transform is off"
    assert viewer._3d_toggle.get_status()[0] is False, "the box reverts to unchecked"
    assert viewer._ax3d is None, "the cube is never created"
