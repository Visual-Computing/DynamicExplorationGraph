"""3D cube view controller, turntable camera, and 3D rendering for the graph viewer."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if TYPE_CHECKING:
    from viewer import GraphViewer

#: The cube's opening zoom: the fitted framing scaled in, so the cloud fills the frame with a margin.
DEFAULT_ZOOM = 1.6


class _CroppedGifWriter(PillowWriter):
    """A GIF writer that crops every frame to one axes' drawn rectangle, matching the save button's crop."""

    max_side = 1024

    def __init__(self, fps: int, ax: Axes) -> None:
        super().__init__(fps)
        self._ax = ax

    def grab_frame(self) -> None:
        from PIL import Image

        # Render through the Agg buffer (the same path savefig uses). The Tk canvas's own
        # buffer_rgba() omits the 3D axes' composited raster, leaving the plot region grey.
        buffer, (width, height) = self.fig.canvas.print_to_buffer()
        image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1).convert("RGB")
        extent = self._ax.get_window_extent(self.fig.canvas.get_renderer())
        left, right = int(extent.x0), int(extent.x1)
        # The buffer's row 0 is the figure's top; the extent's y is measured from the bottom.
        top, bottom = int(height - extent.y1), int(height - extent.y0)
        frame = image.crop((left, top, right, bottom))
        # Scale the frame so its longest side is `max_side`, keeping the aspect ratio.
        scale = self.max_side / max(frame.width, frame.height)
        if scale < 1.0:
            frame = frame.resize(
                (round(frame.width * scale), round(frame.height * scale)), Image.LANCZOS
            )
        self._frames.append(frame)

    def finish(self) -> None:
        # `loop=None` writes no NETSCAPE2.0 loop block, the convention players read as "play exactly once".
        # A loop count of 1 writes the block with a counter, which some players treat as "one extra loop".
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=None)


class Viewer3D:
    """Manages the 3D cube view, turntable camera navigation, and 3D rendering."""

    def __init__(self, viewer: GraphViewer) -> None:
        self.viewer = viewer
        self.ax3d: Axes | None = None
        self._empty_press = False
        self._limits_set = False
        self._fit_span = 1.0

    def ensure_axes(self) -> Axes:
        """Creates the 3D axes over the 2D plot's rectangle on first use, looking straight down."""
        if self.ax3d is None:
            self.ax3d = self.viewer._fig.add_axes([0.045, 0.1, 0.675, 0.84], projection="3d")
            self.ax3d.set_proj_type("ortho")
            self.ax3d.set_facecolor("#ffffff")
            self.ax3d.view_init(elev=90, azim=-90)
            self.install_turntable(self.ax3d)
        return self.ax3d

    def install_turntable(self, ax: Axes) -> None:
        """Replaces matplotlib's quaternion rotation with a turntable: mouse-x → azimuth, mouse-y → elevation."""
        state: dict = {}

        def on_press(event: MouseEvent) -> None:
            if event.inaxes is not ax or event.button != 1 or event.dblclick:
                return
            state["x"] = event.x
            state["y"] = event.y
            state["azim"] = ax.azim
            state["elev"] = ax.elev

        def on_motion(event: MouseEvent) -> None:
            if not state or event.x is None or event.y is None:
                return
            dx = event.x - state["x"]
            dy = event.y - state["y"]
            if abs(dx) > 3 or abs(dy) > 3:
                self._empty_press = False
                self.viewer._3d_empty_press = False
            ax.view_init(
                elev=max(5, min(90, state["elev"] - dy * 0.3)),
                azim=state["azim"] - dx * 0.3,
            )
            self.viewer._sync_3d_angles()
            self.viewer._fig.canvas.draw_idle()

        def on_release(_event: MouseEvent) -> None:
            state.clear()

        def on_draw(_event) -> None:
            # Matplotlib's built-in right-drag zoom changes the axis limits without a callback, so refresh
            # the live angle/zoom read-out on every draw to keep it in step with the camera.
            self.viewer._sync_3d_angles()

        self.viewer._fig.canvas.mpl_connect("button_press_event", on_press)
        self.viewer._fig.canvas.mpl_connect("motion_notify_event", on_motion)
        self.viewer._fig.canvas.mpl_connect("button_release_event", on_release)
        self.viewer._fig.canvas.mpl_connect("draw_event", on_draw)

    def set_visible(self, visible: bool) -> None:
        """Swaps the 2D plot for the 3D cube (and back), creating the 3D axes only when shown."""
        if visible:
            self.ensure_axes()
        self.viewer._ax.set_visible(not visible)
        if self.ax3d is not None:
            self.ax3d.set_visible(visible)

    def rotate(self, d_elev: float, d_azim: float) -> None:
        if self.ax3d is None:
            return
        elev, azim = self.ax3d.elev, self.ax3d.azim
        self.ax3d.view_init(elev=max(-90, min(90, elev + d_elev)), azim=azim + d_azim)
        self.viewer._sync_3d_angles()
        self.viewer._fig.canvas.draw_idle()

    def zoom_in(self) -> None:
        self.scale(0.8)

    def zoom_out(self) -> None:
        self.scale(1.25)

    def on_scroll(self, event) -> None:
        """Zooms the cube with the mouse wheel in a gentle step, clamped to the allowed range."""
        if self.ax3d is None or event.inaxes is not self.ax3d or not event.step:
            return
        self.set_zoom(self.zoom() * (1.0 + 0.1 * event.step))

    def scale(self, factor: float) -> None:
        if self.ax3d is None:
            return
        for setter, getter in (
            (self.ax3d.set_xlim3d, self.ax3d.get_xlim3d),
            (self.ax3d.set_ylim3d, self.ax3d.get_ylim3d),
            (self.ax3d.set_zlim3d, self.ax3d.get_zlim3d),
        ):
            lo, hi = getter()
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            setter(mid - half, mid + half)
        self.viewer._fig.canvas.draw_idle()

    def zoom(self) -> float:
        """The cube's zoom level: the fit span over the current x span, so 1.0 is the fitted view."""
        if self.ax3d is None:
            return 1.0
        lo, hi = self.ax3d.get_xlim3d()
        span = hi - lo
        return self._fit_span / span if span > 0 else 1.0

    def set_zoom(self, z: float) -> None:
        """Scales the cube to a zoom level, clamped to [0.2, 20], keeping the view centred."""
        if self.ax3d is None:
            return
        self._apply_zoom(max(0.2, min(20.0, z)))
        self.viewer._sync_3d_angles()
        self.viewer._fig.canvas.draw_idle()

    def _apply_zoom(self, z: float) -> None:
        """Rescales every axis so the x span matches the zoom level: x/y around the centre, z anchored at the ground, no draw."""
        lo, hi = self.ax3d.get_xlim3d()
        current = self._fit_span / (hi - lo) if hi > lo else 1.0
        factor = current / z
        for setter, getter in (
            (self.ax3d.set_xlim3d, self.ax3d.get_xlim3d),
            (self.ax3d.set_ylim3d, self.ax3d.get_ylim3d),
        ):
            low, high = getter()
            mid = (low + high) / 2.0
            half = (high - low) / 2.0 * factor
            setter(mid - half, mid + half)
        # Anchor z at its lower bound (the ground plane) so zooming never reveals negative height.
        low, high = self.ax3d.get_zlim3d()
        self.ax3d.set_zlim3d(low, low + (high - low) * factor)

    def animate_to(self, elev: float, azim: float, zoom: float | None = None, steps: int = 24, delay: float = 0.01) -> None:
        """
        Glides the camera from its current angle (and zoom) to a target, elevation clamped and azimuth wrapped.

        Elevation is interpolated linearly; azimuth travels the shortest way round the circle, so a
        move from 170° to -170° crosses 180° rather than sweeping back through 0°; zoom interpolates in
        log space so a doubling of scale reads as a constant glide. The frames are drawn in a plain loop
        — no timer, no thread — so the whole move completes synchronously and a headless test sees the
        camera land on the target the moment the call returns.
        """
        if self.ax3d is None:
            return
        start_elev, start_azim = self.ax3d.elev, self.ax3d.azim
        target_elev = max(-90.0, min(90.0, elev))
        target_azim = ((azim + 180.0) % 360.0) - 180.0
        # Fold the azimuth delta into (-180, 180] so the camera turns the short way, not the long way.
        delta = ((target_azim - start_azim + 180.0) % 360.0) - 180.0
        start_zoom = self.zoom()
        target_zoom = start_zoom if zoom is None else max(0.2, min(20.0, zoom))
        log_start, log_target = math.log(start_zoom), math.log(target_zoom)
        frames = max(1, int(steps))
        for step in range(1, frames + 1):
            t = step / frames
            self.ax3d.view_init(elev=start_elev + (target_elev - start_elev) * t, azim=start_azim + delta * t)
            if zoom is not None:
                self._apply_zoom(math.exp(log_start + (log_target - log_start) * t))
            self.viewer._fig.canvas.draw()
            self.viewer._fig.canvas.flush_events()
            if delay:
                time.sleep(delay)
        self.ax3d.view_init(elev=target_elev, azim=target_azim)
        if zoom is not None:
            self._apply_zoom(target_zoom)
        self.viewer._sync_3d_angles()
        self.viewer._fig.canvas.draw()

    def animate_to_gif(self, elev: float, azim: float, zoom: float | None = None, path: str = "cube.gif", steps: int = 48, fps: int = 20) -> None:
        """
        Glides the camera to a target exactly as `animate_to` does, recording each frame into a GIF.

        The same interpolation drives both: elevation linear, azimuth the short way round, zoom in log
        space. `FuncAnimation` renders the frames through the figure's own canvas and `PillowWriter`
        encodes them, so the file is the cube the user sees. The camera lands on the target when it ends.
        """
        from matplotlib.animation import FuncAnimation

        if self.ax3d is None:
            return
        start_elev, start_azim = self.ax3d.elev, self.ax3d.azim
        target_elev = max(-90.0, min(90.0, elev))
        target_azim = ((azim + 180.0) % 360.0) - 180.0
        delta = ((target_azim - start_azim + 180.0) % 360.0) - 180.0
        start_zoom = self.zoom()
        target_zoom = start_zoom if zoom is None else max(0.2, min(20.0, zoom))
        log_start, log_target = math.log(start_zoom), math.log(target_zoom)
        frames = max(1, int(steps))
        gif_path = Path(path)
        start_png = gif_path.with_suffix(".start.png")
        end_png = gif_path.with_suffix(".end.png")

        def update(step: int) -> list:
            t = step / (frames - 1) if frames > 1 else 1.0
            self.ax3d.view_init(elev=start_elev + (target_elev - start_elev) * t, azim=start_azim + delta * t)
            if zoom is not None:
                self._apply_zoom(math.exp(log_start + (log_target - log_start) * t))
            # Redraw and flush so the on-screen window plays the glide live while the writer records it.
            self.viewer._fig.canvas.draw()
            self.viewer._fig.canvas.flush_events()
            return []

        # The camera currently sits at the start, so frame 0 (t = 0) is the start view the glide begins from.
        self._save_cropped_png(start_png)
        animation = FuncAnimation(self.viewer._fig, update, frames=frames, blit=False)
        # Crop to the 2D axes' drawn rectangle — the same region the save button crops to.
        animation.save(path, writer=_CroppedGifWriter(fps, self.viewer._ax))
        # Tear the animation down before the final draw: it keeps a draw_event handler and a live timer
        # that fire `update` on the next canvas draw and snap the camera back to its first frame.
        for attr in ("_first_draw_id", "_draw_id"):
            cid = getattr(animation, attr, None)
            if cid is not None:
                self.viewer._fig.canvas.mpl_disconnect(cid)
        if animation.event_source is not None:
            animation.event_source.stop()
        animation._stop()
        self.ax3d.view_init(elev=target_elev, azim=target_azim)
        if zoom is not None:
            self._apply_zoom(target_zoom)
        self.viewer._sync_3d_angles()
        self.viewer._fig.canvas.draw()
        self._save_cropped_png(end_png)

    def _save_cropped_png(self, path: str | Path) -> None:
        """Writes the plot rectangle alone — the same crop the save button and the GIF use — to a PNG."""
        from PIL import Image

        self.viewer._fig.canvas.draw()
        buffer, (width, height) = self.viewer._fig.canvas.print_to_buffer()
        image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1).convert("RGB")
        extent = self.viewer._ax.get_window_extent(self.viewer._fig.canvas.get_renderer())
        left, right = int(extent.x0), int(extent.x1)
        top, bottom = int(height - extent.y1), int(height - extent.y0)
        image.crop((left, top, right, bottom)).save(path)

    def reset_view(self) -> None:
        if self.ax3d is None:
            return
        self._limits_set = False
        self.redraw()
        self.ax3d.view_init(elev=90, azim=-90)
        self.viewer._sync_3d_angles()
        self.viewer._fig.canvas.draw_idle()

    def reset_limits(self) -> None:
        """Drops the remembered zoom so the next repaint re-fits the cube to a freshly built cloud."""
        self._limits_set = False

    def redraw(self) -> None:
        """Paints the vertices, edges, search path and a z=0 ground plane into the 3D cube view, keeping the camera where the user left it."""
        from viewer import (
            DEG_VIEW,
            EDGE_COLOR,
            HOVER_EDGE_COLOR,
            KNNG_COLOR,
            KNNG_VIEW,
            NONE_VIEW,
            NSG_COLOR,
            NSG_VIEW,
            NSW_COLOR,
            NSW_VIEW,
            OVERLAP_COLORS,
            edge_colours,
            vertex_colors,
        )

        ax = self.ensure_axes()
        elev, azim = ax.elev, ax.azim
        limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
        ax.cla()
        ax.set_proj_type("ortho")
        ax.set_facecolor("#ffffff")
        points = self.viewer.model.points
        edges = self.viewer.model.edges
        low, high = points.min(axis=0), points.max(axis=0)
        # Centre the view on the coordinate origin: x/y span symmetrically about (0, 0) out to the farthest
        # point, so the origin sits in the middle of the image, and z is anchored at the ground plane (0)
        # up to the tallest point.
        cx, cy = 0.0, 0.0
        half_x = max(abs(high[0]), abs(low[0])) * 1.05
        half_y = max(abs(high[1]), abs(low[1])) * 1.05
        if self._limits_set:
            ax.set_xlim3d(*limits[0])
            ax.set_ylim3d(*limits[1])
            ax.set_zlim3d(*limits[2])
        else:
            ax.set_xlim3d(cx - half_x, cx + half_x)
            ax.set_ylim3d(cy - half_y, cy + half_y)
            ax.set_zlim3d(0.0, high[2] * 1.05)
            self._fit_span = 2.0 * half_x
            self._limits_set = True
            self._apply_zoom(DEFAULT_ZOOM)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_title(f"3D · {self.viewer.model.num_points} vertices · {edges.shape[0]} edges", fontsize=11, loc="left")
        ax.view_init(elev=elev, azim=azim)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_visible(False)
            pane.set_fill(False)
        ax.grid(False)
        ax.xaxis.line.set_visible(False)
        ax.yaxis.line.set_visible(False)
        ax.zaxis.line.set_visible(False)
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            top=False,
            right=False,
        )
        ax.zaxis.set_ticklabels([])
        ax.zaxis.set_ticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        self.paint_ground(ax)
        if self.viewer.show_coords:
            plot = self.viewer.model.plot_points
            px0, px1 = plot[:, 0].min(), plot[:, 0].max()
            py0, py1 = plot[:, 1].min(), plot[:, 1].max()
            pad = max(px1 - px0, py1 - py0) * 0.05
            ax.plot([px0 - pad, px1 + pad], [cy, cy], [0, 0], color="#000000", linewidth=0.8, linestyle="--")
            ax.plot([cx, cx], [py0 - pad, py1 + pad], [0, 0], color="#000000", linewidth=0.8, linestyle="--")
            ax.plot([cx, cx], [cy, cy], [0, high[2]], color="#000000", linewidth=0.8, linestyle="--")
        display_edges = self.viewer._display_edges()
        if self.viewer.view == DEG_VIEW:
            base_colors = (
                edge_colours(self.viewer.model, self.viewer.overlays) if self.viewer.show_colours else EDGE_COLOR
            )
        elif self.viewer.view == KNNG_VIEW:
            base_colors = KNNG_COLOR
        elif self.viewer.view == NSW_VIEW:
            base_colors = NSW_COLOR
        elif self.viewer.view == NSG_VIEW:
            base_colors = NSG_COLOR
        else:
            base_colors = (
                EDGE_COLOR if self.viewer.view == NONE_VIEW else OVERLAP_COLORS.get(self.viewer.view, EDGE_COLOR)
            )

        if self.viewer.show_edges and display_edges.size:
            ax.add_collection3d(Line3DCollection(points[display_edges], colors=base_colors, linewidths=0.55, zorder=2))

        # Light the hovered vertex's edges in green, in the cube and on the ground plane.
        if self.viewer.hovered is not None and display_edges.size:
            touching = np.isin(display_edges[:, 0], self.viewer.hovered) | np.isin(display_edges[:, 1], self.viewer.hovered)
            if np.any(touching):
                hov_pts = points[display_edges[touching]]
                ax.add_collection3d(Line3DCollection(hov_pts, colors=HOVER_EDGE_COLOR, linewidths=2.0, zorder=5))
                hov_grd = np.pad(self.viewer.model.plot_points[display_edges[touching]], ((0, 0), (0, 0), (0, 1)))
                ax.add_collection3d(
                    Line3DCollection(hov_grd, colors=HOVER_EDGE_COLOR, linewidths=1.2, alpha=0.5, zorder=3)
                )
        self.paint_path(ax)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=26,
            c=vertex_colors(self.viewer.model),
            alpha=0.9,
            edgecolors="#2a2e35",
            linewidths=0.35,
        )
        self.paint_marks(ax)
        self.paint_labels(ax)
        self.viewer._panel_text.set_text(self.viewer._panel_lines())
        self.viewer._fig.canvas.draw_idle()

    def paint_labels(self, ax: Axes) -> None:
        """Paints vertex IDs next to each 3D vertex when the vertex ids overlay is active."""
        from viewer import ID_LIMIT

        if not self.viewer.show_ids:
            return
        if self.viewer.model.num_points > ID_LIMIT:
            ax.text2D(
                0.012,
                0.012,
                f"vertex ids are hidden above {ID_LIMIT} vertices",
                transform=ax.transAxes,
                fontsize=8,
                color="#80868b",
            )
            return
        pts = self.viewer.model.points
        for index in range(self.viewer.model.num_points):
            x, y, z = pts[index]
            ax.text(x, y, z, str(index), fontsize=6, color="#5f6368", ha="left", va="bottom", zorder=7)

    def nearest_to_free_query(self) -> int | None:
        """The vertex nearest the free-space query in feature space, using the same padded query the search runs."""
        if self.viewer.free_query is None or self.viewer.model.num_points == 0:
            return None
        query = self.viewer.model.query_vector(self.viewer.free_query)
        delta = self.viewer.model.points - query
        return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))

    def paint_marks(self, ax: Axes) -> None:
        """Draws the selection ring, entry marker, hover ring and free-query cross on the 3D cloud and z=0 plane."""
        from viewer import ENTRY_COLOR, FREE_QUERY_COLOR, HOVER_EDGE_COLOR, SELECT_RING_COLOR

        plot = self.viewer.model.plot_points
        pts = self.viewer.model.points
        if self.viewer.show_query:
            if self.viewer.selected is not None:
                p = plot[self.viewer.selected]
                p3 = pts[self.viewer.selected]
                ax.scatter(
                    p3[0], p3[1], p3[2], s=200, facecolors="none", edgecolors=SELECT_RING_COLOR, linewidths=1.6, zorder=7
                )
                ax.scatter(
                    p[0], p[1], 0, s=200, facecolors="none", edgecolors=SELECT_RING_COLOR, linewidths=1.6, zorder=7
                )
                ax.plot(
                    [p[0], p3[0]],
                    [p[1], p3[1]],
                    [0, p3[2]],
                    color=SELECT_RING_COLOR,
                    linewidth=0.8,
                    linestyle=":",
                    zorder=6,
                )
            if self.viewer.entry is not None:
                p = plot[self.viewer.entry]
                p3 = pts[self.viewer.entry]
                ax.scatter(
                    p3[0],
                    p3[1],
                    p3[2],
                    s=170,
                    marker="s",
                    facecolors="white",
                    edgecolors=ENTRY_COLOR,
                    linewidths=2.2,
                    alpha=1.0,
                    zorder=7,
                )
                ax.scatter(
                    p[0],
                    p[1],
                    0,
                    s=170,
                    marker="s",
                    facecolors="white",
                    edgecolors=ENTRY_COLOR,
                    linewidths=2.2,
                    alpha=1.0,
                    zorder=7,
                )
            if self.viewer.free_query is not None and (
                self.viewer.query is None or getattr(self.viewer.query, "query", None) is None
            ):
                ax.scatter(
                    self.viewer.free_query[0],
                    self.viewer.free_query[1],
                    0,
                    s=170,
                    marker="+",
                    color=FREE_QUERY_COLOR,
                    linewidths=1.6,
                    zorder=7,
                )
                nearest = self.nearest_to_free_query()
                if nearest is not None:
                    p = self.viewer.model.points[nearest]
                    ax.plot(
                        [self.viewer.free_query[0], p[0]],
                        [self.viewer.free_query[1], p[1]],
                        [0, p[2]],
                        color=FREE_QUERY_COLOR,
                        linewidth=1.4,
                        linestyle="--",
                        zorder=6,
                    )
        if self.viewer.hovered is not None and (
            not self.viewer.show_query or self.viewer.hovered != self.viewer.selected
        ):
            p = plot[self.viewer.hovered]
            p3 = pts[self.viewer.hovered]
            ax.scatter(
                p3[0], p3[1], p3[2], s=200, facecolors="none", edgecolors=HOVER_EDGE_COLOR, linewidths=1.6, zorder=7
            )
            ax.scatter(p[0], p[1], 0, s=200, facecolors="none", edgecolors=HOVER_EDGE_COLOR, linewidths=1.6, zorder=7)
            ax.plot(
                [p[0], p3[0]], [p[1], p3[1]], [0, p3[2]], color=HOVER_EDGE_COLOR, linewidth=0.8, linestyle=":", zorder=6
            )

    def paint_ground(self, ax: Axes) -> None:
        """Draws the original 2D cloud projected onto z=0 as a flat reference plane."""
        from viewer import vertex_colors

        plot = self.viewer.model.plot_points
        x0, x1 = plot[:, 0].min(), plot[:, 0].max()
        y0, y1 = plot[:, 1].min(), plot[:, 1].max()
        pad = max(x1 - x0, y1 - y0) * 0.05
        xs = np.array([x0 - pad, x1 + pad, x1 + pad, x0 - pad])
        ys = np.array([y0 - pad, y0 - pad, y1 + pad, y1 + pad])
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.08, color="#9aa0a6", shade=False, linewidth=0, zorder=0)
        ax.scatter(
            plot[:, 0],
            plot[:, 1],
            np.zeros(len(plot)),
            s=14,
            c=vertex_colors(self.viewer.model),
            alpha=0.7,
            edgecolors="#2a2e35",
            linewidths=0.3,
            zorder=2,
        )

    def paint_path(self, ax: Axes) -> None:
        """Draws the open query's traversal route in 3D feature space and projected onto the z=0 plane."""
        from viewer import ENTRY_COLOR, EXACT_COLOR, EXACT_FILL, MARKER_EDGE, TERMINAL_COLOR

        if not self.viewer.show_query or self.viewer.query is None:
            return
        query = self.viewer.query
        path = query.path
        if len(path) == 0:
            return
        pts = self.viewer.model.points[path]
        plot = self.viewer.model.plot_points[path]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=TERMINAL_COLOR, linewidth=2.2, zorder=5)
        ax.plot(plot[:, 0], plot[:, 1], np.zeros(len(plot)), color=TERMINAL_COLOR, linewidth=1.8, alpha=0.7, zorder=3)
        entry = pts[0]
        ax.scatter(*entry, s=170, marker="s", facecolors="white", edgecolors=ENTRY_COLOR, linewidths=1.4, zorder=6)
        ax.scatter(
            plot[0, 0],
            plot[0, 1],
            0,
            s=170,
            marker="s",
            facecolors="white",
            edgecolors=ENTRY_COLOR,
            linewidths=1.4,
            zorder=6,
        )

        star_pt = self.viewer.model.points[query.exact]
        star_plot = self.viewer.model.plot_points[query.exact]
        ax.scatter(*star_pt, s=150, marker="*", facecolors="none", edgecolors=EXACT_FILL, linewidths=1.6, zorder=6)
        ax.scatter(
            star_plot[0],
            star_plot[1],
            0,
            s=150,
            marker="*",
            facecolors="none",
            edgecolors=EXACT_FILL,
            linewidths=1.6,
            zorder=6,
        )

        if query.deg_indices.size > 0:
            answer = int(query.deg_indices[0])
            ans_pt = self.viewer.model.points[answer]
            ans_plot = self.viewer.model.plot_points[answer]
            ans_color = EXACT_COLOR if answer == query.exact else TERMINAL_COLOR
            face, edge = (EXACT_FILL, MARKER_EDGE) if answer == query.exact else (ans_color, ans_color)
            ax.scatter(*ans_pt, s=170, marker="o", facecolors=face, edgecolors=edge, linewidths=1.2, alpha=1.0, zorder=6)
            ax.scatter(
                ans_plot[0], ans_plot[1], 0, s=170, marker="o", facecolors=face, edgecolors=edge, linewidths=1.2, alpha=1.0, zorder=6
            )
            if answer != query.exact:
                ax.plot(
                    [ans_pt[0], star_pt[0]],
                    [ans_pt[1], star_pt[1]],
                    [ans_pt[2], star_pt[2]],
                    color=TERMINAL_COLOR,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=5,
                )
                ax.plot(
                    [ans_plot[0], star_plot[0]],
                    [ans_plot[1], star_plot[1]],
                    [0, 0],
                    color=TERMINAL_COLOR,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=5,
                )

        # The query marker: bound directly to the path visualization
        from viewer import FREE_QUERY_COLOR

        query_pt = getattr(query, "query", None)
        if query_pt is not None:
            # Query was from a free coordinate (orange cross on z=0 plane)
            ax.scatter(
                query_pt[0],
                query_pt[1],
                0,
                s=170,
                marker="+",
                color=FREE_QUERY_COLOR,
                linewidths=1.6,
                zorder=7,
            )
            # Connect the orange query cross to the search's answer node
            if query.deg_indices.size > 0:
                ax.plot(
                    [query_pt[0], ans_plot[0]],
                    [query_pt[1], ans_plot[1]],
                    [0, 0],
                    color=FREE_QUERY_COLOR,
                    linewidth=1.2,
                    linestyle=":",
                    zorder=6,
                )

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Projects (n, 3) data coordinates to (n, 2) display pixels through the 3D axes' projection pipeline."""
        ax = self.ax3d
        vxs, vys, _vzs, _vis = proj3d._scale_proj_transform_clip(xyz[:, 0], xyz[:, 1], xyz[:, 2], ax)
        return ax.transData.transform(np.column_stack([vxs, vys]))

    def nearest(self, event: MouseEvent) -> int | None:
        """Picks the 3D vertex under the cursor: screen-closest within the radius, camera-depth tiebreak."""
        from viewer import PICK_RADIUS_PX

        if self.ax3d is None or self.viewer.model.num_points == 0 or event.x is None or event.y is None:
            return None
        pts = self.viewer.model.points
        vxs, vys, vzs, _vis = proj3d._scale_proj_transform_clip(pts[:, 0], pts[:, 1], pts[:, 2], self.ax3d)
        display = self.ax3d.transData.transform(np.column_stack([vxs, vys])) - np.array([event.x, event.y])
        squared = np.einsum("ij,ij->i", display, display)
        near = np.flatnonzero(squared <= PICK_RADIUS_PX**2)
        if near.size == 0:
            return None
        min_sq = float(np.min(squared[near]))
        candidates = near[squared[near] <= min_sq + 9.0]
        depths = np.ma.filled(vzs[candidates], np.inf)
        return int(candidates[int(np.argmin(depths))])

    def plane_point_from_click(self, event: MouseEvent) -> np.ndarray | None:
        """The point where the cursor's ray hits the z=0 ground plane, or None if outside cloud bounds."""
        if self.ax3d is None or event.x is None or event.y is None:
            return None
        vxy = self.ax3d.transData.inverted().transform([[event.x, event.y]])[0]
        vx, vy = vxy[0], vxy[1]
        M = self.ax3d.get_proj()
        A = np.array(
            [
                [M[0, 0] - vx * M[3, 0], M[0, 1] - vx * M[3, 1]],
                [M[1, 0] - vy * M[3, 0], M[1, 1] - vy * M[3, 1]],
            ]
        )
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            return None
        b = np.array(
            [
                vx * M[3, 3] - M[0, 3],
                vy * M[3, 3] - M[1, 3],
            ]
        )
        hit = np.linalg.solve(A, b)
        plot = self.viewer.model.plot_points
        x0, x1 = plot[:, 0].min(), plot[:, 0].max()
        y0, y1 = plot[:, 1].min(), plot[:, 1].max()
        pad = max(x1 - x0, y1 - y0) * 0.05
        if not (x0 - pad <= hit[0] <= x1 + pad and y0 - pad <= hit[1] <= y1 + pad):
            return None
        return np.array([hit[0], hit[1]], dtype=np.float32)

    def on_click(self, event: MouseEvent) -> None:
        """Left-click selects nearest 3D vertex; double-click runs query or places free query; right-click sets entry."""
        if event.button not in (1, 3) or event.inaxes is not self.ax3d:
            return
        index = self.nearest(event)
        if event.button == 3:
            self.viewer.set_entry(index)
            return
        if index is not None:
            if event.dblclick:
                self.viewer.set_free_query(self.viewer.model.plot_points[index].astype(np.float32))
                self.viewer.run_query()
            elif index == self.viewer.selected:
                self.viewer.select(None)
            else:
                self.viewer.select(index)
        elif event.dblclick:
            point = self.plane_point_from_click(event)
            if point is not None:
                self.viewer.set_free_query(point)
                self.viewer.run_query()
            else:
                # Double click into empty white space clears the query
                self.viewer.query = None
                self.viewer.redraw()
        else:
            self._empty_press = True
            self.viewer._3d_empty_press = True

    def on_motion(self, event: MouseEvent) -> None:
        """Updates the hovered vertex as the cursor moves over the 3D axes, mirroring the 2D hover."""
        if event.inaxes is not self.ax3d:
            return
        index = self.nearest(event)
        if index == self.viewer.hovered:
            return
        self.viewer.hovered = index
        self.redraw()
