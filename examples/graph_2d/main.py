"""
2D graph explorer: invents a point cloud, builds a DEG over it, and draws the result interactively.

The graph is built with `k` outgoing edges per vertex (4 by default, and never fewer) over FP32
features, in L2 or inner-product space, which is the smallest configuration that still yields a
navigable proximity graph. The UI lets you switch distribution, size, k and metric live, inspect
single vertices, pick the start node of a traversal and run the search from it to a selected vertex,
and draw either the built graph or any of the reference graphs beside it. Every build also compares
the DEG against the Delaunay graph, Gabriel, the RNG, the MST and the MRNG over the same cloud;
clouds stop at 1200 points because those are not cheap.

Usage:
  uv run main.py                                   # interactive window on 200 uniform points
  uv run main.py --preset moons --vertices 600     # another distribution and size
  uv run main.py --k 8                             # more edges per vertex
  uv run main.py --search-metric IP                # walk the L2 graph by inner product
  uv run main.py --mips                            # build on the MIPS→L2 spherical lift, plot stays 2D
  uv run main.py --no-show --output deg.png        # headless render, no GUI
"""

from __future__ import annotations

import argparse

from dataset import MAX_POINTS, MIN_POINTS, PRESETS, make_points
from deg_graph import DEFAULT_K, METRIC_BY_LABEL, METRIC_LABELS, MIN_K, GraphModel, build_model, describe
from deglib.distances import Metric
from deglib.optimization import mips_l2_transform

DEFAULT_VERTICES = 200


def metric_option(label: str) -> Metric:
    """Parses the CLI's short metric name, failing the way argparse wants a bad value to fail."""
    if label not in METRIC_BY_LABEL:
        raise argparse.ArgumentError(None, f"invalid metric {label!r} — choose from {' or '.join(METRIC_BY_LABEL)}")
    return METRIC_BY_LABEL[label]


def build_scene(
    preset: str, num_points: int, k: int, seed: int, metric: Metric, mips: bool = False, threads: int = 1
) -> GraphModel:
    """
    Generates the point cloud and the DEG over it — the single source of truth for both UI and CLI.

    With `mips` the cloud is lifted into the MIPS→L2 spherical space first, which appends a third
    coordinate of equal norm so an inner-product search becomes an L2 search. The graph, the searches and
    the ground truth then run on those 3D features, while the original 2D cloud is handed to the model as
    its drawing coordinates so the picture keeps its shape.
    """
    points, groups = make_points(preset, num_points, seed)
    features = mips_l2_transform(points)[0] if mips else points
    return build_model(features, groups, k=k, seed=seed, threads=threads, metric=metric, plot_points=points, mips=mips)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and visualise a DEG on synthetic 2D data")
    parser.add_argument(
        "--preset", choices=PRESETS, default="uniform", help="Synthetic 2D distribution (default: uniform)"
    )
    parser.add_argument(
        "--vertices",
        type=int,
        default=DEFAULT_VERTICES,
        help=f"Number of generated points, {MIN_POINTS} to {MAX_POINTS} (default: {DEFAULT_VERTICES})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Edges per vertex, even and at least {MIN_K} (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--metric",
        type=metric_option,
        metavar="{L2,IP}",
        default=Metric.FP32_L2,
        help=f"Metric the graph is built with ({' or '.join(METRIC_BY_LABEL)}, default: "
        f"{METRIC_LABELS[Metric.FP32_L2]})",
    )
    parser.add_argument(
        "--search-metric",
        type=metric_option,
        metavar="{L2,IP}",
        default=None,
        help="Metric the traversal ranks neighbours by, the build metric unless given. The graph is not "
        "rebuilt — a read-only view re-ranks the same edges, so both answers can be compared on one cloud",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed for the point cloud and the graph build")
    parser.add_argument(
        "--mips",
        action="store_true",
        help="Transform the data into the MIPS→L2 spherical space, adding a third dimension used by the "
        "graph and searches while the plot stays 2D",
    )
    parser.add_argument("--output", type=str, default=None, help="Render the graph to this image file")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window")
    args = parser.parse_args()

    try:
        model = build_scene(args.preset, args.vertices, args.k, args.seed, args.metric, args.mips)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")

    print(f"DEG over {args.preset} points:")
    for line in describe(model):
        print(f"  {line}")

    if args.no_show or args.output:
        from viewer import render_static

        target = render_static(model, args.output or "graph_2d.png")
        print(f"Rendered graph to {target}")
        return

    from viewer import GraphViewer

    GraphViewer(
        build_scene, args.preset, args.vertices, args.k, args.seed, args.metric, args.search_metric, args.mips
    ).run()


if __name__ == "__main__":
    main()
