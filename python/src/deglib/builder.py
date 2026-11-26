from collections.abc import Iterable, Sequence
import enum
import os
import sys
import time
from typing import Callable, Literal
import numpy as np
import deglib_cpp

from deglib.distances import Metric, _infer_dim_from_array
from deglib.graph import (
    DynamicExplorationGraph,
    _FP16_METRICS,
    _FP32_METRICS,
    compress as compress_graph,
    create_empty,
)
from deglib.optimization import presort as run_presort
from deglib.transformation import (
    FeatureTransformation,
    apply_on_database,
    is_high_level_metric_name,
    make_quantizer_pipeline,
    resolve_spatial_metric,
)
from deglib.utils import InvalidShapeException, assure_array


class OptimizationTarget(enum.IntEnum):
    """
    Graph construction optimization strategy.

    - ``Auto``: Automatically predicts ``LowLID`` vs ``StreamingData`` from dataset geometry.
    - ``StreamingData``: Sequential single-threaded insertion/deletion for shifting data streams.
    - ``HighLID``: Parallel batch extension replacing worst neighbor edge (for LID > 15).
    - ``LowLID``: Default parallel batch extension minimizing metric distortion (for LID <= 15).
    """

    StreamingData = deglib_cpp.OptimizationTarget.StreamingData
    HighLID = deglib_cpp.OptimizationTarget.HighLID
    LowLID = deglib_cpp.OptimizationTarget.LowLID
    Auto = -1

    @classmethod
    def resolve(cls, target: "OptimizationTarget | str | int") -> "OptimizationTarget":
        if isinstance(target, cls):
            return target
        if isinstance(target, deglib_cpp.OptimizationTarget):
            return cls(int(target))
        if isinstance(target, int) and not isinstance(target, bool):
            return cls(target)
        if isinstance(target, str):
            key = target.strip().lower().replace("-", "").replace("_", "")
            mapping = {
                "auto": cls.Auto,
                "lowlid": cls.LowLID,
                "highlid": cls.HighLID,
                "streamingdata": cls.StreamingData,
                "streaming": cls.StreamingData,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unknown OptimizationTarget: {target!r}. Available: {[m.name for m in cls]}")


class BuilderStatus:
    """
    Progress snapshot reported by ``GraphBuilder.build``.
    """

    def __init__(self, status_cpp: deglib_cpp.BuilderStatus):
        self._status_cpp = status_cpp

    @property
    def step(self) -> int:
        return self._status_cpp.step

    @property
    def added(self) -> int:
        return self._status_cpp.added

    @property
    def deleted(self) -> int:
        return self._status_cpp.deleted

    @property
    def improved(self) -> int:
        return self._status_cpp.improved

    @property
    def tries(self) -> int:
        return self._status_cpp.tries

    @property
    def step_added_ids(self) -> list[int]:
        return self._status_cpp.step_added_ids

    @property
    def step_deleted_ids(self) -> list[int]:
        return self._status_cpp.step_deleted_ids

    @property
    def total_added_ids(self) -> list[int]:
        return self._status_cpp.total_added_ids

    @property
    def total_deleted_ids(self) -> list[int]:
        return self._status_cpp.total_deleted_ids


class ProgressCallback:
    """
    Utility callback rendering a terminal progress bar for ``GraphBuilder.build``.
    """

    def __init__(
        self,
        num_new_entries: int,
        num_remove_entries: int,
        bar_length: int = 60,
        min_print_interval: float = 0.1,
    ):
        self.num_new_entries = max(1, int(num_new_entries))
        self.num_remove_entries = max(0, int(num_remove_entries))
        self.bar_length = max(10, int(bar_length))
        self.min_print_interval = float(min_print_interval)
        self.last_print = 0.0
        self.total = self.num_new_entries + self.num_remove_entries

    def __call__(self, status: BuilderStatus) -> None:
        now = time.monotonic()
        done = status.added + status.deleted
        if done < self.total and (now - self.last_print) < self.min_print_interval:
            return
        self.last_print = now
        frac = min(1.0, done / self.total) if self.total > 0 else 1.0
        filled = int(self.bar_length * frac)
        bar = "#" * filled + "-" * (self.bar_length - filled)
        sys.stdout.write(
            f"\rBuilding [{bar}] {frac * 100:5.1f}% (step={status.step}, +{status.added}, -{status.deleted})"
        )
        if done >= self.total:
            sys.stdout.write("\n")
        sys.stdout.flush()


class GraphBuilder:
    """
    Incremental builder for adding and removing vertices on mutable ``DynamicExplorationGraph`` instances.
    """

    def __init__(
        self,
        graph: DynamicExplorationGraph,
        seed: int = 7,
        optimization_target: OptimizationTarget | str = OptimizationTarget.LowLID,
        extend_k: int = 0,
        extend_eps: float = 0.1,
        improve_k: int = 0,
        improve_eps: float = 0.001,
        max_path_length: int = 5,
        improve_tries: int = 0,
        threads: int = 0,
    ):
        opt_target = OptimizationTarget.resolve(optimization_target)
        if opt_target == OptimizationTarget.Auto:
            opt_target = OptimizationTarget.LowLID
        edges = graph.get_edges_per_vertex()
        if extend_k <= edges:
            extend_k = edges * 2
        if improve_k == 0:
            improve_k = edges

        self.builder_cpp = deglib_cpp.GraphBuilder(
            graph.dynamic_exploration_graph_cpp,
            seed,
            deglib_cpp.OptimizationTarget(int(opt_target)),
            extend_k,
            extend_eps,
            improve_k,
            improve_eps,
            max_path_length,
            improve_tries,
        )
        self.graph = graph
        self.optimization_target = opt_target
        if threads > 0:
            self.builder_cpp.set_threads(int(threads))

    def add_entry(self, external_label: int | Iterable[int] | np.ndarray, feature: np.ndarray) -> None:
        """
        Queue one or more ``(label, feature)`` entries to insert on the next ``build()`` call.
        """
        if not isinstance(feature, np.ndarray):
            feature = np.asarray(feature)

        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        elif feature.ndim != 2:
            raise InvalidShapeException(f"Expected feature to have 1 or 2 dimensions, got shape {feature.shape}")

        if self.graph.has_transformations():
            accepted = self.graph.get_metric().get_accepted_dtypes()
            if feature.dtype == np.float32:
                feature = np.ascontiguousarray(feature)
            elif feature.dtype == np.float16 and np.dtype(np.float16) not in accepted:
                feature = np.ascontiguousarray(feature, dtype=np.float32)
            else:
                feature = assure_array(feature, "feature", accepted)
        else:
            feature = assure_array(feature, "feature", self.graph.get_metric().get_accepted_dtypes())

        if isinstance(external_label, (int, np.integer)) and not isinstance(external_label, bool):
            labels_arr = np.array([int(external_label)], dtype=np.uint32)
        else:
            labels_arr = np.ascontiguousarray(
                list(external_label) if not isinstance(external_label, np.ndarray) else external_label, dtype=np.uint32
            )
            if labels_arr.ndim != 1:
                raise InvalidShapeException(f"external_label must be scalar or 1D, got shape {labels_arr.shape}")

        if labels_arr.shape[0] != feature.shape[0]:
            raise ValueError(
                f"Number of labels ({labels_arr.shape[0]}) does not match number of feature rows ({feature.shape[0]})"
            )

        self.builder_cpp.add_entry(labels_arr, feature)

    def remove_entry(self, external_label: int | Iterable[int] | np.ndarray) -> None:
        """
        Queue a single external label or batch of external labels for deletion on the next ``build()``.
        """
        if isinstance(external_label, (int, np.integer)) and not isinstance(external_label, bool):
            self.builder_cpp.remove_entry(int(external_label))
        else:
            arr = np.asarray(
                list(external_label) if not isinstance(external_label, np.ndarray) else external_label,
                dtype=np.uint32,
            )
            for lbl in arr.ravel():
                self.builder_cpp.remove_entry(int(lbl))

    def get_num_new_entries(self) -> int:
        return int(self.builder_cpp.get_num_new_entries())

    def get_num_remove_entries(self) -> int:
        return int(self.builder_cpp.get_num_remove_entries())

    def set_threads(self, threads: int) -> None:
        """
        Set worker thread count for graph building (``0`` selects all available CPU cores).
        """
        eff = (os.cpu_count() or 1) if threads <= 0 else int(threads)
        self.builder_cpp.set_threads(eff)

    def set_batch_size(self, tasks_per_batch: int, task_size: int) -> None:
        self.builder_cpp.set_batch_size(int(tasks_per_batch), int(task_size))

    def get_batch_size(self) -> int:
        return int(self.builder_cpp.get_batch_size())

    def build(
        self,
        callback: Callable[[BuilderStatus], None] | None = None,
        show_progress: bool = False,
        infinite: bool = False,
    ) -> BuilderStatus:
        """
        Build or update the graph with all queued additions and removals.
        """
        cb = callback
        if cb is None and show_progress and not infinite:
            cb = ProgressCallback(self.get_num_new_entries(), self.get_num_remove_entries())

        wrapped_cb = None
        if cb is not None:

            def _cb_wrapper(status_cpp: deglib_cpp.BuilderStatus) -> None:
                cb(BuilderStatus(status_cpp))

            wrapped_cb = _cb_wrapper

        status_cpp = self.builder_cpp.build(wrapped_cb, bool(infinite))
        return BuilderStatus(status_cpp)

    def stop(self) -> None:
        self.builder_cpp.stop()


def build(
    data: np.ndarray,
    labels: Iterable[int] | np.ndarray | None = None,
    metric: Literal["l2", "euclidean", "cosine", "normalized", "ip"] | Metric | str = "l2",
    compress: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str | None = None,
    edges_per_vertex: int = 32,
    optimization_target: OptimizationTarget | str = OptimizationTarget.Auto,
    extend_k: int = 0,
    extend_eps: float = 0.1,
    improve_k: int = 0,
    improve_eps: float = 0.001,
    max_path_length: int = 5,
    improve_tries: int = 0,
    seed: int = 7,
    presort: bool | np.ndarray = False,
    optimize: bool = True,
    threads: int = 0,
    show_progress: bool | None = None,
) -> DynamicExplorationGraph:
    """
    Build and optimize a ``DynamicExplorationGraph`` from a 2D NumPy feature matrix.

    Parameters
    ----------
    data : np.ndarray
        2D feature matrix of shape ``(N, dim)``. Supports ``float32`` and ``float16`` for
        high-level metrics and ``compress``, or pre-quantized dtypes (e.g. ``int8``, ``uint8``)
        when an explicit quantized ``Metric`` is passed.
    labels : iterable of int or np.ndarray, optional
        1D sequence of external ``uint32`` vertex IDs of length ``N``. Defaults to ``0..N-1``.
    metric : {"l2", "euclidean", "cosine", "normalized", "ip"} or Metric or str, default="l2"
        Distance metric or high-level spatial mode:

        - ``"l2"`` / ``"euclidean"``: Euclidean L2 distance (``FP32_L2`` or ``FP16_L2``).
        - ``"cosine"`` / ``"normalized"``: Cosine similarity via ``FP32_InnerProduct`` /
          ``FP16_InnerProduct``; automatically attaches ``L2NormalizeTransform`` if input
          vectors are not already unit-norm.
        - ``"ip"`` (also ``"innerproduct"``, ``"inner_product"``, ``"mips"``): Maximum Inner
          Product Search. Builds the graph topology in ``MipsTransform (d+1) + L2`` space
          and swaps stored features back to unnormalized ``InnerProduct`` (or ``compress``
          target) on the returned ``ReadOnlyGraph``.
        - Explicit ``Metric`` or metric string (e.g. ``Metric.Int8_L2``, ``"Uint7_L2"``).
    compress : {"32bit", "16bit", "8bit", "4bit", "2bit", "auto"} or Metric or str, optional
        Optional post-build compression target:

        - ``None``: Keep native build precision (unnormalized ``FP32_InnerProduct`` /
          ``FP16_InnerProduct`` for ``metric="ip"``).
        - ``"32bit"`` (also ``"fp32"``, ``"float32"``): Keep/target 32-bit float precision.
        - ``"16bit"`` (also ``"fp16"``, ``"float16"``): ``FP16_L2`` or ``FP16_InnerProduct``.
        - ``"8bit"`` / ``"auto"``: ``Uint7_L2`` (L2), ``Int8_InnerProduct`` or ``Uint7_L2`` (cosine/normalized, auto-selected),
          or ``FastHadamardRotation + Int8_InnerProduct`` (``metric="ip"``).
        - ``"4bit"``: Always ``Uint4_L2`` (with ``MipsTransform`` for ``metric="ip"``).
        - ``"2bit"`` (also ``"evp"``): ``EVP_InnerProduct``.
        - Any explicit ``Metric`` or metric name string.
    edges_per_vertex : int, default=32
        Number of outgoing neighbors per vertex (``k_graph``).
    optimization_target : OptimizationTarget or str, default=OptimizationTarget.Auto
        Graph construction strategy: ``"auto"`` (predicts ``LowLID`` vs ``StreamingData``),
        ``"lowlid"``, ``"highlid"``, or ``"streaming"`` / ``"streamingdata"``.
    extend_k : int, default=0
        Candidate pool size during neighbor extension (``0`` uses builder default).
    extend_eps : float, default=0.1
        Exploration tolerance ``eps`` during neighbor extension.
    improve_k : int, default=0
        Candidate pool size during edge improvement (``0`` disables).
    improve_eps : float, default=0.001
        Exploration tolerance ``eps`` during edge improvement.
    max_path_length : int, default=5
        Maximum path length considered during RNG edge pruning/improvement.
    improve_tries : int, default=0
        Number of improvement passes per step.
    seed : int, default=7
        Random seed used during graph construction.
    presort : bool or np.ndarray, default=False
        If ``True``, runs FLAS 1D pre-sorting before construction (reordering data and labels
        together). Alternatively, pass a precomputed 1D ``uint32`` permutation array.
    optimize : bool, default=True
        If ``True``, runs ``graph.optimize()`` (prefetch & entry-point tuning) after build.
    threads : int, default=0
        Number of worker threads (``0`` = all CPU cores).
    show_progress : bool, optional
        Show a live progress bar on stdout (defaults to ``sys.stdout.isatty()``).

    Returns
    -------
    DynamicExplorationGraph
        Mutable ``SizeBoundedGraph`` when ``compress=None`` (or ``"32bit"``), or
        compressed ``ReadOnlyGraph`` when ``compress`` is specified.

    See Also
    --------
    deglib.builder.GraphBuilder : Low-level incremental/streaming graph builder.
    deglib.graph.create_empty : Create an empty fixed-capacity mutable graph.
    deglib.graph.create_dynamic_empty : Create an empty chunked mutable graph.
    deglib.presort : Standalone FLAS 1D pre-sorting.
    deglib.compress : Post-build graph quantization.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise InvalidShapeException(f"data must be a 2D numpy array, got {getattr(data, 'shape', type(data))}")

    is_float_input = data.dtype in (np.dtype(np.float32), np.dtype(np.float16))
    if not is_float_input:
        if compress is not None:
            raise ValueError(f"compress requires float32 or float16 input data, got dtype {data.dtype}")
        if isinstance(presort, (bool, np.bool_)) and presort:
            raise ValueError(
                "presort=True requires float32 or float16 input vectors; "
                "pass a precomputed sorted_indices array instead."
            )

    # High-level build pipeline (see docs/python_metric_quantization_matrix.md):
    #   1. Resolve spatial metric/prefix (L2NormalizeTransform / MipsTransform) on float inputs.
    #   2. Predict OptimizationTarget.Auto on the post-prefix vectors if requested.
    #   3. Run optional 1D FLAS presort on the post-prefix vectors.
    #   4. Quantize (if explicit quantized Metric) and build mutable SizeBoundedGraph.
    #   5. Finalize: swap MIPS(d+1) back to d-dim IP (for metric="ip") and apply `compress`.
    n_samples = int(data.shape[0])
    effective_show_progress = sys.stdout.isatty() if show_progress is None else bool(show_progress)

    xfrms: list[FeatureTransformation] = []
    is_high_level_ip = False
    if is_high_level_metric_name(metric):
        resolved_metric, xfrms = resolve_spatial_metric(data, metric)
        is_high_level_ip = metric.strip().lower() in ("ip", "innerproduct", "inner_product", "mips")
    else:
        resolved_metric = Metric.resolve(metric)

    base_metric = Metric(int(resolved_metric))
    opt_target = OptimizationTarget.resolve(optimization_target)

    if labels is None:
        labels_arr = np.arange(n_samples, dtype=np.uint32)
    else:
        raw_labels = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels)
        if np.issubdtype(raw_labels.dtype, np.floating) or raw_labels.dtype == np.bool_:
            raise TypeError(f"labels must be integer values, got dtype {raw_labels.dtype}")
        labels_arr = np.ascontiguousarray(raw_labels, dtype=np.uint32)
        if labels_arr.ndim != 1 or labels_arr.shape[0] != n_samples:
            raise ValueError(f"labels must be 1D of length {n_samples}, got shape {labels_arr.shape}")

    if n_samples == 0:
        if is_high_level_ip:
            empty_metric = (
                Metric.FP16_InnerProduct if data.dtype == np.dtype(np.float16) else Metric.FP32_InnerProduct
            )
            return create_empty(
                capacity=0,
                dim=int(data.shape[1]),
                edges_per_vertex=edges_per_vertex,
                metric=empty_metric,
                transformations=None,
            )
        inferred_dim = int(xfrms[-1].output_dim) if xfrms else _infer_dim_from_array(resolved_metric, data)
        return create_empty(
            capacity=0,
            dim=inferred_dim,
            edges_per_vertex=edges_per_vertex,
            metric=resolved_metric,
            transformations=xfrms,
        )

    work_data = np.ascontiguousarray(data)

    # Apply spatial prefix transforms (MipsTransform / L2NormalizeTransform) once if present,
    # so Auto target prediction, presort, and graph build all operate in the true build space.
    spatial_data = apply_on_database(xfrms, work_data, threads=threads) if xfrms else None
    if spatial_data is not None and spatial_data.dtype == np.float32 and base_metric in _FP16_METRICS:
        fp32_base = Metric.FP32_InnerProduct if base_metric.name.endswith("InnerProduct") else Metric.FP32_L2
        bound_inst = getattr(resolved_metric, "_instruction", None)
        resolved_metric = fp32_base.with_instruction(bound_inst) if bound_inst is not None else fp32_base
        base_metric = Metric(int(resolved_metric))
    effective_float_data = spatial_data if spatial_data is not None else work_data

    # 1. Resolve OptimizationTarget.Auto using tuning.py defaults
    if opt_target == OptimizationTarget.Auto:
        if is_float_input and base_metric in (_FP32_METRICS | _FP16_METRICS):
            from deglib.tuning import predict_optimization_target

            opt_target = predict_optimization_target(
                effective_float_data,
                metric=resolved_metric,
                threads=threads,
            ).target
        else:
            opt_target = OptimizationTarget.LowLID

    # 2. Optional FLAS 1D pre-sorting (reorders both data and labels together)
    if isinstance(presort, (bool, np.bool_)):
        if presort:
            if not is_float_input:
                raise ValueError(
                    "presort=True requires float32 or float16 input vectors; "
                    "pass a precomputed sorted_indices array instead."
                )
            presort_vecs = spatial_data if spatial_data is not None else work_data
            flas_base = Metric.FP16_L2 if presort_vecs.dtype == np.float16 else Metric.FP32_L2
            bound_inst = getattr(resolved_metric, "_instruction", None)
            flas_metric = flas_base.with_instruction(bound_inst) if bound_inst is not None else flas_base
            sorted_idx = run_presort(
                presort_vecs,
                metric=flas_metric,
                threads=threads,
                show_progress=effective_show_progress,
            )
            work_data = np.ascontiguousarray(work_data[sorted_idx])
            if spatial_data is not None:
                spatial_data = np.ascontiguousarray(spatial_data[sorted_idx])
            labels_arr = np.ascontiguousarray(labels_arr[sorted_idx])
    elif presort is not None:
        sorted_idx = np.ascontiguousarray(presort, dtype=np.uint32)
        if sorted_idx.ndim != 1 or sorted_idx.shape[0] != n_samples:
            raise ValueError(
                f"presort sorted_indices must be a 1D array of length {n_samples}, got shape {sorted_idx.shape}"
            )
        work_data = np.ascontiguousarray(work_data[sorted_idx])
        if spatial_data is not None:
            spatial_data = np.ascontiguousarray(spatial_data[sorted_idx])
        labels_arr = np.ascontiguousarray(labels_arr[sorted_idx])

    # 3. Transformations / auto-quantization (run in parallel BEFORE GraphBuilder so add_entry stays fast)
    if spatial_data is not None:
        build_data = assure_array(spatial_data, "data", resolved_metric.get_accepted_dtypes())
        graph_dim = int(build_data.shape[1])
    else:
        is_float_src = work_data.dtype == np.float32 or (
            work_data.dtype == np.float16 and base_metric not in _FP16_METRICS
        )
        if base_metric not in _FP32_METRICS and is_float_src:
            graph_dim = int(work_data.shape[1])
            xfrms = make_quantizer_pipeline(
                work_data,
                resolved_metric,
                use_rerank=False,
                threads=threads,
            )
            build_data = apply_on_database(xfrms, work_data, threads=threads)
        else:
            build_data = assure_array(work_data, "data", resolved_metric.get_accepted_dtypes())
            graph_dim = _infer_dim_from_array(resolved_metric, build_data)

    # 4. Construct graph without transformations during add_entry, then attach afterwards.
    graph = create_empty(
        capacity=n_samples,
        dim=graph_dim,
        edges_per_vertex=edges_per_vertex,
        metric=resolved_metric,
        transformations=None,
    )
    builder = GraphBuilder(
        graph=graph,
        seed=seed,
        optimization_target=opt_target,
        extend_k=extend_k,
        extend_eps=extend_eps,
        improve_k=improve_k,
        improve_eps=improve_eps,
        max_path_length=max_path_length,
        improve_tries=improve_tries,
        threads=threads,
    )
    builder.add_entry(labels_arr, build_data)
    builder.build(show_progress=effective_show_progress)

    if xfrms:
        graph.set_transformations(xfrms)

    # 5. Finalize graph (for high-level IP, swap MIPS build features back to unnormalized IP)
    if is_high_level_ip:
        comp_key = (
            compress.strip().lower().replace("-", "").replace("_", "")
            if isinstance(compress, str)
            else None
        )
        if compress is None or comp_key in ("32bit", "fp32", "float32"):
            target_ip = (
                Metric.FP16_InnerProduct if data.dtype == np.dtype(np.float16) else Metric.FP32_InnerProduct
            )
            graph = graph.with_features(work_data, metric=target_ip, order="internal", threads=threads)
        elif comp_key in ("16bit", "fp16", "float16"):
            graph = graph.to_readonly().with_features(
                work_data, metric=Metric.FP16_InnerProduct, order="internal", threads=threads
            )
        elif comp_key in ("8bit", "auto"):
            from deglib.transformation import make_fast_hadamard_rotation, make_scalar_quantizer_int8

            rot = make_fast_hadamard_rotation(work_data)
            rot_data = apply_on_database(rot, work_data, threads=threads)
            quant = make_scalar_quantizer_int8(rot_data)
            graph = graph.to_readonly().with_features(
                work_data,
                metric=Metric.Int8_InnerProduct,
                transformations=[rot, quant],
                order="internal",
                threads=threads,
            )
        elif comp_key == "4bit":
            graph = compress_graph(graph, Metric.Uint4_L2, threads=threads)
        else:
            graph = compress_graph(graph, compress, threads=threads)
    elif compress is not None:
        comp_target: Metric | str = compress
        if (
            isinstance(compress, str)
            and compress.strip().lower() in ("8bit", "auto")
            and base_metric.name.endswith("InnerProduct")
        ):
            from deglib.tuning import predict_quantization

            comp_target = predict_quantization(
                effective_float_data,
                metric=resolved_metric,
                threads=threads,
            ).metric
        graph = compress_graph(graph, comp_target, threads=threads)

    if optimize and graph.size() > 0:
        graph.optimize()
    return graph


__all__ = [
    "OptimizationTarget",
    "BuilderStatus",
    "ProgressCallback",
    "GraphBuilder",
    "build",
]
