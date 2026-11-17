from collections.abc import Iterable, Sequence
import enum
import os
import sys
import time
from typing import Callable
import numpy as np
import deglib_cpp

from deglib.distances import Metric, _BoundMetric, _infer_dim_from_array
from deglib.graph import (
    DynamicExplorationGraph,
    _FP16_METRICS,
    _FP32_METRICS,
    create_empty,
)
from deglib.optimization import presort as run_presort
from deglib.transformation import (
    FeatureTransformation,
    _normalize_transforms,
    apply_on_database,
    make_quantizer_pipeline,
)
from deglib.utils import InvalidShapeException, assure_array


class OptimizationTarget(enum.IntEnum):
    """
    Graph construction optimization strategy.

    - ``StreamingData``: Sequential single-threaded insertion/deletion for shifting data streams.
    - ``HighLID``: Parallel batch extension replacing worst neighbor edge (for LID > 15).
    - ``LowLID``: Default parallel batch extension minimizing metric distortion (for LID <= 15).
    """

    StreamingData = deglib_cpp.OptimizationTarget.StreamingData
    HighLID = deglib_cpp.OptimizationTarget.HighLID
    LowLID = deglib_cpp.OptimizationTarget.LowLID

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
        seed: int | None = None,
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
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    optimization_target: OptimizationTarget | str = OptimizationTarget.LowLID,
    presort: bool | np.ndarray = False,
    presort_decay: float = 0.9,
    optimize: bool = True,
    readonly: bool = False,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    capacity: int = -1,
    extend_k: int = 0,
    extend_eps: float = 0.1,
    improve_k: int = 0,
    improve_eps: float = 0.001,
    max_path_length: int = 5,
    improve_tries: int = 0,
    seed: int | None = None,
    threads: int = 0,
    callback: Callable[[BuilderStatus], None] | None = None,
    show_progress: bool = False,
) -> DynamicExplorationGraph:
    """
    Build and optimize a ``DynamicExplorationGraph`` from a 2D NumPy feature matrix.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise InvalidShapeException(f"data must be a 2D numpy array, got {getattr(data, 'shape', type(data))}")

    resolved_metric = Metric.resolve(metric)
    base_metric = Metric(int(resolved_metric))
    opt_target = OptimizationTarget.resolve(optimization_target)
    xfrms = _normalize_transforms(transformations)
    for i, t in enumerate(xfrms):
        if not isinstance(t, FeatureTransformation):
            raise TypeError(f"transformations[{i}] must be a FeatureTransformation, got {type(t).__name__}")

    n_samples = int(data.shape[0])
    target_capacity = max(n_samples, int(capacity)) if capacity >= 0 else n_samples

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
        inferred_dim = int(xfrms[-1].output_dim) if xfrms else _infer_dim_from_array(resolved_metric, data)
        g = create_empty(
            capacity=target_capacity,
            dim=inferred_dim,
            edges_per_vertex=edges_per_vertex,
            metric=resolved_metric,
            transformations=xfrms,
        )
        return g.to_readonly(optimize=False, threads=threads) if readonly else g

    work_data = np.ascontiguousarray(data)

    # 1. Optional FLAS 1D pre-sorting (reorders both data and labels together)
    if isinstance(presort, (bool, np.bool_)):
        if presort:
            if work_data.dtype == np.float32:
                presort_vecs = work_data
            elif work_data.dtype == np.float16:
                presort_vecs = np.ascontiguousarray(work_data, dtype=np.float32)
            else:
                raise ValueError(
                    "presort=True requires float32 or float16 input vectors; "
                    "pass a precomputed sorted_indices array instead."
                )
            flas_base = Metric.FP32_InnerProduct if base_metric.name.endswith("InnerProduct") else Metric.FP32_L2
            flas_metric = (
                flas_base.with_instruction(resolved_metric._instruction)
                if isinstance(resolved_metric, _BoundMetric)
                else flas_base
            )
            sorted_idx = run_presort(
                presort_vecs,
                metric=flas_metric,
                radius_decay=presort_decay,
                threads=threads,
                show_progress=show_progress,
            )
            work_data = np.ascontiguousarray(work_data[sorted_idx])
            labels_arr = np.ascontiguousarray(labels_arr[sorted_idx])
    elif presort is not None:
        sorted_idx = np.ascontiguousarray(presort, dtype=np.uint32)
        if sorted_idx.ndim != 1 or sorted_idx.shape[0] != n_samples:
            raise ValueError(
                f"presort sorted_indices must be a 1D array of length {n_samples}, got shape {sorted_idx.shape}"
            )
        work_data = np.ascontiguousarray(work_data[sorted_idx])
        labels_arr = np.ascontiguousarray(labels_arr[sorted_idx])

    # 2. Transformations / auto-quantization (run in parallel BEFORE GraphBuilder so add_entry stays fast)
    if len(xfrms) > 0:
        last_t = xfrms[-1]
        if last_t.output_dtype != np.float32:
            needs_apply = work_data.dtype == np.float32 or (
                work_data.dtype == np.float16 and base_metric not in _FP16_METRICS
            )
            if needs_apply and work_data.shape[1] == xfrms[0].input_dim:
                build_data = apply_on_database(xfrms, work_data, threads=threads)
                graph_dim = int(last_t.output_dim)
            else:
                build_data = assure_array(work_data, "data", resolved_metric.get_accepted_dtypes())
                graph_dim = _infer_dim_from_array(resolved_metric, build_data)
        else:
            if last_t.output_dim != xfrms[0].input_dim:
                if work_data.shape[1] == xfrms[0].input_dim:
                    build_data = apply_on_database(xfrms, work_data, threads=threads)
                    graph_dim = int(last_t.output_dim)
                elif work_data.shape[1] == last_t.output_dim:
                    build_data = assure_array(work_data, "data", np.dtype(np.float32))
                    graph_dim = int(build_data.shape[1])
                else:
                    raise InvalidShapeException(
                        f"data width {work_data.shape[1]} matches neither transformation input_dim "
                        f"({xfrms[0].input_dim}) nor output_dim ({last_t.output_dim})"
                    )
            elif any(t.name == "L2NormalizeTransform" for t in xfrms):
                build_data = apply_on_database(xfrms, work_data, threads=threads)
                graph_dim = int(build_data.shape[1])
            else:
                build_data = assure_array(work_data, "data", resolved_metric.get_accepted_dtypes())
                graph_dim = _infer_dim_from_array(resolved_metric, build_data)
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

    # 3. Construct graph without transformations during add_entry, then attach afterwards.
    graph = create_empty(
        capacity=target_capacity,
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
    builder.build(callback=callback, show_progress=show_progress)

    if xfrms:
        graph.set_transformations(xfrms)

    if readonly:
        return graph.to_readonly(optimize=optimize, threads=threads)
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
