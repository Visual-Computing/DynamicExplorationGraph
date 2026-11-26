from dataclasses import dataclass
import numpy as np
import deglib_cpp

from deglib.builder import OptimizationTarget
from deglib.distances import Metric

cpp_tuning = deglib_cpp.tuning


@dataclass(frozen=True)
class DatasetProfile:
    """Base-only geometric profile of a vector dataset."""

    sample_count: int = 0
    dim: int = 0
    participation_ratio: float = 0.0
    rel_contrast_1: float = 0.0
    pp_iqr: float = 0.0
    d16_skew: float = 0.0
    ball_cnt_p90_p10_r15: float = 0.0
    hadamard_kurt: float = 0.0
    metric: Metric = Metric.FP32_L2

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", Metric.resolve(self.metric))


@dataclass(frozen=True)
class TargetPrediction:
    """Result of :func:`predict_optimization_target`."""

    target: OptimizationTarget
    lowlid_advantage_pct: float
    profile: DatasetProfile

    def __str__(self) -> str:
        return f"{self.target.name} (expected {self.lowlid_advantage_pct:+.1f}% QPS for LowLID)"


@dataclass(frozen=True)
class QuantizationPrediction:
    """Result of :func:`predict_quantization`."""

    metric: Metric
    uint7_advantage_pct: float
    profile: DatasetProfile

    def __str__(self) -> str:
        if (int(self.profile.metric) & 0x0F) != 0x02:
            return f"{self.metric.name} (required for L2 metric)"
        return f"{self.metric.name} (expected {self.uint7_advantage_pct:+.1f}% QPS for Uint7_L2)"


def _from_cpp_profile(raw: cpp_tuning.DatasetProfile) -> DatasetProfile:
    return DatasetProfile(
        sample_count=int(raw.sample_count),
        dim=int(raw.dim),
        participation_ratio=float(raw.participation_ratio),
        rel_contrast_1=float(raw.rel_contrast_1),
        pp_iqr=float(raw.pp_iqr),
        d16_skew=float(raw.d16_skew),
        ball_cnt_p90_p10_r15=float(raw.ball_cnt_p90_p10_r15),
        hadamard_kurt=float(raw.hadamard_kurt),
        metric=Metric(int(raw.metric)),
    )


def _to_cpp_profile(profile: DatasetProfile | cpp_tuning.DatasetProfile) -> cpp_tuning.DatasetProfile:
    if isinstance(profile, cpp_tuning.DatasetProfile):
        return profile
    return cpp_tuning.DatasetProfile(
        sample_count=int(profile.sample_count),
        dim=int(profile.dim),
        participation_ratio=float(profile.participation_ratio),
        rel_contrast_1=float(profile.rel_contrast_1),
        pp_iqr=float(profile.pp_iqr),
        d16_skew=float(profile.d16_skew),
        ball_cnt_p90_p10_r15=float(profile.ball_cnt_p90_p10_r15),
        hadamard_kurt=float(profile.hadamard_kurt),
        metric=Metric.resolve(profile.metric)._to_cpp_metric(),
    )


def _prepare_vectors(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array, got {arr.ndim}D with shape {arr.shape}")
    if arr.dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        raise TypeError(f"data must have dtype float32 or float16, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _validate_counts(sample_count: int, query_count: int, seed: int, threads: int) -> None:
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}")
    if query_count < 0:
        raise ValueError(f"query_count must be non-negative, got {query_count}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if threads < 0:
        raise ValueError(f"threads must be non-negative, got {threads}")


def _predict_from_data_or_profile(
    cpp_fn,
    data_or_profile: np.ndarray | DatasetProfile,
    metric: Metric | str | int,
    sample_count: int,
    query_count: int,
    seed: int,
    threads: int,
):
    if isinstance(data_or_profile, (DatasetProfile, cpp_tuning.DatasetProfile)):
        return cpp_fn(_to_cpp_profile(data_or_profile))
    _validate_counts(sample_count, query_count, seed, threads)
    arr = _prepare_vectors(data_or_profile)
    cpp_metric = Metric.resolve(metric)._to_cpp_metric()
    return cpp_fn(
        arr,
        metric=cpp_metric,
        sample_count=int(sample_count),
        query_count=int(query_count),
        seed=int(seed),
        threads=int(threads),
    )


def measure_dataset_profile(
    data: np.ndarray,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> DatasetProfile:
    """
    Measure base-only geometric dataset properties used for pre-build tuning.

    Args:
        data: 2D NumPy array (``float32`` or ``float16``) of shape ``(N, dim)``.
        metric: Metric defining the geometry (``InnerProduct`` triggers L2 row normalization).
        sample_count: Maximum base vectors to sample (``0`` = all rows).
        query_count: Number of sampled query rows for 1-NN estimation (``0`` = all sampled rows).
        seed: Random seed for subsampling and random pairwise contrast.
        threads: Number of worker threads (``0`` = all CPU cores).

    Returns:
        :class:`DatasetProfile` containing ``participation_ratio``, ``rel_contrast_1``,
        ``pp_iqr`` (Two-NN local ID IQR), ``d16_skew``, ``ball_cnt_p90_p10_r15``, ``hadamard_kurt``, and the resolved ``metric``.
    """
    _validate_counts(sample_count, query_count, seed, threads)
    arr = _prepare_vectors(data)
    cpp_metric = Metric.resolve(metric)._to_cpp_metric()
    raw = cpp_tuning.measure_dataset_profile(
        arr,
        metric=cpp_metric,
        sample_count=int(sample_count),
        query_count=int(query_count),
        seed=int(seed),
        threads=int(threads),
    )
    return _from_cpp_profile(raw)


def predict_optimization_target(
    data_or_profile: np.ndarray | DatasetProfile,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> TargetPrediction:
    """
    Predict whether ``OptimizationTarget.LowLID`` outperforms ``OptimizationTarget.StreamingData``.

    Accepts either a 2D ``float32``/``float16`` feature matrix or a precomputed :class:`DatasetProfile`.
    """
    raw = _predict_from_data_or_profile(
        cpp_tuning.predict_optimization_target,
        data_or_profile,
        metric,
        sample_count,
        query_count,
        seed,
        threads,
    )
    return TargetPrediction(
        target=OptimizationTarget(int(raw.target)),
        lowlid_advantage_pct=float(raw.lowlid_advantage_pct),
        profile=_from_cpp_profile(raw.profile),
    )


def predict_quantization(
    data_or_graph: np.ndarray | object,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> QuantizationPrediction:
    """
    Predict whether ``Metric.Uint7_L2`` outperforms ``Metric.Int8_InnerProduct``.

    Dispatches automatically based on input type:
    - **2D NumPy array** (``deglib.build``, ``deglib.reranker_with_data``): uses **K5-Top** on raw unnormalized vectors.
    - **``DynamicExplorationGraph``** (``deglib.compress``, ``deglib.searcher``, ``deglib.reranker``): uses the
      L2-invariant **A1+A2** ensemble on unit-sphere ``cosine``/``normalized`` graphs (passing ``is_cosine`` from
      ``L2NormalizeTransform``), ``Int8_InnerProduct`` on unnormalized ``IP`` graphs, and ``Uint7_L2`` on ``L2`` graphs.
    """
    if hasattr(data_or_graph, "dynamic_exploration_graph_cpp"):
        _validate_counts(sample_count, query_count, seed, threads)
        raw = cpp_tuning.predict_quantization(
            data_or_graph.dynamic_exploration_graph_cpp,
            int(sample_count),
            int(seed),
            int(threads),
        )
    else:
        _validate_counts(sample_count, query_count, seed, threads)
        arr = _prepare_vectors(data_or_graph)
        cpp_metric = Metric.resolve(metric)._to_cpp_metric()
        raw = cpp_tuning.predict_quantization(
            arr,
            metric=cpp_metric,
            sample_count=int(sample_count),
            query_count=int(query_count),
            seed=int(seed),
            threads=int(threads),
        )
    return QuantizationPrediction(
        metric=Metric(int(raw.metric)),
        uint7_advantage_pct=float(raw.uint7_advantage_pct),
        profile=_from_cpp_profile(raw.profile),
    )


def predict_quantization_normalized(
    data: np.ndarray,
    is_cosine: bool,
    metric: Metric | str | int = Metric.FP32_InnerProduct,
    sample_count: int = 16000,
    seed: int = 7,
    threads: int = 0,
) -> QuantizationPrediction:
    """
    Predict whether ``Metric.Uint7_L2`` outperforms ``Metric.Int8_InnerProduct`` from already L2-normalized vectors (**A1+A2**).

    Intended for ``deglib.compress`` / ``deglib.searcher`` / ``deglib.reranker`` when original raw vectors are unavailable.
    ``is_cosine`` is ``True`` when an ``L2NormalizeTransform`` was attached and ``False`` when inputs were already unit-norm.
    Non-InnerProduct (L2) datasets unconditionally select ``Metric.Uint7_L2``.
    """
    _validate_counts(sample_count, 0, seed, threads)
    arr = _prepare_vectors(data)
    cpp_metric = Metric.resolve(metric)._to_cpp_metric()
    raw = cpp_tuning.predict_quantization_normalized(
        arr,
        is_cosine=bool(is_cosine),
        metric=cpp_metric,
        sample_count=int(sample_count),
        seed=int(seed),
        threads=int(threads),
    )
    return QuantizationPrediction(
        metric=Metric(int(raw.metric)),
        uint7_advantage_pct=float(raw.uint7_advantage_pct),
        profile=_from_cpp_profile(raw.profile),
    )


__all__ = [
    "DatasetProfile",
    "TargetPrediction",
    "QuantizationPrediction",
    "measure_dataset_profile",
    "predict_optimization_target",
    "predict_quantization",
    "predict_quantization_normalized",
]
