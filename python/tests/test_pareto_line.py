import numpy as np
import pytest

import deglib
import deglib_cpp
from deglib import Metric, OptimizationTarget
from deglib.search import ParetoLine, RecallTable


def _make_clustered_data(n: int, dim: int, clusters: int = 20, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(clusters, dim)).astype(np.float32)
    idx = np.arange(n) % clusters
    noise = rng.normal(0.0, 0.35, size=(n, dim)).astype(np.float32)
    return np.ascontiguousarray(centers[idx] + noise, dtype=np.float32)


def _verify_pareto_invariants(line: ParetoLine) -> None:
    assert isinstance(line, ParetoLine)
    assert all(isinstance(t, RecallTable) for t in line.recall_tables)
    entries = line.entries
    assert len(line) == len(entries)

    if not entries:
        assert line.recall_bounds is None
        assert line.eps_or_ef_values() == []
        assert line.params() == []
        return

    assert line.recall_bounds is not None
    assert line.recall_bounds[0] == pytest.approx(entries[0][3])
    assert line.recall_bounds[1] == pytest.approx(entries[-1][3])

    for i, entry in enumerate(entries):
        assert len(entry) == 6
        tgt, param, rf, meas, lat_us, qps = entry
        assert isinstance(tgt, float)
        assert isinstance(param, float)
        assert isinstance(rf, float)
        assert isinstance(meas, float)
        assert isinstance(lat_us, float)
        assert isinstance(qps, float)
        assert (10.0 <= param <= 500.0) or (0.001 - 1e-6 <= param <= 0.20 + 1e-6)
        assert rf >= 1.0 - 1e-6
        assert 0.0 <= meas <= 1.0
        assert lat_us > 0.0
        assert qps == pytest.approx(1e6 / lat_us, rel=1e-4)

        if i > 0:
            assert meas > entries[i - 1][3]
            assert lat_us > entries[i - 1][4]
            assert qps < entries[i - 1][5]

        looked_up = line.param_for(meas)
        assert looked_up is not None
        assert looked_up[0] == pytest.approx(param, abs=1e-6)
        assert looked_up[1] == pytest.approx(rf, abs=1e-6)

    for i in range(len(entries) - 1):
        r_a = entries[i][3]
        r_b = entries[i + 1][3]
        lo_q = r_a + 0.25 * (r_b - r_a)
        mid_q = 0.5 * (r_a + r_b)
        hi_q = r_a + 0.75 * (r_b - r_a)
        assert line.param_for(lo_q) == pytest.approx((entries[i][1], entries[i][2]), abs=1e-6)
        assert line.param_for(mid_q) == pytest.approx((entries[i + 1][1], entries[i + 1][2]), abs=1e-6)
        assert line.param_for(hi_q) == pytest.approx((entries[i + 1][1], entries[i + 1][2]), abs=1e-6)

    params = line.eps_or_ef_values()
    assert len(params) <= len(entries)
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            assert abs(params[i] - params[j]) > 1e-6

    pairs = line.params()
    assert len(pairs) <= len(entries)
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            assert abs(pairs[i][0] - pairs[j][0]) > 1e-6 or abs(pairs[i][1] - pairs[j][1]) > 1e-6


def test_exports_and_wrapper_type():
    assert deglib.search.ParetoLine is ParetoLine
    assert deglib.search.ParetoLine is ParetoLine
    assert ParetoLine.__module__ == "deglib.search"
    assert not hasattr(ParetoLine, "_from_cpp")
    for mod in (deglib, deglib.search, deglib_cpp, deglib_cpp.search):
        assert not hasattr(mod, "ParetoEntry")
        assert not hasattr(mod, "RecallTableMode")
        assert not hasattr(mod, "Mode")


def test_to_pareto_line_with_and_without_recall_tables():
    data = _make_clustered_data(450, 16, clusters=22, seed=19)
    graph = deglib.build(
        data,
        edges_per_vertex=16,
        metric=Metric.FP32_L2,
        optimization_target=OptimizationTarget.LowLID,
        extend_k=32,
        optimize=False,
        threads=2,
    )

    # 1a. Single precomputed RecallTable (Ef-only) via graph.to_pareto_line
    ef_table = graph.to_recall_table(
        k=10, mode="ef", sample_count=60, recall_range=(0.80, 0.99), recall_step=0.05, threads=1
    )
    line_ef_only = graph.to_pareto_line(recall_tables=ef_table)
    _verify_pareto_invariants(line_ef_only)
    assert len(line_ef_only.recall_tables) == 1
    assert line_ef_only.recall_tables[0].is_ef
    assert all(param >= 1.0 and rf == pytest.approx(1.0) for _, param, rf, _, _, _ in line_ef_only.entries)

    # 1b. Two precomputed RecallTables via graph.to_pareto_line
    tables = graph.to_recall_tables(k=10, sample_count=60, recall_range=(0.80, 0.99), recall_step=0.05, threads=1)
    line_from_tables = graph.to_pareto_line(recall_tables=tables)
    _verify_pareto_invariants(line_from_tables)
    assert line_from_tables.k == 10
    assert line_from_tables.sample_count == 60
    assert len(line_from_tables.recall_tables) == 2
    assert len(line_from_tables.recall_tables[0].entries) == len(tables[0].entries)
    for a, b in zip(line_from_tables.recall_tables[0].entries, tables[0].entries):
        assert a == pytest.approx(b)
    assert len(line_from_tables.recall_tables[1].entries) == len(tables[1].entries)
    for a, b in zip(line_from_tables.recall_tables[1].entries, tables[1].entries):
        assert a == pytest.approx(b)

    # 2. Direct calculation (recall_tables omitted) via range and explicit target_recalls
    line_direct = graph.to_pareto_line(k=10, sample_count=60, recall_range=(0.80, 0.99), recall_step=0.05, threads=1)
    _verify_pareto_invariants(line_direct)
    assert len(line_direct.recall_tables) == 2
    assert all(t.k == 10 for t in line_direct.recall_tables)

    line_explicit = graph.to_pareto_line(k=10, sample_count=60, target_recalls=[0.95, 0.85, 0.90], threads=1, rounds=2)
    _verify_pareto_invariants(line_explicit)
    assert len(line_explicit) >= 1
    assert "ParetoLine(" in str(line_explicit)
    assert str(line_explicit) == repr(line_explicit)

    # param_for clamping / bounds behavior
    r_min, r_max = line_direct.recall_bounds
    if r_min > 0.10:
        assert line_direct.param_for(0.05, clamp=False) is None
        clamped_lo = line_direct.param_for(0.05, clamp=True)
        assert clamped_lo is not None
        assert clamped_lo[0] == pytest.approx(line_direct.entries[0][1])
        assert clamped_lo[1] == pytest.approx(line_direct.entries[0][2])
    if r_max < 0.995:
        assert line_direct.param_for(0.999, clamp=False) is None
        clamped_hi = line_direct.param_for(0.999, clamp=True)
        assert clamped_hi is not None
        assert clamped_hi[0] == pytest.approx(line_direct.entries[-1][1])
        assert clamped_hi[1] == pytest.approx(line_direct.entries[-1][2])


def test_pareto_line_quantized_graphs():
    rng = np.random.default_rng(29)

    l2_data = _make_clustered_data(350, 32, clusters=16, seed=29)
    l2_graph = deglib.build(l2_data, edges_per_vertex=16, metric=Metric.FP32_L2, extend_k=32, optimize=False, threads=2)
    u4_graph = l2_graph.with_metric(Metric.Uint4_L2)
    u4_line = u4_graph.to_pareto_line(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    _verify_pareto_invariants(u4_line)

    ip_raw = rng.normal(0.0, 1.0, size=(350, 32)).astype(np.float32)
    ip_raw /= np.linalg.norm(ip_raw, axis=1, keepdims=True)
    ip_graph = deglib.build(
        ip_raw,
        edges_per_vertex=16,
        metric=Metric.FP32_InnerProduct,
        optimization_target=OptimizationTarget.HighLID,
        extend_k=32,
        optimize=False,
        threads=2,
    )
    i8_graph = ip_graph.with_metric(Metric.Int8_InnerProduct)
    i8_line = i8_graph.to_pareto_line(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    _verify_pareto_invariants(i8_line)


def test_empty_graph_and_validation_errors():
    empty = deglib.graph.create_empty(16, 8, edges_per_vertex=4, metric=Metric.FP32_L2)
    empty_line = empty.to_pareto_line(k=10)
    assert len(empty_line) == 0
    assert empty_line.k == 10
    assert empty_line.sample_count == 0
    assert len(empty_line.recall_tables) == 2
    assert empty_line.recall_bounds is None
    assert empty_line.param_for(0.95, clamp=True) is None
    assert empty_line.eps_or_ef_values() == []
    assert "<empty>" in str(empty_line)

    data = _make_clustered_data(60, 8, clusters=6, seed=41)
    graph = deglib.build(data, edges_per_vertex=8, metric=Metric.FP32_L2, optimize=False, threads=1)
    tables = graph.to_recall_tables(k=10, sample_count=30, threads=1)
    other_k_table = graph.to_recall_table(k=5, mode="ef", sample_count=30, threads=1)

    empty_seq_line = graph.to_pareto_line(recall_tables=[])
    assert len(empty_seq_line) == 0

    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=[tables[0], other_k_table])
    with pytest.raises(ValueError):
        graph.to_pareto_line()
    with pytest.raises(ValueError):
        graph.to_pareto_line(k=10, recall_tables=tables)
    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=tables, recall_range=(0.8, 0.99))
    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=tables, recall_step=0.05)
    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=tables, target_recalls=[0.9])
    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=tables, sample_count=-1)
    with pytest.raises(ValueError):
        graph.to_pareto_line(recall_tables=tables, rounds=0)
    with pytest.raises(ValueError):
        graph.to_pareto_line(k=10, rounds=0)
    with pytest.raises(TypeError):
        graph.to_pareto_line(recall_tables="not a table")
    with pytest.raises(TypeError):
        graph.to_pareto_line(recall_tables=[123])
    with pytest.raises(TypeError):
        ParetoLine("not a cpp line")
