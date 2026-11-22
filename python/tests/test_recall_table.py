import numpy as np
import pytest

import deglib
import deglib_cpp
from deglib import (
    Metric,
    OptimizationTarget,
    RecallTable,
)


def _make_clustered_data(n: int, dim: int, clusters: int = 20, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(clusters, dim)).astype(np.float32)
    idx = np.arange(n) % clusters
    noise = rng.normal(0.0, 0.35, size=(n, dim)).astype(np.float32)
    return np.ascontiguousarray(centers[idx] + noise, dtype=np.float32)


def test_top_level_exports():
    assert deglib.RecallTable is RecallTable
    assert deglib.search.RecallTable is RecallTable
    assert RecallTable.__module__ == "deglib.search"
    for mod in (deglib, deglib.search, deglib_cpp, deglib_cpp.search):
        assert not hasattr(mod, "EfRecallEntry")
        assert not hasattr(mod, "EpsRecallEntry")
    assert not hasattr(deglib, "RecallTableMode")
    assert not hasattr(deglib.search, "RecallTableMode")

    with pytest.raises(TypeError):
        RecallTable("invalid")


def test_to_recall_table_defaults_and_step():
    data = _make_clustered_data(400, 16, clusters=20, seed=7)
    graph = deglib.build(
        data,
        edges_per_vertex=16,
        metric=Metric.FP32_L2,
        optimization_target=OptimizationTarget.LowLID,
        extend_k=32,
        optimize=False,
        threads=2,
    )

    table = graph.to_recall_table(k=10, sample_count=60, threads=1)
    assert isinstance(table, RecallTable)
    assert table.k == 10
    assert table.sample_count == 60
    assert table.has_ef
    assert table.has_eps
    assert table.ef_recall_bounds is not None
    assert table.eps_recall_bounds is not None
    assert table.ef_recall_bounds[0] <= table.ef_recall_bounds[1]
    assert table.eps_recall_bounds[0] <= table.eps_recall_bounds[1]

    for target, ef, measured in table.ef_entries:
        assert isinstance(target, float)
        assert isinstance(ef, int)
        assert isinstance(measured, float)
        assert 10 <= ef <= 300
        assert 0.0 <= measured <= 1.0
        assert measured >= target - 1e-5

    for target, eps, measured in table.eps_entries:
        assert isinstance(target, float)
        assert isinstance(eps, float)
        assert isinstance(measured, float)
        assert 0.01 - 1e-6 <= eps <= 0.20 + 1e-6
        assert 0.0 <= measured <= 1.0
        assert measured >= target - 1e-5

    efs = table.ef_values()
    epss = table.eps_values()
    combined = table.eps_or_ef_values()
    assert efs == sorted(set(efs))
    assert len(epss) == len(table.eps_values())
    assert epss == sorted(epss)
    assert combined == pytest.approx([float(v) for v in efs] + epss)

    # Verify default 0.5% recall_step granularity over achievable span
    t_half_pct = graph.to_recall_table(k=10, sample_count=60, recall_range=(0.80, 0.95), threads=1)
    if t_half_pct.ef_recall_bounds[0] <= 0.85 and t_half_pct.ef_recall_bounds[1] >= 0.92:
        diffs = [
            round(t_half_pct.ef_entries[i][0] - t_half_pct.ef_entries[i - 1][0], 4)
            for i in range(1, len(t_half_pct.ef_entries))
        ]
        assert any(abs(d - 0.005) <= 1e-4 for d in diffs)

    assert "RecallTable(" in str(table)
    assert str(table) == repr(table)


def test_range_step_and_explicit_target_recalls():
    data = _make_clustered_data(450, 16, clusters=22, seed=19)
    graph = deglib.build(
        data,
        edges_per_vertex=16,
        metric=Metric.FP32_L2,
        extend_k=32,
        optimize=False,
        threads=2,
    )

    t_range = graph.to_recall_table(
        k=10,
        sample_count=60,
        recall_range=(0.80, 0.99),
        recall_step=0.05,
        threads=1,
    )
    assert t_range.has_ef and t_range.has_eps

    targets_list = [0.95, 0.85, 0.95, 0.90]
    t_list = graph.to_recall_table(k=10, sample_count=60, target_recalls=targets_list, threads=1)
    t_np = graph.to_recall_table(
        k=10, sample_count=60, target_recalls=np.array(targets_list, dtype=np.float32), threads=1
    )
    assert [ef for _, ef, _ in t_list.ef_entries] == [ef for _, ef, _ in t_np.ef_entries]
    assert [eps for _, eps, _ in t_list.eps_entries] == pytest.approx([eps for _, eps, _ in t_np.eps_entries])


def test_quantized_graphs_uint4_and_int8_ip():
    rng = np.random.default_rng(29)

    l2_data = _make_clustered_data(350, 32, clusters=16, seed=29)
    l2_graph = deglib.build(l2_data, edges_per_vertex=16, metric=Metric.FP32_L2, extend_k=32, optimize=False, threads=2)
    u4_graph = l2_graph.to_readonly(metric=Metric.Uint4_L2, optimize=False)
    u4_table = u4_graph.to_recall_table(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    assert u4_table.has_ef
    assert u4_table.has_eps
    assert len(u4_table.ef_entries) >= 1
    assert len(u4_table.eps_entries) >= 1

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
    i8_graph = ip_graph.to_readonly(metric=Metric.Int8_InnerProduct, optimize=False)
    i8_table = i8_graph.to_recall_table(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    assert i8_table.has_ef
    assert i8_table.has_eps
    assert len(i8_table.ef_entries) >= 1
    assert len(i8_table.eps_entries) >= 1


def test_unreachable_targets_bounds_and_lookup_clamping():
    rng = np.random.default_rng(37)
    data = rng.normal(0.0, 1.0, size=(1500, 32)).astype(np.float32)
    raw = deglib.create_random_graph(data, edges_per_vertex=6, metric=Metric.FP32_L2, seed=37, optimize=False)

    table = raw.to_recall_table(
        k=30,
        sample_count=50,
        target_recalls=[0.95, 0.98, 0.999],
        threads=1,
    )
    assert table.ef_recall_bounds is not None
    assert table.eps_recall_bounds is not None
    r_min, r_max = table.ef_recall_bounds
    assert r_max < 0.94
    assert len(table.ef_entries) == 1
    target, ef, measured = table.ef_entries[0]
    assert target == pytest.approx(r_max)
    assert measured == pytest.approx(r_max)

    assert table.ef_for(0.999, clamp=False) is None
    assert table.ef_for(0.999, clamp=True) == ef

    if r_min > 0.10:
        assert table.ef_for(0.05, clamp=False) is None
        assert table.ef_for(0.05, clamp=True) == 10


def test_empty_graph_and_validation_errors():
    empty = deglib.create_empty(16, 8, edges_per_vertex=4, metric=Metric.FP32_L2)
    empty_table = empty.to_recall_table(k=10)
    assert empty_table.k == 10
    assert empty_table.sample_count == 0
    assert not empty_table.has_ef
    assert not empty_table.has_eps
    assert empty_table.ef_for(0.95, clamp=True) is None
    assert "<empty>" in str(empty_table)

    data = _make_clustered_data(50, 8, clusters=5, seed=41)
    graph = deglib.build(data, edges_per_vertex=8, metric=Metric.FP32_L2, optimize=False, threads=1)

    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, target_recalls=[0.9, 0.95], recall_range=(0.9, 0.99))
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, target_recalls=[0.9, 0.95], recall_step=0.05)
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, target_recalls=np.array([[0.9, 0.95]]))
    with pytest.raises(ValueError):
        graph.to_recall_table(k=0)
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, sample_count=0)
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, recall_range=(0.9,))
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, recall_range=(0.95, 0.90))
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, recall_step=0.0)
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, target_recalls=[])
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, target_recalls=[1.2])
