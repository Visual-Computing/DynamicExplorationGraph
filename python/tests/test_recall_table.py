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


def _verify_table_invariants(table: RecallTable) -> None:
    assert isinstance(table, RecallTable)
    assert table.is_ef != table.is_eps
    assert len(table) == len(table.entries)
    assert table.rerank_factor == pytest.approx(1.0)

    if len(table) == 0:
        return

    assert table.recall_bounds is not None
    assert table.recall_bounds[0] <= table.recall_bounds[1]

    for i, entry in enumerate(table.entries):
        assert len(entry) == 3
        target, param, measured = entry
        assert isinstance(target, float)
        assert isinstance(param, float)
        assert isinstance(measured, float)
        if table.is_ef:
            assert 10.0 <= param <= 500.0
            assert abs(param - round(param)) <= 1e-6
        else:
            assert 0.001 - 1e-6 <= param <= 0.20 + 1e-6
        assert 0.0 <= measured <= 1.0
        assert measured >= target - 1e-5
        if i > 0:
            assert target > table.entries[i - 1][0]

    vals = table.eps_or_ef_values()
    assert len(vals) >= 1
    for i in range(1, len(vals)):
        assert vals[i - 1] + 1e-6 < vals[i]


def test_top_level_exports():
    assert deglib.RecallTable is RecallTable
    assert deglib.search.RecallTable is RecallTable
    assert RecallTable.__module__ == "deglib.search"
    for mod in (deglib, deglib.search, deglib_cpp, deglib_cpp.search):
        assert not hasattr(mod, "EfRecallEntry")
        assert not hasattr(mod, "EpsRecallEntry")
        assert not hasattr(mod, "RecallTableMode")
        assert not hasattr(mod, "Mode")
    assert not hasattr(RecallTable, "_from_cpp")
    assert not hasattr(RecallTable, "Mode")
    assert not hasattr(RecallTable, "mode")
    assert not hasattr(deglib_cpp.search.RecallTable, "mode")

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

    ef_table = graph.to_recall_table(k=10, mode="ef", sample_count=60, threads=1)
    eps_table = graph.to_recall_table(k=10, mode="eps", sample_count=60, threads=1)
    assert ef_table.k == 10
    assert ef_table.sample_count == 60
    assert ef_table.is_ef and not ef_table.is_eps
    assert eps_table.is_eps and not eps_table.is_ef
    _verify_table_invariants(ef_table)
    _verify_table_invariants(eps_table)

    # to_recall_tables builds both modes in one pass matching single-mode calls
    tables = graph.to_recall_tables(k=10, sample_count=60, threads=1)
    assert len(tables) == 2
    assert tables[0].is_ef and tables[1].is_eps
    _verify_table_invariants(tables[0])
    _verify_table_invariants(tables[1])
    assert tables[0].entries == pytest.approx(ef_table.entries)
    assert tables[1].entries == pytest.approx(eps_table.entries)

    # Verify default 0.5% recall_step granularity over achievable span
    half_pct_tables = graph.to_recall_tables(k=10, sample_count=60, recall_range=(0.80, 0.95), threads=1)
    t_ef = half_pct_tables[0]
    if t_ef.recall_bounds[0] <= 0.85 and t_ef.recall_bounds[1] >= 0.92:
        diffs = [round(t_ef.entries[i][0] - t_ef.entries[i - 1][0], 4) for i in range(1, len(t_ef.entries))]
        assert any(abs(d - 0.005) <= 1e-4 for d in diffs)

    assert "RecallTable(mode=ef" in str(ef_table)
    assert "RecallTable(mode=eps" in str(eps_table)
    assert str(ef_table) == repr(ef_table)


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
        mode="ef",
        sample_count=60,
        recall_range=(0.80, 0.99),
        recall_step=0.05,
        threads=1,
    )
    _verify_table_invariants(t_range)

    targets_list = [0.95, 0.85, 0.95, 0.90]
    t_list = graph.to_recall_tables(k=10, sample_count=60, target_recalls=targets_list, threads=1)
    t_np = graph.to_recall_tables(
        k=10, sample_count=60, target_recalls=np.array(targets_list, dtype=np.float32), threads=1
    )
    assert len(t_list) == 2 and len(t_np) == 2
    assert [p for _, p, _ in t_list[0].entries] == pytest.approx([p for _, p, _ in t_np[0].entries])
    assert [p for _, p, _ in t_list[1].entries] == pytest.approx([p for _, p, _ in t_np[1].entries])


def test_quantized_graphs_uint4_and_int8_ip():
    rng = np.random.default_rng(29)

    l2_data = _make_clustered_data(350, 32, clusters=16, seed=29)
    l2_graph = deglib.build(l2_data, edges_per_vertex=16, metric=Metric.FP32_L2, extend_k=32, optimize=False, threads=2)
    u4_graph = l2_graph.to_readonly(metric=Metric.Uint4_L2, optimize=False)
    u4_tables = u4_graph.to_recall_tables(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    assert len(u4_tables) == 2
    _verify_table_invariants(u4_tables[0])
    _verify_table_invariants(u4_tables[1])
    assert len(u4_tables[0]) >= 1
    assert len(u4_tables[1]) >= 1

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
    i8_tables = i8_graph.to_recall_tables(k=10, sample_count=50, recall_range=(0.70, 0.99), recall_step=0.10, threads=1)
    assert len(i8_tables) == 2
    _verify_table_invariants(i8_tables[0])
    _verify_table_invariants(i8_tables[1])
    assert len(i8_tables[0]) >= 1
    assert len(i8_tables[1]) >= 1


def test_unreachable_targets_bounds_and_lookup_clamping():
    rng = np.random.default_rng(37)
    data = rng.normal(0.0, 1.0, size=(2500, 64)).astype(np.float32)
    raw = deglib.create_random_graph(data, edges_per_vertex=6, metric=Metric.FP32_L2, seed=37, optimize=False)

    table = raw.to_recall_table(
        k=30,
        mode="ef",
        sample_count=50,
        target_recalls=[0.95, 0.98, 0.999],
        threads=1,
    )
    assert table.recall_bounds is not None
    r_min, r_max = table.recall_bounds
    assert r_max < 0.94
    assert len(table.entries) == 1
    target, ef, measured = table.entries[0]
    assert target == pytest.approx(r_max)
    assert measured == pytest.approx(r_max)

    assert table.param_for(0.999, clamp=False) is None
    assert table.param_for(0.999, clamp=True) == pytest.approx(ef)

    if r_min > 0.10:
        assert table.param_for(0.05, clamp=False) is None
        assert table.param_for(0.05, clamp=True) == pytest.approx(10.0)


def test_empty_graph_and_validation_errors():
    empty = deglib.create_empty(16, 8, edges_per_vertex=4, metric=Metric.FP32_L2)
    empty_ef = empty.to_recall_table(k=10, mode="ef")
    assert empty_ef.k == 10
    assert empty_ef.sample_count == 0
    assert empty_ef.is_ef
    assert len(empty_ef) == 0
    assert empty_ef.recall_bounds is None
    assert empty_ef.param_for(0.95, clamp=True) is None
    assert "mode=ef" in str(empty_ef)
    assert "<empty>" in str(empty_ef)

    empty_both = empty.to_recall_tables(k=10)
    assert len(empty_both) == 2
    assert empty_both[0].is_ef and empty_both[1].is_eps
    assert "mode=eps" in str(empty_both[1])

    data = _make_clustered_data(50, 8, clusters=5, seed=41)
    graph = deglib.build(data, edges_per_vertex=8, metric=Metric.FP32_L2, optimize=False, threads=1)

    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, mode="invalid")
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, mode="ef", target_recalls=[0.9, 0.95], recall_range=(0.9, 0.99))
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, mode="ef", target_recalls=[0.9, 0.95], recall_step=0.05)
    with pytest.raises(ValueError):
        graph.to_recall_table(k=10, mode="ef", target_recalls=np.array([[0.9, 0.95]]))
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=0)
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, sample_count=0)
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, recall_range=(0.9,))
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, recall_range=(0.95, 0.90))
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, recall_step=0.0)
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, target_recalls=[])
    with pytest.raises(ValueError):
        graph.to_recall_tables(k=10, target_recalls=[1.2])
