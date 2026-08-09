import pytest
import numpy as np

import deglib
import deglib.analysis


def _build_test_graph(samples=50, dims=16, edges_per_vertex=10):
    """Build a graph with external labels that differ from internal indices."""
    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    labels = np.arange(samples, dtype=np.uint32) * 10
    graph = deglib.builder.build_from_data(
        data, labels=labels, edges_per_vertex=edges_per_vertex,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=10, extend_eps=0.001
    )
    return graph, data


def test_calc_search_reachability():
    """Test that calc_search_reachability returns a valid ratio."""
    graph, _ = _build_test_graph()
    result = deglib.analysis.calc_search_reachability(graph)
    assert 0.0 <= result <= 1.0


def test_calc_exploration_reach():
    """Test that calc_exploration_reach returns a valid ratio."""
    graph, _ = _build_test_graph()
    result = deglib.analysis.calc_exploration_reach(graph)
    assert 0.0 <= result <= 1.0


def test_analyze_graph_returns_dict():
    """Test that analyze_graph returns a dictionary with all expected keys."""
    graph, _ = _build_test_graph()
    stats = deglib.analysis.analyze_graph(graph)

    expected_keys = {
        "vertex_count", "edge_count", "feature_dims", "edges_per_vertex",
        "avg_out_degree", "min_out_degree", "max_out_degree",
        "avg_in_degree", "min_in_degree", "max_in_degree",
        "source_vertices", "search_reachability", "exploration_reachability",
        "memory_bytes"
    }
    assert set(stats.keys()) == expected_keys

    assert stats["vertex_count"] == graph.size()
    assert stats["feature_dims"] == 16
    assert stats["edges_per_vertex"] == 10
    assert stats["edge_count"] > 0
    assert 0.0 <= stats["search_reachability"] <= 1.0
    assert 0.0 <= stats["exploration_reachability"] <= 1.0
    assert stats["memory_bytes"] > 0


def test_analyze_graph_with_readonly_graph():
    """Test that analyze_graph works with ReadOnlyGraph."""
    graph, _ = _build_test_graph()
    readonly_graph = graph.to_readonly()
    stats = deglib.analysis.analyze_graph(readonly_graph)
    assert stats["vertex_count"] == readonly_graph.size()


def test_analyze_graph_values_consistent():
    """Test that analyze_graph values are internally consistent."""
    graph, _ = _build_test_graph()
    stats = deglib.analysis.analyze_graph(graph)

    # avg_out_degree should be edge_count / vertex_count
    expected_avg_out = stats["edge_count"] / max(stats["vertex_count"], 1)
    assert np.isclose(stats["avg_out_degree"], expected_avg_out)

    # min_out_degree <= avg_out_degree <= max_out_degree
    assert stats["min_out_degree"] <= stats["avg_out_degree"]
    assert stats["avg_out_degree"] <= stats["max_out_degree"]

    # min_in_degree <= avg_in_degree <= max_in_degree
    assert stats["min_in_degree"] <= stats["avg_in_degree"]
    assert stats["avg_in_degree"] <= stats["max_in_degree"]

    # total in-degree should equal total edges
    # avg_in_degree * vertex_count should equal edge_count
    assert np.isclose(stats["avg_in_degree"] * stats["vertex_count"], stats["edge_count"])
