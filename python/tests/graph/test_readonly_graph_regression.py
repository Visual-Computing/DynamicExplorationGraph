import time
import numpy as np
import pytest

import deglib
from deglib.distances import FloatSpace, Metric


def deglib_prng_next(state: int) -> tuple[int, int]:
    """Pure 32-bit integer xorshift PRNG, identical to C++ deglib_prng_next."""
    x = state & 0xFFFFFFFF
    x = (x ^ ((x << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ (x >> 17)) & 0xFFFFFFFF
    x = (x ^ ((x << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return x, x


def generate_synthetic_clustered_dataset(count: int, dim: int, query_count: int, num_clusters: int = 100):
    """Fast synthetic dataset generation for Python regression tests."""
    rng = np.random.default_rng(42)
    centroids = rng.integers(-1000, 1001, size=(num_clusters, dim), dtype=np.int32).astype(np.float32)

    cluster_indices = np.arange(count) % num_clusters
    base_noise = rng.integers(-100, 101, size=(count, dim), dtype=np.int32).astype(np.float32)
    base = centroids[cluster_indices] + base_noise

    query_cluster_indices = np.arange(query_count) % num_clusters
    query_noise = rng.integers(-100, 101, size=(query_count, dim), dtype=np.int32).astype(np.float32)
    query = centroids[query_cluster_indices] + query_noise

    return base.astype(np.float32), query.astype(np.float32)


def compute_groundtruth_l2(base: np.ndarray, query: np.ndarray, k: int):
    """Compute exact brute-force L2 groundtruth for top-k neighbors."""
    query_count = query.shape[0]
    gt = []
    for q in range(query_count):
        dists = np.sum(np.square(base - query[q]), axis=1)
        top_k = np.argsort(dists)[:k]
        gt.append(set(top_k))
    return gt


def compute_groundtruth_innerproduct(base: np.ndarray, query: np.ndarray, k: int):
    """Compute exact brute-force InnerProduct groundtruth for top-k neighbors (distance = 1 - dot_product)."""
    query_count = query.shape[0]
    gt = []
    for q in range(query_count):
        dists = 1.0 - np.dot(base, query[q])
        top_k = np.argsort(dists)[:k]
        gt.append(set(top_k))
    return gt


def test_readonly_graph_regression_fp32_l2():
    """
    ReadOnlyGraph Search & Explore Regression Benchmark (FP32_L2, 100x averaged)
    Mirrors ReadOnlyGraphRegression.SearchAndExplore_FP32_L2 in test_readonly_graph_regression.cpp
    """
    dim = 1024
    base_count = 10000
    query_count = 100
    num_clusters = 100
    edges_per_vertex = 16
    extend_eps = 0.1
    search_eps = 0.001
    search_k = 100
    benchmark_runs = 100
    explore_max_calcs = 100

    base_data, query_data = generate_synthetic_clustered_dataset(base_count, dim, query_count, num_clusters)

    explore_data = np.zeros((query_count, dim), dtype=np.float32)
    for i in range(query_count):
        entry_node = i % base_count
        explore_data[i] = base_data[entry_node]

    search_gt_data = compute_groundtruth_l2(base_data, query_data, search_k)
    explore_gt_data = compute_groundtruth_l2(base_data, explore_data, search_k)

    # Build DynamicExplorationGraph
    mutable_graph = deglib.create_empty(base_count, FloatSpace.create(dim, Metric.FP32_L2), edges_per_vertex)

    builder = deglib.GraphBuilder(
        mutable_graph,
        seed=42,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=edges_per_vertex,
        extend_eps=extend_eps,
        improve_k=0,
        improve_eps=0.0,
    )
    builder.set_thread_count(1)
    builder.add_entry(range(base_count), base_data)
    builder.build()

    # Convert to ReadOnlyGraph
    graph = mutable_graph.to_readonly()

    # 1. Search Benchmark
    last_search_results = None
    t_start_search = time.perf_counter()

    for run in range(benchmark_runs):
        indices, _ = graph.search(query_data, eps=search_eps, k=search_k, threads=1)
        if run == 0:
            last_search_results = indices

    t_end_search = time.perf_counter()
    total_search_ms = (t_end_search - t_start_search) * 1000.0

    # Ground truth & recall calculation for search
    correct_search = 0
    total_gt = 0
    for q in range(query_count):
        gt_set = search_gt_data[q]
        total_gt += len(gt_set)
        for found_idx in last_search_results[q]:
            if found_idx in gt_set:
                correct_search += 1

    total_queries = query_count * benchmark_runs
    search_qps = (total_queries / total_search_ms) * 1000.0
    search_recall = (correct_search / total_gt) * 100.0

    print(
        f"\n[BENCHMARK 100x] ReadOnlyGraph FP32_L2 search(): "
        f"{total_search_ms:.3f} ms total for {total_queries} queries ("
        f"{(total_search_ms / total_queries):.4f} ms/q), "
        f"{search_qps:.2f} QPS, recall={search_recall:.2f}%"
    )

    # 2. Explore Benchmark
    last_explore_results = None
    t_start_explore = time.perf_counter()

    for run in range(benchmark_runs):
        entry_indices = np.array([(run * 13 + i) % base_count for i in range(query_count)], dtype=np.uint32)
        indices, _ = graph.explore(
            entry_indices, k=search_k, include_entry=True, max_distance_computation_count=explore_max_calcs, threads=1
        )
        if run == 0:
            last_explore_results = indices

    t_end_explore = time.perf_counter()
    total_explore_ms = (t_end_explore - t_start_explore) * 1000.0

    # Ground truth & recall calculation for explore
    correct_explore = 0
    total_explore_gt = 0
    for i in range(query_count):
        gt_set = explore_gt_data[i]
        total_explore_gt += len(gt_set)
        for found_idx in last_explore_results[i]:
            if found_idx in gt_set:
                correct_explore += 1

    total_explorations = query_count * benchmark_runs
    explore_qps = (total_explorations / total_explore_ms) * 1000.0
    explore_recall = (correct_explore / total_explore_gt) * 100.0

    print(
        f"[BENCHMARK 100x] ReadOnlyGraph FP32_L2 explore(): "
        f"{total_explore_ms:.3f} ms total for {total_explorations} explorations ("
        f"{(total_explore_ms / total_explorations):.4f} ms/q), "
        f"{explore_qps:.2f} QPS, recall={explore_recall:.2f}%"
    )

    assert search_recall > 0.0
    assert explore_recall > 0.0


def test_readonly_graph_regression_fp32_inner_product():
    """
    ReadOnlyGraph Search & Explore Regression Benchmark (FP32_InnerProduct, 100x averaged)
    Mirrors ReadOnlyGraphRegression.SearchAndExplore_FP32_InnerProduct in test_readonly_graph_regression.cpp
    """
    dim = 1024
    base_count = 10000
    query_count = 100
    num_clusters = 100
    edges_per_vertex = 16
    extend_eps = 0.1
    search_eps = 0.001
    search_k = 100
    benchmark_runs = 100
    explore_max_calcs = 100

    base_data, query_data = generate_synthetic_clustered_dataset(base_count, dim, query_count, num_clusters)

    explore_data = np.zeros((query_count, dim), dtype=np.float32)
    for i in range(query_count):
        entry_node = i % base_count
        explore_data[i] = base_data[entry_node]

    search_gt_data = compute_groundtruth_innerproduct(base_data, query_data, search_k)
    explore_gt_data = compute_groundtruth_innerproduct(base_data, explore_data, search_k)

    # Build DynamicExplorationGraph
    mutable_graph = deglib.create_empty(base_count, FloatSpace.create(dim, Metric.FP32_InnerProduct), edges_per_vertex)

    builder = deglib.GraphBuilder(
        mutable_graph,
        seed=42,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=edges_per_vertex,
        extend_eps=extend_eps,
        improve_k=0,
        improve_eps=0.0,
    )
    builder.set_thread_count(1)
    builder.add_entry(range(base_count), base_data)
    builder.build()

    # Convert to ReadOnlyGraph
    graph = mutable_graph.to_readonly()

    # 1. Search Benchmark
    last_search_results = None
    t_start_search = time.perf_counter()

    for run in range(benchmark_runs):
        indices, _ = graph.search(query_data, eps=search_eps, k=search_k, threads=1)
        if run == 0:
            last_search_results = indices

    t_end_search = time.perf_counter()
    total_search_ms = (t_end_search - t_start_search) * 1000.0

    # Ground truth & recall calculation for search
    correct_search = 0
    total_gt = 0
    for q in range(query_count):
        gt_set = search_gt_data[q]
        total_gt += len(gt_set)
        for found_idx in last_search_results[q]:
            if found_idx in gt_set:
                correct_search += 1

    total_queries = query_count * benchmark_runs
    search_qps = (total_queries / total_search_ms) * 1000.0
    search_recall = (correct_search / total_gt) * 100.0

    print(
        f"\n[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct search(): "
        f"{total_search_ms:.3f} ms total for {total_queries} queries ("
        f"{(total_search_ms / total_queries):.4f} ms/q), "
        f"{search_qps:.2f} QPS, recall={search_recall:.2f}%"
    )

    # 2. Explore Benchmark
    last_explore_results = None
    t_start_explore = time.perf_counter()

    for run in range(benchmark_runs):
        entry_indices = np.array([(run * 13 + i) % base_count for i in range(query_count)], dtype=np.uint32)
        indices, _ = graph.explore(
            entry_indices, k=search_k, include_entry=True, max_distance_computation_count=explore_max_calcs, threads=1
        )
        if run == 0:
            last_explore_results = indices

    t_end_explore = time.perf_counter()
    total_explore_ms = (t_end_explore - t_start_explore) * 1000.0

    # Ground truth & recall calculation for explore
    correct_explore = 0
    total_explore_gt = 0
    for i in range(query_count):
        gt_set = explore_gt_data[i]
        total_explore_gt += len(gt_set)
        for found_idx in last_explore_results[i]:
            if found_idx in gt_set:
                correct_explore += 1

    total_explorations = query_count * benchmark_runs
    explore_qps = (total_explorations / total_explore_ms) * 1000.0
    explore_recall = (correct_explore / total_explore_gt) * 100.0

    print(
        f"[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct explore(): "
        f"{total_explore_ms:.3f} ms total for {total_explorations} explorations ("
        f"{(total_explore_ms / total_explorations):.4f} ms/q), "
        f"{explore_qps:.2f} QPS, recall={explore_recall:.2f}%"
    )

    assert search_recall > 0.0
    assert explore_recall > 0.0


def test_readonly_graph_regression_evp_inner_product():
    """
    ReadOnlyGraph Search & Explore Regression Benchmark (EVP_InnerProduct, 100x averaged)
    Mirrors ReadOnlyGraphRegression.SearchAndExplore_EVP_InnerProduct in test_readonly_graph_regression.cpp
    """
    dim = 1024
    base_count = 10000
    query_count = 100
    num_clusters = 100
    edges_per_vertex = 16
    extend_eps = 0.1
    search_eps = 0.001
    search_k = 100
    benchmark_runs = 100
    explore_max_calcs = 100
    non_zeros = 512

    base_data, query_data = generate_synthetic_clustered_dataset(base_count, dim, query_count, num_clusters)

    explore_data = np.zeros((query_count, dim), dtype=np.float32)
    for i in range(query_count):
        entry_node = i % base_count
        explore_data[i] = base_data[entry_node]

    # Compute groundtruth using original float data (InnerProduct)
    search_gt_data = compute_groundtruth_innerproduct(base_data, query_data, search_k)
    explore_gt_data = compute_groundtruth_innerproduct(base_data, explore_data, search_k)

    # Quantize data to EVP using deglib.optimization.quantize_batch
    base_quant = deglib.optimization.quantize_batch(base_data, non_zeros=non_zeros, num_threads=8)
    query_quant = deglib.optimization.quantize_batch(query_data, non_zeros=non_zeros, num_threads=8)

    # Build DynamicExplorationGraph with EVP_InnerProduct metric
    mutable_graph = deglib.create_empty(base_count, FloatSpace.create(dim, Metric.EVP_InnerProduct), edges_per_vertex)

    builder = deglib.GraphBuilder(
        mutable_graph,
        seed=42,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=edges_per_vertex,
        extend_eps=extend_eps,
        improve_k=0,
        improve_eps=0.0,
    )
    builder.set_thread_count(1)
    builder.add_entry(range(base_count), base_quant)
    builder.build()

    # Convert to ReadOnlyGraph
    graph = mutable_graph.to_readonly()

    # 1. Search Benchmark
    last_search_results = None
    t_start_search = time.perf_counter()

    for run in range(benchmark_runs):
        indices, _ = graph.search(query_quant, eps=search_eps, k=search_k, threads=1)
        if run == 0:
            last_search_results = indices

    t_end_search = time.perf_counter()
    total_search_ms = (t_end_search - t_start_search) * 1000.0

    # Ground truth & recall calculation for search using original float GT
    correct_search = 0
    total_gt = 0
    for q in range(query_count):
        gt_set = search_gt_data[q]
        total_gt += len(gt_set)
        for found_idx in last_search_results[q]:
            if found_idx in gt_set:
                correct_search += 1

    total_queries = query_count * benchmark_runs
    search_qps = (total_queries / total_search_ms) * 1000.0
    search_recall = (correct_search / total_gt) * 100.0

    print(
        f"\n[BENCHMARK 100x] ReadOnlyGraph EVP_InnerProduct search(): "
        f"{total_search_ms:.3f} ms total for {total_queries} queries ("
        f"{(total_search_ms / total_queries):.4f} ms/q), "
        f"{search_qps:.2f} QPS, recall={search_recall:.2f}%"
    )

    # 2. Explore Benchmark
    last_explore_results = None
    t_start_explore = time.perf_counter()

    for run in range(benchmark_runs):
        entry_indices = np.array([(run * 13 + i) % base_count for i in range(query_count)], dtype=np.uint32)
        indices, _ = graph.explore(
            entry_indices, k=search_k, include_entry=True, max_distance_computation_count=explore_max_calcs, threads=1
        )
        if run == 0:
            last_explore_results = indices

    t_end_explore = time.perf_counter()
    total_explore_ms = (t_end_explore - t_start_explore) * 1000.0

    # Ground truth & recall calculation for explore using original float GT
    correct_explore = 0
    total_explore_gt = 0
    for i in range(query_count):
        gt_set = explore_gt_data[i]
        total_explore_gt += len(gt_set)
        for found_idx in last_explore_results[i]:
            if found_idx in gt_set:
                correct_explore += 1

    total_explorations = query_count * benchmark_runs
    explore_qps = (total_explorations / total_explore_ms) * 1000.0
    explore_recall = (correct_explore / total_explore_gt) * 100.0

    print(
        f"[BENCHMARK 100x] ReadOnlyGraph EVP_InnerProduct explore(): "
        f"{total_explore_ms:.3f} ms total for {total_explorations} explorations ("
        f"{(total_explore_ms / total_explorations):.4f} ms/q), "
        f"{explore_qps:.2f} QPS, recall={explore_recall:.2f}%"
    )

    assert search_recall > 0.0
    assert explore_recall > 0.0
