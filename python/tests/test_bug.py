#!/usr/bin/env python3

import numpy as np
import time


def main():
    # Read .fvecs and .ivecs files

    # Load data
    sift_base = read_fvecs("./sift_subset_100k/sift_base.fvecs")              # base vectors
    sift_query = read_fvecs("./sift_subset_100k/sift_query.fvecs")            # queries
    sift_query_gt = read_ivecs("./sift_subset_100k/sift_groundtruth.ivecs")   # ground truth indices

    print(f"Base shape: {sift_base.shape}, Query shape: {sift_query.shape}")

    graph, recalls, qps = run_deg_ann(
        sift_base, sift_query, sift_query_gt,
        k=100,
        edges_per_vertex=16,            # Graph degree (d)
        extend_k=32,                    # Build: candidate pool size
        extend_eps=0.1,                 # Build: search radius parameter
        eps_list=[0.0001, 0.001, 0.01], # Try multiple search radius settings
    )


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
    return np.fromfile(filename, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
    return np.fromfile(filename, dtype=np.int32).reshape(-1, dim + 1)[:, 1:]


def recall_at_k(retrieved_indices, ground_truth, k=100):
    """
    Computes Recall@k for nearest neighbor search.

    Parameters:
        retrieved_indices: np.ndarray of shape (num_queries, k)
        ground_truth: np.ndarray of shape (num_queries, 1) – true nearest neighbor indices
        k: int – number of top predictions to check

    Returns:
        recall: float – fraction of queries where ground truth was in top-k
    """
    correct = 0
    for i in range(ground_truth.shape[0]):
        if ground_truth[i, 0] in retrieved_indices[i, :k]:
            correct += 1
    return correct / ground_truth.shape[0]


def run_deg_ann(
    base_vectors,
    query_vectors,
    ground_truth,
    k=100,
    edges_per_vertex=30,
    extend_k=60,
    extend_eps=0.1,
    eps_list=[0.001, 0.01, 0.05, 0.1],
):
    """
    Run DEG ANN search across multiple eps values.

    Parameters:
        - base_vectors: (N, d) base vectors
        - query_vectors: (Q, d) query vectors
        - ground_truth: (Q, k) ground truth indices
        - k: top-k neighbors to retrieve
        - edges_per_vertex: number of edges per vertex in the graph
        - extend_k: candidate pool size during graph build
        - extend_eps: effort during graph build
        - eps_list: list of eps values to benchmark
    Returns:
        - graph: the DEG search graph
        - recalls: list of recall@k values
        - qps_list: list of queries/sec values
    """

    # Ensure C-contiguous arrays
    base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    ground_truth = np.ascontiguousarray(ground_truth, dtype=np.int32)

    # Build graph
    t0 = time.time()

    from deglib.distances import Metric
    from deglib.graph import SizeBoundedGraph
    from deglib.builder import EvenRegularGraphBuilder, LID
    from deglib.std import Mt19937

    capacity = base_vectors.shape[0]
    graph = SizeBoundedGraph.create_empty(capacity, base_vectors.shape[1], edges_per_vertex, Metric.L2)
    builder = EvenRegularGraphBuilder(
        graph, None, lid=LID.Low, extend_k=extend_k, extend_eps=extend_eps, improve_k=0,
        improve_eps=0, max_path_length=0, swap_tries=0,
        additional_swap_tries=0
    )
    labels = np.arange(base_vectors.shape[0], dtype=np.uint32)
    builder.set_thread_count(1)
    builder.set_batch_size(1)
    builder.add_entry(labels, base_vectors)
    builder.build(callback="progress")

    # graph = deglib.builder.build_from_data(
    #     data=base_vectors,
    #     edges_per_vertex=edges_per_vertex,
    #     extend_k=extend_k,
    #     extend_eps=extend_eps,
    #     improve_k=0,
    #     lid=deglib.builder.LID.Low,
    #     callback="progress"
    # )
    t1 = time.time()
    build_time = t1 - t0

    recalls = []
    qps_list = []

    for eps in eps_list:
        # Vectorized search
        t0 = time.time()
        results, _ = graph.search(query=query_vectors, eps=eps, k=k, threads=2)
        t1 = time.time()
        search_time = t1 - t0
        qps = len(query_vectors) / search_time

        recall = recall_at_k(results, ground_truth, k)
        recalls.append(recall)
        qps_list.append(qps)

        print(
            f"[DEG] d={edges_per_vertex}, k_ext={extend_k}, eps_ext={extend_eps:.3f}, "
            f"eps={eps:.3f}, k={k:3d} | "
            f"Recall@{k:<3d}: {recall * 100:6.2f}% | "
            f"QPS: {qps:7.2f} | "
            f"Build: {build_time:5.2f}s | "
            f"Search: {search_time:6.2f}s"
        )

    return graph, recalls, qps_list


if __name__ == '__main__':
    main()

