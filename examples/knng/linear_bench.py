import time
import multiprocessing as mp
import numpy as np
import deglib
from deglib.distances import FloatSpace, Metric
from dataset import ensure_small_dataset, load_hdf5_dataset

DEFAULT_K_TOP = 15
DEFAULT_EVP_K = 50
DEFAULT_NON_ZEROS = 700

def worker(chunk_indices, quant_vectors, dim, evp_k):
    space = FloatSpace.create(dim, Metric.EVP_InnerProduct)
    n = len(chunk_indices)
    out = np.empty((n, evp_k), dtype=np.uint32)
    for i, idx in enumerate(chunk_indices):
        dists = space.compute_distances(quant_vectors[idx], quant_vectors)
        dists[idx] = 1e9  # exclude self
        part = np.argpartition(dists, evp_k)[:evp_k]
        out[i] = part[np.argsort(dists[part])]
    return out

def run_evp_linear_benchmark(num_threads: int = 8, max_queries: int | None = None):
    dataset_path = ensure_small_dataset()
    print(f"Loading small dataset from {dataset_path}...")
    train_vectors, ground_truth = load_hdf5_dataset(dataset_path)
    n_vecs, dims = train_vectors.shape
    n_queries = n_vecs if max_queries is None else min(max_queries, n_vecs)
    print(f"Dataset: {n_vecs} vectors, {dims} dims. Running EVP linear search for {n_queries} queries (threads={num_threads}).")

    # 1. Quantization Phase
    t0 = time.perf_counter()
    quant_vectors = deglib.optimization.EvpQuantizer(DEFAULT_NON_ZEROS).quantize(train_vectors, num_threads=num_threads)
    t_quant = time.perf_counter() - t0
    print(f"Quantization completed in {t_quant:.3f}s")

    # 2. Pure EVP Linear Scan Phase (EVP_InnerProduct top 50)
    t0 = time.perf_counter()
    query_indices = np.arange(n_queries, dtype=np.uint32)
    chunks = np.array_split(query_indices, num_threads)
    with mp.Pool(num_threads) as pool:
        chunk_results = pool.starmap(worker, [(chunk, quant_vectors, dims, DEFAULT_EVP_K) for chunk in chunks])
    candidate_indices = np.vstack(chunk_results)
    t_search = time.perf_counter() - t0
    print(f"EVP Linear Search ({n_queries} queries) completed in {t_search:.3f}s ({(t_search/n_queries)*1000:.2f}ms/query)")

    # 3. Pure EVP Recall@15 Evaluation (Direct EVP top 15, NO reranking!)
    correct_pure_15 = 0
    correct_cand_50 = 0
    total_gt = 0
    for i in range(n_queries):
        gt_row = [idx for idx in ground_truth[i] if idx != i][:DEFAULT_K_TOP]
        gt_set = set(gt_row)
        evp_top15 = set(candidate_indices[i][:DEFAULT_K_TOP])
        evp_top50 = set(candidate_indices[i])
        correct_pure_15 += len(gt_set.intersection(evp_top15))
        correct_cand_50 += len(gt_set.intersection(evp_top50))
        total_gt += len(gt_set)

    recall_pure_15 = correct_pure_15 / max(total_gt, 1)
    recall_cand_50 = correct_cand_50 / max(total_gt, 1)

    # 4. FP16 Candidate Rerank Phase
    t0 = time.perf_counter()
    rerank_space = FloatSpace.create(dims, Metric.FP16_InnerProduct)
    final_knng_edges = deglib.search.rerank(
        space=rerank_space,
        queries=train_vectors[:n_queries],
        candidate_indices=candidate_indices,
        base_vectors=train_vectors,
        k_top=DEFAULT_K_TOP,
        num_threads=num_threads,
        return_distances=False,
        unsorted=True,
    )
    t_rerank = time.perf_counter() - t0
    print(f"FP16 Reranking completed in {t_rerank:.3f}s")

    correct_rerank_15 = 0
    for i in range(n_queries):
        gt_row = [idx for idx in ground_truth[i] if idx != i][:DEFAULT_K_TOP]
        gt_set = set(gt_row)
        pred_set = set(final_knng_edges[i])
        correct_rerank_15 += len(gt_set.intersection(pred_set))

    recall_rerank_15 = correct_rerank_15 / max(total_gt, 1)

    t_total = t_quant + t_search + t_rerank

    print("\n==================================================")
    print(f"---> 1. PURE EVP Recall@15 (NO Reranking):         {recall_pure_15 * 100.0:.2f}%")
    print(f"---> 2. EVP Top-50 Candidate Recall@15:           {recall_cand_50 * 100.0:.2f}%")
    print(f"---> 3. EVP + FP16 Rerank Recall@15:              {recall_rerank_15 * 100.0:.2f}%")
    print("==================================================")
    print("\n--- Timing Summary ---")
    print(f"Quantization Time:   {t_quant:.3f}s")
    print(f"Linear Search Time:  {t_search:.3f}s")
    print(f"Rerank Time:         {t_rerank:.3f}s")
    print(f"Total Time:          {t_total:.3f}s")

    return {
        "quant_time": t_quant,
        "search_time": t_search,
        "rerank_time": t_rerank,
        "total_time": t_total,
        "recall_pure_15": recall_pure_15,
        "recall_cand_50": recall_cand_50,
        "recall_rerank_15": recall_rerank_15,
    }

if __name__ == "__main__":
    mp.freeze_support()
    run_evp_linear_benchmark(num_threads=8)
