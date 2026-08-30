import numpy as np
import pytest
import deglib
from deglib import Metric, FloatSpace, create_searcher


def test_searcher_direct_f32():
    np.random.seed(42)
    dim = 16
    n = 100
    data = np.random.randn(n, dim).astype(np.float32)

    graph = deglib.builder.build_from_data(data, edges_per_vertex=16, metric=Metric.FP32_L2)
    searcher = create_searcher(graph)

    query = data[0]
    res = searcher.search(query, k=5, eps=0.1)
    assert len(res) == 5
    assert res[0] == 0

    # With return_distances
    indices, distances = searcher.search(query, k=5, eps=0.1, return_distances=True)
    assert len(indices) == 5
    assert len(distances) == 5
    assert indices[0] == 0
    assert distances[0] == pytest.approx(0.0, abs=1e-5)
    # Check ascending order
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1]

    # With unsorted
    indices_unsorted, distances_unsorted = searcher.search(query, k=5, eps=0.1, return_distances=True, unsorted=True)
    assert len(indices_unsorted) == 5
    assert set(indices_unsorted) == set(indices)

    # Unified search method with 1D, (1, dim), and (dim, 1) queries
    res_1xDim = searcher.search(query.reshape(1, -1), k=5, eps=0.1)
    assert res_1xDim.shape == (5,)
    assert res_1xDim[0] == 0

    res_Dimx1 = searcher.search(query.reshape(-1, 1), k=5, eps=0.1)
    assert res_Dimx1.shape == (5,)
    assert res_Dimx1[0] == 0

    # Unified search method with 2D batch queries
    res_batch_unified = searcher.search(data[:10], k=5, eps=0.1, threads=2)
    assert res_batch_unified.shape == (10, 5)
    assert res_batch_unified[0, 0] == 0

    indices_batch, dists_batch = searcher.search(data[:10], k=5, eps=0.1, threads=2, return_distances=True)
    assert indices_batch.shape == (10, 5)
    assert dists_batch.shape == (10, 5)
    assert indices_batch[0, 0] == 0
    assert dists_batch[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_searcher_quantized_int8_with_rerank():
    np.random.seed(42)
    dim = 32
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)

    # 1. Base graph in FP32
    graph = deglib.builder.build_from_data(data, edges_per_vertex=16, metric=Metric.FP32_L2)

    # 2. INT8 Quantization
    quantizer = deglib.optimization.make_scalar_quantizer_int8(data)
    int8_data = quantizer.quantize(data)
    target_space = FloatSpace.create(dim=dim, metric=Metric.Int8_L2)
    ro_graph = graph.to_readonly(target_space, int8_data)

    # 3. Refine features in FP16
    base_fp16 = deglib.distances.floats_to_fp16(data)
    rerank_space = FloatSpace.create(dim=dim, metric=Metric.FP16_L2)

    # 4. Create Searcher
    searcher = create_searcher(
        graph=ro_graph,
        quantizer=quantizer,
        refine_space=rerank_space,
        refine_data=base_fp16,
    )

    query = data[5]
    res = searcher.search(query, k=5, eps=0.2, rerank_factor=1.5)
    assert len(res) == 5
    assert res[0] == 5

    # With return_distances
    indices, distances = searcher.search(query, k=5, eps=0.2, rerank_factor=1.5, return_distances=True)
    assert indices[0] == 5
    assert distances[0] == pytest.approx(0.0, abs=1e-3)
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1]

    # Batch search
    res_batch = searcher.search(data[:5], k=5, eps=0.2, rerank_factor=1.5, threads=2)
    assert res_batch.shape == (5, 5)
    for i in range(5):
        assert res_batch[i, 0] == i


def test_searcher_quantized_uint8_with_rerank():
    np.random.seed(42)
    dim = 32
    n = 200
    data = np.random.uniform(0.0, 10.0, size=(n, dim)).astype(np.float32)

    graph = deglib.builder.build_from_data(data, edges_per_vertex=16, metric=Metric.FP32_L2)

    quantizer = deglib.optimization.make_scalar_quantizer_uint8(data)
    uint8_data = quantizer.quantize(data)
    target_space = FloatSpace.create(dim=dim, metric=Metric.Uint8_L2)
    ro_graph = graph.to_readonly(target_space, uint8_data)

    searcher = create_searcher(
        graph=ro_graph,
        quantizer=quantizer,
        refine_space=FloatSpace.create(dim=dim, metric=Metric.FP32_L2),
        refine_data=data,
    )

    query = data[12]
    res = searcher.search(query, k=5, eps=0.2, rerank_factor=1.5)
    assert len(res) == 5
    assert res[0] == 12

    indices, distances = searcher.search(query, k=5, eps=0.2, rerank_factor=1.5, return_distances=True)
    assert indices[0] == 12
    assert distances[0] == pytest.approx(0.0, abs=1e-5)


def test_searcher_evp_quantizer():
    np.random.seed(42)
    dim = 64
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)
    # L2 normalize for MIPS / InnerProduct
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    graph = deglib.builder.build_from_data(data, edges_per_vertex=16, metric=Metric.FP32_InnerProduct)

    # 16 non-zeros for EVP
    non_zeros = 16
    evp_data = deglib.optimization.quantize_batch(data, non_zeros=non_zeros)
    target_space = FloatSpace.create(dim=dim, metric=Metric.EVP_InnerProduct)
    ro_graph = graph.to_readonly(target_space, evp_data)

    searcher = create_searcher(
        graph=ro_graph,
        quantizer=non_zeros,
        refine_space=FloatSpace.create(dim=dim, metric=Metric.FP32_InnerProduct),
        refine_data=data,
    )

    query = data[3]
    res = searcher.search(query, k=3, eps=0.3, rerank_factor=2.0)
    assert len(res) == 3
    assert res[0] == 3

    # FP16 query
    query_fp16 = deglib.distances.floats_to_fp16(query)
    res_fp16, dists_fp16 = searcher.search(query_fp16, k=3, eps=0.3, rerank_factor=2.0, return_distances=True)
    assert len(res_fp16) == 3
    assert res_fp16[0] == 3
    assert dists_fp16[0] == pytest.approx(0.0, abs=1e-3)
