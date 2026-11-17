import os
import struct
import tempfile

import numpy as np

import deglib
from deglib import (
    Metric,
    create_empty,
    create_searcher,
    load_graph,
    load_mutable_graph,
    load_readonly_graph,
    make_evp_quantizer,
    make_mips_transform,
    make_scalar_quantizer_fp16,
    make_scalar_quantizer_int8,
    make_scalar_quantizer_uint4,
    make_scalar_quantizer_uint8,
)
from deglib.transformation import apply_on_database


def test_scalar_int8_transform_serialization_and_transform():
    dim = 16
    abs_max = 2.0
    data = np.random.uniform(-abs_max, abs_max, size=(10, dim)).astype(np.float32)
    transform = make_scalar_quantizer_int8(data, quantile=1.0)
    transformations = [transform]

    assert len(transformations) == 1
    assert transform.input_dim == dim
    assert transform.output_dim == dim
    assert transform.output_dtype == np.int8

    scale = 127.0 / transform.abs_max
    out = apply_on_database(transformations, data)
    assert out.dtype == np.int8
    expected = np.clip(np.round(data * scale), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(out, expected)


def test_scalar_fp16_transform_serialization_and_transform():
    dim = 16
    # Audio-like large-range data exceeding FP16 max (65504) -> forces power-of-two scaling.
    data = np.random.uniform(-100000.0, 100000.0, size=(10, dim)).astype(np.float32)
    transform = make_scalar_quantizer_fp16(data, quantile=1.0)
    transformations = [transform]

    assert transform.input_dim == dim
    assert transform.output_dim == dim
    assert transform.output_dtype == np.float16
    assert transform.scale < 1.0

    out = apply_on_database(transformations, data)
    assert out.dtype == np.float16
    assert out.shape == (10, dim)
    assert not np.any(np.isinf(out))
    assert not np.any(np.isnan(out))

    # Roundtrip via dequantize approximates the scaled input within FP16 precision.
    deq = transform.dequantize(out)
    np.testing.assert_allclose(deq, data, rtol=2e-3, atol=1.0)


def test_scalar_uint8_transform_serialization_and_transform():
    dim = 16
    min_val = -3.0
    max_val = 3.0
    data = np.random.uniform(min_val, max_val, size=(10, dim)).astype(np.float32)
    transform = make_scalar_quantizer_uint8(data, quantile=1.0)
    transformations = [transform]

    assert transform.output_dtype == np.uint8

    scale = 255.0 / (transform.max_val - transform.min_val)
    out = apply_on_database(transformations, data)
    assert out.dtype == np.uint8
    expected = np.clip(np.round((data - transform.min_val) * scale), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_mips_transform():
    dim = 8
    max_norm = 10.0
    data = np.random.randn(5, dim).astype(np.float32)
    transform = make_mips_transform(data, max_norm=max_norm)
    transformations = [transform]

    assert transform.input_dim == dim
    assert transform.output_dim == dim + 1
    assert transform.output_dtype == np.float32
    out = apply_on_database(transformations, data)
    assert out.shape == (5, dim + 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, :dim], data)


def test_builder_and_graph_search_with_int8_transforms():
    np.random.seed(42)
    N, dim = 60, 16
    data = np.random.randn(N, dim).astype(np.float32)

    # Setup Int8 Quantizer
    sq = make_scalar_quantizer_int8(data, quantile=1.0)
    transformations = [sq]

    # Create empty graph configured with transformations
    graph = create_empty(N, dim, edges_per_vertex=16, metric=Metric.Int8_L2, transformations=transformations)

    assert graph.has_transformations()
    assert graph.get_transformations()[0].output_dtype == np.int8

    # Ingestion via GraphBuilder: pass raw float32 data!
    builder = deglib.GraphBuilder(graph, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    assert graph.size() == N

    # Transparent search using float32 query
    query = data[10]
    indices, dists = graph.search(query, eps_or_ef=0.2, k=5)
    assert len(indices) == 5
    # The exact vector itself should be nearest
    assert indices[0] == 10

    # Batch search with float32 queries
    batch_queries = data[:5]
    batch_indices, batch_dists = graph.search(batch_queries, eps_or_ef=0.2, k=5)
    assert batch_indices.shape == (5, 5)
    for i in range(5):
        assert batch_indices[i, 0] == i


def test_builder_and_graph_search_with_int8_inner_product_transform():
    """Int8 IP quantization must keep zero-centered values; nonnegative S7 codes distort IP ranking."""
    np.random.seed(123)
    N, dim = 200, 32
    data = np.random.randn(N, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    sq = make_scalar_quantizer_int8(data, quantile=1.0)
    transformations = [sq]
    graph = create_empty(N, dim, edges_per_vertex=16, metric=Metric.Int8_InnerProduct, transformations=transformations)

    builder = deglib.GraphBuilder(graph, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    indices, _ = graph.search(data[:20], eps_or_ef=64, k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(20))


def test_container_persistence_roundtrip():
    np.random.seed(123)
    N, dim = 50, 16
    data = np.random.randn(N, dim).astype(np.float32)

    sq = make_scalar_quantizer_int8(data, quantile=1.0)
    transformations = [sq]

    graph = create_empty(N, dim, edges_per_vertex=16, metric=Metric.Int8_L2, transformations=transformations)
    builder = deglib.GraphBuilder(graph, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_container.deg")
        graph.save_graph(save_path)

        # 1. Verify standard header at offset 0 and DEGX trailer in file
        with open(save_path, "rb") as f:
            header_8b = f.read(8)
            assert header_8b[0] == int(Metric.Int8_L2)
            dim_in_file = struct.unpack("<H", header_8b[1:3])[0]
            assert dim_in_file == dim

            f.seek(0)
            content = f.read()
            assert b"DEGX" in content
            assert b"XFRM" in content

        # 2. Load via load_graph
        loaded_g = load_graph(save_path)
        assert loaded_g.has_transformations()
        assert loaded_g.size() == N

        # Transparent search on loaded graph
        query = data[7]
        indices, _ = loaded_g.search(query, eps_or_ef=0.2, k=5)
        assert indices[0] == 7

        # 3. Load via load_readonly_graph
        ro_g = load_readonly_graph(save_path)
        assert ro_g.has_transformations()
        assert ro_g.size() == N
        ro_indices, _ = ro_g.search(query, eps_or_ef=0.2, k=5)
        assert ro_indices[0] == 7

        # 4. Searcher with loaded graph
        searcher = create_searcher(
            graph=ro_g,
            rerank_metric=Metric.FP32_L2,
            rerank_data=data,
        )
        searcher.optimize(n_clusters=min(N, 16), n_iter=5, sample_size=N)
        res_searcher = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5)
        assert res_searcher[0] == 7


def test_evp_transforms_end_to_end():
    np.random.seed(42)
    N, dim = 50, 32
    # EVP requires unit-norm vectors for inner product
    raw_data = np.random.randn(N, dim).astype(np.float32)
    norms = np.linalg.norm(raw_data, axis=1, keepdims=True)
    data = (raw_data / norms).astype(np.float32)

    evp = make_evp_quantizer(data, non_zeros=8)
    transformations = [evp]

    graph = create_empty(N, dim, edges_per_vertex=16, metric=Metric.EVP_InnerProduct, transformations=transformations)

    builder = deglib.GraphBuilder(graph, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    assert graph.has_transformations()

    # Search with float query
    query = data[15]
    indices, _ = graph.search(query, eps_or_ef=0.2, k=5)
    assert indices[0] == 15

    # Container save & load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "evp_graph.deg")
        graph.save_graph(path)

        loaded = load_readonly_graph(path)
        assert loaded.has_transformations()
        loaded_indices, _ = loaded.search(query, eps_or_ef=0.2, k=5)
        assert loaded_indices[0] == 15


def test_legacy_v1_compatibility(tmp_path):
    N, dim, edges_per_vertex = 20, 16, 4
    data = np.random.randn(N, dim).astype(np.float32)

    legacy_path = tmp_path / "legacy.deg"
    v1_bytes = bytearray()
    metric_val = int(Metric.FP32_L2)
    v1_bytes.extend(struct.pack("<BHI B", metric_val, dim, N, edges_per_vertex))

    for i in range(N):
        v1_bytes.extend(data[i].tobytes())
        neighbors = [int((i + j + 1) % N) for j in range(edges_per_vertex)]
        v1_bytes.extend(struct.pack(f"<{edges_per_vertex}I", *neighbors))
        weights = [0.0] * edges_per_vertex
        v1_bytes.extend(struct.pack(f"<{edges_per_vertex}f", *weights))
        v1_bytes.extend(struct.pack("<I", i))

    legacy_path.write_bytes(bytes(v1_bytes))

    # Load legacy graph
    loaded_ro = load_readonly_graph(str(legacy_path))
    assert not loaded_ro.has_transformations()
    assert loaded_ro.size() == N
    # Search on legacy graph works with float query
    indices, _ = loaded_ro.search(data[0], eps_or_ef=0.2, k=3)
    assert len(indices) == 3


def test_transformation_graph_search_fp16():
    N, dim, edges_per_vertex = 30, 16, 4
    np.random.seed(42)
    data_f32 = np.random.randn(N, dim).astype(np.float32)

    transform = make_scalar_quantizer_int8(data_f32, quantile=1.0)

    graph = create_empty(N, dim, edges_per_vertex=edges_per_vertex, metric=Metric.Int8_L2, transformations=[transform])
    builder = deglib.GraphBuilder(graph, extend_k=8, extend_eps=0.1)
    builder.add_entry(np.arange(N, dtype=np.uint32), data_f32)
    builder.build()

    ro_graph = graph.to_readonly()

    # Query in FP16
    queries_f16 = data_f32[:5].astype(np.float16)

    # 1. Single query FP16
    idx_single, dist_single = ro_graph.search(queries_f16[0], eps_or_ef=0.1, k=3)
    assert idx_single[0] == 0

    # 2. Batch query FP16 (eps mode)
    idx_batch, dist_batch = ro_graph.search(queries_f16, eps_or_ef=0.1, k=3)
    assert idx_batch.shape == (5, 3)
    assert dist_batch.shape == (5, 3)
    np.testing.assert_array_equal(idx_batch[0], idx_single)
    np.testing.assert_allclose(dist_batch[0], dist_single, rtol=1e-5)

    # 3. Batch query FP16 (ef mode)
    idx_batch_ef, dist_batch_ef = ro_graph.search(queries_f16, eps_or_ef=16.0, k=3)
    assert idx_batch_ef.shape == (5, 3)
    assert dist_batch_ef.shape == (5, 3)
    idx_single_ef, dist_single_ef = ro_graph.search(queries_f16[0], eps_or_ef=16.0, k=3)
    np.testing.assert_array_equal(idx_batch_ef[0], idx_single_ef)
    np.testing.assert_allclose(dist_batch_ef[0], dist_single_ef, rtol=1e-5)


def test_uint4_transformation_graph_search():
    N, dim, edges_per_vertex = 40, 16, 4
    np.random.seed(42)
    data_f32 = np.random.uniform(0.0, 10.0, size=(N, dim)).astype(np.float32)

    transform = make_scalar_quantizer_uint4(data_f32, quantile=1.0)
    quantized_data = transform.transform_database(data_f32)
    assert quantized_data.shape == (N, (dim + 1) // 2)

    # Build FP32 graph first, then convert to readonly with Uint4
    graph = create_empty(N, dim, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)
    builder = deglib.GraphBuilder(graph, extend_k=8, extend_eps=0.1)
    builder.add_entry(np.arange(N, dtype=np.uint32), data_f32)
    builder.build()

    ro_graph = graph.to_readonly(Metric.Uint4_L2, quantized_data, transformations=[transform])
    searcher = create_searcher(ro_graph)

    # Search with original float query
    query = data_f32[0]
    indices = searcher.search(query, k=5, eps_or_ef=0.1)
    assert len(indices) == 5
    assert indices[0] == 0
