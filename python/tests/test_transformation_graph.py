import os
import struct
import tempfile
import numpy as np
import pytest

import deglib
from deglib import (
    FloatSpace,
    Metric,
    create_empty,
    load_graph,
    load_readonly_graph,
    Searcher,
    create_searcher,
    ScalarQuantizerInt8,
    ScalarQuantizerUint8,
    ScalarQuantizerInt8PerDim,
    ScalarQuantizerUint8PerDim,
    EvpQuantizer,
    MipsTransform,
)
from deglib.transformation import (
    serialize_transformations,
    deserialize_transformations,
    apply_database,
    apply_query,
)


def test_scalar_int8_transform_serialization_and_transform():
    dim = 16
    abs_max = 2.0
    transform = ScalarQuantizerInt8(dim, abs_max=abs_max)
    transformations = [transform]

    assert len(transformations) == 1
    assert transform.input_dim == dim
    assert transform.output_dim == dim
    assert transform.output_dtype == np.int8

    scale = 127.0 / abs_max
    data = np.random.uniform(-abs_max, abs_max, size=(10, dim)).astype(np.float32)
    out = apply_database(transformations, data)
    assert out.dtype == np.int8
    expected = np.clip(np.round(data * scale), -128, 127).astype(np.int8)
    np.testing.assert_array_equal(out, expected)

    # Test serialization roundtrip
    serialized = serialize_transformations(transformations)
    deserialized = deserialize_transformations(serialized)
    assert len(deserialized) == 1
    assert deserialized[0].output_dtype == np.int8

    out_deser = apply_database(deserialized, data)
    np.testing.assert_array_equal(out_deser, expected)


def test_scalar_uint8_transform_serialization_and_transform():
    dim = 16
    min_val = -3.0
    max_val = 3.0
    transform = ScalarQuantizerUint8(dim, min_val=min_val, max_val=max_val)
    transformations = [transform]

    assert transform.output_dtype == np.uint8

    scale = 255.0 / (max_val - min_val)
    data = np.random.uniform(min_val, max_val, size=(10, dim)).astype(np.float32)
    out = apply_database(transformations, data)
    assert out.dtype == np.uint8
    expected = np.clip(np.round((data - min_val) * scale), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)

    # Roundtrip
    deser = deserialize_transformations(serialize_transformations(transformations))
    np.testing.assert_array_equal(apply_database(deser, data), expected)


def test_per_dim_transforms():
    dim = 4
    abs_maxs = [1.0, 2.0, 3.0, 4.0]
    mins = [-1.0, -2.0, -3.0, -4.0]
    maxs = [1.0, 2.0, 3.0, 4.0]

    # Int8 PerDim
    t_int8 = [ScalarQuantizerInt8PerDim(dim, abs_maxs=abs_maxs)]
    assert t_int8[0].output_dtype == np.int8
    deser_int8 = deserialize_transformations(serialize_transformations(t_int8))
    assert len(deser_int8) == 1
    assert deser_int8[0].output_dtype == np.int8

    # Uint8 PerDim
    t_uint8 = [ScalarQuantizerUint8PerDim(dim, mins=mins, maxs=maxs)]
    assert t_uint8[0].output_dtype == np.uint8
    deser_uint8 = deserialize_transformations(serialize_transformations(t_uint8))
    assert len(deser_uint8) == 1
    assert deser_uint8[0].output_dtype == np.uint8


def test_mips_transform():
    dim = 8
    max_norm = 10.0
    transform = MipsTransform(dim, max_norm=max_norm)
    transformations = [transform]

    assert transform.input_dim == dim
    assert transform.output_dim == dim + 1
    assert transform.output_dtype == np.float32

    data = np.random.randn(5, dim).astype(np.float32)
    out = apply_database(transformations, data)
    assert out.shape == (5, dim + 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, :dim], data)

    deser = deserialize_transformations(serialize_transformations(transformations))
    out_deser = apply_database(deser, data)
    np.testing.assert_allclose(out_deser, out)


def test_builder_and_graph_search_with_int8_transforms():
    np.random.seed(42)
    N, dim = 60, 16
    data = np.random.randn(N, dim).astype(np.float32)

    # Setup Int8 Quantizer
    abs_max = float(np.max(np.abs(data)))
    transformations = [ScalarQuantizerInt8(dim, abs_max=abs_max)]

    # Create empty graph configured with transformations
    space = FloatSpace.create(dim, Metric.Int8_L2)
    graph = create_empty(N, space, edges_per_vertex=16, transformations=transformations)

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


def test_container_persistence_roundtrip():
    np.random.seed(123)
    N, dim = 50, 16
    data = np.random.randn(N, dim).astype(np.float32)

    abs_max = float(np.max(np.abs(data)))
    transformations = [ScalarQuantizerInt8(dim, abs_max=abs_max)]

    space = FloatSpace.create(dim, Metric.Int8_L2)
    graph = create_empty(N, space, edges_per_vertex=16, transformations=transformations)
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
        refine_space = FloatSpace.create(dim, Metric.FP32_L2)
        searcher = create_searcher(
            graph=ro_g,
            refine_space=refine_space,
            refine_data=data,
        )
        searcher.optimize(n_clusters=min(N, 16), n_iter=5, sample_size=N, k=5)
        res_searcher = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5)
        assert res_searcher[0] == 7


def test_evp_transforms_end_to_end():
    np.random.seed(42)
    N, dim = 50, 32
    # EVP requires unit-norm vectors for inner product
    raw_data = np.random.randn(N, dim).astype(np.float32)
    norms = np.linalg.norm(raw_data, axis=1, keepdims=True)
    data = (raw_data / norms).astype(np.float32)

    evp = EvpQuantizer(dim, 8)
    evp.fit(data)
    transformations = [evp]

    space = FloatSpace.create(dim, Metric.EVP_InnerProduct)
    graph = create_empty(N, space, edges_per_vertex=16, transformations=transformations)

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

    transform = ScalarQuantizerInt8(dim, abs_max=float(np.max(np.abs(data_f32))))
    space = FloatSpace.create(dim, Metric.Int8_L2)

    graph = create_empty(N, space, edges_per_vertex=edges_per_vertex, transformations=[transform])
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

