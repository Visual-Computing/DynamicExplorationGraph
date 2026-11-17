import numpy as np
import pytest

from deglib.transformation import apply_on_database, apply_on_query, make_mips_transform


def test_mips_transform_basic():
    rng = np.random.default_rng(42)
    n, d = 100, 16
    data = rng.standard_normal((n, d)).astype(np.float32)

    mips = make_mips_transform(data)
    assert mips.name == "MipsTransform"
    assert mips.input_dim == d
    assert mips.output_dim == d + 1
    assert mips.max_norm > 0.0

    transformed = mips.transform_database(data)
    assert transformed.shape == (n, d + 1)
    assert transformed.dtype == np.float32

    np.testing.assert_allclose(transformed[:, :d], data)
    norms_sq = np.sum(transformed**2, axis=1)
    np.testing.assert_allclose(norms_sq, mips.max_norm**2, rtol=1e-5, atol=1e-5)

    # Multi-stage apply_on_database matches transform_database
    via_pipe = apply_on_database([mips], data)
    np.testing.assert_allclose(via_pipe, transformed)


def test_mips_transform_distance_equivalence():
    rng = np.random.default_rng(123)
    n, d, q = 50, 32, 10
    data = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((q, d)).astype(np.float32)

    mips = make_mips_transform(data)
    transformed_data = mips.transform_database(data)
    transformed_queries = mips.transform_query(queries)

    for i in range(q):
        l2_dists = np.sum((transformed_data - transformed_queries[i]) ** 2, axis=1)
        inner_products = data @ queries[i]
        l2_order = np.argsort(l2_dists)
        mips_order = np.argsort(-inner_products)
        assert l2_order[0] == mips_order[0]
        np.testing.assert_array_equal(l2_order[:5], mips_order[:5])


def test_mips_transform_query_1d_and_explicit_max_norm():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((20, 8)).astype(np.float32)
    mips = make_mips_transform(data, max_norm=100.0)
    assert mips.max_norm == pytest.approx(100.0)

    q_1d = rng.standard_normal(8).astype(np.float32)
    out_1d = mips.transform_query(q_1d)
    assert out_1d.shape == (9,)
    np.testing.assert_allclose(out_1d[:8], q_1d)
    assert out_1d[8] == 0.0

    via_pipe = apply_on_query(mips, q_1d)
    np.testing.assert_allclose(via_pipe, out_1d)
