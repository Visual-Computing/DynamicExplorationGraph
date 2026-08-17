import numpy as np
import pytest

from deglib.optimization import mips_l2_transform, mips_l2_transform_query


def test_mips_l2_transform_basic():
    np.random.seed(42)
    N, d = 100, 32
    data = np.random.randn(N, d).astype(np.float32)

    transformed, max_norm = mips_l2_transform(data)

    assert transformed.shape == (N, d + 1)
    assert transformed.dtype == np.float32
    assert max_norm > 0.0

    # 1. Original features should match first d columns
    np.testing.assert_allclose(transformed[:, :d], data, rtol=1e-5)

    # 2. All transformed vectors must have identical squared norm M^2
    norms_sq = np.sum(transformed**2, axis=1)
    expected_max_norm_sq = max_norm**2
    np.testing.assert_allclose(norms_sq, expected_max_norm_sq, rtol=1e-4)


def test_mips_l2_transform_distance_equivalence():
    np.random.seed(42)
    N, Q, d = 20, 5, 16
    data = np.random.randn(N, d).astype(np.float32)
    queries = np.random.randn(Q, d).astype(np.float32)

    transformed_data, max_norm = mips_l2_transform(data)
    transformed_queries = mips_l2_transform_query(queries)
    M_sq = max_norm**2

    # Verify that for query q' = [q, 0] and db vector x'_i = [x_i, sqrt(M^2 - ||x_i||^2)]:
    # ||q' - x'_i||^2 = ||q||^2 + M^2 - 2 * <q, x_i>
    for q_idx in range(Q):
        q_norm_sq = np.sum(queries[q_idx] ** 2)
        for i in range(N):
            l2_dist_sq = np.sum((transformed_queries[q_idx] - transformed_data[i]) ** 2)
            ip_val = np.dot(queries[q_idx], data[i])
            expected_l2_dist_sq = q_norm_sq + M_sq - 2.0 * ip_val
            np.testing.assert_allclose(l2_dist_sq, expected_l2_dist_sq, rtol=1e-4)


def test_mips_l2_transform_query():
    np.random.seed(42)
    Q, d = 10, 32
    queries = np.random.randn(Q, d).astype(np.float32)

    transformed_q = mips_l2_transform_query(queries)

    assert transformed_q.shape == (Q, d + 1)
    assert transformed_q.dtype == np.float32
    np.testing.assert_allclose(transformed_q[:, :d], queries, rtol=1e-5)
    np.testing.assert_allclose(transformed_q[:, d], 0.0, atol=1e-6)

    # Single 1D query vector
    single_q = queries[0]
    transformed_single_q = mips_l2_transform_query(single_q)
    assert transformed_single_q.shape == (d + 1,)
    np.testing.assert_allclose(transformed_single_q[:d], single_q, rtol=1e-5)
    assert transformed_single_q[d] == 0.0
