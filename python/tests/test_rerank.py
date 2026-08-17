import pytest
import numpy as np

import deglib
from deglib.distances import FloatSpace, Metric


class TestRerankUnit:
    def setup_method(self):
        self.dims = 16
        self.space = FloatSpace.create(self.dims, Metric.FP32_L2)

    def test_single_query_single_candidate(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((1, self.dims), dtype=np.float32)
        candidates = np.array([[0]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_single_query_multiple_candidates(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 3.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        # Closest 2: idx 2 (dist=0.25), idx 0 (dist=1.0)
        assert result[0, 0] == 2
        assert result[0, 1] == 0

    def test_return_distances(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 3.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        indices, distances = deglib.search.rerank(self.space, queries, candidates, base, k_top=2, return_distances=True)
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        assert indices[0, 0] == 2
        assert indices[0, 1] == 0
        np.testing.assert_allclose(distances[0, 0], 0.25, rtol=1e-5)
        np.testing.assert_allclose(distances[0, 1], 1.0, rtol=1e-5)

    def test_k_top_zero_returns_all(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 3.0
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=0)
        assert result.shape == (1, 3)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[0, 2] == 2

    def test_k_top_larger_than_candidates(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((1, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        candidates = np.array([[0]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=5)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_uses_queries_as_targets_when_base_none(self):
        queries = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        candidates = np.array([[0, 1], [0, 1]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, k_top=2)
        assert result.shape == (2, 2)
        # Query 0: idx 0 (dist=0), idx 1 (dist=1)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        # Query 1: idx 1 (dist=0), idx 0 (dist=1)
        assert result[1, 0] == 1
        assert result[1, 1] == 0

    def test_invalid_candidate_index_skipped(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((1, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        # idx 99 is out of bounds, should be skipped
        candidates = np.array([[0, 99]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == np.iinfo(np.uint32).max  # unfilled candidate is padded with uint32 max

    def test_multiple_queries(self):
        queries = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        base = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        candidates = np.array([[0, 1], [0, 1]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=2)
        assert result.shape == (2, 2)
        # Query 0: idx 0 (dist=0), idx 1 (dist=2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        # Query 1: idx 1 (dist=32), idx 0 (dist=50)
        assert result[1, 0] == 1
        assert result[1, 1] == 0

    def test_inner_product_metric(self):
        space = FloatSpace.create(self.dims, Metric.FP32_InnerProduct)
        query = np.zeros((1, self.dims), dtype=np.float32)
        query[0, 0] = 1.0
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        result = deglib.search.rerank(space, query, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        # Inner product: higher is better → idx 1 (ip=2), idx 0 (ip=1)
        assert result[0, 0] == 1
        assert result[0, 1] == 0

    def test_return_distances_all(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 3.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        indices, distances = deglib.search.rerank(self.space, queries, candidates, base, k_top=3, return_distances=True)
        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)
        # Verify distances are correct and sorted ascending
        np.testing.assert_allclose(distances[0, 0], 0.25, rtol=1e-5)  # idx 2
        np.testing.assert_allclose(distances[0, 1], 1.0, rtol=1e-5)  # idx 0
        np.testing.assert_allclose(distances[0, 2], 9.0, rtol=1e-5)  # idx 1

    def test_multi_threaded(self):
        queries = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        base = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        candidates = np.array([[0, 1], [0, 1]], dtype=np.uint32)

        result = deglib.search.rerank(self.space, queries, candidates, base, k_top=2, num_threads=4)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 1
        assert result[1, 1] == 0

    def test_unsorted(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 3.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        # Sorted returns idx 2 then idx 0
        sorted_res = deglib.search.rerank(self.space, queries, candidates, base, k_top=2, unsorted=False)
        assert sorted_res[0, 0] == 2
        assert sorted_res[0, 1] == 0

        # Unsorted returns top 2 without sorting
        unsorted_res = deglib.search.rerank(self.space, queries, candidates, base, k_top=2, unsorted=True)
        assert unsorted_res.shape == (1, 2)
        assert set(unsorted_res[0]) == {0, 2}
