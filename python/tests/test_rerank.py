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


class TestRerankerClass:
    def setup_method(self):
        self.dims = 16
        self.space = FloatSpace.create(self.dims, Metric.FP32_L2)

    def test_reranker_class(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 3.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        reranker = deglib.Reranker(self.space, base)
        assert reranker.get_num_base_vectors() == 3

        res = reranker.rerank(queries, candidates, k_top=2)
        assert res.shape == (1, 2)
        assert res[0, 0] == 2
        assert res[0, 1] == 0

        indices, distances = reranker.rerank(queries, candidates, k_top=2, return_distances=True)
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        np.testing.assert_allclose(distances[0, 0], 0.25, rtol=1e-5)
        np.testing.assert_allclose(distances[0, 1], 1.0, rtol=1e-5)

    def test_reranker_optimize_preserves_results(self):
        rng = np.random.default_rng(0)
        base = rng.random((64, self.dims), dtype=np.float32)
        query = rng.random((1, self.dims), dtype=np.float32)
        candidates = np.array([list(range(64))], dtype=np.uint32)

        reranker = deglib.Reranker(self.space, base)
        before = reranker.rerank(query, candidates, k_top=5)

        # Prefetch auto-tuning must not change reranking results.
        reranker.optimize(sample_count=16, num_candidates=32, k=5)

        after = reranker.rerank(query, candidates, k_top=5)
        np.testing.assert_array_equal(before, after)

    def test_reranker_prefetch_defaults_and_override(self):
        rng = np.random.default_rng(3)
        base = rng.random((64, self.dims), dtype=np.float32)
        query = rng.random((1, self.dims), dtype=np.float32)
        candidates = np.array([list(range(64))], dtype=np.uint32)

        reranker = deglib.Reranker(self.space, base)

        # Prefetching must be on out of the box, otherwise out-of-cache reranking silently loses throughput.
        assert reranker.get_po() > 0
        assert reranker.get_pl() == 0

        reranker.set_prefetch(12, 4)
        assert reranker.get_po() == 12
        assert reranker.get_pl() == 4

        reranker.set_prefetch(0, -1)
        assert reranker.get_po() == 0
        assert reranker.get_pl() == 0

        # Both configs must rank identically; prefetching is a speed knob only.
        unprefetched = reranker.rerank(query, candidates, k_top=5)
        reranker.set_prefetch(8, 0)
        prefetched = reranker.rerank(query, candidates, k_top=5)
        np.testing.assert_array_equal(unprefetched, prefetched)

    def test_reranker_rejects_mismatched_base_vectors(self):
        mismatched = np.zeros((3, self.dims + 1), dtype=np.float32)
        with pytest.raises(ValueError):
            deglib.Reranker(self.space, mismatched)

    def test_reranker_matches_free_function(self):
        queries = np.zeros((2, self.dims), dtype=np.float32)
        base = np.zeros((4, self.dims), dtype=np.float32)
        for i in range(4):
            base[i, 0] = float(i)
        candidates = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint32)

        via_class = deglib.Reranker(self.space, base).rerank(queries, candidates, k_top=3)
        via_function = deglib.search.rerank(self.space, queries, candidates, base, k_top=3)
        np.testing.assert_array_equal(via_class, via_function)
