import numpy as np
import pytest

import deglib
from deglib.distances import Metric


class TestRerank:
    def setup_method(self):
        self.dims = 16
        self.metric = Metric.FP32_L2

    def test_single_query_single_candidate(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.ones((3, self.dims), dtype=np.float32)
        candidates = np.array([[0]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_single_query_multiple_candidates(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0  # L2^2 = 1.0
        base[1, 0] = 2.0  # L2^2 = 4.0
        base[2, 0] = 0.5  # L2^2 = 0.25
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        assert result[0, 0] == 2
        assert result[0, 1] == 0

    def test_return_distances_true(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        indices, distances = deglib.rerank(self.metric, queries, candidates, base, k_top=2, return_distances=True)
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        assert indices[0, 0] == 2
        assert indices[0, 1] == 0
        np.testing.assert_allclose(distances[0], [0.25, 1.0], rtol=1e-5)

    def test_k_top_zero_returns_all(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 0.5
        base[1, 0] = 1.0
        base[2, 0] = 2.0
        candidates = np.array([[2, 0, 1]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=0)
        assert result.shape == (1, 3)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[0, 2] == 2

    def test_k_top_greater_than_candidates(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.ones((2, self.dims), dtype=np.float32)
        candidates = np.array([[0]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=5)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_uses_queries_as_targets_when_base_none(self):
        queries = np.zeros((2, self.dims), dtype=np.float32)
        queries[1, 0] = 1.0
        candidates = np.array([[0, 1], [1, 0]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, k_top=2)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 1
        assert result[1, 1] == 0

    def test_uint32_max_padded_candidates_are_ignored(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.ones((2, self.dims), dtype=np.float32)
        candidates = np.array([[np.iinfo(np.uint32).max, 0]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == np.iinfo(np.uint32).max

    def test_multiple_queries_with_multiple_candidates(self):
        queries = np.zeros((2, self.dims), dtype=np.float32)
        queries[1, 0] = 3.0
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 0.0
        base[1, 0] = 1.4
        base[2, 0] = 3.1
        candidates = np.array([[2, 1, 0], [0, 1, 2]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=2)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 2
        assert result[1, 1] == 1

    def test_inner_product_metric(self):
        query = np.zeros((1, self.dims), dtype=np.float32)
        query[0, 0] = 1.0
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        result = deglib.rerank(Metric.FP32_InnerProduct, query, candidates, base, k_top=2)
        assert result.shape == (1, 2)
        assert result[0, 0] == 1
        assert result[0, 1] == 0

    def test_return_distances_values(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        indices, distances = deglib.rerank(self.metric, queries, candidates, base, k_top=3, return_distances=True)
        assert indices.shape == (1, 3)
        assert distances.shape == (1, 3)
        np.testing.assert_allclose(distances[0], [0.25, 1.0, 4.0], rtol=1e-5)

    def test_multithreaded_rerank(self):
        queries = np.zeros((2, self.dims), dtype=np.float32)
        queries[1, 0] = 3.0
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[1, 0] = 1.4
        base[2, 0] = 3.1
        candidates = np.array([[2, 1, 0], [0, 1, 2]], dtype=np.uint32)

        result = deglib.rerank(self.metric, queries, candidates, base, k_top=2, threads=4)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 2
        assert result[1, 1] == 1

    def test_unsorted_flag(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        sorted_res = deglib.rerank(self.metric, queries, candidates, base, k_top=2, unsorted=False)
        assert sorted_res[0, 0] == 2
        assert sorted_res[0, 1] == 0

        unsorted_res = deglib.rerank(self.metric, queries, candidates, base, k_top=2, unsorted=True)
        assert unsorted_res.shape == (1, 2)
        assert set(unsorted_res[0]) == {0, 2}


class TestRerankerClass:
    def setup_method(self):
        self.dims = 16
        self.metric = Metric.FP32_L2

    def test_reranker_class(self):
        queries = np.zeros((1, self.dims), dtype=np.float32)
        base = np.zeros((3, self.dims), dtype=np.float32)
        base[0, 0] = 1.0
        base[1, 0] = 2.0
        base[2, 0] = 0.5
        candidates = np.array([[0, 1, 2]], dtype=np.uint32)

        reranker = deglib.Reranker(self.metric, base)
        assert reranker.get_num_base_vectors() == 3

        res = reranker.rerank(queries, candidates, k_top=2)
        assert res.shape == (1, 2)
        assert res[0, 0] == 2
        assert res[0, 1] == 0

        indices, distances = reranker.rerank(queries, candidates, k_top=2, return_distances=True)
        np.testing.assert_allclose(distances[0], [0.25, 1.0], rtol=1e-5)

    def test_reranker_optimize_preserves_results(self):
        rng = np.random.default_rng(7)
        base = rng.standard_normal((64, self.dims), dtype=np.float32)
        query = base[5:6].copy()
        candidates = np.arange(64, dtype=np.uint32)[None, :]

        reranker = deglib.Reranker(self.metric, base)
        before = reranker.rerank(query, candidates, k_top=5)

        reranker.optimize(sample_count=16, num_candidates=32)
        after = reranker.rerank(query, candidates, k_top=5)
        np.testing.assert_array_equal(before, after)

    def test_reranker_prefetch_defaults(self):
        rng = np.random.default_rng(3)
        base = rng.standard_normal((16, self.dims), dtype=np.float32)
        reranker = deglib.Reranker(self.metric, base)
        assert reranker.get_po() > 0
        assert reranker.get_pl() == 0

    def test_reranker_auto_quantizes_float32_to_fp16(self):
        rng = np.random.default_rng(11)
        base = rng.standard_normal((50, self.dims), dtype=np.float32)
        reranker = deglib.Reranker(Metric.FP16_InnerProduct, base, optimize=True, threads=2)
        candidates = np.arange(50, dtype=np.uint32)[None, :]
        res = reranker.rerank(base[3:4], candidates, k_top=1)
        assert res[0, 0] == 3

    def test_reranker_rejects_invalid_base_vectors(self):
        bad_shape = np.zeros(self.dims, dtype=np.float32)
        with pytest.raises(Exception):
            deglib.Reranker(self.metric, bad_shape)

        wrong_dtype = np.zeros((4, self.dims), dtype=np.float64)
        with pytest.raises(Exception):
            deglib.Reranker(self.metric, wrong_dtype)

    def test_reranker_matches_free_function(self):
        rng = np.random.default_rng(9)
        queries = rng.standard_normal((2, self.dims), dtype=np.float32)
        base = rng.standard_normal((4, self.dims), dtype=np.float32)
        candidates = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint32)

        via_class = deglib.Reranker(self.metric, base).rerank(queries, candidates, k_top=3)
        via_function = deglib.rerank(self.metric, queries, candidates, base, k_top=3)
        np.testing.assert_array_equal(via_class, via_function)

    def test_reranker_preserves_instruction_set(self):
        rng = np.random.default_rng(17)
        dim = 65
        n_base = 24
        base = rng.standard_normal((n_base, dim), dtype=np.float32)
        queries = rng.standard_normal((4, dim), dtype=np.float32)
        candidates = np.broadcast_to(np.arange(n_base, dtype=np.uint32), (queries.shape[0], n_base)).copy()

        expected_top = deglib.Reranker(
            Metric.FP32_L2.with_instruction(deglib.InstructionSet.Scalar),
            base,
        ).rerank(queries, candidates, k_top=5)

        graph = deglib.create_empty(n_base, dim, edges_per_vertex=8, metric=Metric.FP32_L2)
        builder = deglib.GraphBuilder(graph, extend_k=8, extend_eps=0.1)
        builder.add_entry(np.arange(n_base, dtype=np.uint32), base)
        builder.build()

        test_isas = [deglib.InstructionSet.Scalar]
        if deglib.cpu.has_avx2():
            test_isas.append(deglib.InstructionSet.AVX2)
        if deglib.cpu.has_avx512():
            test_isas.append(deglib.InstructionSet.AVX512)

        for isa in test_isas:
            r_fp32 = deglib.Reranker(Metric.FP32_L2.with_instruction(isa), base)
            assert r_fp32.get_instruction() == isa
            np.testing.assert_array_equal(r_fp32.rerank(queries, candidates, k_top=5), expected_top)

            r_fp16 = deglib.Reranker(Metric.FP16_L2.with_instruction(isa), base)
            assert r_fp16.get_instruction() == isa

            r_graph = graph.to_reranker(Metric.FP32_L2.with_instruction(isa))
            assert r_graph.get_instruction() == isa
            np.testing.assert_array_equal(r_graph.rerank(queries, candidates, k_top=5), expected_top)

            r_graph_fp16 = graph.to_reranker(Metric.FP16_L2.with_instruction(isa))
            assert r_graph_fp16.get_instruction() == isa
