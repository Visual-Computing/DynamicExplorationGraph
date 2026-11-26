import numpy as np
import pytest

import deglib
from deglib.distances import Metric


def _make_vectors(n: int = 64, d: int = 32, *, normalized: bool = False, dtype=np.float32, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    if normalized:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)
    else:
        # Varying norms so unnormalized IP/MIPS is genuinely distinct from cosine
        scales = rng.uniform(0.5, 4.0, size=(n, 1)).astype(np.float32)
        x = x * scales
    return x.astype(dtype)


def _prefix_names(graph: deglib.DynamicExplorationGraph) -> list[str]:
    return [t.name for t in graph.get_transformations()]


def _expected_fp_metric(metric_in: str, dtype) -> Metric:
    is_fp16 = np.dtype(dtype) == np.dtype(np.float16)
    if metric_in in ("l2", "euclidean"):
        return Metric.FP16_L2 if is_fp16 else Metric.FP32_L2
    return Metric.FP16_InnerProduct if is_fp16 else Metric.FP32_InnerProduct


class TestBuildMatrix:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize(
        ("metric_in", "input_normalized", "compress_in", "expected_metrics", "expect_l2n"),
        [
            # l2 / euclidean
            ("l2", False, None, {Metric.FP32_L2, Metric.FP16_L2}, False),
            ("l2", False, "16bit", {Metric.FP16_L2}, False),
            ("l2", False, "8bit", {Metric.Uint7_L2}, False),
            ("l2", False, "auto", {Metric.Uint7_L2}, False),
            ("l2", False, "4bit", {Metric.Uint4_L2}, False),
            ("l2", False, "2bit", {Metric.EVP_InnerProduct}, False),
            ("l2", False, "evp", {Metric.EVP_InnerProduct}, False),
            # cosine (unnormalized input -> attaches L2NormalizeTransform)
            ("cosine", False, None, {Metric.FP32_InnerProduct, Metric.FP16_InnerProduct}, True),
            ("cosine", False, "16bit", {Metric.FP16_InnerProduct}, True),
            ("cosine", False, "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, True),
            ("cosine", False, "auto", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, True),
            ("cosine", False, "4bit", {Metric.Uint4_L2}, True),
            ("cosine", False, "2bit", {Metric.EVP_InnerProduct}, True),
            ("cosine", False, "evp", {Metric.EVP_InnerProduct}, True),
            # normalized / cosine with already-normalized input -> NO L2NormalizeTransform
            ("normalized", True, None, {Metric.FP32_InnerProduct, Metric.FP16_InnerProduct}, False),
            ("normalized", True, "16bit", {Metric.FP16_InnerProduct}, False),
            ("normalized", True, "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, False),
            ("normalized", True, "auto", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, False),
            ("normalized", True, "4bit", {Metric.Uint4_L2}, False),
            ("normalized", True, "2bit", {Metric.EVP_InnerProduct}, False),
            ("cosine", True, "4bit", {Metric.Uint4_L2}, False),
            # ip with already-normalized input -> auto-switches to normalized (NO MipsTransform, NO L2NormalizeTransform)
            ("ip", True, None, {Metric.FP32_InnerProduct, Metric.FP16_InnerProduct}, False),
            ("ip", True, "16bit", {Metric.FP16_InnerProduct}, False),
            ("ip", True, "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, False),
            ("ip", True, "4bit", {Metric.Uint4_L2}, False),
            ("ip", True, "2bit", {Metric.EVP_InnerProduct}, False),
            # ip with unnormalized input -> d-dim IP (builds via MIPS(d+1) internally, swaps back)
            ("ip", False, None, {Metric.FP32_InnerProduct, Metric.FP16_InnerProduct}, False),
            ("ip", False, "16bit", {Metric.FP16_InnerProduct}, False),
            ("ip", False, "8bit", {Metric.Int8_InnerProduct}, False),
            ("ip", False, "auto", {Metric.Int8_InnerProduct}, False),
            ("ip", False, "4bit", {Metric.Int4_InnerProduct}, False),
            ("ip", False, "2bit", {Metric.EVP_InnerProduct}, False),
            ("ip", False, "evp", {Metric.EVP_InnerProduct}, False),
        ],
    )
    def test_build_matrix(self, dtype, metric_in, input_normalized, compress_in, expected_metrics, expect_l2n):
        x = _make_vectors(64, 32, normalized=input_normalized, dtype=dtype)
        g = deglib.build(
            x,
            metric=metric_in,
            compress=compress_in,
            edges_per_vertex=16,
            extend_k=16,
            presort=False,
            threads=1,
        )
        assert g.is_mutable()
        assert g.get_dim() == 32
        if compress_in is None:
            if metric_in == "cosine" and not input_normalized and dtype == np.float16:
                # L2NormalizeTransform outputs float32, promoting FP16_InnerProduct -> FP32_InnerProduct
                assert g.get_metric() == Metric.FP32_InnerProduct
            else:
                assert g.get_metric() == _expected_fp_metric(metric_in, dtype)
        else:
            assert Metric(int(g.get_metric())) in expected_metrics

        xfrms = _prefix_names(g)
        assert ("L2NormalizeTransform" in xfrms) is expect_l2n
        assert "MipsTransform" not in xfrms

        if metric_in == "ip" and not input_normalized and compress_in in ("8bit", "auto"):
            assert xfrms == ["FastHadamardRotation", "ScalarQuantizerInt8"]

        q = x[0] if (not xfrms and Metric(int(g.get_metric())) in (Metric.FP16_L2, Metric.FP16_InnerProduct)) else x[0].astype(np.float32)
        res = g.search(q, k=5, eps_or_ef=0.2)
        assert res.indices.shape == (5,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("metric_in", ["l2", "cosine", "normalized", "ip"])
    def test_build_32bit_compress_fp32_vs_fp16(self, dtype, metric_in):
        x = _make_vectors(48, 32, normalized=(metric_in == "normalized"), dtype=dtype)
        if dtype == np.float16 and metric_in in ("l2", "normalized"):
            with pytest.raises(ValueError, match="Cannot switch"):
                deglib.build(x, metric=metric_in, compress="32bit", edges_per_vertex=16, extend_k=16, threads=1)
        else:
            g = deglib.build(x, metric=metric_in, compress="32bit", edges_per_vertex=16, extend_k=16, threads=1)
            if metric_in == "ip" and dtype == np.float16:
                assert g.get_metric() == Metric.FP16_InnerProduct
            else:
                assert g.get_metric() == _expected_fp_metric(metric_in, np.float32)
            assert g.get_dim() == 32

    @pytest.mark.parametrize("explicit_l2", [Metric.Uint7_L2, Metric.Int8_L2, Metric.Uint4_L2])
    def test_build_ip_unnormalized_with_explicit_l2_keeps_mips(self, explicit_l2):
        x = _make_vectors(64, 32, normalized=False, dtype=np.float32)
        g = deglib.build(x, metric="ip", compress=explicit_l2, edges_per_vertex=16, extend_k=16, threads=1)
        assert g.is_mutable()
        assert g.get_metric() == explicit_l2
        assert g.get_dim() == 33
        assert _prefix_names(g)[0] == "MipsTransform"

    @pytest.mark.parametrize("metric_in", ["l2", "cosine", "normalized", "ip"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_build_empty_graph(self, metric_in, dtype):
        x = np.empty((0, 32), dtype=dtype)
        g = deglib.build(x, metric=metric_in, compress="4bit", edges_per_vertex=16, threads=1)
        assert g.size() == 0
        assert g.get_dim() == 32
        assert g.get_metric() == _expected_fp_metric(metric_in, dtype)
        assert g.is_mutable()


@pytest.fixture(scope="module")
def base_graphs():
    data_unnorm = _make_vectors(64, 32, normalized=False, dtype=np.float32, seed=11)
    data_norm = _make_vectors(64, 32, normalized=True, dtype=np.float32, seed=12)
    data_unnorm_16 = data_unnorm.astype(np.float16)
    data_norm_16 = data_norm.astype(np.float16)
    return {
        "l2_fp32": deglib.build(data_unnorm, metric="l2", edges_per_vertex=16, extend_k=16, threads=1),
        "l2_fp16": deglib.build(data_unnorm_16, metric="l2", edges_per_vertex=16, extend_k=16, threads=1),
        "cosine_unnorm_fp32": deglib.build(data_unnorm, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1),
        "cosine_unnorm_fp16": deglib.build(data_unnorm_16, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1),
        "normalized_fp32": deglib.build(data_norm, metric="normalized", edges_per_vertex=16, extend_k=16, threads=1),
        "normalized_fp16": deglib.build(data_norm_16, metric="normalized", edges_per_vertex=16, extend_k=16, threads=1),
        "ip_norm_fp32": deglib.build(data_norm, metric="ip", edges_per_vertex=16, extend_k=16, threads=1),
        "ip_unnorm_fp32": deglib.build(data_unnorm, metric="ip", edges_per_vertex=16, extend_k=16, threads=1),
        "ip_unnorm_fp16": deglib.build(data_unnorm_16, metric="ip", edges_per_vertex=16, extend_k=16, threads=1),
        "q_unnorm": data_unnorm[0],
        "q_norm": data_norm[0],
    }


class TestCompressAndSearcherMatrix:

    @pytest.mark.parametrize(
        ("graph_key", "target", "expected_metrics", "expected_dim", "expect_mips", "expect_l2n"),
        [
            # L2 source (FP32 & FP16)
            ("l2_fp32", None, {Metric.FP32_L2}, 32, False, False),
            ("l2_fp32", "32bit", {Metric.FP32_L2}, 32, False, False),
            ("l2_fp32", "16bit", {Metric.FP16_L2}, 32, False, False),
            ("l2_fp32", "8bit", {Metric.Uint7_L2}, 32, False, False),
            ("l2_fp32", "auto", {Metric.Uint7_L2}, 32, False, False),
            ("l2_fp32", "4bit", {Metric.Uint4_L2}, 32, False, False),
            ("l2_fp32", "2bit", {Metric.EVP_InnerProduct}, 32, False, False),
            ("l2_fp32", "evp", {Metric.EVP_InnerProduct}, 32, False, False),
            ("l2_fp16", "16bit", {Metric.FP16_L2}, 32, False, False),
            ("l2_fp16", "8bit", {Metric.Uint7_L2}, 32, False, False),
            ("l2_fp16", "4bit", {Metric.Uint4_L2}, 32, False, False),
            ("l2_fp16", "2bit", {Metric.EVP_InnerProduct}, 32, False, False),
            # Cosine unnormalized (has L2NormalizeTransform)
            ("cosine_unnorm_fp32", "32bit", {Metric.FP32_InnerProduct}, 32, False, True),
            ("cosine_unnorm_fp32", "16bit", {Metric.FP16_InnerProduct}, 32, False, True),
            ("cosine_unnorm_fp32", "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, 32, False, True),
            ("cosine_unnorm_fp32", "auto", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, 32, False, True),
            ("cosine_unnorm_fp32", "4bit", {Metric.Uint4_L2}, 32, False, True),
            ("cosine_unnorm_fp32", "2bit", {Metric.EVP_InnerProduct}, 32, False, True),
            ("cosine_unnorm_fp32", Metric.Uint7_L2, {Metric.Uint7_L2}, 32, False, True),
            ("cosine_unnorm_fp32", Metric.Uint4_L2, {Metric.Uint4_L2}, 32, False, True),
            ("cosine_unnorm_fp16", "4bit", {Metric.Uint4_L2}, 32, False, True),
            # Normalized (already unit-norm -> NO L2NormalizeTransform!)
            ("normalized_fp32", "32bit", {Metric.FP32_InnerProduct}, 32, False, False),
            ("normalized_fp32", "16bit", {Metric.FP16_InnerProduct}, 32, False, False),
            ("normalized_fp32", "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, 32, False, False),
            ("normalized_fp32", "auto", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, 32, False, False),
            ("normalized_fp32", "4bit", {Metric.Uint4_L2}, 32, False, False),
            ("normalized_fp32", "2bit", {Metric.EVP_InnerProduct}, 32, False, False),
            ("normalized_fp32", Metric.Uint7_L2, {Metric.Uint7_L2}, 32, False, False),
            ("normalized_fp32", Metric.Uint4_L2, {Metric.Uint4_L2}, 32, False, False),
            ("normalized_fp16", "4bit", {Metric.Uint4_L2}, 32, False, False),
            # IP with already unit-norm vectors -> behaves like normalized
            ("ip_norm_fp32", "4bit", {Metric.Uint4_L2}, 32, False, False),
            ("ip_norm_fp32", "8bit", {Metric.Uint7_L2, Metric.Int8_InnerProduct}, 32, False, False),
            ("ip_norm_fp32", Metric.Uint7_L2, {Metric.Uint7_L2}, 32, False, False),
            # IP unnormalized -> Int8_InnerProduct / Int4_InnerProduct (d-dim), explicit L2 adds MipsTransform (d+1)
            ("ip_unnorm_fp32", "32bit", {Metric.FP32_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "16bit", {Metric.FP16_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "8bit", {Metric.Int8_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "auto", {Metric.Int8_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "4bit", {Metric.Int4_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "2bit", {Metric.EVP_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", "evp", {Metric.EVP_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp32", Metric.Uint7_L2, {Metric.Uint7_L2}, 33, True, False),
            ("ip_unnorm_fp32", Metric.Int8_L2, {Metric.Int8_L2}, 33, True, False),
            ("ip_unnorm_fp32", Metric.Uint4_L2, {Metric.Uint4_L2}, 33, True, False),
            ("ip_unnorm_fp32", Metric.Uint8_L2, {Metric.Uint8_L2}, 33, True, False),
            ("ip_unnorm_fp16", "4bit", {Metric.Int4_InnerProduct}, 32, False, False),
            ("ip_unnorm_fp16", "8bit", {Metric.Int8_InnerProduct}, 32, False, False),
        ],
    )
    def test_compress_matrix(self, base_graphs, graph_key, target, expected_metrics, expected_dim, expect_mips, expect_l2n):
        src = base_graphs[graph_key]
        if target is None:
            return
        comp = deglib.compress(src, target, threads=1)
        assert Metric(int(comp.get_metric())) in expected_metrics
        assert comp.get_dim() == expected_dim
        assert type(comp.dynamic_exploration_graph_cpp) is type(src.dynamic_exploration_graph_cpp)

        xfrms = _prefix_names(comp)
        assert ("MipsTransform" in xfrms) is expect_mips
        assert ("L2NormalizeTransform" in xfrms) is expect_l2n

        q = base_graphs["q_norm"] if "norm" in graph_key and "unnorm" not in graph_key else base_graphs["q_unnorm"]
        s = deglib.searcher(src, compress=target, threads=1)
        assert Metric(int(s.graph.get_metric())) in expected_metrics
        assert s.search(q, k=5, eps_or_ef=0.2).indices.shape == (5,)

    @pytest.mark.parametrize("backend", ["mutable", "readonly", "dynamic"])
    def test_compress_preserves_backend_on_normalized_4bit(self, base_graphs, backend):
        src = base_graphs["normalized_fp32"]
        if backend == "readonly":
            g = src.to_readonly()
        elif backend == "dynamic":
            g = src.to_dynamic(chunk_size=32, threads=1)
        else:
            g = src

        comp = deglib.compress(g, "4bit", threads=1)
        assert comp.get_metric() == Metric.Uint4_L2
        assert comp.get_dim() == 32
        assert type(comp.dynamic_exploration_graph_cpp) is type(g.dynamic_exploration_graph_cpp)

    def test_compress_invalid_inputs(self, base_graphs):
        g4 = deglib.compress(base_graphs["l2_fp32"], "4bit", threads=1)
        with pytest.raises(ValueError, match="non-float metric"):
            deglib.compress(g4, "8bit", threads=1)

        with pytest.raises(ValueError, match="without explicit transformations"):
            deglib.compress(base_graphs["l2_fp16"], "32bit", threads=1)

        with pytest.raises(ValueError, match="without explicit transformations"):
            deglib.compress(base_graphs["ip_unnorm_fp16"], "32bit", threads=1)

        empty = deglib.build(np.empty((0, 32), dtype=np.float32), metric="l2", threads=1)
        with pytest.raises(ValueError, match="empty graph"):
            deglib.compress(empty, "4bit", threads=1)


class TestRerankerMatrix:
    def test_reranker_and_reranker_with_data_matrix(self):
        x_unnorm = _make_vectors(64, 32, normalized=False, dtype=np.float32, seed=21)
        x_norm = _make_vectors(64, 32, normalized=True, dtype=np.float32, seed=22)

        g_l2 = deglib.build(x_unnorm, metric="l2", edges_per_vertex=16, extend_k=16, threads=1)
        g_cos = deglib.build(x_unnorm, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1)
        g_norm = deglib.build(x_norm, metric="normalized", edges_per_vertex=16, extend_k=16, threads=1)
        g_ip = deglib.build(x_unnorm, metric="ip", edges_per_vertex=16, extend_k=16, threads=1)

        assert deglib.reranker(g_l2, compress="16bit", threads=1).metric == Metric.FP16_L2
        assert deglib.reranker(g_l2, compress="8bit", threads=1).metric == Metric.Uint7_L2
        assert deglib.reranker(g_cos, compress="16bit", threads=1).metric == Metric.FP16_InnerProduct
        assert deglib.reranker(g_cos, compress="8bit", threads=1).metric in {Metric.Uint7_L2, Metric.Int8_InnerProduct}
        assert deglib.reranker(g_norm, compress="8bit", threads=1).metric in {Metric.Uint7_L2, Metric.Int8_InnerProduct}
        assert deglib.reranker(g_ip, compress="16bit", threads=1).metric == Metric.FP16_InnerProduct
        assert deglib.reranker(g_ip, compress="8bit", threads=1).metric == Metric.Int8_InnerProduct

        # reranker_with_data on quantized graphs with presort + custom labels
        labels = np.arange(100, 164, dtype=np.uint32)[::-1].copy()
        for metric_in, data in [("l2", x_unnorm), ("cosine", x_unnorm), ("normalized", x_norm), ("ip", x_unnorm)]:
            g = deglib.build(data, labels=labels, metric=metric_in, edges_per_vertex=16, extend_k=16, presort=True, threads=1)
            g4 = deglib.compress(g, "4bit", threads=1)
            rr = deglib.reranker_with_data(g4, data=data, labels=labels, compress="16bit", metric=metric_in, threads=1)
            s = deglib.searcher(g4, reranker=rr, threads=1)
            res = s.search(data[3], k=5, eps_or_ef=0.2, rerank_factor=2.0)
            expected_idx = int(np.argmax(data @ data[3])) if metric_in == "ip" else 3
            assert int(res.indices[0]) == int(labels[expected_idx])
