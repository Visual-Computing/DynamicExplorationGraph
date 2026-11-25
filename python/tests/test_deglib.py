import numpy as np
import pytest

import deglib
from deglib.builder import OptimizationTarget
from deglib.distances import Metric


class TestHighLevelBuild:
    def test_build_defaults_l2_and_auto_target(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 16)).astype(np.float32)

        g = deglib.build(x, edges_per_vertex=16, extend_k=16, threads=1)
        assert g.size() == 80
        assert g.get_metric() == Metric.FP32_L2
        assert not g.has_transformations()
        res = g.search(x[0], k=5, eps_or_ef=0.2)
        assert int(res.indices[0]) == 0

    def test_build_cosine_normalized_vs_unnormalized(self):
        rng = np.random.default_rng(2)
        raw = rng.normal(size=(80, 16)).astype(np.float32) * 5.0
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        g_norm = deglib.build(unit, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1)
        assert g_norm.get_metric() == Metric.FP32_InnerProduct
        assert not g_norm.has_transformations()

        g_raw = deglib.build(raw, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1)
        assert g_raw.get_metric() == Metric.FP32_InnerProduct
        assert g_raw.has_transformations()
        assert g_raw.get_transformations()[0].name == "L2NormalizeTransform"
        res = g_raw.search(raw[3], k=5, eps_or_ef=0.2)
        assert int(res.indices[0]) == 3

    def test_build_cosine_presort_uses_normalized_vectors(self, monkeypatch):
        import deglib.builder as builder_mod

        rng = np.random.default_rng(4)
        x = (rng.normal(size=(80, 16)) * 5.0).astype(np.float32)

        captured = {}
        orig_presort = builder_mod.run_presort

        def spy_presort(vectors, *args, **kwargs):
            captured["norms"] = np.linalg.norm(vectors, axis=1)
            return orig_presort(vectors, *args, **kwargs)

        monkeypatch.setattr(builder_mod, "run_presort", spy_presort)
        deglib.build(x, metric="cosine", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        assert np.allclose(captured["norms"], 1.0, atol=1e-5)

    def test_compress_int8_ip_keeps_auto_vnni_instruction(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(64, 64)).astype(np.float32)
        g = deglib.build(x, metric="cosine", optimization_target="LowLID", edges_per_vertex=16, extend_k=16, threads=1)
        g8 = deglib.compress(g, "Int8_InnerProduct", threads=1)
        expected = Metric.Int8_InnerProduct._to_cpp_space(64).get_instruction()
        assert g8.get_instruction().name == expected

    def test_build_ip_builds_mips_and_swaps_to_unnormalized_ip(self, monkeypatch):
        import deglib.builder as builder_mod

        rng = np.random.default_rng(3)
        x = rng.normal(size=(80, 16)).astype(np.float32)
        # Scale rows to widely varying norms so unnormalized MIPS differs from cosine
        scales = np.linspace(0.5, 10.0, 80, dtype=np.float32)[:, None]
        x = x * scales

        captured = {}
        orig_presort = builder_mod.run_presort
        orig_add_entry = builder_mod.GraphBuilder.add_entry

        def spy_presort(vectors, *args, **kwargs):
            captured["presort_dim"] = vectors.shape[1]
            captured["presort_metric"] = kwargs.get("metric")
            return orig_presort(vectors, *args, **kwargs)

        def spy_add_entry(self_builder, labels, features):
            captured["build_metric"] = self_builder.graph.get_metric()
            captured["build_dim"] = features.shape[1]
            return orig_add_entry(self_builder, labels, features)

        monkeypatch.setattr(builder_mod, "run_presort", spy_presort)
        monkeypatch.setattr(builder_mod.GraphBuilder, "add_entry", spy_add_entry)

        g = deglib.build(x, metric="ip", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        assert captured["presort_metric"] == Metric.FP32_L2
        assert captured["presort_dim"] == 17
        assert captured["build_metric"] == Metric.FP32_L2
        assert captured["build_dim"] == 17

        assert g.is_mutable()
        assert g.get_metric() == Metric.FP32_InnerProduct
        assert g.get_dim() == 16
        assert not g.has_transformations()

        # Verify stored features are unnormalized (exact MIPS top-1 differs from cosine)
        q = x[0]
        assert int(g.search(q, k=1, eps_or_ef=0.2).indices[0]) == int(np.argmax(x @ q))

        g16 = deglib.build(x, metric="ip", compress="16bit", edges_per_vertex=16, extend_k=16, threads=1)
        assert not g16.is_mutable()
        assert g16.get_metric() == Metric.FP16_InnerProduct
        assert [t.name for t in g16.get_transformations()] == ["ScalarQuantizerFP16"]
        assert int(np.argmax(x @ q)) in g16.search(q, k=5, eps_or_ef=0.2).indices

        g8 = deglib.build(x, metric="ip", compress="8bit", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        assert not g8.is_mutable()
        assert g8.get_metric() == Metric.Int8_InnerProduct
        assert g8.get_dim() == 16
        assert [t.name for t in g8.get_transformations()] == ["FastHadamardRotation", "ScalarQuantizerInt8"]

        g4 = deglib.build(x, metric="ip", compress="4bit", edges_per_vertex=16, extend_k=16, threads=1)
        assert not g4.is_mutable()
        assert g4.get_metric() == Metric.Int4_InnerProduct
        assert [t.name for t in g4.get_transformations()] == [
            "L2NormalizeTransform",
            "FastHadamardRotation",
            "ScalarQuantizerInt4",
        ]

    def test_build_fp16_presort_and_compress(self):
        rng = np.random.default_rng(4)
        x16 = rng.normal(size=(80, 16)).astype(np.float16)

        g = deglib.build(x16, metric="l2", compress="16bit", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        assert g.get_metric() == Metric.FP16_L2
        res = g.search(x16[2], k=5, eps_or_ef=0.2)
        assert int(res.indices[0]) == 2

    @pytest.mark.parametrize(
        ("metric_in", "compress_in", "expected_metric"),
        [
            ("l2", "16bit", Metric.FP16_L2),
            ("l2", "8bit", Metric.Uint7_L2),
            ("l2", "4bit", Metric.Uint4_L2),
            ("l2", "2bit", Metric.EVP_InnerProduct),
            ("cosine", "4bit", Metric.Int4_InnerProduct),
            ("cosine", "2bit", Metric.EVP_InnerProduct),
            ("ip", "4bit", Metric.Int4_InnerProduct),
            ("ip", "2bit", Metric.EVP_InnerProduct),
        ],
    )
    def test_build_with_compress_modes(self, metric_in, compress_in, expected_metric):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(80, 32)).astype(np.float32)

        g = deglib.build(x, metric=metric_in, compress=compress_in, edges_per_vertex=16, extend_k=16, threads=1)
        assert g.size() == 80
        assert g.get_metric() == expected_metric

    def test_build_prequantized_validation(self):
        rng = np.random.default_rng(6)
        u8 = rng.integers(0, 255, size=(40, 16), dtype=np.uint8)

        with pytest.raises(ValueError, match="compress requires float32 or float16"):
            deglib.build(u8, metric=Metric.Uint8_L2, compress="8bit")
        with pytest.raises(ValueError, match="presort=True requires float32 or float16"):
            deglib.build(u8, metric=Metric.Uint8_L2, presort=True)
        with pytest.raises(ValueError, match="explicit deglib.Metric"):
            deglib.build(u8, metric="l2")

        g = deglib.build(u8, metric=Metric.Uint8_L2, edges_per_vertex=8, extend_k=8, threads=1)
        assert g.get_metric() == Metric.Uint8_L2

    def test_optimization_target_auto_resolve(self):
        assert OptimizationTarget.resolve("auto") == OptimizationTarget.Auto
        assert OptimizationTarget.resolve(OptimizationTarget.Auto) == OptimizationTarget.Auto


class TestCompressRerankerSearcher:
    def test_compress_and_reranker_error_on_quantized(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(80, 32)).astype(np.float32)

        g = deglib.build(x, metric="l2", edges_per_vertex=16, extend_k=16, threads=1)
        rr = deglib.reranker(g, compress="16bit", threads=1)
        assert rr.get_num_base_vectors() == 80

        g4 = deglib.compress(g, "4bit", threads=1)
        assert g4.get_metric() == Metric.Uint4_L2

        with pytest.raises(ValueError, match="reranker_with_data"):
            deglib.reranker(g4, compress="16bit")

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("metric_in", ["l2", "cosine", "ip"])
    def test_reranker_with_data_presort_and_searcher_prefix_split(self, dtype, metric_in):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(100, 32)).astype(dtype)
        # Custom non-trivial labels (offset + reversed) to verify label -> row mapping
        labels = np.arange(1000, 1100, dtype=np.uint32)[::-1].copy()

        g = deglib.build(
            x,
            labels=labels,
            metric=metric_in,
            edges_per_vertex=16,
            extend_k=16,
            presort=True,
            threads=1,
        )
        g4 = deglib.compress(g, "4bit", threads=1)
        rr = deglib.reranker_with_data(
            g4,
            data=x,
            labels=labels,
            compress="16bit",
            metric=metric_in,
            threads=1,
        )
        s = deglib.searcher(g4, reranker=rr, threads=1)

        q = x[5].astype(np.float32)
        res = s.search(q, k=5, eps_or_ef=0.2, rerank_factor=3.0)
        assert int(res.indices[0]) == int(labels[5])

    def test_searcher_compress_shortcut(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 32)).astype(np.float32)
        g = deglib.build(x, metric="cosine", edges_per_vertex=16, extend_k=16, threads=1)
        rr = deglib.reranker(g, compress="16bit", threads=1)
        s = deglib.searcher(g, compress="8bit", reranker=rr, threads=1)

        res = s.search(x[0], k=5, eps_or_ef=0.2, rerank_factor=2.0)
        assert int(res.indices[0]) == 0

    def test_searcher_on_ip_graph_without_external_data(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(80, 32)).astype(np.float32) * 3.0
        g = deglib.build(x, metric="ip", edges_per_vertex=16, extend_k=16, presort=True, threads=1)

        s32 = deglib.searcher(g, compress="32bit", threads=1)
        s16 = deglib.searcher(g, compress="16bit", threads=1)
        rr16 = deglib.reranker(g, compress="16bit", threads=1)
        s8 = deglib.searcher(g, compress="8bit", reranker=rr16, threads=1)
        g4 = deglib.compress(g, "4bit", threads=1)
        assert g4.get_metric() == Metric.Int4_InnerProduct
        assert g4.get_transformations()[0].name == "L2NormalizeTransform"

        s_u7_l2 = g.to_searcher(metric=Metric.Uint7_L2, threads=1)
        assert s_u7_l2.graph.get_metric() == Metric.Uint7_L2
        assert s_u7_l2.graph.get_transformations()[0].name == "MipsTransform"

        q = x[0]
        assert int(np.argmax(x @ q)) in s32.search(q, k=5, eps_or_ef=0.2).indices
        assert int(np.argmax(x @ q)) in s16.search(q, k=5, eps_or_ef=0.2).indices
        assert int(np.argmax(x @ q)) in s8.search(q, k=5, eps_or_ef=0.2, rerank_factor=2.0).indices

    def test_ip_graph_save_load_searcher_roundtrip(self, tmp_path):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(80, 32)).astype(np.float32) * 3.0

        g = deglib.build(x, metric="ip", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        p_fp32 = tmp_path / "ip_fp32.deg"
        g.save_graph(p_fp32)
        loaded = deglib.load_graph(p_fp32)
        assert loaded.get_metric() == Metric.FP32_InnerProduct
        assert not loaded.has_transformations()

        rr16 = deglib.reranker(loaded, compress="16bit", threads=1)
        q = x[0]
        top1_mips = int(np.argmax(x @ q))
        for comp, rf in (("32bit", 1.0), ("16bit", 1.0), ("8bit", 2.0)):
            s = deglib.searcher(loaded, compress=comp, reranker=(rr16 if comp == "8bit" else None), threads=1)
            assert top1_mips in s.search(q, k=5, eps_or_ef=0.2, rerank_factor=rf).indices

        s4 = deglib.searcher(loaded, compress="4bit", reranker=rr16, threads=1)
        assert s4.graph.get_metric() == Metric.Int4_InnerProduct
        assert s4.graph.get_transformations()[0].name == "L2NormalizeTransform"
        assert s4.search(q, k=5, eps_or_ef=0.2, rerank_factor=2.0).indices.shape == (5,)

        g8 = deglib.build(x, metric="ip", compress="8bit", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        p_int8 = tmp_path / "ip_int8.deg"
        g8.to_mutable().save_graph(p_int8)
        loaded8 = deglib.load_graph(p_int8)
        assert loaded8.get_metric() == Metric.Int8_InnerProduct
        assert [t.name for t in loaded8.get_transformations()] == ["FastHadamardRotation", "ScalarQuantizerInt8"]
        s8_loaded = deglib.searcher(loaded8, threads=1)
        assert top1_mips in s8_loaded.search(q, k=5, eps_or_ef=0.2).indices

        g4 = deglib.build(x, metric="ip", compress="4bit", edges_per_vertex=16, extend_k=16, threads=1)
        p_int4 = tmp_path / "ip_int4.deg"
        g4.to_mutable().save_graph(p_int4)
        loaded4 = deglib.load_graph(p_int4)
        assert loaded4.get_metric() == Metric.Int4_InnerProduct
        assert [t.name for t in loaded4.get_transformations()] == [
            "L2NormalizeTransform",
            "FastHadamardRotation",
            "ScalarQuantizerInt4",
        ]
        s4_loaded = deglib.searcher(loaded4, threads=1)
        assert s4_loaded.search(q, k=5, eps_or_ef=0.2).indices.shape == (5,)

        g_l2_4 = deglib.build(x, metric="l2", compress="4bit", edges_per_vertex=16, extend_k=16, threads=1)
        p_u4 = tmp_path / "l2_uint4.deg"
        g_l2_4.to_mutable().save_graph(p_u4)
        loaded_u4 = deglib.load_graph(p_u4)
        assert loaded_u4.get_metric() == Metric.Uint4_L2
        assert loaded_u4.get_transformations()[-1].name == "ScalarQuantizerUint4"
        s_u4_loaded = deglib.searcher(loaded_u4, threads=1)
        assert s_u4_loaded.search(q, k=5, eps_or_ef=0.2).indices.shape == (5,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("metric_in", ["euclidean", "cosine", "ip"])
    def test_presort_high_level_metrics(self, dtype, metric_in):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(40, 16)).astype(dtype) * 3.0
        idx = deglib.presort(x, metric=metric_in, threads=1)
        assert idx.shape == (40,)
        assert idx.dtype == np.uint32
        assert set(idx.tolist()) == set(range(40))
