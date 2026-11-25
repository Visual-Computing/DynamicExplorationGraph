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
        assert g4.is_mutable()
        assert g4.get_metric() == Metric.Uint4_L2
        assert g4.get_dim() == 17
        assert g4.get_transformations()[0].name == "MipsTransform"

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
            ("cosine", "4bit", Metric.Uint4_L2),
            ("cosine", "2bit", Metric.EVP_InnerProduct),
            ("ip", "4bit", Metric.Uint4_L2),
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
        assert g4.get_metric() == Metric.Uint4_L2
        assert g4.get_transformations()[0].name == "MipsTransform"

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
        assert s4.graph.get_metric() == Metric.Uint4_L2
        assert s4.graph.get_transformations()[0].name == "MipsTransform"
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
        p_uint4 = tmp_path / "ip_uint4.deg"
        g4.to_mutable().save_graph(p_uint4)
        loaded4 = deglib.load_graph(p_uint4)
        assert loaded4.get_metric() == Metric.Uint4_L2
        assert loaded4.get_dim() == 33
        assert loaded4.get_transformations()[0].name == "MipsTransform"
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

    def test_top_level_namespace_minimal(self):
        expected = {
            "__version__",
            "analysis",
            "builder",
            "cpu",
            "distances",
            "graph",
            "optimization",
            "search",
            "transformation",
            "tuning",
            "build",
            "presort",
            "compress",
            "reranker",
            "reranker_with_data",
            "searcher",
            "load_graph",
            "DynamicExplorationGraph",
            "Searcher",
            "Reranker",
            "SearchResults",
            "Metric",
            "OptimizationTarget",
        }
        assert set(deglib.__all__) == expected
        for low_level in ("create_empty", "GraphBuilder", "load_readonly_graph", "make_quantizer_pipeline", "InstructionSet"):
            assert not hasattr(deglib, low_level)

    @pytest.mark.parametrize("metric_in,expected_metric", [
        ("l2", Metric.FP32_L2),
        ("cosine", Metric.FP32_InnerProduct),
        ("ip", Metric.FP32_InnerProduct),
    ])
    def test_build_empty_graph_edge_cases(self, metric_in, expected_metric):
        empty = np.empty((0, 16), dtype=np.float32)
        g = deglib.build(empty, metric=metric_in, edges_per_vertex=16, threads=1)
        assert g.size() == 0
        assert g.get_dim() == 16
        assert g.get_metric() == expected_metric
        assert g.is_mutable()
        with pytest.raises(ValueError):
            deglib.compress(g, "8bit", threads=1)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("compress_mode,expected_metric", [
        ("32bit", None),
        ("16bit", Metric.FP16_L2),
        ("8bit", Metric.Uint7_L2),
        ("4bit", Metric.Uint4_L2),
        ("2bit", Metric.EVP_InnerProduct),
    ])
    def test_build_l2_compress_matrix(self, dtype, compress_mode, expected_metric):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(64, 32)).astype(dtype)
        if dtype == np.float16 and compress_mode == "32bit":
            with pytest.raises(ValueError):
                deglib.build(x, metric="l2", compress=compress_mode, edges_per_vertex=16, extend_k=16, threads=1)
            return
        g = deglib.build(x, metric="l2", compress=compress_mode, edges_per_vertex=16, extend_k=16, threads=1)
        assert g.is_mutable()
        if compress_mode == "32bit":
            assert g.get_metric() == Metric.FP32_L2
        else:
            assert g.get_metric() == expected_metric
        assert g.get_dim() == 32

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_build_cosine_auto_and_8bit_uses_predict_quantization(self, dtype):
        rng = np.random.default_rng(12)
        x = (rng.normal(size=(64, 32)) * 2.5).astype(dtype)
        g = deglib.build(x, metric="cosine", compress="8bit", edges_per_vertex=16, extend_k=16, threads=1)
        assert g.is_mutable()
        assert Metric(int(g.get_metric())) in (Metric.Uint7_L2, Metric.Int8_InnerProduct)
        assert g.get_transformations()[0].name == "L2NormalizeTransform"

    def test_build_ip_2bit_and_explicit_l2_use_mips(self):
        rng = np.random.default_rng(13)
        x = (rng.normal(size=(64, 16)) * np.linspace(1.0, 5.0, 64, dtype=np.float32)[:, None]).astype(np.float32)

        g2 = deglib.build(x, metric="ip", compress="2bit", edges_per_vertex=16, extend_k=16, threads=1)
        assert g2.get_metric() == Metric.EVP_InnerProduct
        assert g2.get_dim() == 17
        assert [t.name for t in g2.get_transformations()] == ["MipsTransform", "EvpQuantizer"]

        gu7 = deglib.build(x, metric="ip", compress=Metric.Uint7_L2, edges_per_vertex=16, extend_k=16, threads=1)
        assert gu7.get_metric() == Metric.Uint7_L2
        assert gu7.get_dim() == 17
        assert gu7.get_transformations()[0].name == "MipsTransform"

    def test_compress_edge_cases_and_reranker_modes(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(64, 64)).astype(np.float32)
        g_l2 = deglib.build(x, metric="l2", optimization_target="LowLID", edges_per_vertex=16, extend_k=16, threads=1)
        assert deglib.compress(g_l2, "32bit", threads=1) is g_l2
        g_l2_ro = g_l2.to_readonly()
        assert not deglib.compress(g_l2_ro, "8bit", threads=1).is_mutable()

        g_l2_16 = deglib.compress(g_l2, "16bit", threads=1)
        assert g_l2_16.is_mutable()
        with pytest.raises(ValueError):
            deglib.compress(g_l2_16, "32bit", threads=1)

        g_ip = deglib.build(x, metric="ip", optimization_target="LowLID", edges_per_vertex=16, extend_k=16, threads=1)
        rr8 = deglib.reranker(g_ip, compress="8bit", threads=1)
        assert rr8.metric == Metric.Int8_InnerProduct
        assert rr8.get_instruction().name == Metric.Int8_InnerProduct._to_cpp_space(64).get_instruction()

        rr_u7 = deglib.reranker(g_ip, compress="uint7", threads=1)
        assert rr_u7.metric == Metric.Uint7_InnerProduct

        # reranker_with_data label mapping edge cases (shuffled labels + missing label KeyError)
        shuffled = np.arange(64, dtype=np.uint32)[::-1]
        rr_shuf = deglib.reranker_with_data(g_ip, x[::-1], labels=shuffled, compress="16bit", threads=1)
        assert rr_shuf.get_num_base_vectors() == 64

        with pytest.raises(KeyError):
            deglib.reranker_with_data(g_ip, x[:60], labels=np.arange(60, dtype=np.uint32), compress="16bit", threads=1)

    def test_searcher_ip_uint7_with_fp16_reranker(self):
        rng = np.random.default_rng(15)
        x = (rng.normal(size=(80, 16)) * np.linspace(0.5, 8.0, 80, dtype=np.float32)[:, None]).astype(np.float32)
        g = deglib.build(x, metric="ip", edges_per_vertex=16, extend_k=16, presort=True, threads=1)
        rr = deglib.reranker(g, compress="16bit", threads=1)
        s = deglib.searcher(g, compress=Metric.Uint7_L2, reranker=rr, threads=1)
        q = x[0]
        res = s.search(q, k=5, eps_or_ef=0.2, rerank_factor=2.0, return_distances=True)
        assert int(np.argmax(x @ q)) in res.indices
        assert res.rerank_distance_count > 0

    @pytest.mark.parametrize("metric_alias,expected_metric", [
        ("l2", Metric.FP32_L2),
        ("euclidean", Metric.FP32_L2),
        ("cosine", Metric.FP32_InnerProduct),
        ("normalized", Metric.FP32_InnerProduct),
        ("ip", Metric.FP32_InnerProduct),
        ("innerproduct", Metric.FP32_InnerProduct),
        ("inner_product", Metric.FP32_InnerProduct),
        ("mips", Metric.FP32_InnerProduct),
    ])
    @pytest.mark.parametrize("presort_mode", [False, True, "indices"])
    def test_build_metric_aliases_and_presort_modes(self, metric_alias, expected_metric, presort_mode):
        rng = np.random.default_rng(16)
        x = (rng.normal(size=(48, 16)) * 3.0).astype(np.float32)
        labels = np.arange(1000, 1048, dtype=np.uint32)
        presort_arg = np.arange(48, dtype=np.uint32)[::-1] if presort_mode == "indices" else presort_mode
        g = deglib.build(
            x,
            labels=labels,
            metric=metric_alias,
            edges_per_vertex=12,
            extend_k=12,
            optimization_target="LowLID",
            presort=presort_arg,
            optimize=False,
            threads=1,
        )
        assert g.size() == 48
        assert g.get_metric() == expected_metric
        assert set(g.get_labels().tolist()) == set(labels.tolist())

    @pytest.mark.parametrize("opt_target", ["Auto", "LowLID", "HighLID", "StreamingData"])
    def test_build_optimization_targets(self, opt_target):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(48, 16)).astype(np.float32)
        g = deglib.build(x, metric="l2", optimization_target=opt_target, edges_per_vertex=12, extend_k=12, threads=1)
        assert g.size() == 48

    def test_build_invalid_inputs_raise(self):
        rng = np.random.default_rng(18)
        x_f32 = rng.normal(size=(32, 16)).astype(np.float32)
        x_i8 = rng.integers(-64, 64, size=(32, 16), dtype=np.int8)

        with pytest.raises(ValueError):
            deglib.build(x_i8, metric=Metric.Int8_L2, compress="8bit", threads=1)
        with pytest.raises(ValueError):
            deglib.build(x_i8, metric=Metric.Int8_L2, presort=True, threads=1)
        with pytest.raises(ValueError):
            deglib.build(x_f32, labels=np.arange(10, dtype=np.uint32), threads=1)
        with pytest.raises(TypeError):
            deglib.build(x_f32, labels=np.arange(32, dtype=np.float32), threads=1)
        with pytest.raises(Exception):
            deglib.build(x_f32[0], threads=1)
        with pytest.raises(ValueError):
            deglib.build(x_f32, metric="unknown_metric", threads=1)
        with pytest.raises(ValueError):
            deglib.build(x_f32, metric="l2", compress="unknown_compress", threads=1)

    @pytest.mark.parametrize("src_kind", ["fp32_l2", "fp16_l2", "fp32_ip_unit", "fp32_ip_mips", "fp16_ip_mips"])
    @pytest.mark.parametrize("target", ["16bit", "8bit", "auto", "4bit", "2bit", "evp", Metric.Uint7_L2, Metric.Int8_L2])
    def test_compress_full_matrix_preserves_mutability(self, src_kind, target):
        rng = np.random.default_rng(19)
        raw = rng.normal(size=(48, 32)).astype(np.float32)
        if "unit" in src_kind:
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        elif "mips" in src_kind:
            raw = raw * np.linspace(0.5, 4.0, 48, dtype=np.float32)[:, None]

        arr = raw.astype(np.float16) if src_kind.startswith("fp16") else raw
        metric_arg = "l2" if "l2" in src_kind else ("cosine" if "unit" in src_kind else "ip")
        g = deglib.build(arr, metric=metric_arg, optimization_target="LowLID", edges_per_vertex=12, extend_k=12, threads=1)

        c_mut = deglib.compress(g, target, threads=1)
        assert c_mut.is_mutable()

        c_ro = deglib.compress(g.to_readonly(), target, threads=1)
        assert not c_ro.is_mutable()
        assert c_ro.get_metric() == c_mut.get_metric()
        assert c_ro.get_dim() == c_mut.get_dim()

        if "ip_mips" in src_kind and isinstance(target, Metric):
            assert c_mut.get_transformations()[0].name == "MipsTransform"
            assert c_mut.get_dim() == 33
        elif "ip_unit" in src_kind and isinstance(target, Metric):
            assert all(t.name != "MipsTransform" for t in c_mut.get_transformations())
            assert c_mut.get_dim() == 32

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_compress_8bit_dispatches_by_metric_and_unit_norm(self, dtype):
        rng = np.random.default_rng(21)
        raw = rng.normal(size=(64, 32)).astype(np.float32)

        # 1. L2 -> always Uint7_L2
        g_l2 = deglib.build(raw.astype(dtype), metric="l2", edges_per_vertex=12, extend_k=12, threads=1)
        assert deglib.compress(g_l2, "8bit", threads=1).get_metric() == Metric.Uint7_L2

        # 2. Unnormalized IP -> Int8_InnerProduct
        raw_ip = (raw * np.linspace(1.0, 5.0, 64, dtype=np.float32)[:, None]).astype(dtype)
        g_ip = deglib.build(raw_ip, metric="ip", edges_per_vertex=12, extend_k=12, threads=1)
        assert deglib.compress(g_ip, "8bit", threads=1).get_metric() == Metric.Int8_InnerProduct

        # 3. Cosine / normalized (unit sphere) -> matches predict_quantization
        unit = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(dtype)
        g_cos = deglib.build(unit, metric="normalized", edges_per_vertex=12, extend_k=12, threads=1)
        expected_metric = deglib.tuning.predict_quantization(g_cos, threads=1).metric
        assert expected_metric in (Metric.Uint7_L2, Metric.Int8_InnerProduct)
        assert deglib.compress(g_cos, "8bit", threads=1).get_metric() == expected_metric
        assert deglib.compress(g_cos, "auto", threads=1).get_metric() == expected_metric

    @pytest.mark.parametrize("base_metric", ["l2", "ip"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("rr_compress", [None, "32bit", "16bit", "8bit", "int8", "uint7", "uint8"])
    def test_reranker_all_compress_modes(self, base_metric, dtype, rr_compress):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(48, 64)).astype(dtype)
        g = deglib.build(x, metric=base_metric, optimization_target="LowLID", edges_per_vertex=12, extend_k=12, threads=1)
        rr = deglib.reranker(g, compress=rr_compress, threads=1)
        assert rr.get_num_base_vectors() == 48
        if base_metric == "ip" and rr_compress in ("8bit", "int8", "uint8"):
            expected_isa = rr.metric._to_cpp_space(64).get_instruction()
            assert rr.get_instruction().name == expected_isa

    @pytest.mark.parametrize("graph_compress", ["8bit", "4bit", "2bit"])
    @pytest.mark.parametrize("rr_metric", [None, "l2", "cosine", "ip"])
    @pytest.mark.parametrize("label_mode", ["none", "sorted_sparse", "shuffled"])
    def test_reranker_with_data_matrix(self, graph_compress, rr_metric, label_mode):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(60, 32)).astype(np.float32)
        if label_mode == "none":
            g = deglib.build(x, metric="cosine", compress=graph_compress, edges_per_vertex=12, extend_k=12, threads=1)
            rr = deglib.reranker_with_data(g, x, compress="16bit", metric=rr_metric, threads=1)
        elif label_mode == "sorted_sparse":
            lbls = np.arange(100, 220, 2, dtype=np.uint32)
            g = deglib.build(x, labels=lbls, metric="cosine", compress=graph_compress, edges_per_vertex=12, extend_k=12, threads=1)
            rr = deglib.reranker_with_data(g, x, labels=lbls, compress="16bit", metric=rr_metric, threads=1)
        else:
            lbls = rng.permutation(np.arange(10, 70, dtype=np.uint32))
            g = deglib.build(x, labels=lbls, metric="cosine", compress=graph_compress, edges_per_vertex=12, extend_k=12, threads=1)
            rr = deglib.reranker_with_data(g, x, labels=lbls, compress="16bit", metric=rr_metric, threads=1)
        assert rr.get_num_base_vectors() == 60

    @pytest.mark.parametrize("search_compress", [None, "16bit", "8bit", "auto", "4bit", "2bit", Metric.Uint7_L2])
    @pytest.mark.parametrize("optimize", [True, False])
    @pytest.mark.parametrize("eps_or_ef,rerank_factor,return_distances,unsorted", [
        (0.2, 1.0, False, True),
        (16.0, 2.0, True, False),
    ])
    def test_searcher_full_parameter_matrix(self, search_compress, optimize, eps_or_ef, rerank_factor, return_distances, unsorted):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(64, 32)).astype(np.float32)
        g = deglib.build(x, metric="cosine", edges_per_vertex=12, extend_k=12, threads=1)
        rr = deglib.reranker(g, compress="16bit", threads=1) if rerank_factor > 1.0 else None
        s = deglib.searcher(g, compress=search_compress, reranker=rr, optimize=optimize, threads=1)
        assert not s.graph.is_mutable()
        assert "backend=MutableGraph" in repr(g)
        assert "backend=ReadOnlyGraph" in repr(s.graph)
        assert "ReadOnlyGraph" in repr(s)
        if search_compress is None:
            g_ro = g.to_readonly()
            assert deglib.searcher(g_ro, optimize=False, threads=1).graph is g_ro

        r1 = s.search(x[0], k=5, eps_or_ef=eps_or_ef, rerank_factor=rerank_factor, return_distances=return_distances, unsorted=unsorted)
        assert r1.indices.shape == (5,)
        assert (r1.distances is not None) == return_distances

        rb = s.search(x[:4], k=5, eps_or_ef=eps_or_ef, rerank_factor=rerank_factor, threads=2, return_distances=return_distances, unsorted=unsorted)
        assert rb.indices.shape == (4, 5)
        assert (rb.distances is not None) == return_distances
