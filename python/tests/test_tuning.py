import numpy as np
import pytest

import deglib
from deglib.builder import OptimizationTarget
from deglib.distances import Metric
from deglib.tuning import (
    DatasetProfile,
    QuantizationPrediction,
    TargetPrediction,
    measure_dataset_profile,
    predict_optimization_target,
    predict_quantization,
    predict_quantization_normalized,
)


def _make_vectors(n: int = 300, d: int = 32, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def test_tuning_f32_and_profile_reuse():
    data = _make_vectors(350, 24, seed=11)
    profile = measure_dataset_profile(data, metric=Metric.FP32_L2, sample_count=300, query_count=200, seed=7, threads=2)
    assert isinstance(profile, DatasetProfile)
    assert profile.sample_count == 300
    assert profile.dim == 24
    assert isinstance(profile.metric, Metric)
    assert profile.metric == Metric.FP32_L2
    assert 0.0 < profile.participation_ratio <= 1.0
    assert profile.rel_contrast_1 > 0.0
    assert profile.pp_iqr > 0.0
    assert np.isfinite(profile.d16_skew)
    assert profile.ball_cnt_p90_p10_r15 >= 1.0
    assert np.isfinite(profile.hadamard_kurt)

    pred_from_profile = predict_optimization_target(profile)
    pred_direct = predict_optimization_target(
        data, metric=Metric.FP32_L2, sample_count=300, query_count=200, seed=7, threads=2
    )
    assert isinstance(pred_from_profile, TargetPrediction)
    assert isinstance(pred_from_profile.target, OptimizationTarget)
    assert pred_from_profile.profile.metric == Metric.FP32_L2
    assert pred_from_profile.target == pred_direct.target
    assert pred_from_profile.lowlid_advantage_pct == pytest.approx(pred_direct.lowlid_advantage_pct, abs=1e-6)
    assert pred_from_profile.target.name in str(pred_from_profile)

    qpred_direct = predict_quantization(
        data, metric=Metric.FP32_L2, sample_count=300, query_count=200, seed=7, threads=2
    )
    assert isinstance(qpred_direct, QuantizationPrediction)
    assert isinstance(qpred_direct.metric, Metric)
    assert qpred_direct.metric == Metric.Uint7_L2
    assert qpred_direct.uint7_advantage_pct == 0.0
    assert "L2" in str(qpred_direct)

    # Module re-exports match deglib.tuning
    assert deglib.tuning.measure_dataset_profile is measure_dataset_profile
    assert deglib.tuning.predict_optimization_target is predict_optimization_target
    assert deglib.tuning.predict_quantization is predict_quantization
    assert deglib.tuning.predict_quantization_normalized is predict_quantization_normalized
    assert deglib.tuning.DatasetProfile is DatasetProfile
    assert deglib.tuning.TargetPrediction is TargetPrediction
    assert deglib.tuning.QuantizationPrediction is QuantizationPrediction


def test_tuning_fp16_matches_fp32():
    f32 = _make_vectors(300, 32, seed=21)
    f16 = f32.astype(np.float16)

    p32 = predict_optimization_target(f32, metric=Metric.FP32_L2, sample_count=250, query_count=150, seed=7, threads=2)
    p16 = predict_optimization_target(f16, metric=Metric.FP32_L2, sample_count=250, query_count=150, seed=7, threads=2)

    assert p16.profile.sample_count == p32.profile.sample_count
    assert p16.profile.dim == p32.profile.dim
    assert p16.profile.participation_ratio == pytest.approx(p32.profile.participation_ratio, abs=0.01)
    assert p16.profile.rel_contrast_1 == pytest.approx(p32.profile.rel_contrast_1, abs=0.01)
    assert p16.profile.pp_iqr == pytest.approx(p32.profile.pp_iqr, abs=0.25)
    assert p16.profile.d16_skew == pytest.approx(p32.profile.d16_skew, abs=0.05)
    assert p16.profile.ball_cnt_p90_p10_r15 == pytest.approx(p32.profile.ball_cnt_p90_p10_r15, abs=0.1)
    assert p16.profile.hadamard_kurt == pytest.approx(p32.profile.hadamard_kurt, abs=0.02)
    assert p16.lowlid_advantage_pct == pytest.approx(p32.lowlid_advantage_pct, abs=0.5)


def test_tuning_string_metrics_and_inner_product_invariance():
    base = _make_vectors(250, 16, seed=31)
    scales = np.random.default_rng(32).uniform(0.2, 15.0, size=(250, 1)).astype(np.float32)
    rescaled = base * scales

    p_enum = measure_dataset_profile(base, metric=Metric.FP32_InnerProduct, sample_count=0, query_count=0, threads=2)
    p_str = measure_dataset_profile(rescaled, metric="fp32_cosine", sample_count=0, query_count=0, threads=2)
    p_l2_str = measure_dataset_profile(base, metric="fp32_l2", sample_count=0, query_count=0, threads=2)

    assert p_str.participation_ratio == pytest.approx(p_enum.participation_ratio, abs=1e-5)
    assert p_str.rel_contrast_1 == pytest.approx(p_enum.rel_contrast_1, abs=1e-5)
    assert p_str.pp_iqr == pytest.approx(p_enum.pp_iqr, abs=1e-2)
    assert p_str.d16_skew == pytest.approx(p_enum.d16_skew, abs=1e-4)
    assert p_str.ball_cnt_p90_p10_r15 == pytest.approx(p_enum.ball_cnt_p90_p10_r15, abs=1e-4)
    assert p_str.hadamard_kurt == pytest.approx(p_enum.hadamard_kurt, abs=1e-4)
    assert p_enum.metric == Metric.FP32_InnerProduct
    assert p_str.metric == Metric.FP32_InnerProduct
    assert p_l2_str.metric == Metric.FP32_L2
    assert p_l2_str.sample_count == 250

    qn_base = predict_quantization_normalized(base, is_cosine=False, metric=Metric.FP32_InnerProduct, sample_count=0, threads=2)
    qn_scaled = predict_quantization_normalized(rescaled, is_cosine=False, metric="fp32_cosine", sample_count=0, threads=2)
    qn_cos = predict_quantization_normalized(base, is_cosine=True, metric="fp32_cosine", sample_count=0, threads=2)
    q_l2 = predict_quantization(base, metric="fp32_l2", sample_count=0, query_count=0, threads=2)
    qn_l2 = predict_quantization_normalized(base, is_cosine=False, metric="fp32_l2", sample_count=0, threads=2)
    assert qn_scaled.metric == qn_base.metric
    assert qn_scaled.uint7_advantage_pct == pytest.approx(qn_base.uint7_advantage_pct, abs=1e-4)
    assert qn_cos.uint7_advantage_pct > qn_base.uint7_advantage_pct
    assert q_l2.metric == Metric.Uint7_L2
    assert q_l2.uint7_advantage_pct == 0.0
    assert qn_l2.metric == Metric.Uint7_L2
    assert qn_l2.uint7_advantage_pct == 0.0


def test_tuning_determinism_threads_and_non_contiguous():
    data = _make_vectors(400, 32, seed=41)
    non_contig = data[::2, ::2]
    assert not non_contig.flags.c_contiguous

    p1 = predict_optimization_target(
        data, metric=Metric.FP32_InnerProduct, sample_count=300, query_count=200, seed=7, threads=1
    )
    p4 = predict_optimization_target(
        data, metric=Metric.FP32_InnerProduct, sample_count=300, query_count=200, seed=7, threads=4
    )
    assert p1.target == p4.target
    assert p1.profile.participation_ratio == pytest.approx(p4.profile.participation_ratio, abs=1e-5)
    assert p1.profile.rel_contrast_1 == pytest.approx(p4.profile.rel_contrast_1, abs=1e-5)
    assert p1.profile.pp_iqr == pytest.approx(p4.profile.pp_iqr, abs=1e-2)
    assert p1.profile.d16_skew == pytest.approx(p4.profile.d16_skew, abs=1e-4)
    assert p1.profile.ball_cnt_p90_p10_r15 == pytest.approx(p4.profile.ball_cnt_p90_p10_r15, abs=1e-4)
    assert p1.profile.hadamard_kurt == pytest.approx(p4.profile.hadamard_kurt, abs=1e-4)

    p_slice = predict_optimization_target(
        non_contig, metric=Metric.FP32_L2, sample_count=0, query_count=0, seed=7, threads=2
    )
    assert p_slice.profile.sample_count == 200
    assert p_slice.profile.dim == 16
    assert np.isfinite(p_slice.lowlid_advantage_pct)


def test_tuning_duplicates_and_edge_cases():
    clean = _make_vectors(300, 16, seed=51)
    with_dups = np.vstack([clean, clean[:15]])

    p_clean = measure_dataset_profile(clean, metric=Metric.FP32_L2, sample_count=0, query_count=0, threads=2)
    p_dups = measure_dataset_profile(with_dups, metric=Metric.FP32_L2, sample_count=0, query_count=0, threads=2)
    assert p_dups.pp_iqr == pytest.approx(p_clean.pp_iqr, abs=10.0)
    assert p_dups.d16_skew == pytest.approx(p_clean.d16_skew, abs=0.5)
    assert p_dups.ball_cnt_p90_p10_r15 == pytest.approx(p_clean.ball_cnt_p90_p10_r15, abs=1.0)
    assert p_dups.hadamard_kurt == pytest.approx(p_clean.hadamard_kurt, abs=0.05)
    assert p_dups.rel_contrast_1 == pytest.approx(p_clean.rel_contrast_1, abs=0.05)

    empty = np.empty((0, 8), dtype=np.float32)
    pred_empty = predict_optimization_target(empty)
    assert pred_empty.target == OptimizationTarget.StreamingData
    assert pred_empty.lowlid_advantage_pct == 0.0

    single = np.ones((1, 8), dtype=np.float32)
    pred_single = predict_optimization_target(single)
    assert pred_single.target == OptimizationTarget.StreamingData
    assert pred_single.lowlid_advantage_pct == 0.0


def test_tuning_invalid_inputs():
    for fn in (measure_dataset_profile, predict_optimization_target, predict_quantization):
        with pytest.raises(ValueError):
            fn(np.ones((10,), dtype=np.float32))
        with pytest.raises(ValueError):
            fn(np.ones((2, 4, 8), dtype=np.float32))
        with pytest.raises(ValueError):
            fn(np.ones((10, 0), dtype=np.float32))
        with pytest.raises(TypeError):
            fn(np.ones((10, 8), dtype=np.float64))
        with pytest.raises(TypeError):
            fn(np.ones((10, 8), dtype=np.int32))
        with pytest.raises(ValueError):
            fn(_make_vectors(20, 8), sample_count=-1)
        with pytest.raises(ValueError):
            fn(_make_vectors(20, 8), query_count=-1)
        with pytest.raises(ValueError):
            fn(_make_vectors(20, 8), seed=-1)
        with pytest.raises(ValueError):
            fn(_make_vectors(20, 8), threads=-1)


def test_predict_quantization_fp16_threads_slice_and_edge_cases():
    f32 = _make_vectors(400, 32, seed=61)
    f16 = f32.astype(np.float16)

    q32 = predict_quantization(
        f32, metric=Metric.FP32_InnerProduct, sample_count=250, query_count=150, seed=7, threads=2
    )
    q16 = predict_quantization(
        f16, metric=Metric.FP16_InnerProduct, sample_count=250, query_count=150, seed=7, threads=2
    )
    assert q16.profile.metric == Metric.FP16_InnerProduct
    assert q16.metric == q32.metric
    assert q16.uint7_advantage_pct == pytest.approx(q32.uint7_advantage_pct, abs=0.25)

    qn32 = predict_quantization_normalized(
        f32, is_cosine=True, metric=Metric.FP32_InnerProduct, sample_count=250, seed=7, threads=2
    )
    qn16 = predict_quantization_normalized(
        f16, is_cosine=True, metric=Metric.FP16_InnerProduct, sample_count=250, seed=7, threads=2
    )
    assert qn16.profile.metric == Metric.FP16_InnerProduct
    assert qn16.metric == qn32.metric
    assert qn16.uint7_advantage_pct == pytest.approx(qn32.uint7_advantage_pct, abs=0.25)

    q_t1 = predict_quantization(
        f32, metric=Metric.FP32_InnerProduct, sample_count=250, query_count=150, seed=7, threads=1
    )
    q_t4 = predict_quantization(
        f32, metric=Metric.FP32_InnerProduct, sample_count=250, query_count=150, seed=7, threads=4
    )
    assert q_t1.metric == q_t4.metric
    assert q_t1.uint7_advantage_pct == pytest.approx(q_t4.uint7_advantage_pct, abs=1e-4)

    non_contig = f32[::2, ::2]
    assert not non_contig.flags.c_contiguous
    q_slice = predict_quantization(
        non_contig, metric=Metric.FP32_InnerProduct, sample_count=0, query_count=0, seed=7, threads=2
    )
    assert q_slice.profile.sample_count == 200
    assert q_slice.profile.dim == 16
    assert np.isfinite(q_slice.uint7_advantage_pct)

    for edge in (
        np.empty((0, 8), dtype=np.float32),
        np.ones((1, 8), dtype=np.float32),
        np.full((64, 8), 3.5, dtype=np.float32),
    ):
        pred = predict_quantization(edge, metric=Metric.FP32_InnerProduct)
        assert pred.metric == Metric.Uint7_L2
        assert pred.uint7_advantage_pct == 0.0
        pred_n = predict_quantization_normalized(edge, is_cosine=False, metric=Metric.FP32_InnerProduct)
        assert pred_n.metric == Metric.Uint7_L2
        assert pred_n.uint7_advantage_pct == 0.0


def test_reranker_8bit_predictor_dispatch():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(64, 32)).astype(np.float32)

    g_l2 = deglib.build(x, metric="l2", edges_per_vertex=12, extend_k=12, threads=1)
    assert deglib.reranker(g_l2, compress="8bit", threads=1).metric == Metric.Uint7_L2
    assert deglib.reranker_with_data(g_l2, x, compress="8bit", threads=1).metric == Metric.Uint7_L2

    g_ip = deglib.build(x, metric="ip", edges_per_vertex=12, extend_k=12, threads=1)
    assert deglib.reranker(g_ip, compress="8bit", threads=1).metric == Metric.Int8_InnerProduct
    assert deglib.reranker_with_data(g_ip, x, compress="8bit", metric="ip", threads=1).metric == Metric.Int8_InnerProduct

    g_cos = deglib.build(x, metric="cosine", edges_per_vertex=12, extend_k=12, threads=1)
    rr_cos = deglib.reranker(g_cos, compress="8bit", threads=1)
    rr_cos_data = deglib.reranker_with_data(g_cos, x, compress="8bit", metric="cosine", threads=1)
    assert rr_cos.metric in (Metric.Uint7_L2, Metric.Int8_InnerProduct)
    assert rr_cos_data.metric in (Metric.Uint7_L2, Metric.Int8_InnerProduct)
