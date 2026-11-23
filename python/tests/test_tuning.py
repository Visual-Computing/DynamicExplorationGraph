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
    assert 0.0 < profile.q_mode_u7 <= 1.0
    assert 0.0 < profile.q_mode_u4 <= 1.0

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

    qpred_from_profile = predict_quantization(profile)
    qpred_direct = predict_quantization(
        data, metric=Metric.FP32_L2, sample_count=300, query_count=200, seed=7, threads=2
    )
    assert isinstance(qpred_from_profile, QuantizationPrediction)
    assert isinstance(qpred_from_profile.metric, Metric)
    assert qpred_from_profile.metric == Metric.Uint7_L2
    assert qpred_from_profile.uint7_advantage_pct == 0.0
    assert qpred_from_profile == qpred_direct
    assert "L2" in str(qpred_from_profile)

    # Top-level deglib re-exports match deglib.tuning
    assert deglib.measure_dataset_profile is measure_dataset_profile
    assert deglib.predict_optimization_target is predict_optimization_target
    assert deglib.predict_quantization is predict_quantization
    assert deglib.DatasetProfile is DatasetProfile
    assert deglib.TargetPrediction is TargetPrediction
    assert deglib.QuantizationPrediction is QuantizationPrediction


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
    assert p16.profile.q_mode_u7 == pytest.approx(p32.profile.q_mode_u7, abs=0.005)
    assert p16.profile.q_mode_u4 == pytest.approx(p32.profile.q_mode_u4, abs=0.01)
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
    assert p_str.pp_iqr == pytest.approx(p_enum.pp_iqr, abs=1e-4)
    assert p_str.q_mode_u7 == pytest.approx(p_enum.q_mode_u7, abs=1e-5)
    assert p_str.q_mode_u4 == pytest.approx(p_enum.q_mode_u4, abs=1e-5)
    assert p_enum.metric == Metric.FP32_InnerProduct
    assert p_str.metric == Metric.FP32_InnerProduct
    assert p_l2_str.metric == Metric.FP32_L2
    assert p_l2_str.sample_count == 250

    q_enum = predict_quantization(p_enum)
    q_str = predict_quantization(rescaled, metric="fp32_cosine", sample_count=0, query_count=0, threads=2)
    q_l2 = predict_quantization(base, metric="fp32_l2", sample_count=0, query_count=0, threads=2)
    assert q_str.metric == q_enum.metric
    assert q_str.uint7_advantage_pct == pytest.approx(q_enum.uint7_advantage_pct, abs=1e-4)
    assert q_l2.metric == Metric.Uint7_L2
    assert q_l2.uint7_advantage_pct == 0.0


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
    assert p1.profile.pp_iqr == pytest.approx(p4.profile.pp_iqr, abs=1e-4)
    assert p1.profile.q_mode_u7 == pytest.approx(p4.profile.q_mode_u7, abs=1e-5)
    assert p1.profile.q_mode_u4 == pytest.approx(p4.profile.q_mode_u4, abs=1e-5)

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
    assert p_dups.q_mode_u7 == pytest.approx(p_clean.q_mode_u7, abs=0.02)
    assert p_dups.q_mode_u4 == pytest.approx(p_clean.q_mode_u4, abs=0.02)
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


def test_predict_quantization_vibe_reference_profiles():
    refs = [
        (
            "laion-clip",
            DatasetProfile(16000, 512, 0.174291, 1.907535, 68.472295, 0.028788, 0.223782, Metric.FP32_InnerProduct),
            Metric.Int8_InnerProduct,
            -2.22,
        ),
        (
            "yandex",
            DatasetProfile(16000, 200, 0.551745, 1.963387, 49.415739, 0.025665, 0.199440, Metric.FP32_InnerProduct),
            Metric.Int8_InnerProduct,
            -3.55,
        ),
        (
            "imagenet-align",
            DatasetProfile(16000, 640, 0.214643, 2.047639, 54.426652, 0.026883, 0.208970, Metric.FP32_InnerProduct),
            Metric.Int8_InnerProduct,
            -3.79,
        ),
        (
            "gooaq-distilroberta",
            DatasetProfile(16000, 768, 0.287027, 1.592527, 55.598128, 0.027625, 0.205864, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            1.33,
        ),
        (
            "yahoo-minilm",
            DatasetProfile(16000, 384, 0.390076, 1.578273, 67.708323, 0.031112, 0.206428, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            1.02,
        ),
        (
            "agnews-mxbai",
            DatasetProfile(16000, 1024, 0.121258, 1.614592, 67.004191, 0.026256, 0.204236, Metric.FP32_L2),
            Metric.Uint7_L2,
            0.0,
        ),
        (
            "ccnews-nomic",
            DatasetProfile(16000, 768, 0.149226, 1.730994, 75.723209, 0.026181, 0.203433, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            0.38,
        ),
        (
            "landmark-dino",
            DatasetProfile(16000, 768, 0.120246, 1.728638, 86.464614, 0.026126, 0.203020, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            0.79,
        ),
        (
            "arxiv-nomic",
            DatasetProfile(16000, 768, 0.081903, 1.670677, 71.741581, 0.025987, 0.201974, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            2.33,
        ),
        (
            "inaturalist-resnet",
            DatasetProfile(16000, 2048, 0.033901, 2.115867, 65.657329, 0.713799, 0.892152, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            12.14,
        ),
        (
            "celeba-resnet",
            DatasetProfile(16000, 2048, 0.010431, 2.179041, 81.038963, 0.666204, 0.839986, Metric.FP32_InnerProduct),
            Metric.Uint7_L2,
            63.02,
        ),
    ]

    advs = []
    for name, profile, expected_metric, expected_adv in refs:
        pred = predict_quantization(profile)
        assert pred.metric == expected_metric, name
        assert pred.uint7_advantage_pct == pytest.approx(expected_adv, abs=0.20), name
        assert pred.metric.name in str(pred)
        advs.append(pred.uint7_advantage_pct)

    # celeba > inaturalist > arxiv > gooaq > yahoo > landmark > ccnews > 0 > laion > yandex > imagenet
    assert advs[10] > advs[9] > advs[8] > advs[3] > advs[4] > advs[7] > advs[6] > 0.0 > advs[0] > advs[1] > advs[2]

    # L2 rule overrides high-CV geometry unconditionally
    l2_override = predict_quantization(
        DatasetProfile(16000, 640, 0.214643, 2.047639, 54.426652, 0.026883, 0.208970, "fp32_l2")
    )
    assert l2_override.metric == Metric.Uint7_L2
    assert l2_override.uint7_advantage_pct == 0.0


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
    assert q16.uint7_advantage_pct == pytest.approx(q32.uint7_advantage_pct, abs=0.2)

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
