import math

import torch

from quant_videogen.compress import (
    QuantizeFunctions,
    compress_kv_cache,
    get_quantize_fn,
    get_quantize_type,
)
from quant_videogen.hqmq_fake import FakeHQMQCacheAdapter
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.uncompress import uncompress_kv_cache


def _small_config(quant_type: str) -> QuantizeConfig:
    return QuantizeConfig(
        quant_type=quant_type,
        hqmq_secondary_size=8,
        hqmq_radius_bits=3,
        hqmq_seed=123,
    )


def test_hqmq_fake_round_trip_shape_dtype_and_diagnostics():
    torch.manual_seed(0)
    config = _small_config("hqmq-fake")
    quantize_fn = get_quantize_fn(config.quant_type, config)
    assert isinstance(quantize_fn, FakeHQMQCacheAdapter)

    k = torch.randn(1, 2, 5, 10, dtype=torch.float32)
    v = torch.randn(1, 2, 5, 10, dtype=torch.float32)
    k_quant, v_quant = compress_kv_cache(k, v, config.quant_type, config, quantize_fn, layer_idx=1)
    k_round, v_round = uncompress_kv_cache(k_quant, v_quant)

    assert k_round.shape == k.shape
    assert v_round.shape == v.shape
    assert k_round.dtype == k.dtype
    assert v_round.dtype == v.dtype
    assert torch.isfinite(k_round).all()
    assert torch.isfinite(v_round).all()

    assert len(quantize_fn.last_diagnostics) == 2
    for diagnostic in quantize_fn.last_diagnostics:
        assert diagnostic["finite"] is True
        assert math.isfinite(diagnostic["relative_l2"])
        assert diagnostic["relative_l2"] >= 0
        assert diagnostic["padded_head_dim"] == 12


def test_hqmq_fake_is_deterministic_for_fixed_seed():
    torch.manual_seed(1)
    config = _small_config("hqmq-fake")
    k = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float32)

    fn_a = get_quantize_fn(config.quant_type, config)
    fn_b = get_quantize_fn(config.quant_type, config)
    k_a, v_a = compress_kv_cache(k, v, config.quant_type, config, fn_a, layer_idx=2)
    k_b, v_b = compress_kv_cache(k, v, config.quant_type, config, fn_b, layer_idx=2)

    assert torch.allclose(k_a, k_b)
    assert torch.allclose(v_a, v_b)


def test_hqmq_fake_med3x_records_outlier_rate():
    torch.manual_seed(2)
    config = _small_config("hqmq-fake-med3x")
    quantize_fn = get_quantize_fn(config.quant_type, config)
    k = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    k[:, :, 0, :4] *= 100

    compress_kv_cache(k, v, config.quant_type, config, quantize_fn, layer_idx=0)
    rates = [d["outlier_rate"] for d in quantize_fn.last_diagnostics]

    assert len(rates) == 2
    assert all(rate is not None and math.isfinite(rate) for rate in rates)
    assert all(0.0 <= rate <= 1.0 for rate in rates)


def test_existing_qvg_dispatch_names_still_classify():
    expected = {
        "naive-int4": QuantizeFunctions.NAIVE,
        "kmeans-int4": QuantizeFunctions.KMEANS,
        "nstages-kmeans-int2": QuantizeFunctions.NSTAGE_KMEANS,
        "triton-nstages-kmeans-int2": QuantizeFunctions.TRITON_PRQ,
        "hqmq-fake": QuantizeFunctions.HQMQ_FAKE,
        "hqmq-fake-med3x": QuantizeFunctions.HQMQ_FAKE,
    }
    for quant_type, quantize_type in expected.items():
        assert get_quantize_type(quant_type) == quantize_type

    for quant_type in ("naive-int4", "triton-nstages-kmeans-int2"):
        config = QuantizeConfig(quant_type=quant_type)
        assert callable(get_quantize_fn(quant_type, config))


if __name__ == "__main__":
    test_hqmq_fake_round_trip_shape_dtype_and_diagnostics()
    test_hqmq_fake_is_deterministic_for_fixed_seed()
    test_hqmq_fake_med3x_records_outlier_rate()
    test_existing_qvg_dispatch_names_still_classify()
