import math
import importlib.util
from pathlib import Path
import sys
import types

import torch

from quant_videogen.compress import (
    QuantizeFunctions,
    compress_kv_cache,
    get_quantize_fn,
    get_quantize_type,
)
from quant_videogen.hqmq_fake import FakeHQMQCacheAdapter
from quant_videogen.kv_cache import ChunkedKVCache
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.uncompress import uncompress_kv_cache


def _small_config(quant_type: str, **overrides) -> QuantizeConfig:
    kwargs = {
        "quant_type": quant_type,
        "hqmq_secondary_size": 8,
        "hqmq_radius_bits": 3,
        "hqmq_seed": 123,
    }
    kwargs.update(overrides)
    return QuantizeConfig(
        **kwargs,
    )


def _dense_quantize_padded_reference(
    adapter: FakeHQMQCacheAdapter,
    x: torch.Tensor,
    *,
    layer_idx: int,
    kv: int,
) -> torch.Tensor:
    bsz, heads, seq_len, head_dim = x.shape
    n_chunks = head_dim // adapter.chunk_dim
    chunked = x.reshape(bsz, heads, seq_len, n_chunks, adapter.chunk_dim)
    radius = chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction = chunked / radius

    joint = adapter._joint_codebook(layer_idx=layer_idx, kv=kv, heads=heads, device=x.device)
    sims = torch.einsum("bhtnc,hkc->bhtnk", direction, joint)
    idx = sims.argmax(dim=-1)
    joint_expanded = joint.view(1, heads, 1, 1, joint.shape[1], adapter.chunk_dim).expand(
        bsz, heads, seq_len, n_chunks, joint.shape[1], adapter.chunk_dim
    )
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, adapter.chunk_dim)
    direction_quant = joint_expanded.gather(dim=-2, index=gather_idx).squeeze(-2)
    radius_quant = adapter._quantize_radius(radius)
    return (radius_quant * direction_quant).reshape(bsz, heads, seq_len, head_dim)


def _load_self_forcing_causal_inference_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "Self-Forcing"
        / "pipeline"
        / "causal_inference.py"
    )

    previous_modules = {}
    stub_names = [
        "utils",
        "utils.wan_wrapper",
        "demo_utils",
        "demo_utils.memory",
        "termcolor",
    ]
    for name in stub_names:
        previous_modules[name] = sys.modules.get(name)

    utils_module = types.ModuleType("utils")
    wan_wrapper_module = types.ModuleType("utils.wan_wrapper")
    wan_wrapper_module.WanDiffusionWrapper = object
    wan_wrapper_module.WanTextEncoder = object
    wan_wrapper_module.WanVAEWrapper = object

    demo_utils_module = types.ModuleType("demo_utils")
    memory_module = types.ModuleType("demo_utils.memory")
    memory_module.gpu = None
    memory_module.get_cuda_free_memory_gb = lambda *_args, **_kwargs: 0
    memory_module.DynamicSwapInstaller = object
    memory_module.move_model_to_device_with_memory_preservation = (
        lambda *_args, **_kwargs: None
    )
    termcolor_module = types.ModuleType("termcolor")
    termcolor_module.cprint = lambda *args, **_kwargs: print(*args)

    sys.modules["utils"] = utils_module
    sys.modules["utils.wan_wrapper"] = wan_wrapper_module
    sys.modules["demo_utils"] = demo_utils_module
    sys.modules["demo_utils.memory"] = memory_module
    sys.modules["termcolor"] = termcolor_module

    try:
        spec = importlib.util.spec_from_file_location(
            "self_forcing_causal_inference_for_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


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


def test_hqmq_fake_tiled_matches_dense_reference():
    torch.manual_seed(3)
    config = _small_config("hqmq-fake", hqmq_max_sim_elements=128)
    adapter = get_quantize_fn(config.quant_type, config)
    x = torch.randn(1, 2, 7, 12, dtype=torch.float32)

    tiled = adapter._quantize_padded(x, layer_idx=1, kv=0, heads=2)
    dense = _dense_quantize_padded_reference(adapter, x, layer_idx=1, kv=0)

    assert torch.equal(tiled, dense)
    assert adapter.last_tile_stats is not None
    assert adapter.last_tile_stats["sim_tile_count"] > 1
    assert adapter.last_tile_stats["max_sim_tile_elements"] <= config.hqmq_max_sim_elements
    assert adapter.last_tile_stats["dense_sim_elements"] > config.hqmq_max_sim_elements


def test_hqmq_fake_diagnostics_record_bounded_tiles():
    torch.manual_seed(4)
    config = _small_config("hqmq-fake", hqmq_max_sim_elements=96)
    adapter = get_quantize_fn(config.quant_type, config)
    k = torch.randn(1, 2, 6, 8, dtype=torch.float32)
    v = torch.randn(1, 2, 6, 8, dtype=torch.float32)

    compress_kv_cache(k, v, config.quant_type, config, adapter, layer_idx=0)

    assert len(adapter.last_diagnostics) == 2
    for diagnostic in adapter.last_diagnostics:
        assert diagnostic["sim_tile_count"] > 1
        assert diagnostic["max_sim_tile_elements"] <= config.hqmq_max_sim_elements
        assert diagnostic["dense_sim_elements"] > config.hqmq_max_sim_elements


def test_self_forcing_dense_hqmq_output_is_permuted_for_bshd_cache():
    torch.manual_seed(5)
    config = _small_config("hqmq-fake")
    adapter = get_quantize_fn(config.quant_type, config)
    source_bshd = torch.randn(1, 6, 2, 8, dtype=torch.float32)

    source_bhsd = source_bshd.permute(0, 2, 1, 3).contiguous()
    dense_quant_bhsd, _ = compress_kv_cache(
        source_bhsd,
        source_bhsd,
        config.quant_type,
        config,
        adapter,
        layer_idx=0,
    )

    cache = ChunkedKVCache(
        batch_size=1,
        frame_seq_length=3,
        num_heads=2,
        head_dim=8,
        max_num_chunks=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layout="BSHD",
    )

    try:
        cache.store_quantized(0, 6, dense_quant_bhsd)
    except AssertionError as exc:
        assert "data tokens (2)" in str(exc)
    else:
        raise AssertionError("BHSD tensor unexpectedly stored directly in BSHD cache")

    causal_inference = _load_self_forcing_causal_inference_module()
    dense_quant_bshd = (
        causal_inference.CausalInferencePipeline
        ._dense_quant_tensor_to_cache_layout(dense_quant_bhsd, cache)
    )

    cache.store_quantized(0, 6, dense_quant_bshd)
    assert torch.equal(cache.read(0, 6), dense_quant_bshd)
    assert torch.equal(
        dense_quant_bshd,
        dense_quant_bhsd.permute(0, 2, 1, 3).contiguous(),
    )


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
    test_hqmq_fake_tiled_matches_dense_reference()
    test_hqmq_fake_diagnostics_record_bounded_tiles()
    test_self_forcing_dense_hqmq_output_is_permuted_for_bshd_cache()
    test_existing_qvg_dispatch_names_still_classify()
