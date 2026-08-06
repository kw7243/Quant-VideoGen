from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import torch


class _patched_environ:
    def __init__(self, updates: dict[str, str | None]):
        self.updates = updates
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, _exc_type, _exc, _tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_tool_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _attention_fixture():
    query = torch.zeros(1, 1, 2, 4)
    query[0, 0, 0, 0] = 4
    query[0, 0, 1, 1] = 4

    cached_key = torch.zeros(1, 1, 3, 4)
    cached_key[0, 0, 0, 0] = 4
    cached_key[0, 0, 1, 1] = 4
    cached_key[0, 0, 2, 2] = 4
    current_key = torch.zeros(1, 1, 1, 4)
    current_key[0, 0, 0, 3] = 4

    cached_value = torch.zeros(1, 1, 3, 4)
    cached_value[0, 0, 0, 0] = 10
    cached_value[0, 0, 1, 1] = 10
    cached_value[0, 0, 2, 2] = 10
    current_value = torch.zeros(1, 1, 1, 4)
    current_value[0, 0, 0, 3] = 10

    return {
        "query": query,
        "current_key": current_key,
        "current_value": current_value,
        "cached_key": cached_key,
        "cached_value": cached_value,
    }


def test_attention_metrics_detect_k_only_perturbation():
    probe = _load_tool_module("attention_error_probe.py", "qvg_attention_error_probe")
    tensors = _attention_fixture()
    k_swapped = tensors["cached_key"].clone()
    k_swapped[:, :, [0, 1], :] = k_swapped[:, :, [1, 0], :]

    metrics = probe.compute_attention_error_metrics(
        query=tensors["query"],
        current_key=tensors["current_key"],
        current_value=tensors["current_value"],
        cached_key_ref=tensors["cached_key"],
        cached_value_ref=tensors["cached_value"],
        cached_key_variant=k_swapped,
        top_k=1,
    )

    assert metrics["attention_logit_error"]["relative_l2"] > 0.5
    assert metrics["attention_probability_divergence"]["js_mean"] > 0.5
    assert metrics["attention_output_relative_error"]["relative_l2"] > 0.9
    assert metrics["top_attended_token_overlap"]["top1_match_fraction"] == 0.0


def test_attention_metrics_detect_v_only_perturbation_without_logit_change():
    probe = _load_tool_module("attention_error_probe.py", "qvg_attention_error_probe")
    tensors = _attention_fixture()
    v_swapped = tensors["cached_value"].clone()
    v_swapped[:, :, [0, 1], :] = v_swapped[:, :, [1, 0], :]

    metrics = probe.compute_attention_error_metrics(
        query=tensors["query"],
        current_key=tensors["current_key"],
        current_value=tensors["current_value"],
        cached_key_ref=tensors["cached_key"],
        cached_value_ref=tensors["cached_value"],
        cached_value_variant=v_swapped,
        top_k=1,
    )

    assert metrics["attention_logit_error"]["relative_l2"] == 0.0
    assert metrics["attention_probability_divergence"]["js_mean"] == 0.0
    assert metrics["attention_output_relative_error"]["relative_l2"] > 0.9
    assert metrics["top_attended_token_overlap"]["top1_match_fraction"] == 1.0


def test_probe_artifact_and_cache_audit_smoke(tmp_path):
    probe = _load_tool_module("attention_error_probe.py", "qvg_attention_error_probe")
    tensors = _attention_fixture()
    k_swapped = tensors["cached_key"].clone()
    k_swapped[:, :, [0, 1], :] = k_swapped[:, :, [1, 0], :]
    v_swapped = tensors["cached_value"].clone()
    v_swapped[:, :, [0, 1], :] = v_swapped[:, :, [1, 0], :]

    artifact = tmp_path / "attention_probe_input.pt"
    torch.save(
        {
            "schema_version": probe.INPUT_SCHEMA_VERSION,
            "records": [
                {
                    "name": "synthetic",
                    "layout": "BHSD",
                    "query": tensors["query"],
                    "current_key": tensors["current_key"],
                    "current_value": tensors["current_value"],
                    "cached_key_ref": tensors["cached_key"],
                    "cached_value_ref": tensors["cached_value"],
                    "variants": [
                        {"name": "k_swap", "cached_key": k_swapped},
                        {"name": "v_swap", "cached_value": v_swapped},
                    ],
                }
            ],
        },
        artifact,
    )

    summary = probe.run_probe_artifact(artifact, top_k=1)
    assert summary["schema_version"] == probe.SCHEMA_VERSION
    assert summary["records"][0]["variants"][0]["name"] == "k_swap"
    assert summary["records"][0]["variants"][1]["name"] == "v_swap"

    smoke = probe.smoke_cache_audit_summary(
        Path(__file__).resolve().parents[1] / "analysis" / "cache_audit_frames27_bounded_stats.json"
    )
    assert smoke["attention_probe_computability"]["can_compute_attention_logit_error"] is False
    assert "query" in smoke["attention_probe_computability"]["missing_required_tensors"]
    assert smoke["available_without_tensor_materialization"]["self_attention_cache_tensor_count"] == 18


def test_probe_identity_variant_if_missing_for_reference_only_capture(tmp_path):
    probe = _load_tool_module("attention_error_probe.py", "qvg_attention_error_probe_identity")
    tensors = _attention_fixture()
    artifact = tmp_path / "attention_probe_reference_only.pt"
    torch.save(
        {
            "schema_version": probe.INPUT_SCHEMA_VERSION,
            "records": [
                {
                    "name": "reference_only",
                    "layout": "BHSD",
                    "query": tensors["query"],
                    "current_key": tensors["current_key"],
                    "current_value": tensors["current_value"],
                    "cached_key_ref": tensors["cached_key"],
                    "cached_value_ref": tensors["cached_value"],
                }
            ],
        },
        artifact,
    )

    summary = probe.run_probe_artifact(artifact, top_k=1, identity_variant_if_missing=True)

    variant = summary["records"][0]["variants"][0]
    assert variant["name"] == "identity_reference"
    assert variant["metrics"]["attention_logit_error"]["relative_l2"] == 0.0
    assert variant["metrics"]["attention_output_relative_error"]["relative_l2"] == 0.0
    assert summary["config"]["identity_variant_if_missing"] is True


def test_self_forcing_capture_schema_is_bounded_and_bhsd(tmp_path):
    capture = _load_tool_module("attention_probe_capture.py", "qvg_attention_probe_capture")
    config = capture.CaptureConfig(
        enabled=True,
        output_dir=tmp_path,
        layers={29},
        heads={1},
        chunks=None,
        max_file_bytes=1024 * 1024,
        variant_name="bf16",
    )
    query = torch.randn(1, 2, 2, 4)
    key_all = torch.randn(1, 5, 2, 4)
    value_all = torch.randn(1, 5, 2, 4)

    record = capture.build_self_forcing_probe_record(
        layer_idx=29,
        current_start=2 * 1560,
        local_start_index=3,
        local_end_index=5,
        frame_seq_length=1560,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=config,
        metadata={"quant_type": "none"},
    )

    assert record["layout"] == "BHSD"
    assert record["query"].shape == (1, 1, 2, 4)
    assert record["cached_key_ref"].shape == (1, 1, 3, 4)
    assert record["current_key"].shape == (1, 1, 2, 4)
    assert record["metadata"]["selected_heads"] == [1]
    assert record["metadata"]["selected_chunks"] == "all"

    manifest = capture.write_self_forcing_probe_capture(
        layer_idx=29,
        current_start=2 * 1560,
        local_start_index=3,
        local_end_index=5,
        frame_seq_length=1560,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=config,
    )
    assert manifest["status"] == "written"
    assert Path(manifest["output_path"]).exists()

    too_small = capture.CaptureConfig(
        enabled=True,
        output_dir=tmp_path,
        layers={29},
        heads={1},
        chunks=None,
        max_file_bytes=1,
    )
    skipped = capture.write_self_forcing_probe_capture(
        layer_idx=29,
        current_start=2 * 1560,
        local_start_index=3,
        local_end_index=5,
        frame_seq_length=1560,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=too_small,
    )
    assert skipped["status"] == "skipped_byte_guard"


def test_capture_env_defaults_and_overrides(tmp_path):
    capture = _load_tool_module("attention_probe_capture.py", "qvg_attention_probe_capture_env")
    keys = {
        "QVG_ATTENTION_PROBE_ENABLED": "1",
        "QVG_ATTENTION_PROBE_DIR": str(tmp_path),
        "QVG_ATTENTION_PROBE_MAX_FILE_BYTES": "12345",
        "QVG_ATTENTION_PROBE_MAX_CACHE_TOKENS": "7",
        "QVG_ATTENTION_PROBE_VARIANT": "hqmq-fake",
        "QVG_ATTENTION_PROBE_LAYERS": None,
        "QVG_ATTENTION_PROBE_HEADS": None,
        "QVG_ATTENTION_PROBE_CHUNKS": None,
    }
    with _patched_environ(keys):
        config = capture.CaptureConfig.from_env()
    assert config.enabled is True
    assert config.output_dir == tmp_path
    assert config.layers == {27, 28, 29}
    assert config.heads == {0, 9}
    assert config.chunks == {24, 25, 26}
    assert config.max_file_bytes == 12345
    assert config.max_cache_tokens == 7
    assert config.variant_name == "hqmq-fake"

    with _patched_environ(
        {
            "QVG_ATTENTION_PROBE_LAYERS": "all",
            "QVG_ATTENTION_PROBE_HEADS": "1,3",
            "QVG_ATTENTION_PROBE_CHUNKS": "4,5",
        }
    ):
        config = capture.CaptureConfig.from_env()
    assert config.layers is None
    assert config.heads == {1, 3}
    assert config.chunks == {4, 5}


def test_capture_selection_skip_and_chunk_slicing(tmp_path):
    capture = _load_tool_module("attention_probe_capture.py", "qvg_attention_probe_capture_selection")
    query = torch.randn(1, 2, 2, 4)
    key_all = torch.randn(1, 10, 2, 4)
    value_all = torch.randn(1, 10, 2, 4)

    disabled = capture.CaptureConfig(enabled=False, output_dir=tmp_path, layers={29}, chunks={4})
    skipped = capture.write_self_forcing_probe_capture(
        layer_idx=29,
        current_start=8,
        local_start_index=8,
        local_end_index=10,
        frame_seq_length=2,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=disabled,
    )
    assert skipped["status"] == "skipped_selection"

    layer_mismatch = capture.CaptureConfig(enabled=True, output_dir=tmp_path, layers={28}, chunks={4})
    skipped = capture.write_self_forcing_probe_capture(
        layer_idx=29,
        current_start=8,
        local_start_index=8,
        local_end_index=10,
        frame_seq_length=2,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=layer_mismatch,
    )
    assert skipped["status"] == "skipped_selection"

    bounded = capture.CaptureConfig(
        enabled=True,
        output_dir=tmp_path,
        layers={29},
        heads={1},
        chunks={1, 3, 4},
        max_file_bytes=1024 * 1024,
    )
    record = capture.build_self_forcing_probe_record(
        layer_idx=29,
        current_start=8,
        local_start_index=8,
        local_end_index=10,
        frame_seq_length=2,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=bounded,
    )
    assert record["cached_key_ref"].shape == (1, 1, 4, 4)
    assert record["current_key"].shape == (1, 1, 2, 4)
    assert record["metadata"]["cached_chunk_indices"] == [1, 3]
    assert record["metadata"]["current_chunk_indices"] == [4]


def test_capture_head_guard_cache_truncation_and_post_write_guard(tmp_path):
    capture = _load_tool_module("attention_probe_capture.py", "qvg_attention_probe_capture_guards")
    query = torch.randn(1, 2, 2, 4)
    key_all = torch.randn(1, 6, 2, 4)
    value_all = torch.randn(1, 6, 2, 4)

    invalid_head = capture.CaptureConfig(enabled=True, output_dir=tmp_path, heads={99}, chunks=None)
    try:
        capture.build_self_forcing_probe_record(
            layer_idx=0,
            current_start=4,
            local_start_index=4,
            local_end_index=6,
            frame_seq_length=2,
            query_rope_bshd=query,
            key_all_rope_bshd=key_all,
            value_all_bshd=value_all,
            config=invalid_head,
        )
        raise AssertionError("expected invalid head selection to raise")
    except ValueError as exc:
        assert "matched no heads" in str(exc)

    truncated = capture.CaptureConfig(
        enabled=True,
        output_dir=tmp_path,
        heads={0},
        chunks=None,
        max_cache_tokens=2,
    )
    record = capture.build_self_forcing_probe_record(
        layer_idx=0,
        current_start=4,
        local_start_index=4,
        local_end_index=6,
        frame_seq_length=2,
        query_rope_bshd=query,
        key_all_rope_bshd=key_all,
        value_all_bshd=value_all,
        config=truncated,
    )
    assert record["cached_key_ref"].shape == (1, 1, 2, 4)
    assert record["metadata"]["dropped_prefix_cache_tokens"] == 2

    output_path = tmp_path / "post_write_guard.pt"
    manifest = capture.write_probe_input(output_path, [{"metadata": {}}], max_file_bytes=1)
    assert manifest["status"] == "skipped_byte_guard"
    assert manifest["estimated_tensor_bytes"] == 0
    assert not output_path.exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_probe_artifact_and_cache_audit_smoke(Path(tmpdir))
        test_probe_identity_variant_if_missing_for_reference_only_capture(Path(tmpdir))
        test_self_forcing_capture_schema_is_bounded_and_bhsd(Path(tmpdir))
        test_capture_env_defaults_and_overrides(Path(tmpdir))
        test_capture_selection_skip_and_chunk_slicing(Path(tmpdir))
        test_capture_head_guard_cache_truncation_and_post_write_guard(Path(tmpdir))
    test_attention_metrics_detect_k_only_perturbation()
    test_attention_metrics_detect_v_only_perturbation_without_logit_change()
