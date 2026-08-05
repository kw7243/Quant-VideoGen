from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import torch


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


def test_self_forcing_capture_schema_is_bounded_and_bhsd(tmp_path):
    capture = _load_tool_module("attention_probe_capture.py", "qvg_attention_probe_capture")
    config = capture.CaptureConfig(
        enabled=True,
        output_dir=tmp_path,
        layers={29},
        heads={1},
        chunks={2},
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
        chunks={2},
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


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_probe_artifact_and_cache_audit_smoke(Path(tmpdir))
        test_self_forcing_capture_schema_is_bounded_and_bhsd(Path(tmpdir))
    test_attention_metrics_detect_k_only_perturbation()
    test_attention_metrics_detect_v_only_perturbation_without_logit_change()
