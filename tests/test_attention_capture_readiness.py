from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def _load_tool_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _submission(root: Path, *, max_file_bytes: int = 1024, chunks: str = "24,25,26") -> dict:
    return {
        "job_id": "1316140",
        "staged_copy": "/tmp/staged",
        "source_repo": "/tmp/source",
        "job_script": "scripts/Self-Forcing/attention_capture_bf16.sbatch",
        "account": "vision-torralba-urops-meng",
        "qos": "vision-torralba-urops-meng",
        "partition": "vision-torralba-h200",
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem": "96G",
        "time_limit": "01:00:00",
        "log_stdout": str(root / "logs" / "qvg_sf_attncap-1316140.out"),
        "log_stderr": str(root / "logs" / "qvg_sf_attncap-1316140.err"),
        "output_dir": str(root / "outputs" / "bf16_attncap_27f_prompt1_1316140"),
        "capture_dir": str(root / "captures" / "bf16_attncap_27f_prompt1_1316140"),
        "capture_manifest": str(root / "capture_manifests" / "bf16_attncap_27f_prompt1_1316140.json"),
        "quant_type": "none",
        "prompt_limit": 1,
        "num_output_frames": 27,
        "local_attn_size": 27,
        "attention_probe": {
            "enabled": 1,
            "layers": "27,28,29",
            "heads": "0,9",
            "chunks": chunks,
            "max_file_bytes": max_file_bytes,
            "variant": "bf16_reference",
        },
        "preflight": {},
    }


def _write_submission(root: Path, *, max_file_bytes: int = 1024, chunks: str = "24,25,26") -> dict:
    data = _submission(root, max_file_bytes=max_file_bytes, chunks=chunks)
    _write_json(root / "submission_1316140.json", data)
    return data


def _make_complete_artifacts(
    root: Path,
    *,
    has_variants: bool,
    max_file_bytes: int = 1024,
    capture_size: int = 64,
    selected_heads: list[int] | None = None,
    selected_chunks: list[int] | None = None,
    cached_chunk_indices: list[int] | None = None,
    current_chunk_indices: list[int] | None = None,
    variant_names: list[str] | None = None,
    record_chunks: list[int] | None = None,
) -> dict:
    chunks = ",".join(str(item) for item in (selected_chunks if selected_chunks is not None else [24, 25, 26]))
    submission = _write_submission(root, max_file_bytes=max_file_bytes, chunks=chunks)
    Path(submission["log_stdout"]).parent.mkdir(parents=True, exist_ok=True)
    Path(submission["log_stdout"]).write_text("capture_file_count=1\n", encoding="utf-8")
    Path(submission["log_stderr"]).write_text("", encoding="utf-8")
    output_dir = Path(submission["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample.mp4").write_bytes(b"video")
    capture_dir = Path(submission["capture_dir"])
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_path = capture_dir / "attention_probe_l27_chunk0024_start0_bf16_reference.pt"
    capture_path.write_bytes(b"x" * capture_size)
    records = []
    for layer_idx in (27, 28, 29):
        for chunk_idx in (record_chunks if record_chunks is not None else [24, 25, 26]):
            records.append(
                {
                    "name": f"layer_{layer_idx}_chunk_{chunk_idx}",
                    "layer_idx": layer_idx,
                    "chunk_idx": chunk_idx,
                    "selected_heads": selected_heads if selected_heads is not None else [0, 9],
                    "selected_chunks": selected_chunks if selected_chunks is not None else [24, 25, 26],
                    "cached_chunk_indices": cached_chunk_indices if cached_chunk_indices is not None else [24],
                    "current_chunk_indices": current_chunk_indices if current_chunk_indices is not None else [chunk_idx],
                    "selected_cached_tokens": 4,
                    "selected_current_tokens": 2,
                    "query_shape": [1, 2, 2, 128],
                    "cached_key_ref_shape": [1, 2, 4, 128],
                    "current_key_shape": [1, 2, 2, 128],
                    "has_variants": has_variants,
                    "variant_names": variant_names,
                }
            )
    manifest = {
        "schema_version": "qvg-attention-capture-manifest.v1",
        "capture_dir": submission["capture_dir"],
        "capture_file_count": 1,
        "capture_total_size_bytes": capture_path.stat().st_size,
        "captures": [
            {
                "path": str(capture_path),
                "size_bytes": capture_path.stat().st_size,
                "record_count": len(records),
                "records": records,
            }
        ],
    }
    _write_json(Path(submission["capture_manifest"]), manifest)
    return submission


def _evaluate(readiness, root: Path, *, max_file_bytes: int = 1024, expected_chunks=None):
    return readiness.evaluate_readiness(
        task_root=root,
        job_id="1316140",
        expected_chunks=expected_chunks if expected_chunks is not None else (24, 25, 26),
        expected_max_capture_file_bytes=max_file_bytes,
        probe_python="python3",
        tool_path=root / "tools" / "attention_error_probe.py",
    )


def test_pending_artifacts_are_not_ready_without_crashing(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_pending")
    _write_submission(tmp_path)

    summary = _evaluate(readiness, tmp_path)

    assert summary["status"] == "not_ready"
    assert summary["ready"] is False
    assert "stdout_log_missing" in summary["reasons"]
    assert "stderr_log_missing" in summary["reasons"]
    assert "capture_manifest_missing" in summary["reasons"]
    assert summary["next_attention_error_probe_command"] is None
    gate = summary["focused_attention_error_gate"]
    assert gate["status"] == "not_ready"
    assert gate["decision_state"] == "no_claim"
    assert gate["probe_commands"] == []


def test_complete_reference_only_capture_emits_identity_probe_command(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_reference")
    _make_complete_artifacts(tmp_path, has_variants=False)

    summary = _evaluate(readiness, tmp_path)

    assert summary["status"] == "ready"
    assert summary["variant_distortion_ready"] is False
    assert summary["analysis_mode"] == "identity_reference"
    assert "--identity-variant-if-missing" in summary["next_attention_error_probe_command"]
    assert "--record-limit 1" in summary["next_attention_error_probe_command"]
    assert summary["reasons"] == []
    gate = summary["focused_attention_error_gate"]
    assert gate["status"] == "incomplete"
    assert "cached_chunk_23_missing" in gate["reasons"]


def test_complete_variant_capture_emits_distortion_probe_command(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_variant")
    _make_complete_artifacts(tmp_path, has_variants=True)

    summary = _evaluate(readiness, tmp_path)

    assert summary["status"] == "ready"
    assert summary["variant_distortion_ready"] is True
    assert summary["analysis_mode"] == "variant_distortion"
    assert "--identity-variant-if-missing" not in summary["next_attention_error_probe_command"]
    gate = summary["focused_attention_error_gate"]
    assert gate["status"] == "incomplete"
    assert "cached_chunk_23_missing" in gate["reasons"]


def test_capture_byte_guard_blocks_oversized_capture(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_byteguard")
    _make_complete_artifacts(tmp_path, has_variants=True, max_file_bytes=16, capture_size=32)

    summary = _evaluate(readiness, tmp_path, max_file_bytes=16)

    assert summary["status"] == "not_ready"
    assert "capture_file_exceeds_byte_guard" in summary["reasons"]
    assert summary["next_attention_error_probe_command"] is None


def test_capture_selection_mismatch_is_machine_readable(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_selection")
    _make_complete_artifacts(tmp_path, has_variants=True, selected_heads=[1])

    summary = _evaluate(readiness, tmp_path)

    assert summary["status"] == "not_ready"
    assert "capture_selection_mismatch" in summary["reasons"]
    selection_checks = [check for check in summary["checks"] if check["name"] == "capture_selection"]
    assert selection_checks[0]["details"]["selected_head_mismatches"]


def test_focused_exact_reference_capture_is_identity_only(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_focus_ref")
    _make_complete_artifacts(
        tmp_path,
        has_variants=False,
        selected_chunks=[24],
        cached_chunk_indices=[23],
        current_chunk_indices=[24],
        record_chunks=[24],
    )

    summary = _evaluate(readiness, tmp_path, expected_chunks=(24,))

    gate = summary["focused_attention_error_gate"]
    assert summary["status"] == "ready"
    assert gate["status"] == "incomplete"
    assert gate["decision_state"] == "identity_reference_only"
    assert gate["support"]["target_head_supported"] is True
    assert gate["support"]["control_head_supported"] is True
    assert gate["support"]["cached_transition_supported"] is True
    assert gate["support"]["current_transition_supported"] is True
    assert gate["probe_commands"][0]["role"] == "reference_only_no_distortion_claim"
    assert "--identity-variant-if-missing" in gate["probe_commands"][0]["command"]


def test_focused_gate_emits_separated_k_and_v_commands(tmp_path):
    readiness = _load_tool_module("attention_capture_readiness.py", "qvg_attention_capture_readiness_focus_variants")
    _make_complete_artifacts(
        tmp_path,
        has_variants=True,
        selected_chunks=[24],
        cached_chunk_indices=[23],
        current_chunk_indices=[24],
        variant_names=["hqmq_k_only", "hqmq_v_only"],
        record_chunks=[24],
    )

    summary = _evaluate(readiness, tmp_path, expected_chunks=(24,))

    gate = summary["focused_attention_error_gate"]
    assert gate["status"] == "ready"
    assert gate["decision_state"] == "attention_error_probe_ready"
    commands = {item["role"]: item["command"] for item in gate["probe_commands"]}
    assert set(commands) == {"k_only", "v_only"}
    assert "--variant-name hqmq_k_only" in commands["k_only"]
    assert "--variant-name hqmq_v_only" in commands["v_only"]
    assert "--identity-variant-if-missing" not in commands["k_only"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_pending_artifacts_are_not_ready_without_crashing(Path(tmpdir) / "pending")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_complete_reference_only_capture_emits_identity_probe_command(Path(tmpdir) / "reference")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_complete_variant_capture_emits_distortion_probe_command(Path(tmpdir) / "variant")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_capture_byte_guard_blocks_oversized_capture(Path(tmpdir) / "byteguard")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_capture_selection_mismatch_is_machine_readable(Path(tmpdir) / "selection")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_focused_exact_reference_capture_is_identity_only(Path(tmpdir) / "focus-reference")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_focused_gate_emits_separated_k_and_v_commands(Path(tmpdir) / "focus-variants")
