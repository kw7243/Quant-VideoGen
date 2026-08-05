from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import torch


def _load_cache_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "cache_audit.py"
    spec = importlib.util.spec_from_file_location("qvg_cache_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_synthetic_artifact(path: Path) -> None:
    k = torch.arange(1 * 72 * 2 * 8, dtype=torch.float32).reshape(1, 72, 2, 8) / 100
    v = torch.flip(k, dims=[1])
    artifact = {
        "provenance": {
            "source": "synthetic-cache-audit-fixture",
            "quant_type": "hqmq-fake",
            "num_output_frames": 2,
        },
        "kv_cache": [
            {
                "k": k,
                "v": v,
                "global_end_index": torch.tensor([72], dtype=torch.long),
                "local_end_index": torch.tensor([72], dtype=torch.long),
            }
        ],
        "crossattn_cache": [
            {
                "k": torch.ones(1, 16, 2, 8, dtype=torch.float32),
                "v": torch.zeros(1, 16, 2, 8, dtype=torch.float32),
                "is_init": True,
            }
        ],
    }
    torch.save(artifact, path)


def test_cache_audit_schema_and_stats(tmp_path):
    cache_audit = _load_cache_audit_module()
    artifact = tmp_path / "synthetic_kv_cache.pt"
    _write_synthetic_artifact(artifact)

    summary = cache_audit.audit_artifact(
        artifact,
        stats_mode="auto",
        max_full_load_bytes=10 * 1024**2,
        max_total_stat_bytes=10 * 1024**2,
        max_tensor_stat_bytes=10 * 1024**2,
        max_time_points=8,
    )

    assert summary["schema_version"] == "qvg-cache-audit.v1"
    assert summary["input"]["path"].endswith("synthetic_kv_cache.pt")
    assert summary["load"]["status"] == "stats_computed"
    assert summary["provenance"]["provenance"]["quant_type"] == "hqmq-fake"

    tensors = {item["path"]: item for item in summary["tensors"]}
    k_info = tensors["kv_cache[0].k"]
    assert k_info["shape"] == [1, 72, 2, 8]
    assert k_info["layout"] == "BSHD"
    assert k_info["cache_kind"] == "self_attention"
    assert k_info["layer_idx"] == 0
    assert k_info["kv_role"] == "key"
    assert k_info["stats_status"] == "computed"
    assert "per_head" in k_info["stats"]
    assert k_info["stats"]["tuple4_radius"]["supported"] is True
    assert k_info["stats"]["temporal_deltas"]["supported"] is True
    assert k_info["stats"]["per_cache_position"]["sampled_count"] == 8

    layer_key = "self_attention:layer_0:key"
    assert layer_key in summary["layout_summary"]["by_layer_role"]


def test_cache_audit_metadata_only_guard(tmp_path):
    cache_audit = _load_cache_audit_module()
    artifact = tmp_path / "synthetic_kv_cache.pt"
    _write_synthetic_artifact(artifact)

    summary = cache_audit.audit_artifact(
        artifact,
        stats_mode="auto",
        max_full_load_bytes=1,
        max_total_stat_bytes=10 * 1024**2,
        max_tensor_stat_bytes=10 * 1024**2,
    )

    assert summary["load"]["status"] == "metadata_only"
    assert summary["load"]["stats_decision"]["compute_stats"] is False
    assert all(item["stats_status"] == "metadata_only" for item in summary["tensors"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cache_audit_schema_and_stats(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cache_audit_metadata_only_guard(Path(tmpdir))
