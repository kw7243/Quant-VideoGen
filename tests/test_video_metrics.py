import json
import tempfile
from dataclasses import replace
from pathlib import Path

from quant_videogen import video_metrics


def _write_fixture_mp4(path: Path, frames, fps: float = 4.0) -> bool:
    cv2 = video_metrics._try_import("cv2")
    np = video_metrics._try_import("numpy")
    if cv2 is None or np is None:
        return False

    height, width, _channels = frames[0].shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return True


def _fixture_frames(offset: int = 0):
    np = video_metrics._require_import("numpy")
    frames = []
    for idx in range(4):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[..., 0] = 20 + idx * 15 + offset
        frame[..., 1] = np.arange(16, dtype=np.uint8).reshape(16, 1) + idx + offset
        frame[..., 2] = np.arange(16, dtype=np.uint8).reshape(1, 16) + idx * 2
        frames.append(frame)
    return frames


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _entry(label: str = "bf16", quant_type: str = "bf16", **overrides):
    values = {
        "path": f"/tmp/{label}.mp4",
        "label": label,
        "prompt_id": "0",
        "prompt_hash": "prompt-sha256",
        "config_id": "sf-27f",
        "seed": "123",
        "run_label": "pair-smoke",
        "frame_count": "105",
        "local_attn_size": "27",
        "initial_latent_hash": None,
        "latent_identity_mode": video_metrics.LATENT_INFERRED_SEED_ONLY,
        "scheduler_id": "self-forcing-dmd-default",
        "checkpoint_id": "self_forcing_dmd.pt",
        "quant_type": quant_type,
        "code_commit": "abc123",
    }
    values.update(overrides)
    return video_metrics.VideoEntry(**values)


def test_inspect_missing_video_records_skip():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "inspect.jsonl"
        rc = video_metrics.main(["inspect", str(Path(tmp) / "missing.mp4"), "--output-jsonl", str(output)])
        assert rc == 0
        records = _read_jsonl(output)
        assert records[0]["status"] == "skip"
        assert records[0]["research_stage"] == video_metrics.STAGE0_INFRASTRUCTURE
        assert records[0]["skip_reason"] == "missing_file"


def test_compare_manifest_fixture_writes_pair_and_summary():
    cv2 = video_metrics._try_import("cv2")
    np = video_metrics._try_import("numpy")
    if cv2 is None or np is None:
        print("skipped video metrics fixture: cv2 or numpy unavailable")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref = tmp_path / "bf16.mp4"
        cand = tmp_path / "hqmq.mp4"
        assert _write_fixture_mp4(ref, _fixture_frames(offset=0))
        assert _write_fixture_mp4(cand, _fixture_frames(offset=3))

        manifest = tmp_path / "manifest.jsonl"
        rows = [
            {
                "path": str(ref),
                "label": "bf16",
                "prompt_id": "0",
                "config_id": "smoke",
                "seed": "123",
                "run_label": "fixture",
                "motion_category": "synthetic-low-motion",
                "frame_count": "4",
                "local_attn_size": "4",
                "latent_identity_mode": video_metrics.LATENT_EXPLICIT_HASH,
                "initial_latent_hash": "fixture-latent-hash",
                "scheduler": "fixture-scheduler",
                "checkpoint_id": "fixture-checkpoint",
                "quant_type": "bf16",
                "code_commit": "fixture-commit",
            },
            {
                "path": str(cand),
                "label": "hqmq-fake",
                "prompt_id": "0",
                "config_id": "smoke",
                "seed": "123",
                "run_label": "fixture",
                "motion_category": "synthetic-low-motion",
                "frame_count": "4",
                "local_attn_size": "4",
                "latent_identity_mode": video_metrics.LATENT_EXPLICIT_HASH,
                "initial_latent_hash": "fixture-latent-hash",
                "scheduler": "fixture-scheduler",
                "checkpoint_id": "fixture-checkpoint",
                "quant_type": "hqmq-fake",
                "code_commit": "fixture-commit",
            },
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

        output = tmp_path / "metrics.jsonl"
        summary = tmp_path / "summary.json"
        rc = video_metrics.main(
            [
                "compare",
                "--manifest",
                str(manifest),
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        assert rc == 0

        records = _read_jsonl(output)
        pair = records[0]
        aggregate = records[1]
        assert pair["record_type"] == "pair_metrics"
        assert pair["research_stage"] == video_metrics.STAGE3_ROLLOUT
        assert pair["metric_scope"] == "paired_bf16_fidelity"
        assert pair["status"] == "ok"
        assert pair["pair_key"] == {
            "prompt_id": "0",
            "config_id": "smoke",
            "seed": "123",
            "run_label": "fixture",
        }
        assert pair["controlled_fields"] == {
            "prompt_category": "synthetic-low-motion",
            "rollout_length": None,
            "frame_count": "4",
            "local_attn_size": "4",
            "initial_latent_id": None,
            "initial_latent_hash": "fixture-latent-hash",
            "latent_identity_mode": video_metrics.LATENT_EXPLICIT_HASH,
            "scheduler_id": "fixture-scheduler",
            "scheduler_config_hash": None,
            "checkpoint_id": "fixture-checkpoint",
            "checkpoint_hash": None,
            "code_commit": "fixture-commit",
        }
        assert pair["pairing_protocol"]["status"] == "ok"
        assert pair["pairing_protocol"]["reference"]["latent_identity_mode"] == video_metrics.LATENT_EXPLICIT_HASH
        assert pair["reference"]["media"]["width"] == 16
        assert pair["candidate"]["media"]["frame_count"] == 4
        assert pair["metrics"]["frame_count"] == 4
        assert pair["metrics"]["mse"] > 0
        assert pair["metrics"]["psnr_db"] is not None
        assert 0.0 <= pair["metrics"]["ssim"] <= 1.0
        assert pair["metrics"]["temporal_delta_mse"] >= 0

        assert aggregate["record_type"] == "aggregate_summary"
        assert aggregate["research_stage"] == video_metrics.STAGE3_ROLLOUT
        assert aggregate["ok_pair_count"] == 1
        assert aggregate["skip_pair_count"] == 0
        assert aggregate["invalid_pair_count"] == 0
        assert json.loads(summary.read_text())["metrics"]["mse"]["count"] == 1


def test_pairing_protocol_rejects_identity_mismatches_before_video_decode():
    reference = _entry()
    mismatches = [
        (replace(_entry(label="hqmq", quant_type="hqmq-fake"), prompt_id="1"), "prompt_id_mismatch"),
        (replace(_entry(label="hqmq", quant_type="hqmq-fake"), seed="124"), "seed_mismatch"),
        (replace(_entry(label="hqmq", quant_type="hqmq-fake"), config_id="sf-81f"), "config_id_mismatch"),
        (replace(_entry(label="hqmq", quant_type="hqmq-fake"), frame_count="104"), "frame_count_mismatch"),
        (replace(_entry(label="hqmq", quant_type="hqmq-fake"), checkpoint_id="other.pt"), "checkpoint_id_mismatch"),
        (
            replace(
                _entry(label="hqmq", quant_type="hqmq-fake"),
                latent_identity_mode=video_metrics.LATENT_EXPLICIT_HASH,
                initial_latent_hash="latent-hash",
            ),
            "latent_identity_mode_mismatch",
        ),
    ]

    for candidate, expected_issue in mismatches:
        record = video_metrics.compare_entries(reference, candidate)
        assert record["status"] == "invalid"
        assert record["invalid_reason"] == "invalid_pairing_protocol"
        assert expected_issue in record["pairing_protocol"]["issues"]
        assert record["reference"].get("media") is None
        assert record["candidate"].get("media") is None


def test_latent_identity_modes_are_explicit_in_schema():
    reference = _entry(
        latent_identity_mode=video_metrics.LATENT_EXPLICIT_HASH,
        initial_latent_hash="latent-hash",
    )
    candidate = _entry(
        label="hqmq",
        quant_type="hqmq-fake",
        latent_identity_mode=video_metrics.LATENT_EXPLICIT_HASH,
        initial_latent_hash="latent-hash",
    )
    assert video_metrics.validate_pair_identity(reference, candidate)["status"] == "ok"

    candidate = replace(candidate, initial_latent_hash="other-latent")
    protocol = video_metrics.validate_pair_identity(reference, candidate)
    assert protocol["status"] == "invalid"
    assert "initial_latent_hash_mismatch" in protocol["issues"]

    candidate = replace(candidate, latent_identity_mode=video_metrics.LATENT_EXPLICIT_HASH, initial_latent_hash=None)
    protocol = video_metrics.validate_pair_identity(reference, candidate)
    assert protocol["status"] == "invalid"
    assert "candidate_missing_explicit_latent_hash" in protocol["issues"]


def test_validate_manifest_writes_pairing_ledger_without_video_decode():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = tmp_path / "manifest.jsonl"
        rows = [
            _entry(path=str(tmp_path / "missing_ref.mp4")).pairing_ledger() | {"label": "bf16"},
            _entry(path=str(tmp_path / "missing_cand.mp4"), label="hqmq", quant_type="hqmq-fake").pairing_ledger() | {"label": "hqmq"},
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

        output = tmp_path / "pairing.jsonl"
        summary = tmp_path / "pairing-summary.json"
        rc = video_metrics.main(
            [
                "validate-manifest",
                "--manifest",
                str(manifest),
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        assert rc == 0
        records = _read_jsonl(output)
        assert [record["record_type"] for record in records] == [
            "pairing_ledger_entry",
            "pairing_ledger_entry",
            "pairing_validation",
            "pairing_validation_summary",
        ]
        assert records[2]["status"] == "ok"
        assert not records[0]["entry"]["artifact_path"].endswith("exists.mp4")
        assert json.loads(summary.read_text())["status_counts"]["ok"] == 3


def test_compare_mismatched_frames_records_skip():
    cv2 = video_metrics._try_import("cv2")
    np = video_metrics._try_import("numpy")
    if cv2 is None or np is None:
        print("skipped mismatch fixture: cv2 or numpy unavailable")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref = tmp_path / "bf16.mp4"
        cand = tmp_path / "qvg.mp4"
        assert _write_fixture_mp4(ref, _fixture_frames(offset=0))
        assert _write_fixture_mp4(cand, _fixture_frames(offset=1)[:3])

        output = tmp_path / "metrics.jsonl"
        rc = video_metrics.main(
            [
                "compare",
                "--reference",
                str(ref),
                "--candidate",
                str(cand),
                "--candidate-label",
                "qvg",
                "--reference-quant-type",
                "bf16",
                "--candidate-quant-type",
                "qvg",
                "--prompt-id",
                "0",
                "--config-id",
                "fixture",
                "--seed",
                "123",
                "--run-label",
                "fixture",
                "--frame-count",
                "4",
                "--local-attn-size",
                "4",
                "--latent-identity-mode",
                video_metrics.LATENT_INFERRED_SEED_ONLY,
                "--scheduler-id",
                "fixture-scheduler",
                "--checkpoint-id",
                "fixture-checkpoint",
                "--code-commit",
                "fixture-commit",
                "--output-jsonl",
                str(output),
            ]
        )
        assert rc == 0
        records = _read_jsonl(output)
        assert records[0]["status"] == "skip"
        assert records[0]["skip_reason"] == "incompatible_video_shapes"
        assert records[1]["skip_pair_count"] == 1
        assert records[1]["invalid_pair_count"] == 0


if __name__ == "__main__":
    test_inspect_missing_video_records_skip()
    test_compare_manifest_fixture_writes_pair_and_summary()
    test_pairing_protocol_rejects_identity_mismatches_before_video_decode()
    test_latent_identity_modes_are_explicit_in_schema()
    test_validate_manifest_writes_pairing_ledger_without_video_decode()
    test_compare_mismatched_frames_records_skip()
