import json
import tempfile
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
                "num_frames": "4",
                "initial_latent_id": "fixture-latent",
                "scheduler": "fixture-scheduler",
            },
            {
                "path": str(cand),
                "label": "hqmq-fake",
                "prompt_id": "0",
                "config_id": "smoke",
                "seed": "123",
                "run_label": "fixture",
                "motion_category": "synthetic-low-motion",
                "num_frames": "4",
                "initial_latent_id": "fixture-latent",
                "scheduler": "fixture-scheduler",
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
            "rollout_length": "4",
            "initial_latent_id": "fixture-latent",
            "scheduler_id": "fixture-scheduler",
        }
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
        assert json.loads(summary.read_text())["metrics"]["mse"]["count"] == 1


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
                "--output-jsonl",
                str(output),
            ]
        )
        assert rc == 0
        records = _read_jsonl(output)
        assert records[0]["status"] == "skip"
        assert records[0]["skip_reason"] == "incompatible_video_shapes"
        assert records[1]["skip_pair_count"] == 1


if __name__ == "__main__":
    test_inspect_missing_video_records_skip()
    test_compare_manifest_fixture_writes_pair_and_summary()
    test_compare_mismatched_frames_records_skip()
