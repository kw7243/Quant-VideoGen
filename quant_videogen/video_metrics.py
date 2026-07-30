"""CPU-friendly video metadata, pairing, and metric aggregation utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PAIR_KEY_FIELDS = ("prompt_id", "config_id", "seed", "run_label")
DEFAULT_REFERENCE_LABEL = "bf16"
STAGE0_INFRASTRUCTURE = "stage0_infrastructure_validation"
STAGE3_ROLLOUT = "stage3_controlled_rollout_comparison"


@dataclass(frozen=True)
class VideoEntry:
    path: str
    label: str
    prompt_id: str | None = None
    prompt: str | None = None
    config_id: str | None = None
    seed: str | None = None
    run_label: str | None = None
    prompt_category: str | None = None
    rollout_length: str | None = None
    initial_latent_id: str | None = None
    scheduler_id: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "VideoEntry":
        path = row.get("path") or row.get("video_path") or row.get("file")
        label = row.get("label") or row.get("variant") or row.get("quant_type") or row.get("method")
        if not path:
            raise ValueError(f"manifest row is missing a video path field: {row}")
        if not label:
            raise ValueError(f"manifest row is missing a label/variant field: {row}")
        return cls(
            path=str(path),
            label=str(label),
            prompt_id=_optional_str(row.get("prompt_id", row.get("idx"))),
            prompt=_optional_str(row.get("prompt")),
            config_id=_optional_str(row.get("config_id", row.get("config"))),
            seed=_optional_str(row.get("seed")),
            run_label=_optional_str(row.get("run_label", row.get("run"))),
            prompt_category=_optional_str(row.get("prompt_category", row.get("motion_category"))),
            rollout_length=_optional_str(row.get("rollout_length", row.get("num_frames"))),
            initial_latent_id=_optional_str(row.get("initial_latent_id")),
            scheduler_id=_optional_str(row.get("scheduler_id", row.get("scheduler"))),
        )

    def pair_key(self) -> dict[str, str | None]:
        return {field: getattr(self, field) for field in PAIR_KEY_FIELDS}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _jsonable_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_manifest(path: str | Path) -> list[VideoEntry]:
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    else:
        with manifest_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{manifest_path}:{line_number}: invalid JSONL row") from exc
    return [VideoEntry.from_mapping(row) for row in rows]


def inspect_video(path: str | Path) -> dict[str, Any]:
    video_path = Path(path)
    info: dict[str, Any] = {
        "path": str(video_path),
        "exists": video_path.exists(),
        "size_bytes": video_path.stat().st_size if video_path.exists() else None,
        "backend": None,
        "codec": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
    }
    if not video_path.exists():
        info["error"] = "missing_file"
        return info

    cv2 = _try_import("cv2")
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if cap.isOpened():
                frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
                height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
                info.update(
                    {
                        "backend": "opencv",
                        "codec": _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                        "width": width or None,
                        "height": height or None,
                        "fps": fps or None,
                        "frame_count": frame_count or None,
                        "duration_seconds": (frame_count / fps) if frame_count and fps else None,
                    }
                )
                return info
        finally:
            cap.release()

    av = _try_import("av")
    if av is not None:
        try:
            container = av.open(str(video_path))
        except Exception as exc:  # pragma: no cover - backend-specific message
            info["error"] = f"open_failed: {exc}"
            return info
        with container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if streams:
                stream = streams[0]
                fps = float(stream.average_rate) if stream.average_rate else None
                frame_count = int(stream.frames) if stream.frames else None
                duration = float(stream.duration * stream.time_base) if stream.duration else None
                info.update(
                    {
                        "backend": "pyav",
                        "codec": stream.codec_context.name,
                        "width": stream.codec_context.width,
                        "height": stream.codec_context.height,
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration_seconds": duration,
                    }
                )
                return info

    info["error"] = "no_video_backend"
    return info


def load_video_frames(path: str | Path) -> Any:
    """Return RGB float32 frames shaped [T, H, W, C] in [0, 1]."""

    np = _require_import("numpy")
    cv2 = _try_import("cv2")
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        frames = []
        try:
            if not cap.isOpened():
                raise RuntimeError("opencv_open_failed")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            raise RuntimeError("empty_video")
        return np.stack(frames).astype("float32") / 255.0

    av = _try_import("av")
    if av is not None:
        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise RuntimeError("empty_video")
        return np.stack(frames).astype("float32") / 255.0

    raise RuntimeError("no_video_backend")


def compare_entries(reference: VideoEntry, candidate: VideoEntry) -> dict[str, Any]:
    record = {
        "record_type": "pair_metrics",
        "research_stage": STAGE3_ROLLOUT,
        "metric_scope": "paired_bf16_fidelity",
        "status": "ok",
        "pair_key": reference.pair_key(),
        "reference": {"label": reference.label, "path": reference.path},
        "candidate": {"label": candidate.label, "path": candidate.path},
        "controlled_fields": {
            "prompt_category": reference.prompt_category,
            "rollout_length": reference.rollout_length,
            "initial_latent_id": reference.initial_latent_id,
            "scheduler_id": reference.scheduler_id,
        },
        "metrics": {},
        "skipped_metrics": [],
    }
    if reference.prompt:
        record["prompt"] = reference.prompt

    ref_meta = inspect_video(reference.path)
    cand_meta = inspect_video(candidate.path)
    record["reference"]["media"] = ref_meta
    record["candidate"]["media"] = cand_meta

    for side, media in (("reference", ref_meta), ("candidate", cand_meta)):
        if media.get("error"):
            return _skip_record(record, f"{side}_{media['error']}")

    try:
        ref_frames = load_video_frames(reference.path)
        cand_frames = load_video_frames(candidate.path)
    except RuntimeError as exc:
        return _skip_record(record, str(exc))
    except ImportError as exc:
        return _skip_record(record, f"missing_dependency:{exc.name or exc}")

    if ref_frames.shape != cand_frames.shape:
        return _skip_record(
            record,
            "incompatible_video_shapes",
            {
                "reference_shape": list(ref_frames.shape),
                "candidate_shape": list(cand_frames.shape),
            },
        )

    record["metrics"] = compute_frame_metrics(ref_frames, cand_frames)
    return record


def compute_frame_metrics(reference_frames: Any, candidate_frames: Any) -> dict[str, Any]:
    np = _require_import("numpy")
    diff = candidate_frames.astype("float64") - reference_frames.astype("float64")
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr_db, psnr_infinite = _psnr_from_mse(mse, data_range=1.0)

    frame_mse = np.mean(diff * diff, axis=(1, 2, 3))
    frame_ssim = [_global_ssim(reference_frames[idx], candidate_frames[idx], np) for idx in range(reference_frames.shape[0])]

    metrics = {
        "frame_count": int(reference_frames.shape[0]),
        "height": int(reference_frames.shape[1]),
        "width": int(reference_frames.shape[2]),
        "channels": int(reference_frames.shape[3]),
        "mse": mse,
        "mae": mae,
        "psnr_db": _jsonable_number(psnr_db),
        "psnr_is_infinite": psnr_infinite,
        "ssim": float(np.mean(frame_ssim)),
        "frame_mse_mean": float(np.mean(frame_mse)),
        "frame_mse_max": float(np.max(frame_mse)),
    }

    if reference_frames.shape[0] < 2:
        metrics["temporal_delta_mse"] = None
        metrics["temporal_delta_psnr_db"] = None
        metrics["temporal_delta_psnr_is_infinite"] = False
        return metrics

    ref_delta = np.diff(reference_frames.astype("float64"), axis=0)
    cand_delta = np.diff(candidate_frames.astype("float64"), axis=0)
    delta_diff = cand_delta - ref_delta
    temporal_mse = float(np.mean(delta_diff * delta_diff))
    temporal_mae = float(np.mean(np.abs(delta_diff)))
    temporal_psnr_db, temporal_psnr_infinite = _psnr_from_mse(temporal_mse, data_range=2.0)
    metrics.update(
        {
            "temporal_delta_mse": temporal_mse,
            "temporal_delta_mae": temporal_mae,
            "temporal_delta_psnr_db": _jsonable_number(temporal_psnr_db),
            "temporal_delta_psnr_is_infinite": temporal_psnr_infinite,
        }
    )
    return metrics


def build_pairs(entries: Iterable[VideoEntry], reference_label: str) -> list[tuple[VideoEntry, VideoEntry] | dict[str, Any]]:
    groups: dict[tuple[str | None, ...], list[VideoEntry]] = defaultdict(list)
    for entry in entries:
        groups[tuple(getattr(entry, field) for field in PAIR_KEY_FIELDS)].append(entry)

    pairs: list[tuple[VideoEntry, VideoEntry] | dict[str, Any]] = []
    reference_label_norm = reference_label.lower()
    for key, group_entries in sorted(groups.items(), key=lambda item: str(item[0])):
        refs = [entry for entry in group_entries if entry.label.lower() == reference_label_norm]
        candidates = [entry for entry in group_entries if entry.label.lower() != reference_label_norm]
        pair_key = dict(zip(PAIR_KEY_FIELDS, key))
        if not refs:
            pairs.append(_group_skip(pair_key, "missing_reference", group_entries))
            continue
        if not candidates:
            pairs.append(_group_skip(pair_key, "missing_candidate", group_entries))
            continue
        for ref in refs:
            for candidate in candidates:
                pairs.append((ref, candidate))
    return pairs


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("record_type") == "pair_metrics" and record.get("status") == "ok"]
    skip_reasons = Counter(
        record.get("skip_reason", "unknown")
        for record in records
        if record.get("record_type") == "pair_metrics" and record.get("status") == "skip"
    )
    numeric: dict[str, list[float]] = defaultdict(list)
    for record in ok_records:
        for key, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                numeric[key].append(float(value))

    metrics = {
        key: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for key, values in sorted(numeric.items())
        if values
    }
    return {
        "record_type": "aggregate_summary",
        "research_stage": STAGE3_ROLLOUT,
        "status": "ok",
        "pair_count": len([record for record in records if record.get("record_type") == "pair_metrics"]),
        "ok_pair_count": len(ok_records),
        "skip_pair_count": sum(skip_reasons.values()),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "metrics": metrics,
    }


def inspect_command(args: argparse.Namespace) -> int:
    records = []
    for video_path in args.paths:
        media = inspect_video(video_path)
        records.append(
            {
                "record_type": "video_metadata",
                "research_stage": STAGE0_INFRASTRUCTURE,
                "metric_scope": "per_video_media_metadata",
                "status": "skip" if media.get("error") else "ok",
                "path": str(video_path),
                "media": media,
                "skip_reason": media.get("error"),
            }
        )
    _write_jsonl(records, args.output_jsonl)
    return 0


def compare_command(args: argparse.Namespace) -> int:
    if args.manifest:
        entries = read_manifest(args.manifest)
        pair_items = build_pairs(entries, args.reference_label)
        records = [
            item if isinstance(item, dict) else compare_entries(item[0], item[1])
            for item in pair_items
        ]
    else:
        reference = VideoEntry(
            path=args.reference,
            label=args.reference_label,
            prompt_id=args.prompt_id,
            prompt=args.prompt,
            config_id=args.config_id,
            seed=args.seed,
            run_label=args.run_label,
            prompt_category=args.prompt_category,
            rollout_length=args.rollout_length,
            initial_latent_id=args.initial_latent_id,
            scheduler_id=args.scheduler_id,
        )
        candidate = VideoEntry(
            path=args.candidate,
            label=args.candidate_label,
            prompt_id=args.prompt_id,
            prompt=args.prompt,
            config_id=args.config_id,
            seed=args.seed,
            run_label=args.run_label,
            prompt_category=args.prompt_category,
            rollout_length=args.rollout_length,
            initial_latent_id=args.initial_latent_id,
            scheduler_id=args.scheduler_id,
        )
        records = [compare_entries(reference, candidate)]

    summary = aggregate_records(records)
    records_with_summary = records + [summary]
    _write_jsonl(records_with_summary, args.output_jsonl)
    if args.summary_json:
        _write_json(args.summary_json, summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, pair, and compare generated videos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Extract per-video MP4 metadata.")
    inspect_parser.add_argument("paths", nargs="+", help="Video paths to inspect.")
    inspect_parser.add_argument("--output-jsonl", help="Optional JSONL output path.")
    inspect_parser.set_defaults(func=inspect_command)

    compare_parser = subparsers.add_parser("compare", help="Compare paired videos and aggregate metrics.")
    compare_input = compare_parser.add_mutually_exclusive_group(required=True)
    compare_input.add_argument("--manifest", help="JSONL or CSV manifest with path, label, prompt/config/seed/run fields.")
    compare_input.add_argument("--reference", help="Reference video path for explicit one-pair mode.")
    compare_parser.add_argument("--candidate", help="Candidate video path for explicit one-pair mode.")
    compare_parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL, help="Reference label. Default: bf16.")
    compare_parser.add_argument("--candidate-label", default="candidate", help="Candidate label for explicit one-pair mode.")
    compare_parser.add_argument("--prompt-id")
    compare_parser.add_argument("--prompt")
    compare_parser.add_argument("--config-id")
    compare_parser.add_argument("--seed")
    compare_parser.add_argument("--run-label")
    compare_parser.add_argument("--prompt-category")
    compare_parser.add_argument("--rollout-length")
    compare_parser.add_argument("--initial-latent-id")
    compare_parser.add_argument("--scheduler-id")
    compare_parser.add_argument("--output-jsonl", help="Optional JSONL output path.")
    compare_parser.add_argument("--summary-json", help="Optional aggregate summary JSON path.")
    compare_parser.set_defaults(func=compare_command)

    args = parser.parse_args(argv)
    if args.command == "compare" and not args.manifest and not args.candidate:
        parser.error("--candidate is required with --reference")
    return args.func(args)


def _group_skip(pair_key: dict[str, str | None], reason: str, entries: list[VideoEntry]) -> dict[str, Any]:
    return {
        "record_type": "pair_metrics",
        "research_stage": STAGE3_ROLLOUT,
        "metric_scope": "paired_bf16_fidelity",
        "status": "skip",
        "skip_reason": reason,
        "pair_key": pair_key,
        "entries": [asdict(entry) for entry in entries],
        "metrics": {},
        "skipped_metrics": ["all"],
    }


def _skip_record(record: dict[str, Any], reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record["status"] = "skip"
    record["skip_reason"] = reason
    record["metrics"] = {}
    record["skipped_metrics"] = ["all"]
    if extra:
        record.update(extra)
    return record


def _write_jsonl(records: list[dict[str, Any]], output_path: str | None) -> None:
    lines = [json.dumps(record, sort_keys=True, allow_nan=False) for record in records]
    if output_path:
        output = Path(output_path)
        if output.parent != Path("."):
            output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n")
    else:
        for line in lines:
            print(line)


def _write_json(output_path: str, payload: dict[str, Any]) -> None:
    output = Path(output_path)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _psnr_from_mse(mse: float, *, data_range: float) -> tuple[float | None, bool]:
    if mse == 0:
        return None, True
    return 20.0 * math.log10(data_range / math.sqrt(mse)), False


def _global_ssim(frame_a: Any, frame_b: Any, np: Any) -> float:
    a = frame_a.astype("float64")
    b = frame_b.astype("float64")
    c1 = 0.01**2
    c2 = 0.03**2
    channel_values = []
    for channel in range(a.shape[-1]):
        x = a[..., channel]
        y = b[..., channel]
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.mean((x - mu_x) ** 2))
        var_y = float(np.mean((y - mu_y) ** 2))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        channel_values.append(numerator / denominator if denominator else 1.0)
    return float(np.mean(channel_values))


def _try_import(module_name: str) -> Any | None:
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _require_import(module_name: str) -> Any:
    module = _try_import(module_name)
    if module is None:
        raise ImportError(module_name, name=module_name)
    return module


def _decode_fourcc(value: float) -> str | None:
    try:
        intval = int(value)
    except (TypeError, ValueError):
        return None
    chars = [chr((intval >> 8 * idx) & 0xFF) for idx in range(4)]
    text = "".join(chars).strip("\x00 ")
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
