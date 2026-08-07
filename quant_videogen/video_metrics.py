"""CPU-friendly video metadata, pairing, and metric aggregation utilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
PAIRING_PROTOCOL_VERSION = "qvg_pairing_protocol_v1"
LATENT_INFERRED_SEED_ONLY = "inferred_seed_only"
LATENT_EXPLICIT_HASH = "explicit_latent_hash"
LATENT_IDENTITY_MODES = (LATENT_INFERRED_SEED_ONLY, LATENT_EXPLICIT_HASH)


@dataclass(frozen=True)
class VideoEntry:
    path: str
    label: str
    prompt_id: str | None = None
    prompt: str | None = None
    prompt_hash: str | None = None
    config_id: str | None = None
    seed: str | None = None
    run_label: str | None = None
    prompt_category: str | None = None
    rollout_length: str | None = None
    frame_count: str | None = None
    local_attn_size: str | None = None
    initial_latent_id: str | None = None
    initial_latent_hash: str | None = None
    latent_identity_mode: str | None = None
    scheduler_id: str | None = None
    scheduler_config_hash: str | None = None
    checkpoint_id: str | None = None
    checkpoint_hash: str | None = None
    quant_type: str | None = None
    code_commit: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "VideoEntry":
        path = row.get("path") or row.get("artifact_path") or row.get("video_path") or row.get("file")
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
            prompt_hash=_optional_str(row.get("prompt_hash", row.get("prompt_sha256"))),
            config_id=_optional_str(row.get("config_id", row.get("config"))),
            seed=_optional_str(row.get("seed")),
            run_label=_optional_str(row.get("run_label", row.get("run"))),
            prompt_category=_optional_str(row.get("prompt_category", row.get("motion_category"))),
            rollout_length=_optional_str(row.get("rollout_length", row.get("num_frames"))),
            frame_count=_optional_str(row.get("frame_count", row.get("num_frames", row.get("rollout_length")))),
            local_attn_size=_optional_str(row.get("local_attn_size", row.get("local_attention_size"))),
            initial_latent_id=_optional_str(row.get("initial_latent_id")),
            initial_latent_hash=_optional_str(row.get("initial_latent_hash", row.get("latent_hash"))),
            latent_identity_mode=_optional_str(row.get("latent_identity_mode", row.get("latent_mode"))),
            scheduler_id=_optional_str(row.get("scheduler_id", row.get("scheduler"))),
            scheduler_config_hash=_optional_str(row.get("scheduler_config_hash", row.get("scheduler_hash"))),
            checkpoint_id=_optional_str(row.get("checkpoint_id", row.get("checkpoint", row.get("checkpoint_path")))),
            checkpoint_hash=_optional_str(row.get("checkpoint_hash", row.get("checkpoint_sha256"))),
            quant_type=_optional_str(row.get("quant_type", row.get("method", row.get("variant", label)))),
            code_commit=_optional_str(row.get("code_commit", row.get("git_commit", row.get("git_commit_full")))),
        )

    def pair_key(self) -> dict[str, str | None]:
        return {field: getattr(self, field) for field in PAIR_KEY_FIELDS}

    def canonical_prompt_hash(self) -> str | None:
        return self.prompt_hash or _sha256_text(self.prompt)

    def scheduler_identity(self) -> str | None:
        return _joined_identity(("scheduler_id", self.scheduler_id), ("scheduler_config_hash", self.scheduler_config_hash))

    def checkpoint_identity(self) -> str | None:
        return _joined_identity(("checkpoint_id", self.checkpoint_id), ("checkpoint_hash", self.checkpoint_hash))

    def latent_identity(self) -> str | None:
        if self.latent_identity_mode == LATENT_INFERRED_SEED_ONLY:
            return LATENT_INFERRED_SEED_ONLY
        if self.latent_identity_mode == LATENT_EXPLICIT_HASH:
            return _joined_identity(("initial_latent_hash", self.initial_latent_hash), ("initial_latent_id", self.initial_latent_id))
        return None

    def pairing_ledger(self) -> dict[str, Any]:
        return {
            "pairing_protocol_version": PAIRING_PROTOCOL_VERSION,
            "artifact_path": self.path,
            "label": self.label,
            "quant_type": self.quant_type,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.canonical_prompt_hash(),
            "seed": self.seed,
            "latent_identity_mode": self.latent_identity_mode,
            "initial_latent_id": self.initial_latent_id,
            "initial_latent_hash": self.initial_latent_hash,
            "scheduler_id": self.scheduler_id,
            "scheduler_config_hash": self.scheduler_config_hash,
            "config_id": self.config_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_hash": self.checkpoint_hash,
            "frame_count": self.frame_count,
            "local_attn_size": self.local_attn_size,
            "run_label": self.run_label,
            "code_commit": self.code_commit,
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _joined_identity(*parts: tuple[str, str | None]) -> str | None:
    present = [f"{name}:{value}" for name, value in parts if value]
    return "|".join(present) if present else None


def validate_entry_identity(entry: VideoEntry) -> list[str]:
    issues: list[str] = []
    if not entry.path:
        issues.append("missing_artifact_path")
    if not (entry.prompt_id or entry.canonical_prompt_hash()):
        issues.append("missing_prompt_identity")
    if not entry.seed:
        issues.append("missing_seed")
    if not entry.config_id:
        issues.append("missing_config_id")
    if not entry.run_label:
        issues.append("missing_run_label")
    if not entry.frame_count:
        issues.append("missing_frame_count")
    if not entry.local_attn_size:
        issues.append("missing_local_attn_size")
    if not entry.quant_type:
        issues.append("missing_quant_type")
    if not entry.code_commit:
        issues.append("missing_code_commit")
    if not entry.scheduler_identity():
        issues.append("missing_scheduler_identity")
    if not entry.checkpoint_identity():
        issues.append("missing_checkpoint_identity")

    if not entry.latent_identity_mode:
        issues.append("missing_latent_identity_mode")
    elif entry.latent_identity_mode not in LATENT_IDENTITY_MODES:
        issues.append(f"invalid_latent_identity_mode:{entry.latent_identity_mode}")
    elif entry.latent_identity_mode == LATENT_EXPLICIT_HASH and not entry.latent_identity():
        issues.append("missing_explicit_latent_hash")

    return issues


def validate_pair_identity(reference: VideoEntry, candidate: VideoEntry) -> dict[str, Any]:
    issues: list[str] = []
    ref_issues = validate_entry_identity(reference)
    cand_issues = validate_entry_identity(candidate)
    for issue in ref_issues:
        issues.append(f"reference_{issue}")
    for issue in cand_issues:
        issues.append(f"candidate_{issue}")

    issues.extend(_compare_prompt_identity(reference, candidate))
    issues.extend(_compare_identity_family(reference, candidate, "scheduler", "scheduler_id", "scheduler_config_hash"))
    issues.extend(_compare_identity_family(reference, candidate, "checkpoint", "checkpoint_id", "checkpoint_hash"))
    issues.extend(_compare_latent_identity(reference, candidate))

    for field in ("seed", "config_id", "frame_count", "local_attn_size", "run_label", "code_commit"):
        ref_value = getattr(reference, field)
        cand_value = getattr(candidate, field)
        if ref_value and cand_value and ref_value != cand_value:
            issues.append(f"{field}_mismatch")

    if reference.quant_type and candidate.quant_type and reference.quant_type == candidate.quant_type:
        issues.append("quant_type_not_changed")

    return {
        "pairing_protocol_version": PAIRING_PROTOCOL_VERSION,
        "status": "invalid" if issues else "ok",
        "issues": issues,
        "reference": reference.pairing_ledger(),
        "candidate": candidate.pairing_ledger(),
    }


def _compare_prompt_identity(reference: VideoEntry, candidate: VideoEntry) -> list[str]:
    issues: list[str] = []
    ref_hash = reference.canonical_prompt_hash()
    cand_hash = candidate.canonical_prompt_hash()
    shared_identity = False
    if reference.prompt_id and candidate.prompt_id:
        shared_identity = True
        if reference.prompt_id != candidate.prompt_id:
            issues.append("prompt_id_mismatch")
    if ref_hash and cand_hash:
        shared_identity = True
        if ref_hash != cand_hash:
            issues.append("prompt_hash_mismatch")
    if (reference.prompt_id or ref_hash) and (candidate.prompt_id or cand_hash) and not shared_identity:
        issues.append("prompt_identity_not_comparable")
    return issues


def _compare_identity_family(reference: VideoEntry, candidate: VideoEntry, family: str, *fields: str) -> list[str]:
    issues: list[str] = []
    shared_identity = False
    for field in fields:
        ref_value = getattr(reference, field)
        cand_value = getattr(candidate, field)
        if ref_value and cand_value:
            shared_identity = True
            if ref_value != cand_value:
                issues.append(f"{field}_mismatch")
    if any(getattr(reference, field) for field in fields) and any(getattr(candidate, field) for field in fields) and not shared_identity:
        issues.append(f"{family}_identity_not_comparable")
    return issues


def _compare_latent_identity(reference: VideoEntry, candidate: VideoEntry) -> list[str]:
    issues: list[str] = []
    if reference.latent_identity_mode and candidate.latent_identity_mode:
        if reference.latent_identity_mode != candidate.latent_identity_mode:
            issues.append("latent_identity_mode_mismatch")
            return issues
        if reference.latent_identity_mode == LATENT_EXPLICIT_HASH:
            issues.extend(_compare_identity_family(reference, candidate, "latent", "initial_latent_hash", "initial_latent_id"))
    return issues


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
    pairing_protocol = validate_pair_identity(reference, candidate)
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
            "frame_count": reference.frame_count,
            "local_attn_size": reference.local_attn_size,
            "initial_latent_id": reference.initial_latent_id,
            "initial_latent_hash": reference.initial_latent_hash,
            "latent_identity_mode": reference.latent_identity_mode,
            "scheduler_id": reference.scheduler_id,
            "scheduler_config_hash": reference.scheduler_config_hash,
            "checkpoint_id": reference.checkpoint_id,
            "checkpoint_hash": reference.checkpoint_hash,
            "code_commit": reference.code_commit,
        },
        "pairing_protocol": pairing_protocol,
        "metrics": {},
        "skipped_metrics": [],
    }
    if reference.prompt:
        record["prompt"] = reference.prompt

    if pairing_protocol["status"] != "ok":
        return _invalid_record(record, "invalid_pairing_protocol")

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
    invalid_reasons = Counter(
        record.get("invalid_reason", "unknown")
        for record in records
        if record.get("record_type") == "pair_metrics" and record.get("status") == "invalid"
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
        "invalid_pair_count": sum(invalid_reasons.values()),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "metrics": metrics,
    }


def _pairing_validation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(record.get("status", "unknown") for record in records)
    invalid_reasons = Counter(
        record.get("invalid_reason", "unknown")
        for record in records
        if record.get("status") == "invalid" and record.get("invalid_reason")
    )
    skip_reasons = Counter(
        record.get("skip_reason", "unknown")
        for record in records
        if record.get("status") == "skip" and record.get("skip_reason")
    )
    return {
        "record_type": "pairing_validation_summary",
        "research_stage": STAGE3_ROLLOUT,
        "status": "ok" if not statuses.get("invalid") else "invalid",
        "record_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
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
        frame_count = args.frame_count or args.rollout_length
        reference = VideoEntry(
            path=args.reference,
            label=args.reference_label,
            prompt_id=args.prompt_id,
            prompt=args.prompt,
            prompt_hash=args.prompt_hash,
            config_id=args.config_id,
            seed=args.seed,
            run_label=args.run_label,
            prompt_category=args.prompt_category,
            rollout_length=args.rollout_length,
            frame_count=frame_count,
            local_attn_size=args.local_attn_size,
            initial_latent_id=args.initial_latent_id,
            initial_latent_hash=args.initial_latent_hash,
            latent_identity_mode=args.latent_identity_mode,
            scheduler_id=args.scheduler_id,
            scheduler_config_hash=args.scheduler_config_hash,
            checkpoint_id=args.checkpoint_id,
            checkpoint_hash=args.checkpoint_hash,
            quant_type=args.reference_quant_type or args.reference_label,
            code_commit=args.code_commit,
        )
        candidate = VideoEntry(
            path=args.candidate,
            label=args.candidate_label,
            prompt_id=args.prompt_id,
            prompt=args.prompt,
            prompt_hash=args.prompt_hash,
            config_id=args.config_id,
            seed=args.seed,
            run_label=args.run_label,
            prompt_category=args.prompt_category,
            rollout_length=args.rollout_length,
            frame_count=frame_count,
            local_attn_size=args.local_attn_size,
            initial_latent_id=args.initial_latent_id,
            initial_latent_hash=args.initial_latent_hash,
            latent_identity_mode=args.latent_identity_mode,
            scheduler_id=args.scheduler_id,
            scheduler_config_hash=args.scheduler_config_hash,
            checkpoint_id=args.checkpoint_id,
            checkpoint_hash=args.checkpoint_hash,
            quant_type=args.candidate_quant_type or args.candidate_label,
            code_commit=args.code_commit,
        )
        records = [compare_entries(reference, candidate)]

    summary = aggregate_records(records)
    records_with_summary = records + [summary]
    _write_jsonl(records_with_summary, args.output_jsonl)
    if args.summary_json:
        _write_json(args.summary_json, summary)
    return 0


def validate_manifest_command(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    records = []
    for entry in entries:
        issues = validate_entry_identity(entry)
        records.append(
            {
                "record_type": "pairing_ledger_entry",
                "research_stage": STAGE3_ROLLOUT,
                "metric_scope": "paired_artifact_identity",
                "status": "invalid" if issues else "ok",
                "issues": issues,
                "entry": entry.pairing_ledger(),
            }
        )

    pair_items = build_pairs(entries, args.reference_label)
    for item in pair_items:
        if isinstance(item, dict):
            records.append(
                {
                    "record_type": "pairing_validation",
                    "research_stage": STAGE3_ROLLOUT,
                    "metric_scope": "paired_artifact_identity",
                    "status": "skip",
                    "skip_reason": item["skip_reason"],
                    "pair_key": item["pair_key"],
                    "entries": item["entries"],
                }
            )
            continue
        reference, candidate = item
        protocol = validate_pair_identity(reference, candidate)
        records.append(
            {
                "record_type": "pairing_validation",
                "research_stage": STAGE3_ROLLOUT,
                "metric_scope": "paired_artifact_identity",
                "status": protocol["status"],
                "invalid_reason": "invalid_pairing_protocol" if protocol["status"] == "invalid" else None,
                "pair_key": reference.pair_key(),
                "pairing_protocol": protocol,
            }
        )

    summary = _pairing_validation_summary(records)
    _write_jsonl(records + [summary], args.output_jsonl)
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
    compare_parser.add_argument("--reference-quant-type", help="Reference quant/cache type. Defaults to --reference-label.")
    compare_parser.add_argument("--candidate-quant-type", help="Candidate quant/cache type. Defaults to --candidate-label.")
    compare_parser.add_argument("--prompt-id")
    compare_parser.add_argument("--prompt")
    compare_parser.add_argument("--prompt-hash")
    compare_parser.add_argument("--config-id")
    compare_parser.add_argument("--seed")
    compare_parser.add_argument("--run-label")
    compare_parser.add_argument("--prompt-category")
    compare_parser.add_argument("--rollout-length")
    compare_parser.add_argument("--frame-count")
    compare_parser.add_argument("--local-attn-size")
    compare_parser.add_argument("--initial-latent-id")
    compare_parser.add_argument("--initial-latent-hash")
    compare_parser.add_argument("--latent-identity-mode", choices=LATENT_IDENTITY_MODES)
    compare_parser.add_argument("--scheduler-id")
    compare_parser.add_argument("--scheduler-config-hash")
    compare_parser.add_argument("--checkpoint-id")
    compare_parser.add_argument("--checkpoint-hash")
    compare_parser.add_argument("--code-commit")
    compare_parser.add_argument("--output-jsonl", help="Optional JSONL output path.")
    compare_parser.add_argument("--summary-json", help="Optional aggregate summary JSON path.")
    compare_parser.set_defaults(func=compare_command)

    validate_parser = subparsers.add_parser("validate-manifest", help="Validate pairing identity fields without decoding videos.")
    validate_parser.add_argument("--manifest", required=True, help="JSONL or CSV manifest to validate.")
    validate_parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL, help="Reference label. Default: bf16.")
    validate_parser.add_argument("--output-jsonl", help="Optional JSONL ledger output path.")
    validate_parser.add_argument("--summary-json", help="Optional validation summary JSON path.")
    validate_parser.set_defaults(func=validate_manifest_command)

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


def _invalid_record(record: dict[str, Any], reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record["status"] = "invalid"
    record["invalid_reason"] = reason
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
