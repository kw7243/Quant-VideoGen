#!/usr/bin/env python3
"""Check post-capture readiness for the bounded Self-Forcing attention job.

The checker is intentionally CPU-only and metadata-only. It does not query or
modify Slurm state, does not load tensor payloads, and treats missing pending
artifacts as a normal machine-readable ``not_ready`` state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "qvg-attention-capture-readiness.v1"
MANIFEST_SCHEMA_VERSION = "qvg-attention-capture-manifest.v1"
DEFAULT_JOB_ID = "1316140"
DEFAULT_TASK_ROOT = Path("/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run")
DEFAULT_EXPECTED_LAYERS = (27, 28, 29)
DEFAULT_EXPECTED_HEADS = (0, 9)
DEFAULT_EXPECTED_CHUNKS = (24, 25, 26)
DEFAULT_MAX_CAPTURE_FILE_BYTES = 64 * 1024**2
DEFAULT_PROBE_PYTHON = "/data/scratch-fast/kwen1/eval_envs/vbench/bin/python"


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, repr(exc)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _parse_int_list(value: Any) -> list[int] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"all", "*"}:
            return "all"
        if not stripped:
            return []
        try:
            return [int(part.strip()) for part in stripped.split(",") if part.strip()]
        except ValueError:
            return None
    if isinstance(value, (list, tuple, set)):
        parsed = []
        for item in value:
            item_int = _safe_int(item)
            if item_int is None:
                return None
            parsed.append(item_int)
        return parsed
    item_int = _safe_int(value)
    return [item_int] if item_int is not None else None


def _sorted_unique(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _path_info(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    candidate = Path(path).expanduser()
    info: dict[str, Any] = {"path": str(candidate), "exists": candidate.exists()}
    try:
        stat = candidate.stat()
    except OSError:
        return info
    info.update(
        {
            "is_file": candidate.is_file(),
            "is_dir": candidate.is_dir(),
            "size_bytes": stat.st_size,
            "mtime_unix": stat.st_mtime,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    )
    if candidate.is_dir():
        try:
            entries = list(candidate.iterdir())
        except OSError:
            entries = []
        info["entry_count"] = len(entries)
        info["pt_file_count"] = sum(1 for item in entries if item.is_file() and item.suffix == ".pt")
    return info


def _resolve_for_root(path: str | Path, task_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = task_root / candidate
    return candidate.resolve(strict=False)


def _is_task_local(path: str | Path | None, task_root: Path) -> bool:
    if path is None:
        return False
    resolved_root = task_root.resolve(strict=False)
    resolved_path = _resolve_for_root(path, task_root)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def _quote_command(parts: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _default_probe_output(task_root: Path, job_id: str, mode: str) -> Path:
    return task_root / "attention_probe_results" / f"smallest_functional_attention_{job_id}_{mode}.json"


def _make_probe_command(
    *,
    probe_python: str,
    tool_path: Path,
    input_path: str,
    output_path: Path,
    identity_variant_if_missing: bool,
) -> str:
    parts: list[str | Path] = [
        probe_python,
        tool_path,
        "--input",
        input_path,
        "--record-limit",
        "1",
    ]
    if identity_variant_if_missing:
        parts.append("--identity-variant-if-missing")
    parts.extend(["-o", output_path])
    return _quote_command(parts)


def evaluate_readiness(
    *,
    task_root: str | Path = DEFAULT_TASK_ROOT,
    job_id: str = DEFAULT_JOB_ID,
    submission_path: str | Path | None = None,
    expected_layers: Iterable[int] = DEFAULT_EXPECTED_LAYERS,
    expected_heads: Iterable[int] = DEFAULT_EXPECTED_HEADS,
    expected_chunks: Iterable[int] = DEFAULT_EXPECTED_CHUNKS,
    expected_max_capture_file_bytes: int = DEFAULT_MAX_CAPTURE_FILE_BYTES,
    expected_quant_type: str = "none",
    probe_python: str = DEFAULT_PROBE_PYTHON,
    probe_output: str | Path | None = None,
    tool_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(task_root).expanduser().resolve(strict=False)
    submission = Path(submission_path).expanduser() if submission_path else root / f"submission_{job_id}.json"
    expected = {
        "job_id": str(job_id),
        "layers": _sorted_unique(expected_layers),
        "heads": _sorted_unique(expected_heads),
        "chunks": _sorted_unique(expected_chunks),
        "max_capture_file_bytes": int(expected_max_capture_file_bytes),
        "quant_type": str(expected_quant_type),
    }
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    def add_check(name: str, status: str, *, code: str | None = None, details: Mapping[str, Any] | None = None) -> None:
        check: dict[str, Any] = {"name": name, "status": status}
        if code:
            check["code"] = code
        if details:
            check["details"] = dict(details)
        checks.append(check)
        if status in {"not_ready", "invalid"}:
            reasons.append(code or name)
        elif status == "warning":
            warnings.append(code or name)

    submission_info = _path_info(submission)
    artifacts: dict[str, Any] = {"submission_metadata": submission_info}
    metadata: Mapping[str, Any] | None = None
    if not submission_info["exists"]:
        add_check("submission_metadata", "not_ready", code="submission_metadata_missing", details=submission_info)
    elif not submission_info.get("is_file"):
        add_check("submission_metadata", "invalid", code="submission_metadata_not_file", details=submission_info)
    else:
        loaded, error = _read_json(submission)
        if error is not None or not isinstance(loaded, Mapping):
            add_check(
                "submission_metadata",
                "invalid",
                code="submission_metadata_invalid_json",
                details={**submission_info, "error": error},
            )
        else:
            metadata = loaded
            add_check("submission_metadata", "ok", details=submission_info)

    declared_paths: dict[str, str | None] = {}
    actual_probe: dict[str, Any] = {}
    if metadata is not None:
        if str(metadata.get("job_id")) != str(job_id):
            add_check(
                "job_id",
                "invalid",
                code="job_id_mismatch",
                details={"expected": str(job_id), "actual": metadata.get("job_id")},
            )
        else:
            add_check("job_id", "ok", details={"job_id": str(job_id)})

        if str(metadata.get("quant_type")) != expected["quant_type"]:
            add_check(
                "quant_type",
                "invalid",
                code="quant_type_mismatch",
                details={"expected": expected["quant_type"], "actual": metadata.get("quant_type")},
            )
        else:
            add_check("quant_type", "ok", details={"quant_type": metadata.get("quant_type")})

        probe = metadata.get("attention_probe")
        if not isinstance(probe, Mapping):
            add_check("attention_probe_metadata", "invalid", code="attention_probe_metadata_missing")
        else:
            actual_probe = {
                "enabled": _boolish(probe.get("enabled")),
                "layers": _parse_int_list(probe.get("layers")),
                "heads": _parse_int_list(probe.get("heads")),
                "chunks": _parse_int_list(probe.get("chunks")),
                "max_file_bytes": _safe_int(probe.get("max_file_bytes")),
                "variant": probe.get("variant"),
            }
            mismatches = {}
            if actual_probe["enabled"] is not True:
                mismatches["enabled"] = {"expected": True, "actual": probe.get("enabled")}
            for key in ("layers", "heads", "chunks"):
                if actual_probe[key] != expected[key]:
                    mismatches[key] = {"expected": expected[key], "actual": actual_probe[key]}
            if actual_probe["max_file_bytes"] is None or actual_probe["max_file_bytes"] > expected["max_capture_file_bytes"]:
                mismatches["max_file_bytes"] = {
                    "expected_at_most": expected["max_capture_file_bytes"],
                    "actual": actual_probe["max_file_bytes"],
                }
            if mismatches:
                add_check(
                    "attention_probe_metadata",
                    "invalid",
                    code="attention_probe_metadata_mismatch",
                    details=mismatches,
                )
            else:
                add_check("attention_probe_metadata", "ok", details=actual_probe)

        for key in ("log_stdout", "log_stderr", "output_dir", "capture_dir", "capture_manifest"):
            value = metadata.get(key)
            declared_paths[key] = str(value) if value is not None else None
            if value is None:
                add_check("artifact_paths", "invalid", code=f"{key}_missing_from_submission")
            elif not _is_task_local(value, root):
                add_check(
                    "artifact_paths",
                    "invalid",
                    code=f"{key}_not_task_local",
                    details={"key": key, "path": value, "task_root": str(root)},
                )

        preflight = metadata.get("preflight")
        if isinstance(preflight, Mapping):
            missing_preflight = []
            for key, value in preflight.items():
                if value and _is_task_local(value, root):
                    info = _path_info(value)
                    if not info["exists"]:
                        missing_preflight.append(str(key))
                else:
                    missing_preflight.append(str(key))
            if missing_preflight:
                add_check("preflight_files", "warning", code="preflight_files_missing", details={"keys": missing_preflight})
            else:
                add_check("preflight_files", "ok", details={"keys": sorted(str(key) for key in preflight)})

    stdout_info = _path_info(declared_paths.get("log_stdout"))
    stderr_info = _path_info(declared_paths.get("log_stderr"))
    output_dir_info = _path_info(declared_paths.get("output_dir"))
    capture_dir_info = _path_info(declared_paths.get("capture_dir"))
    manifest_info = _path_info(declared_paths.get("capture_manifest"))
    artifacts.update(
        {
            "stdout_log": stdout_info,
            "stderr_log": stderr_info,
            "output_dir": output_dir_info,
            "capture_dir": capture_dir_info,
            "capture_manifest": manifest_info,
        }
    )

    if declared_paths.get("log_stdout") is not None:
        if stdout_info.get("is_file"):
            add_check("stdout_log", "ok", details=stdout_info)
        else:
            add_check("stdout_log", "not_ready", code="stdout_log_missing", details=stdout_info)
    if declared_paths.get("log_stderr") is not None:
        if stderr_info.get("is_file"):
            add_check("stderr_log", "ok", details=stderr_info)
            if int(stderr_info.get("size_bytes") or 0) > 0:
                add_check("stderr_log_content", "warning", code="stderr_log_nonempty", details=stderr_info)
        else:
            add_check("stderr_log", "not_ready", code="stderr_log_missing", details=stderr_info)
    if declared_paths.get("output_dir") is not None:
        if output_dir_info.get("is_dir"):
            add_check("output_dir", "ok", details=output_dir_info)
            if int(output_dir_info.get("entry_count") or 0) == 0:
                add_check("output_dir_content", "warning", code="output_dir_empty", details=output_dir_info)
        else:
            add_check("output_dir", "not_ready", code="output_dir_missing", details=output_dir_info)
    if declared_paths.get("capture_dir") is not None:
        if capture_dir_info.get("is_dir"):
            add_check("capture_dir", "ok", details=capture_dir_info)
        else:
            add_check("capture_dir", "not_ready", code="capture_dir_missing", details=capture_dir_info)

    manifest: Mapping[str, Any] | None = None
    capture_records: list[Mapping[str, Any]] = []
    capture_files: list[dict[str, Any]] = []
    has_variant_records = False
    chosen_capture_path: str | None = None
    if declared_paths.get("capture_manifest") is not None:
        if not manifest_info.get("is_file"):
            add_check("capture_manifest", "not_ready", code="capture_manifest_missing", details=manifest_info)
        else:
            loaded, error = _read_json(Path(declared_paths["capture_manifest"]))
            if error is not None or not isinstance(loaded, Mapping):
                add_check(
                    "capture_manifest",
                    "invalid",
                    code="capture_manifest_invalid_json",
                    details={**manifest_info, "error": error},
                )
            else:
                manifest = loaded
                add_check("capture_manifest", "ok", details=manifest_info)

    if manifest is not None:
        manifest_schema = manifest.get("schema_version")
        if manifest_schema != MANIFEST_SCHEMA_VERSION:
            add_check(
                "capture_manifest_schema",
                "invalid",
                code="capture_manifest_schema_mismatch",
                details={"expected": MANIFEST_SCHEMA_VERSION, "actual": manifest_schema},
            )
        else:
            add_check("capture_manifest_schema", "ok", details={"schema_version": manifest_schema})

        manifest_capture_dir = manifest.get("capture_dir")
        if manifest_capture_dir and declared_paths.get("capture_dir"):
            if _resolve_for_root(manifest_capture_dir, root) != _resolve_for_root(declared_paths["capture_dir"], root):
                add_check(
                    "capture_manifest_dir",
                    "invalid",
                    code="capture_manifest_dir_mismatch",
                    details={"expected": declared_paths["capture_dir"], "actual": manifest_capture_dir},
                )
            else:
                add_check("capture_manifest_dir", "ok", details={"capture_dir": manifest_capture_dir})

        captures = manifest.get("captures")
        if not isinstance(captures, list):
            add_check("capture_manifest_captures", "invalid", code="capture_manifest_captures_missing")
            captures = []
        declared_count = _safe_int(manifest.get("capture_file_count"))
        if declared_count != len(captures):
            add_check(
                "capture_manifest_count",
                "invalid",
                code="capture_manifest_count_mismatch",
                details={"declared": declared_count, "actual": len(captures)},
            )
        elif len(captures) == 0:
            add_check("capture_manifest_count", "not_ready", code="capture_manifest_empty")
        else:
            add_check("capture_manifest_count", "ok", details={"capture_file_count": len(captures)})

        total_size = 0
        oversized = []
        missing_capture_files = []
        mismatched_sizes = []
        for idx, capture in enumerate(captures):
            if not isinstance(capture, Mapping):
                continue
            path = capture.get("path")
            info = _path_info(path)
            declared_size = _safe_int(capture.get("size_bytes"))
            info["declared_size_bytes"] = declared_size
            info["manifest_index"] = idx
            if path and not _is_task_local(path, root):
                info["task_local"] = False
                oversized.append({"path": path, "reason": "not_task_local"})
            else:
                info["task_local"] = True
            if not info.get("is_file"):
                missing_capture_files.append(str(path))
            else:
                actual_size = int(info.get("size_bytes") or 0)
                total_size += actual_size
                if declared_size is not None and declared_size != actual_size:
                    mismatched_sizes.append({"path": path, "declared": declared_size, "actual": actual_size})
                max_file_bytes = actual_probe.get("max_file_bytes") or expected["max_capture_file_bytes"]
                if actual_size > int(max_file_bytes):
                    oversized.append({"path": path, "size_bytes": actual_size, "max_file_bytes": int(max_file_bytes)})
                if chosen_capture_path is None:
                    chosen_capture_path = str(path)
            records = capture.get("records")
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, Mapping):
                        capture_records.append(record)
                        if bool(record.get("has_variants")):
                            has_variant_records = True
                            chosen_capture_path = str(path)
            capture_files.append(info)

        if missing_capture_files:
            add_check(
                "capture_files",
                "not_ready",
                code="capture_files_missing",
                details={"paths": missing_capture_files},
            )
        elif mismatched_sizes:
            add_check(
                "capture_files",
                "invalid",
                code="capture_file_size_mismatch",
                details={"mismatches": mismatched_sizes},
            )
        elif oversized:
            add_check(
                "capture_byte_guards",
                "invalid",
                code="capture_file_exceeds_byte_guard",
                details={"oversized": oversized},
            )
        elif captures:
            add_check(
                "capture_files",
                "ok",
                details={
                    "capture_file_count": len(captures),
                    "capture_total_size_bytes": total_size,
                    "max_capture_file_bytes": actual_probe.get("max_file_bytes") or expected["max_capture_file_bytes"],
                },
            )

        declared_total = _safe_int(manifest.get("capture_total_size_bytes"))
        if declared_total is not None and declared_total != total_size and not missing_capture_files:
            add_check(
                "capture_manifest_total_size",
                "invalid",
                code="capture_manifest_total_size_mismatch",
                details={"declared": declared_total, "actual": total_size},
            )

        if not capture_records:
            add_check("capture_records", "not_ready", code="capture_records_missing")
        else:
            layers_seen = _sorted_unique(
                value
                for value in (_safe_int(record.get("layer_idx")) for record in capture_records)
                if value is not None
            )
            chunks_seen = _sorted_unique(
                value
                for value in (_safe_int(record.get("chunk_idx")) for record in capture_records)
                if value is not None
            )
            selected_head_mismatches = []
            selected_chunk_mismatches = []
            unexpected_layers = sorted(set(layers_seen) - set(expected["layers"]))
            unexpected_chunks = sorted(set(chunks_seen) - set(expected["chunks"]))
            missing_layers = sorted(set(expected["layers"]) - set(layers_seen))
            missing_chunks = sorted(set(expected["chunks"]) - set(chunks_seen))
            for record in capture_records:
                selected_heads = _parse_int_list(record.get("selected_heads"))
                selected_chunks = _parse_int_list(record.get("selected_chunks"))
                if selected_heads != expected["heads"]:
                    selected_head_mismatches.append(
                        {"name": record.get("name"), "expected": expected["heads"], "actual": selected_heads}
                    )
                if selected_chunks != expected["chunks"]:
                    selected_chunk_mismatches.append(
                        {"name": record.get("name"), "expected": expected["chunks"], "actual": selected_chunks}
                    )
            selection_errors: dict[str, Any] = {}
            if unexpected_layers:
                selection_errors["unexpected_layers"] = unexpected_layers
            if unexpected_chunks:
                selection_errors["unexpected_chunks"] = unexpected_chunks
            if missing_layers:
                selection_errors["missing_layers"] = missing_layers
            if missing_chunks:
                selection_errors["missing_chunks"] = missing_chunks
            if selected_head_mismatches:
                selection_errors["selected_head_mismatches"] = selected_head_mismatches
            if selected_chunk_mismatches:
                selection_errors["selected_chunk_mismatches"] = selected_chunk_mismatches
            if selection_errors:
                add_check("capture_selection", "invalid", code="capture_selection_mismatch", details=selection_errors)
            else:
                add_check(
                    "capture_selection",
                    "ok",
                    details={
                        "layers_seen": layers_seen,
                        "chunks_seen": chunks_seen,
                        "selected_heads": expected["heads"],
                        "selected_chunks": expected["chunks"],
                        "record_count": len(capture_records),
                    },
                )

    artifact_ready = not reasons
    mode: str | None = None
    next_command: str | None = None
    command_note: str | None = None
    if artifact_ready and chosen_capture_path is not None:
        mode = "variant_distortion" if has_variant_records else "identity_reference"
        identity = not has_variant_records
        output_path = Path(probe_output) if probe_output is not None else _default_probe_output(root, str(job_id), mode)
        next_command = _make_probe_command(
            probe_python=probe_python,
            tool_path=Path(tool_path) if tool_path is not None else Path(__file__).resolve().with_name("attention_error_probe.py"),
            input_path=chosen_capture_path,
            output_path=output_path,
            identity_variant_if_missing=identity,
        )
        if identity:
            command_note = (
                "The submitted BF16/reference capture is not expected to contain quantized variants. "
                "This command runs an identity reference self-check of the offline attention path; "
                "add paired variants before interpreting distortion metrics."
            )
            notes.append(command_note)
    elif artifact_ready:
        reasons.append("no_capture_path_for_probe_command")
        artifact_ready = False

    status = "ready" if artifact_ready else "not_ready"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": status,
        "ready": bool(artifact_ready),
        "artifact_ready": bool(artifact_ready),
        "variant_distortion_ready": bool(artifact_ready and has_variant_records),
        "analysis_mode": mode,
        "job_id": str(job_id),
        "task_root": str(root),
        "expected": expected,
        "submission": metadata if metadata is not None else None,
        "artifacts": artifacts,
        "capture_files": capture_files,
        "capture_record_count": len(capture_records),
        "has_variant_records": bool(has_variant_records),
        "next_attention_error_probe_command": next_command,
        "next_command_note": command_note,
        "checks": checks,
        "reasons": reasons,
        "warnings": warnings,
        "notes": notes,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT))
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--submission", help="Override submission JSON path")
    parser.add_argument("--expected-layers", default=",".join(str(item) for item in DEFAULT_EXPECTED_LAYERS))
    parser.add_argument("--expected-heads", default=",".join(str(item) for item in DEFAULT_EXPECTED_HEADS))
    parser.add_argument("--expected-chunks", default=",".join(str(item) for item in DEFAULT_EXPECTED_CHUNKS))
    parser.add_argument("--max-capture-file-bytes", type=_positive_int, default=DEFAULT_MAX_CAPTURE_FILE_BYTES)
    parser.add_argument("--expected-quant-type", default="none")
    parser.add_argument("--probe-python", default=DEFAULT_PROBE_PYTHON)
    parser.add_argument("--probe-output", help="Override output JSON path for the next probe command")
    parser.add_argument("-o", "--output", help="Write readiness JSON to this path")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    parser.add_argument("--fail-on-not-ready", action="store_true", help="Return exit code 1 when status is not ready")
    return parser


def _parse_required_int_arg(value: str, name: str) -> list[int]:
    parsed = _parse_int_list(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list")
    return parsed


def _write_json(data: Mapping[str, Any], output: str | None, *, compact: bool) -> None:
    text = json.dumps(data, indent=None if compact else 2, sort_keys=True, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_readiness(
            task_root=args.task_root,
            job_id=args.job_id,
            submission_path=args.submission,
            expected_layers=_parse_required_int_arg(args.expected_layers, "expected-layers"),
            expected_heads=_parse_required_int_arg(args.expected_heads, "expected-heads"),
            expected_chunks=_parse_required_int_arg(args.expected_chunks, "expected-chunks"),
            expected_max_capture_file_bytes=args.max_capture_file_bytes,
            expected_quant_type=args.expected_quant_type,
            probe_python=args.probe_python,
            probe_output=args.probe_output,
        )
    except Exception as exc:
        print(json.dumps({"status": "readiness_check_failed", "error": repr(exc)}), file=sys.stderr)
        return 2
    _write_json(result, args.output, compact=args.compact)
    if args.fail_on_not_ready and not result.get("ready"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
