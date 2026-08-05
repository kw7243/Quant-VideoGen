#!/usr/bin/env python3
"""Bounded capture-schema helpers for future attention-error probe dumps.

This module is intentionally opt-in and side-effect free on import. Future
Self-Forcing instrumentation can call ``CaptureConfig.from_env()`` and then
``write_self_forcing_probe_capture(...)`` near the self-attention call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only without torch.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


CAPTURE_SCHEMA_VERSION = "qvg-attention-probe-capture.v1"
PROBE_INPUT_SCHEMA_VERSION = "qvg-attention-probe-input.v1"
DEFAULT_ENV_PREFIX = "QVG_ATTENTION_PROBE"
DEFAULT_MAX_FILE_BYTES = 64 * 1024**2


def _ensure_torch() -> Any:
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    return torch


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_int_set(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "" or value.strip().lower() in {"*", "all"}:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in tuple(tensor.shape)]


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _safe_jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch is not None and torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": _shape(value),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "estimated_bytes": _tensor_nbytes(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonify(item) for item in value]
    return str(value)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "capture"


@dataclass(frozen=True)
class CaptureConfig:
    """Opt-in capture settings for a future Self-Forcing hook.

    Environment variables use the ``QVG_ATTENTION_PROBE_*`` prefix:

    - ``ENABLED=1`` turns capture on.
    - ``DIR=/path`` chooses the output directory.
    - ``LAYERS=27,28,29``, ``HEADS=0,9``, ``CHUNKS=24,25,26`` bound selection.
    - ``MAX_FILE_BYTES`` rejects oversized records before writing.
    - ``MAX_CACHE_TOKENS`` optionally truncates cached context to the latest N
      tokens. Leave unset for exact full-context attention probes.
    """

    enabled: bool = False
    output_dir: Path = Path("attention_probe_dumps")
    layers: set[int] | None = None
    heads: set[int] | None = None
    chunks: set[int] | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_cache_tokens: int | None = None
    variant_name: str = "capture"

    @classmethod
    def from_env(cls, prefix: str = DEFAULT_ENV_PREFIX) -> "CaptureConfig":
        output_dir = Path(os.getenv(f"{prefix}_DIR", "attention_probe_dumps"))
        max_cache_tokens_env = os.getenv(f"{prefix}_MAX_CACHE_TOKENS")
        max_cache_tokens = int(max_cache_tokens_env) if max_cache_tokens_env else None
        return cls(
            enabled=_parse_bool(os.getenv(f"{prefix}_ENABLED"), default=False),
            output_dir=output_dir,
            layers=_parse_int_set(os.getenv(f"{prefix}_LAYERS")),
            heads=_parse_int_set(os.getenv(f"{prefix}_HEADS")),
            chunks=_parse_int_set(os.getenv(f"{prefix}_CHUNKS")),
            max_file_bytes=int(os.getenv(f"{prefix}_MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES))),
            max_cache_tokens=max_cache_tokens,
            variant_name=os.getenv(f"{prefix}_VARIANT", "capture"),
        )

    def should_capture(self, *, layer_idx: int, chunk_idx: int) -> bool:
        if not self.enabled:
            return False
        if self.layers is not None and layer_idx not in self.layers:
            return False
        if self.chunks is not None and chunk_idx not in self.chunks:
            return False
        return True


def canonical_bhsd(tensor: Any, *, layout: str) -> Any:
    """Return a CPU BHSD clone from BHSD or BSHD input."""

    th = _ensure_torch()
    if not th.is_tensor(tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim != 4:
        raise ValueError(f"expected rank-4 tensor, got shape {tuple(tensor.shape)}")
    if layout == "BHSD":
        result = tensor
    elif layout == "BSHD":
        result = tensor.permute(0, 2, 1, 3)
    else:
        raise ValueError(f"layout must be BHSD or BSHD, got {layout!r}")
    return result.detach().cpu().contiguous()


def _select_heads(tensor: Any, heads: set[int] | None) -> Any:
    if heads is None:
        return tensor
    selected = sorted(head for head in heads if 0 <= head < int(tensor.shape[1]))
    if not selected:
        raise ValueError(f"head selection {sorted(heads)} matched no heads for tensor shape {tuple(tensor.shape)}")
    index = torch.tensor(selected, dtype=torch.long)  # type: ignore[union-attr]
    return tensor.index_select(1, index)


def _tail_tokens(tensor: Any, max_tokens: int | None) -> tuple[Any, int]:
    if max_tokens is None or tensor.shape[2] <= max_tokens:
        return tensor, 0
    start = int(tensor.shape[2] - max_tokens)
    return tensor[:, :, start:, :].contiguous(), start


def _record_tensor_bytes(record: Mapping[str, Any]) -> int:
    total = 0
    for value in record.values():
        if torch is not None and torch.is_tensor(value):
            total += _tensor_nbytes(value)
        elif isinstance(value, Mapping):
            total += _record_tensor_bytes(value)
        elif isinstance(value, (list, tuple)):
            total += sum(_record_tensor_bytes({"item": item}) for item in value)
    return total


def build_self_forcing_probe_record(
    *,
    layer_idx: int,
    current_start: int,
    local_start_index: int,
    local_end_index: int,
    frame_seq_length: int,
    query_rope_bshd: Any,
    key_all_rope_bshd: Any,
    value_all_bshd: Any,
    config: CaptureConfig | None = None,
    query_unrope_bshd: Any | None = None,
    key_all_unrope_bshd: Any | None = None,
    grid_sizes: Any | None = None,
    seq_lens: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one offline-probe record from Self-Forcing attention tensors.

    The Self-Forcing cached path reads all cached tokens, RoPE-applies K over
    the full span, then calls attention with BSHD tensors. This function splits
    that full key/value span into cached and current parts for the offline
    probe's reference schema.
    """

    if local_start_index < 0 or local_end_index <= local_start_index:
        raise ValueError("local_start_index/local_end_index must define a positive current-token span")

    cfg = config or CaptureConfig()
    current_chunk_idx = current_start // frame_seq_length
    q = _select_heads(canonical_bhsd(query_rope_bshd, layout="BSHD"), cfg.heads)
    key_all = _select_heads(canonical_bhsd(key_all_rope_bshd, layout="BSHD"), cfg.heads)
    value_all = _select_heads(canonical_bhsd(value_all_bshd, layout="BSHD"), cfg.heads)

    if local_end_index > key_all.shape[2] or local_end_index > value_all.shape[2]:
        raise ValueError(
            "local_end_index exceeds key/value span: "
            f"local_end_index={local_end_index}, key_tokens={key_all.shape[2]}, value_tokens={value_all.shape[2]}"
        )

    cached_key = key_all[:, :, :local_start_index, :].contiguous()
    cached_value = value_all[:, :, :local_start_index, :].contiguous()
    current_key = key_all[:, :, local_start_index:local_end_index, :].contiguous()
    current_value = value_all[:, :, local_start_index:local_end_index, :].contiguous()
    cached_key, dropped_cache_tokens = _tail_tokens(cached_key, cfg.max_cache_tokens)
    cached_value, dropped_value_tokens = _tail_tokens(cached_value, cfg.max_cache_tokens)
    if dropped_cache_tokens != dropped_value_tokens:
        raise RuntimeError("cached K/V truncation mismatch")

    diagnostics: dict[str, Any] = {}
    if query_unrope_bshd is not None:
        diagnostics["query_unrope_bhsd"] = _select_heads(canonical_bhsd(query_unrope_bshd, layout="BSHD"), cfg.heads)
    if key_all_unrope_bshd is not None:
        key_unrope = _select_heads(canonical_bhsd(key_all_unrope_bshd, layout="BSHD"), cfg.heads)
        diagnostics["cached_key_unrope_bhsd"] = key_unrope[:, :, dropped_cache_tokens:local_start_index, :].contiguous()
        diagnostics["current_key_unrope_bhsd"] = key_unrope[:, :, local_start_index:local_end_index, :].contiguous()

    return {
        "name": f"layer_{layer_idx}_chunk_{current_chunk_idx}",
        "layout": "BHSD",
        "query": q,
        "current_key": current_key,
        "current_value": current_value,
        "cached_key_ref": cached_key,
        "cached_value_ref": cached_value,
        "metadata": {
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "source": "Self-Forcing CausalWanSelfAttention.attn_kv_cache_prerope",
            "layer_idx": int(layer_idx),
            "chunk_idx": int(current_chunk_idx),
            "current_start": int(current_start),
            "local_start_index": int(local_start_index),
            "local_end_index": int(local_end_index),
            "frame_seq_length": int(frame_seq_length),
            "dropped_prefix_cache_tokens": int(dropped_cache_tokens),
            "selected_heads": sorted(cfg.heads) if cfg.heads is not None else "all",
            "variant_name": cfg.variant_name,
            "grid_sizes": _safe_jsonify(grid_sizes),
            "seq_lens": _safe_jsonify(seq_lens),
            **_safe_jsonify(dict(metadata or {})),
        },
        "diagnostics": diagnostics,
    }


def build_probe_input(records: Iterable[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": PROBE_INPUT_SCHEMA_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "metadata": _safe_jsonify(dict(metadata or {})),
        "records": list(records),
    }


def write_probe_input(
    output_path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Write a bounded ``attention_error_probe.py --input`` artifact."""

    th = _ensure_torch()
    output = Path(output_path)
    artifact = build_probe_input(records, metadata=metadata)
    tensor_bytes = _record_tensor_bytes(artifact)
    if tensor_bytes > max_file_bytes:
        return {
            "status": "skipped_byte_guard",
            "output_path": str(output),
            "estimated_tensor_bytes": tensor_bytes,
            "max_file_bytes": int(max_file_bytes),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    th.save(artifact, str(output))
    stat = output.stat()
    if stat.st_size > max_file_bytes:
        output.unlink(missing_ok=True)
        return {
            "status": "skipped_byte_guard",
            "output_path": str(output),
            "estimated_tensor_bytes": tensor_bytes,
            "actual_file_bytes": stat.st_size,
            "max_file_bytes": int(max_file_bytes),
        }
    return {
        "status": "written",
        "output_path": str(output.resolve()),
        "estimated_tensor_bytes": tensor_bytes,
        "actual_file_bytes": stat.st_size,
        "max_file_bytes": int(max_file_bytes),
    }


def write_self_forcing_probe_capture(
    *,
    layer_idx: int,
    current_start: int,
    local_start_index: int,
    local_end_index: int,
    frame_seq_length: int,
    query_rope_bshd: Any,
    key_all_rope_bshd: Any,
    value_all_bshd: Any,
    config: CaptureConfig,
    query_unrope_bshd: Any | None = None,
    key_all_unrope_bshd: Any | None = None,
    grid_sizes: Any | None = None,
    seq_lens: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_chunk_idx = current_start // frame_seq_length
    if not config.should_capture(layer_idx=layer_idx, chunk_idx=current_chunk_idx):
        return {
            "status": "skipped_selection",
            "layer_idx": int(layer_idx),
            "chunk_idx": int(current_chunk_idx),
        }

    record = build_self_forcing_probe_record(
        layer_idx=layer_idx,
        current_start=current_start,
        local_start_index=local_start_index,
        local_end_index=local_end_index,
        frame_seq_length=frame_seq_length,
        query_rope_bshd=query_rope_bshd,
        key_all_rope_bshd=key_all_rope_bshd,
        value_all_bshd=value_all_bshd,
        config=config,
        query_unrope_bshd=query_unrope_bshd,
        key_all_unrope_bshd=key_all_unrope_bshd,
        grid_sizes=grid_sizes,
        seq_lens=seq_lens,
        metadata=metadata,
    )
    filename = (
        f"attention_probe_l{layer_idx:02d}_chunk{current_chunk_idx:04d}_"
        f"start{current_start}_{_slug(config.variant_name)}.pt"
    )
    return write_probe_input(
        config.output_dir / filename,
        [record],
        metadata={
            "source": "Self-Forcing attention probe capture",
            "variant_name": config.variant_name,
        },
        max_file_bytes=config.max_file_bytes,
    )
