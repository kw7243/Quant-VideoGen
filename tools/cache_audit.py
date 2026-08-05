#!/usr/bin/env python3
"""Audit Quant-VideoGen KV-cache artifacts without eager tensor materialization.

The default path first loads ``torch.save`` artifacts under FakeTensorMode to
collect tensor metadata. Numeric statistics are computed only when the artifact
is small enough under explicit byte guards, so multi-GB cache dumps produce a
schema-compatible metadata-only report instead of exhausting host RAM.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import IntEnum
import inspect
import json
import math
from pathlib import Path
import re
import sys
import types
from typing import Any, Iterable, Sequence

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only without torch.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


SCHEMA_VERSION = "qvg-cache-audit.v1"
DEFAULT_MAX_FULL_LOAD_BYTES = 1 * 1024**3
DEFAULT_MAX_TENSOR_STAT_BYTES = 256 * 1024**2
DEFAULT_MAX_TOTAL_STAT_BYTES = 512 * 1024**2
DEFAULT_MAX_SUBSET_BYTES = 128 * 1024**2
DEFAULT_MAX_TIME_POINTS = 64
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
PROVENANCE_KEYS = {
    "args",
    "config",
    "git_commit",
    "job_id",
    "metadata",
    "meta",
    "num_output_frames",
    "provenance",
    "quant_config",
    "quant_type",
    "qvg_cache_audit_subset",
    "run_config",
    "seed",
}


def _ensure_torch() -> Any:
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    return torch


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _safe_int(value: Any) -> int | str:
    try:
        return int(value)
    except Exception:
        return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isfinite(result):
        return result
    return None


def _truncate(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


@contextmanager
def _qvg_pickle_stubs() -> Iterable[None]:
    """Install tiny QVG class stubs so old cache pickles can be inspected.

    Self-Forcing dumps contain ``quant_videogen.kv_cache.ChunkedKVCache``
    instances. Importing the real module pulls in the full compression stack,
    including Triton kernels, which is unnecessary for a metadata audit and can
    fail in lightweight CPU environments.
    """

    if torch is None:
        yield
        return

    module_names = ("quant_videogen", "quant_videogen.kv_cache")
    previous = {name: sys.modules.get(name) for name in module_names}
    created: set[str] = set()

    if "quant_videogen" not in sys.modules:
        package = types.ModuleType("quant_videogen")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["quant_videogen"] = package
        created.add("quant_videogen")

    if "quant_videogen.kv_cache" not in sys.modules:
        kv_module = types.ModuleType("quant_videogen.kv_cache")

        class ChunkedKVCache:  # noqa: D401 - pickle-compatible stub.
            """Attribute container for unpickled cache metadata."""

        ChunkedKVCache.__module__ = "quant_videogen.kv_cache"

        ChunkState = IntEnum(
            "ChunkState",
            {"EMPTY": 0, "BF16": 1, "QUANTIZED": 2},
            module="quant_videogen.kv_cache",
        )

        kv_module.ChunkedKVCache = ChunkedKVCache
        kv_module.ChunkState = ChunkState
        sys.modules["quant_videogen.kv_cache"] = kv_module
        created.add("quant_videogen.kv_cache")

        serialization = getattr(torch, "serialization", None)
        add_safe_globals = getattr(serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            try:
                add_safe_globals([ChunkedKVCache, ChunkState])
            except Exception:
                pass

    try:
        yield
    finally:
        for name in reversed(module_names):
            if name in created:
                sys.modules.pop(name, None)
            elif previous[name] is not None:
                sys.modules[name] = previous[name]  # type: ignore[assignment]


def _torch_load_kwargs(*, weights_only: bool, mmap: bool) -> dict[str, Any]:
    th = _ensure_torch()
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        signature = inspect.signature(th.load)
    except (TypeError, ValueError):
        signature = None

    if signature is None or "weights_only" in signature.parameters:
        kwargs["weights_only"] = weights_only
    if mmap and (signature is None or "mmap" in signature.parameters):
        kwargs["mmap"] = True
    return kwargs


def _load_torch_object(
    artifact: Path,
    *,
    fake: bool,
    weights_only: bool,
    mmap: bool,
    use_stubs: bool,
) -> Any:
    th = _ensure_torch()
    kwargs = _torch_load_kwargs(weights_only=weights_only, mmap=mmap)

    def load() -> Any:
        if fake:
            try:
                from torch._subclasses.fake_tensor import FakeTensorMode
            except Exception as exc:
                raise RuntimeError(f"FakeTensorMode unavailable: {exc}") from exc
            with FakeTensorMode():
                return th.load(str(artifact), **kwargs)
        return th.load(str(artifact), **kwargs)

    if use_stubs:
        with _qvg_pickle_stubs():
            return load()
    return load()


def _load_metadata_object(artifact: Path, *, trust_pickle: bool) -> tuple[Any, dict[str, Any]]:
    attempts = [
        {
            "mode": "fake_tensor_weights_only",
            "fake": True,
            "weights_only": True,
            "mmap": False,
            "use_stubs": True,
        },
    ]
    if trust_pickle:
        attempts.append(
            {
                "mode": "fake_tensor_trusted_pickle_stubs",
                "fake": True,
                "weights_only": False,
                "mmap": False,
                "use_stubs": True,
            }
        )

    errors: list[dict[str, str]] = []
    for attempt in attempts:
        try:
            obj = _load_torch_object(artifact, **{k: v for k, v in attempt.items() if k != "mode"})
            return obj, {
                "metadata_load_mode": attempt["mode"],
                "metadata_load_errors": errors,
                "pickle_trusted": not attempt["weights_only"],
            }
        except Exception as exc:
            errors.append({"mode": attempt["mode"], "error": _truncate(repr(exc), 1000)})

    raise RuntimeError(f"metadata load failed: {errors}")


def _load_full_object(
    artifact: Path,
    *,
    trust_pickle: bool,
    prefer_mmap: bool = True,
) -> tuple[Any, dict[str, Any]]:
    attempts = [
        {
            "mode": "cpu_weights_only_mmap",
            "fake": False,
            "weights_only": True,
            "mmap": prefer_mmap,
            "use_stubs": True,
        },
        {
            "mode": "cpu_weights_only",
            "fake": False,
            "weights_only": True,
            "mmap": False,
            "use_stubs": True,
        },
    ]
    if trust_pickle:
        attempts.extend(
            [
                {
                    "mode": "cpu_trusted_pickle_stubs_mmap",
                    "fake": False,
                    "weights_only": False,
                    "mmap": prefer_mmap,
                    "use_stubs": True,
                },
                {
                    "mode": "cpu_trusted_pickle_stubs",
                    "fake": False,
                    "weights_only": False,
                    "mmap": False,
                    "use_stubs": True,
                },
            ]
        )

    errors: list[dict[str, str]] = []
    for attempt in attempts:
        try:
            obj = _load_torch_object(artifact, **{k: v for k, v in attempt.items() if k != "mode"})
            return obj, {
                "stats_load_mode": attempt["mode"],
                "stats_load_errors": errors,
                "stats_pickle_trusted": not attempt["weights_only"],
            }
        except Exception as exc:
            errors.append({"mode": attempt["mode"], "error": _truncate(repr(exc), 1000)})

    raise RuntimeError(f"full load failed: {errors}")


def _is_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _is_simple(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, bytes))


def _json_key(key: Any) -> str:
    if isinstance(key, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return key
    return repr(key)


def _child_path(path: str, key: Any, *, attr: bool = False) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]" if path else f"[{key}]"
    if attr or (isinstance(key, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key)):
        return f"{path}.{key}" if path else str(key)
    return f"{path}[{_json_key(key)}]" if path else f"[{_json_key(key)}]"


def _safe_jsonify(value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return _truncate(repr(value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, bytes):
        return _truncate(value.decode("utf-8", errors="replace"))
    if torch is not None:
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
    if isinstance(value, IntEnum):
        return value.name
    if _is_tensor(value):
        return {
            "tensor": True,
            "shape": _shape(value),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "estimated_bytes": _tensor_nbytes(value),
        }
    if isinstance(value, dict):
        items = list(value.items())[:64]
        result = {
            str(_json_key(key)): _safe_jsonify(item, depth=depth + 1, max_depth=max_depth)
            for key, item in items
        }
        if len(value) > len(items):
            result["__truncated__"] = len(value) - len(items)
        return result
    if isinstance(value, (list, tuple)):
        items = list(value)[:64]
        result = [_safe_jsonify(item, depth=depth + 1, max_depth=max_depth) for item in items]
        if len(value) > len(items):
            result.append({"__truncated__": len(value) - len(items)})
        return result
    if hasattr(value, "__dict__"):
        return {
            "__class__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "attrs": _safe_jsonify(vars(value), depth=depth + 1, max_depth=max_depth),
        }
    return _truncate(repr(value))


def _shape(tensor: Any) -> list[int | str]:
    return [_safe_int(dim) for dim in tuple(tensor.shape)]


def _numel_from_shape(shape: list[int | str]) -> int | None:
    product = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        product *= dim
    return product


def _tensor_nbytes(tensor: Any) -> int | None:
    shape = _shape(tensor)
    numel = _numel_from_shape(shape)
    if numel is None:
        return None
    try:
        return int(numel * tensor.element_size())
    except Exception:
        return None


def _class_name(value: Any) -> str:
    return value.__class__.__name__


def _is_chunked_cache(value: Any) -> bool:
    return _class_name(value) == "ChunkedKVCache"


def _state_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    try:
        numeric = int(value)
    except Exception:
        return str(value)
    return {0: "EMPTY", 1: "BF16", 2: "QUANTIZED"}.get(numeric, str(numeric))


def _cache_object_record(value: Any, path: str) -> dict[str, Any]:
    chunks = getattr(value, "chunks", None)
    chunk_state = getattr(value, "chunk_state", None)
    spans = getattr(value, "quantized_spans", None)
    states = Counter(_state_name(item) for item in chunk_state or [])
    allocated_chunks = 0
    if isinstance(chunks, list):
        allocated_chunks = sum(1 for item in chunks if item is not None)

    record = {
        "path": path,
        "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
        "batch_size": _safe_jsonify(getattr(value, "batch_size", None)),
        "frame_seq_length": _safe_jsonify(getattr(value, "frame_seq_length", None)),
        "num_heads": _safe_jsonify(getattr(value, "num_heads", None)),
        "head_dim": _safe_jsonify(getattr(value, "head_dim", None)),
        "max_num_chunks": _safe_jsonify(getattr(value, "max_num_chunks", None)),
        "dtype": _safe_jsonify(getattr(value, "dtype", None)),
        "device": _safe_jsonify(getattr(value, "device", None)),
        "layout": _safe_jsonify(getattr(value, "layout", None)),
        "chunk_count": len(chunks) if isinstance(chunks, list) else None,
        "allocated_chunk_count": allocated_chunks,
        "chunk_state_counts": dict(sorted(states.items())),
        "quantized_span_count": len(spans) if isinstance(spans, list) else None,
    }
    if isinstance(spans, list):
        record["quantized_spans"] = [
            {
                "start_chunk": _safe_jsonify(span.get("start_chunk")) if isinstance(span, dict) else None,
                "end_chunk": _safe_jsonify(span.get("end_chunk")) if isinstance(span, dict) else None,
                "keys": sorted(str(key) for key in span.keys()) if isinstance(span, dict) else None,
            }
            for span in spans[:64]
        ]
        if len(spans) > 64:
            record["quantized_spans_truncated"] = len(spans) - 64
    return record


def _extract_context_from_path(path: str, context: dict[str, Any], shape: list[int | str]) -> dict[str, Any]:
    result = dict(context)
    if path.startswith("kv_cache"):
        result["cache_kind"] = "self_attention"
    elif path.startswith("crossattn_cache"):
        result["cache_kind"] = "cross_attention"

    layer_match = re.search(r"(?:kv_cache|crossattn_cache)\[(\d+)\]", path)
    if layer_match:
        result["layer_idx"] = int(layer_match.group(1))

    if re.search(r"(^|\.)k_no_rope($|\.|\[)", path):
        result["kv_role"] = "key_no_rope"
    elif re.search(r"(^|\.)k($|\.|\[)", path):
        result["kv_role"] = "key"
    elif re.search(r"(^|\.)v($|\.|\[)", path):
        result["kv_role"] = "value"

    layout = result.get("layout")
    if not layout:
        layout = _infer_layout(path, shape, result)
        if layout:
            result["layout"] = layout

    chunk_match = re.search(r"\.chunks\[(\d+)\]", path)
    if chunk_match:
        chunk_idx = int(chunk_match.group(1))
        result["chunk_idx"] = chunk_idx
        frame_seq_length = result.get("frame_seq_length")
        if not isinstance(frame_seq_length, int):
            if layout == "BSHD" and len(shape) >= 2 and isinstance(shape[1], int):
                frame_seq_length = shape[1]
                result["frame_seq_length"] = frame_seq_length
            elif layout == "BHSD" and len(shape) >= 3 and isinstance(shape[2], int):
                frame_seq_length = shape[2]
                result["frame_seq_length"] = frame_seq_length
        if isinstance(frame_seq_length, int):
            result["cache_position"] = {
                "chunk_index": chunk_idx,
                "start_token": chunk_idx * frame_seq_length,
                "end_token": (chunk_idx + 1) * frame_seq_length,
            }

    if "quantized_spans" in path:
        result["packed_quantized_span"] = True

    return result


def _infer_layout(path: str, shape: list[int | str], context: dict[str, Any]) -> str | None:
    if len(shape) != 4 or any(not isinstance(dim, int) for dim in shape):
        return None
    if context.get("packed_quantized_span"):
        return None
    if context.get("cache_kind") == "cross_attention":
        return "BSHD"
    if context.get("cache_kind") == "self_attention" and "chunks[" not in path:
        return "BSHD"

    _, dim1, dim2, _ = shape
    if dim1 <= 64 < dim2:
        return "BHSD"
    if dim2 <= 64 < dim1:
        return "BSHD"
    return None


class _Collector:
    def __init__(self) -> None:
        self.tensors: list[dict[str, Any]] = []
        self.cache_objects: list[dict[str, Any]] = []

    def visit_tensor(self, path: str, tensor: Any, context: dict[str, Any]) -> None:
        shape = _shape(tensor)
        resolved_context = _extract_context_from_path(path, context, shape)
        record = {
            "path": path,
            "shape": shape,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
            "estimated_bytes": _tensor_nbytes(tensor),
            "cache_kind": resolved_context.get("cache_kind"),
            "layer_idx": resolved_context.get("layer_idx"),
            "kv_role": resolved_context.get("kv_role"),
            "layout": resolved_context.get("layout"),
            "chunk_idx": resolved_context.get("chunk_idx"),
            "cache_position": resolved_context.get("cache_position"),
            "packed_quantized_span": bool(resolved_context.get("packed_quantized_span", False)),
            "stats_status": "pending",
            "stats": {},
        }
        unsupported: list[str] = []
        if record["kv_role"] not in {"key", "value", "key_no_rope"}:
            unsupported.append("not_kv_role")
        if record["layout"] not in {"BHSD", "BSHD"}:
            unsupported.append("unsupported_layout")
        if record["packed_quantized_span"]:
            unsupported.append("packed_quantized_span")
        if unsupported:
            record["unsupported"] = sorted(set(unsupported))
        self.tensors.append(record)


def _traverse(
    value: Any,
    *,
    path: str,
    collector: _Collector,
    context: dict[str, Any] | None = None,
    seen: set[int] | None = None,
) -> None:
    if context is None:
        context = {}
    if seen is None:
        seen = set()

    if _is_tensor(value):
        collector.visit_tensor(path, value, context)
        return
    if _is_simple(value):
        return

    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if _is_chunked_cache(value):
        collector.cache_objects.append(_cache_object_record(value, path))
        next_context = {
            **context,
            "cache_object_path": path,
            "layout": getattr(value, "layout", None),
            "frame_seq_length": getattr(value, "frame_seq_length", None),
            "num_heads": getattr(value, "num_heads", None),
            "head_dim": getattr(value, "head_dim", None),
        }
        for attr, item in vars(value).items():
            attr_context = dict(next_context)
            if attr == "quantized_spans":
                attr_context["packed_quantized_span"] = True
            _traverse(
                item,
                path=_child_path(path, attr, attr=True),
                collector=collector,
                context=attr_context,
                seen=seen,
            )
        return

    if isinstance(value, dict):
        source_path = value.get("source_path")
        tensor = value.get("tensor")
        if isinstance(source_path, str) and _is_tensor(tensor):
            next_context = dict(context)
            source_context = value.get("source_context")
            if isinstance(source_context, dict):
                next_context.update(source_context)
            next_context["subset_record"] = _safe_jsonify(
                {key: item for key, item in value.items() if key != "tensor"}
            )
            collector.visit_tensor(source_path, tensor, next_context)
            return

        for key, item in value.items():
            key_context = dict(context)
            if str(key) == "quantized_spans":
                key_context["packed_quantized_span"] = True
            _traverse(
                item,
                path=_child_path(path, key),
                collector=collector,
                context=key_context,
                seen=seen,
            )
        return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _traverse(
                item,
                path=_child_path(path, idx),
                collector=collector,
                context=context,
                seen=seen,
            )
        return

    if hasattr(value, "__dict__") and not isinstance(value, type):
        for attr, item in vars(value).items():
            _traverse(
                item,
                path=_child_path(path, attr, attr=True),
                collector=collector,
                context=context,
                seen=seen,
            )


def _collect(obj: Any) -> _Collector:
    collector = _Collector()
    _traverse(obj, path="", collector=collector)
    return collector


def _extract_provenance(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    provenance: dict[str, Any] = {}
    for key, value in obj.items():
        key_str = str(key)
        if key_str.lower() in PROVENANCE_KEYS:
            provenance[key_str] = _safe_jsonify(value)
    return provenance


def _quantiles(values: Any, percentiles: Iterable[int] = PERCENTILES) -> dict[str, float | None]:
    th = _ensure_torch()
    if values.numel() == 0:
        return {str(p): None for p in percentiles}
    values = values.reshape(-1).float()
    finite = th.isfinite(values)
    values = values[finite]
    if values.numel() == 0:
        return {str(p): None for p in percentiles}
    qs = th.tensor([p / 100 for p in percentiles], dtype=th.float32, device=values.device)
    result = th.quantile(values, qs)
    return {str(p): _safe_float(v) for p, v in zip(percentiles, result.tolist(), strict=True)}


def _mean_std(values: Any) -> dict[str, float | None]:
    th = _ensure_torch()
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    values = values.reshape(-1).float()
    finite = th.isfinite(values)
    values = values[finite]
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _safe_float(values.mean().item()),
        "std": _safe_float(values.std(unbiased=False).item()),
        "min": _safe_float(values.min().item()),
        "max": _safe_float(values.max().item()),
    }


def _outlier_fraction(values: Any, *, multiplier: float = 3.0) -> dict[str, float | None]:
    th = _ensure_torch()
    if values.numel() == 0:
        return {"threshold": None, "fraction": None}
    values = values.reshape(-1).float()
    values = values[th.isfinite(values)]
    if values.numel() == 0:
        return {"threshold": None, "fraction": None}
    median = values.median()
    threshold = median * multiplier
    fraction = (values > threshold).float().mean()
    return {"threshold": _safe_float(threshold.item()), "fraction": _safe_float(fraction.item())}


def _sample_indices(count: int, max_points: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_points:
        return list(range(count))
    step = (count - 1) / (max_points - 1)
    return sorted({int(round(i * step)) for i in range(max_points)})


def _canonical_bhsd(tensor: Any, layout: str | None) -> Any | None:
    if tensor.ndim != 4:
        return None
    if layout == "BHSD":
        return tensor
    if layout == "BSHD":
        return tensor.permute(0, 2, 1, 3).contiguous()
    return None


def _cache_tensor_stats(tensor: Any, info: dict[str, Any], *, max_time_points: int) -> dict[str, Any]:
    th = _ensure_torch()
    result: dict[str, Any] = {}

    raw = tensor.detach().cpu()
    values = raw.float().reshape(-1)
    finite = th.isfinite(values)
    result["numeric"] = {
        "numel": int(values.numel()),
        "finite_fraction": _safe_float(finite.float().mean().item()) if values.numel() else None,
        "value": _mean_std(values),
        "abs_percentiles": _quantiles(values.abs()),
        "abs_outlier_med3": _outlier_fraction(values.abs()),
    }

    layout = info.get("layout")
    bhsd = _canonical_bhsd(raw, layout)
    if bhsd is None or info.get("kv_role") not in {"key", "value", "key_no_rope"}:
        result["layout_stats"] = {
            "unsupported_layout": True,
            "reason": "expected a K/V tensor with BHSD or BSHD layout",
        }
        return result

    bhsd = bhsd.float()
    batch, heads, seq_len, head_dim = [int(dim) for dim in bhsd.shape]
    vector_norm = th.linalg.vector_norm(bhsd, dim=-1)
    result["norms"] = {
        "layout": "BHSD",
        "batch": batch,
        "heads": heads,
        "cache_positions": seq_len,
        "head_dim": head_dim,
        "vector_norm_percentiles": _quantiles(vector_norm),
        "vector_norm_outlier_med3": _outlier_fraction(vector_norm),
    }

    per_head = []
    for head_idx in range(heads):
        head_values = vector_norm[:, head_idx, :].reshape(-1)
        per_head.append(
            {
                "head_idx": head_idx,
                **_mean_std(head_values),
                "p95": _quantiles(head_values, (95,))["95"],
                "outlier_fraction_med3": _outlier_fraction(head_values)["fraction"],
            }
        )
    result["per_head"] = per_head

    sampled_positions = []
    position_values = vector_norm.permute(2, 0, 1).reshape(seq_len, -1)
    for idx in _sample_indices(seq_len, max_time_points):
        vals = position_values[idx]
        sampled_positions.append(
            {
                "cache_position": idx,
                **_mean_std(vals),
                "p95": _quantiles(vals, (95,))["95"],
            }
        )
    result["per_cache_position"] = {
        "axis": "sequence",
        "num_positions": seq_len,
        "sampled_count": len(sampled_positions),
        "samples": sampled_positions,
    }

    if head_dim % 4 == 0:
        tuples = bhsd.reshape(batch, heads, seq_len, head_dim // 4, 4)
        radii = th.linalg.vector_norm(tuples, dim=-1)
        radius_quantization: dict[str, Any] = {
            "assumption": "radius-only proxy; directions are preserved and full HQMQ direction-codebook error is not measured",
            "bits": {},
        }
        direction = tuples / radii.unsqueeze(-1).clamp(min=1e-8)
        denom = th.linalg.vector_norm(tuples.reshape(-1)).clamp(min=1e-12)
        for bits in (2, 4):
            qmax = (1 << bits) - 1
            scale = radii.amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            quantized_radii = (radii / scale).round().clamp(0, qmax) * scale
            reconstructed = direction * quantized_radii.unsqueeze(-1)
            rel_l2 = th.linalg.vector_norm((reconstructed - tuples).reshape(-1)) / denom
            relative_radius_error = (quantized_radii - radii).abs() / radii.clamp(min=1e-8)
            radius_quantization["bits"][str(bits)] = {
                "relative_l2": _safe_float(rel_l2.item()),
                "relative_radius_error_percentiles": _quantiles(relative_radius_error),
            }
        result["tuple4_radius"] = {
            "supported": True,
            "tuples_per_token": head_dim // 4,
            "radius_percentiles": _quantiles(radii),
            "radius": _mean_std(radii),
            "radius_outlier_med3": _outlier_fraction(radii),
            "zero_radius_fraction": _safe_float((radii == 0).float().mean().item()),
            "radius_quantization_error_proxy": radius_quantization,
        }
        median = radii.reshape(-1).float().median()
        max_radius = radii.reshape(-1).float().max()
        result["tuple4_radius"]["max_to_median"] = (
            _safe_float((max_radius / median.clamp(min=1e-12)).item()) if radii.numel() else None
        )
    else:
        result["tuple4_radius"] = {
            "supported": False,
            "unsupported_layout": "head_dim_not_divisible_by_4",
            "head_dim": head_dim,
        }

    if seq_len > 1:
        deltas = bhsd[:, :, 1:, :] - bhsd[:, :, :-1, :]
        delta_norm = th.linalg.vector_norm(deltas, dim=-1)
        denom = vector_norm[:, :, 1:].clamp(min=1e-12)
        result["temporal_deltas"] = {
            "supported": True,
            "axis": "sequence",
            "delta_norm_percentiles": _quantiles(delta_norm),
            "relative_delta_percentiles": _quantiles(delta_norm / denom),
            "delta_norm_outlier_med3": _outlier_fraction(delta_norm),
        }
    else:
        result["temporal_deltas"] = {
            "supported": False,
            "unsupported_layout": "single_cache_position",
        }

    return result


def _get_nested(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            result.append(parsed)
    return result


def _summary_stats(values: Iterable[Any]) -> dict[str, float | None]:
    vals = _finite_values(values)
    if not vals:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": _safe_float(sum(vals) / len(vals)),
        "min": _safe_float(min(vals)),
        "max": _safe_float(max(vals)),
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _safe_float(numerator)
    bottom = _safe_float(denominator)
    if top is None or bottom is None or abs(bottom) < 1e-12:
        return None
    return _safe_float(top / bottom)


def _tensor_metric_record(info: dict[str, Any]) -> dict[str, Any] | None:
    if info.get("stats_status") != "computed":
        return None
    if info.get("kv_role") not in {"key", "value", "key_no_rope"}:
        return None
    stats = info.get("stats")
    if not isinstance(stats, dict):
        return None
    if not isinstance(stats.get("norms"), dict):
        return None
    chunk_idx = info.get("chunk_idx")
    cache_position = info.get("cache_position")
    if chunk_idx is None and isinstance(cache_position, dict):
        chunk_idx = cache_position.get("chunk_index")
    record = {
        "path": info.get("path"),
        "cache_kind": info.get("cache_kind"),
        "layer_idx": info.get("layer_idx"),
        "kv_role": info.get("kv_role"),
        "chunk_idx": chunk_idx,
        "estimated_bytes": info.get("estimated_bytes") or 0,
        "vector_norm_p50": _get_nested(stats, ("norms", "vector_norm_percentiles", "50")),
        "vector_norm_p95": _get_nested(stats, ("norms", "vector_norm_percentiles", "95")),
        "vector_norm_p99": _get_nested(stats, ("norms", "vector_norm_percentiles", "99")),
        "vector_norm_outlier_med3_fraction": _get_nested(
            stats, ("norms", "vector_norm_outlier_med3", "fraction")
        ),
        "abs_value_p99": _get_nested(stats, ("numeric", "abs_percentiles", "99")),
        "abs_value_outlier_med3_fraction": _get_nested(
            stats, ("numeric", "abs_outlier_med3", "fraction")
        ),
        "tuple4_radius_p50": _get_nested(stats, ("tuple4_radius", "radius_percentiles", "50")),
        "tuple4_radius_p95": _get_nested(stats, ("tuple4_radius", "radius_percentiles", "95")),
        "tuple4_radius_p99": _get_nested(stats, ("tuple4_radius", "radius_percentiles", "99")),
        "tuple4_radius_max_to_median": _get_nested(stats, ("tuple4_radius", "max_to_median")),
        "tuple4_radius_outlier_med3_fraction": _get_nested(
            stats, ("tuple4_radius", "radius_outlier_med3", "fraction")
        ),
        "radius_only_rel_l2_bits2": _get_nested(
            stats,
            (
                "tuple4_radius",
                "radius_quantization_error_proxy",
                "bits",
                "2",
                "relative_l2",
            ),
        ),
        "radius_only_rel_l2_bits4": _get_nested(
            stats,
            (
                "tuple4_radius",
                "radius_quantization_error_proxy",
                "bits",
                "4",
                "relative_l2",
            ),
        ),
        "temporal_delta_norm_p95": _get_nested(
            stats, ("temporal_deltas", "delta_norm_percentiles", "95")
        ),
        "temporal_relative_delta_p50": _get_nested(
            stats, ("temporal_deltas", "relative_delta_percentiles", "50")
        ),
        "temporal_relative_delta_p95": _get_nested(
            stats, ("temporal_deltas", "relative_delta_percentiles", "95")
        ),
    }
    return record


def _summarize_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "vector_norm_p50",
        "vector_norm_p95",
        "vector_norm_p99",
        "vector_norm_outlier_med3_fraction",
        "abs_value_p99",
        "abs_value_outlier_med3_fraction",
        "tuple4_radius_p50",
        "tuple4_radius_p95",
        "tuple4_radius_p99",
        "tuple4_radius_max_to_median",
        "tuple4_radius_outlier_med3_fraction",
        "radius_only_rel_l2_bits2",
        "radius_only_rel_l2_bits4",
        "temporal_delta_norm_p95",
        "temporal_relative_delta_p50",
        "temporal_relative_delta_p95",
    ]
    result: dict[str, Any] = {
        "tensor_count": len(records),
        "estimated_bytes": sum(int(record.get("estimated_bytes") or 0) for record in records),
        "layers": sorted({record["layer_idx"] for record in records if record.get("layer_idx") is not None}),
        "chunks": sorted({record["chunk_idx"] for record in records if record.get("chunk_idx") is not None}),
    }
    for key in metric_keys:
        result[key] = _summary_stats(record.get(key) for record in records)
    return result


def _stats_aggregates(tensors: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for info in tensors if (record := _tensor_metric_record(info)) is not None]
    result: dict[str, Any] = {
        "computed_tensor_count": len(records),
        "computed_estimated_bytes": sum(int(record.get("estimated_bytes") or 0) for record in records),
        "by_kv_role": {},
        "by_layer_role": {},
        "by_chunk_idx": {},
        "per_head_summary": [],
        "worst_heads": {
            "by_vector_norm_p95_max": [],
            "by_outlier_fraction_max": [],
        },
        "chunk_drift": [],
    }
    if not records:
        return result

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[("role", record.get("kv_role"))].append(record)
        grouped[
            (
                "layer_role",
                record.get("cache_kind"),
                record.get("layer_idx"),
                record.get("kv_role"),
            )
        ].append(record)
        grouped[("chunk", record.get("chunk_idx"))].append(record)

    for key, group in grouped.items():
        group_type = key[0]
        if group_type == "role":
            result["by_kv_role"][str(key[1])] = _summarize_metric_records(group)
        elif group_type == "layer_role":
            _, cache_kind, layer_idx, role = key
            out_key = f"{cache_kind or 'unknown'}:layer_{layer_idx}:{role}"
            result["by_layer_role"][out_key] = _summarize_metric_records(group)
        elif group_type == "chunk":
            result["by_chunk_idx"][str(key[1])] = _summarize_metric_records(group)

    head_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for info in tensors:
        if info.get("stats_status") != "computed":
            continue
        stats = info.get("stats")
        if not isinstance(stats, dict):
            continue
        cache_position = info.get("cache_position")
        chunk_idx = info.get("chunk_idx")
        if chunk_idx is None and isinstance(cache_position, dict):
            chunk_idx = cache_position.get("chunk_index")
        for head in stats.get("per_head", []):
            if not isinstance(head, dict):
                continue
            head_groups[
                (
                    info.get("cache_kind"),
                    info.get("layer_idx"),
                    info.get("kv_role"),
                    head.get("head_idx"),
                )
            ].append(
                {
                    "chunk_idx": chunk_idx,
                    "p95": head.get("p95"),
                    "mean": head.get("mean"),
                    "outlier_fraction_med3": head.get("outlier_fraction_med3"),
                }
            )

    per_head_summary = []
    for (cache_kind, layer_idx, role, head_idx), group in sorted(
        head_groups.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        entry = {
            "cache_kind": cache_kind,
            "layer_idx": layer_idx,
            "kv_role": role,
            "head_idx": head_idx,
            "sample_count": len(group),
            "chunks": sorted({item["chunk_idx"] for item in group if item.get("chunk_idx") is not None}),
            "vector_norm_p95": _summary_stats(item.get("p95") for item in group),
            "vector_norm_mean": _summary_stats(item.get("mean") for item in group),
            "outlier_fraction_med3": _summary_stats(item.get("outlier_fraction_med3") for item in group),
        }
        per_head_summary.append(entry)
    result["per_head_summary"] = per_head_summary
    result["worst_heads"]["by_vector_norm_p95_max"] = sorted(
        per_head_summary,
        key=lambda item: item["vector_norm_p95"]["max"] if item["vector_norm_p95"]["max"] is not None else -1,
        reverse=True,
    )[:12]
    result["worst_heads"]["by_outlier_fraction_max"] = sorted(
        per_head_summary,
        key=lambda item: (
            item["outlier_fraction_med3"]["max"]
            if item["outlier_fraction_med3"]["max"] is not None
            else -1
        ),
        reverse=True,
    )[:12]

    drift_metrics = [
        "vector_norm_p50",
        "vector_norm_p95",
        "tuple4_radius_p95",
        "tuple4_radius_outlier_med3_fraction",
        "radius_only_rel_l2_bits4",
        "temporal_relative_delta_p95",
    ]
    drift_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        drift_groups[(record.get("cache_kind"), record.get("layer_idx"), record.get("kv_role"))].append(record)

    for (cache_kind, layer_idx, role), group in sorted(
        drift_groups.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        ordered = sorted(
            [item for item in group if item.get("chunk_idx") is not None],
            key=lambda item: int(item["chunk_idx"]),
        )
        for left, right in zip(ordered, ordered[1:], strict=False):
            entry: dict[str, Any] = {
                "cache_kind": cache_kind,
                "layer_idx": layer_idx,
                "kv_role": role,
                "from_chunk": left.get("chunk_idx"),
                "to_chunk": right.get("chunk_idx"),
                "selected_chunk_gap": (
                    int(right["chunk_idx"]) - int(left["chunk_idx"])
                    if right.get("chunk_idx") is not None and left.get("chunk_idx") is not None
                    else None
                ),
            }
            for metric in drift_metrics:
                right_value = _safe_float(right.get(metric))
                left_value = _safe_float(left.get(metric))
                entry[f"{metric}_delta"] = (
                    _safe_float(right_value - left_value)
                    if right_value is not None and left_value is not None
                    else None
                )
                entry[f"{metric}_ratio"] = _ratio(right.get(metric), left.get(metric))
            result["chunk_drift"].append(entry)

    return result


def _parse_int_list(value: str | None, available: Sequence[int], *, label: str) -> list[int]:
    unique_available = sorted(set(int(item) for item in available))
    if not unique_available:
        raise ValueError(f"no available {label}s to select")
    if value is None or value.lower() in {"auto", "auto3"}:
        picks = [unique_available[0], unique_available[len(unique_available) // 2], unique_available[-1]]
        return sorted(set(picks))
    result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    missing = sorted(set(result) - set(unique_available))
    if missing:
        raise ValueError(f"requested {label}s are not available: {missing}")
    return result


def _parse_roles(value: str) -> list[str]:
    aliases = {"k": "key", "key": "key", "v": "value", "value": "value"}
    roles = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unsupported role {part!r}; expected key,value")
        roles.append(aliases[key])
    result = sorted(set(roles), key=roles.index)
    if not result:
        raise ValueError("at least one subset role is required")
    return result


def _context_for_subset_record(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_kind": info.get("cache_kind"),
        "layer_idx": info.get("layer_idx"),
        "kv_role": info.get("kv_role"),
        "layout": info.get("layout"),
        "chunk_idx": info.get("chunk_idx"),
        "cache_position": info.get("cache_position"),
    }


def create_subset_artifact(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    subset_layers: str | None = None,
    subset_chunks: str | None = None,
    subset_roles: str = "key,value",
    subset_cache_kind: str = "self_attention",
    max_subset_bytes: int = DEFAULT_MAX_SUBSET_BYTES,
    trust_pickle: bool = True,
) -> dict[str, Any]:
    """Write a small tensor-only artifact that preserves source tensor paths."""

    th = _ensure_torch()
    artifact = Path(artifact_path).expanduser()
    output = Path(output_path).expanduser()
    stat = artifact.stat()

    metadata_obj, metadata_load = _load_metadata_object(artifact, trust_pickle=trust_pickle)
    collector = _collect(metadata_obj)
    candidates = [
        info
        for info in collector.tensors
        if info.get("cache_kind") == subset_cache_kind
        and info.get("kv_role") in {"key", "value"}
        and info.get("layout") in {"BHSD", "BSHD"}
        and info.get("estimated_bytes") is not None
        and info.get("chunk_idx") is not None
    ]
    layers = _parse_int_list(subset_layers, [int(info["layer_idx"]) for info in candidates], label="layer")
    chunks = _parse_int_list(subset_chunks, [int(info["chunk_idx"]) for info in candidates], label="chunk")
    roles = _parse_roles(subset_roles)

    selected = [
        info
        for info in candidates
        if info.get("layer_idx") in layers
        and info.get("chunk_idx") in chunks
        and info.get("kv_role") in roles
    ]
    selected = sorted(
        selected,
        key=lambda item: (
            int(item.get("layer_idx") or -1),
            str(item.get("kv_role") or ""),
            int(item.get("chunk_idx") or -1),
            str(item.get("path") or ""),
        ),
    )
    selected_bytes = sum(int(info.get("estimated_bytes") or 0) for info in selected)
    if not selected:
        raise ValueError("subset selection matched no tensors")
    if selected_bytes > max_subset_bytes:
        raise ValueError(
            f"subset estimated bytes {selected_bytes} exceeds max_subset_bytes {max_subset_bytes}"
        )

    full_obj, full_load = _load_full_object(artifact, trust_pickle=trust_pickle, prefer_mmap=True)
    path_to_tensor: dict[str, Any] = {}

    class TensorCollector(_Collector):
        def visit_tensor(self, path: str, tensor: Any, context: dict[str, Any]) -> None:
            path_to_tensor[path] = tensor

    tensor_collector = TensorCollector()
    _traverse(full_obj, path="", collector=tensor_collector)

    records = []
    for info in selected:
        tensor = path_to_tensor.get(str(info["path"]))
        if tensor is None:
            raise RuntimeError(f"selected tensor missing after full load: {info['path']}")
        records.append(
            {
                "source_path": info["path"],
                "source_context": _context_for_subset_record(info),
                "source_shape": info.get("shape"),
                "source_dtype": info.get("dtype"),
                "source_estimated_bytes": info.get("estimated_bytes"),
                "tensor": tensor.detach().cpu().clone(),
            }
        )

    subset_obj = {
        "qvg_cache_audit_subset": {
            "schema_version": "qvg-cache-audit-subset.v1",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_artifact": str(artifact.resolve()),
            "source_size_bytes": stat.st_size,
            "source_mtime_unix": stat.st_mtime,
            "source_mtime_utc": _utc_iso(stat.st_mtime),
            "metadata_load": metadata_load,
            "stats_load": full_load,
            "selection": {
                "cache_kind": subset_cache_kind,
                "layers": layers,
                "roles": roles,
                "chunks": chunks,
            },
            "byte_guards": {
                "max_subset_bytes": max_subset_bytes,
                "selected_estimated_bytes": selected_bytes,
            },
            "selected_tensor_count": len(records),
            "selected_paths": [str(info["path"]) for info in selected],
        },
        "selected_tensors": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    th.save(subset_obj, str(output))
    output_stat = output.stat()
    manifest = _safe_jsonify(subset_obj["qvg_cache_audit_subset"])
    manifest["output_path"] = str(output.resolve())
    manifest["output_size_bytes"] = output_stat.st_size
    return manifest


def _mark_metadata_only(tensors: list[dict[str, Any]], reason: str) -> None:
    for info in tensors:
        info["stats_status"] = "metadata_only"
        info["stats"] = {"metadata_only": True, "reason": reason}


def _layout_summary(tensors: list[dict[str, Any]], cache_objects: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_role: dict[str, dict[str, Any]] = {}
    for info in tensors:
        layer = info.get("layer_idx")
        role = info.get("kv_role")
        if layer is None or role is None:
            continue
        key = f"{info.get('cache_kind') or 'unknown'}:layer_{layer}:{role}"
        item = by_layer_role.setdefault(
            key,
            {
                "cache_kind": info.get("cache_kind"),
                "layer_idx": layer,
                "kv_role": role,
                "tensor_count": 0,
                "estimated_bytes": 0,
                "layouts": Counter(),
                "stats_status_counts": Counter(),
            },
        )
        item["tensor_count"] += 1
        item["estimated_bytes"] += info.get("estimated_bytes") or 0
        if info.get("layout"):
            item["layouts"][info["layout"]] += 1
        item["stats_status_counts"][info.get("stats_status", "unknown")] += 1

    for item in by_layer_role.values():
        item["layouts"] = dict(sorted(item["layouts"].items()))
        item["stats_status_counts"] = dict(sorted(item["stats_status_counts"].items()))

    total_estimated = sum((info.get("estimated_bytes") or 0) for info in tensors)
    return {
        "tensor_count": len(tensors),
        "cache_object_count": len(cache_objects),
        "total_estimated_tensor_bytes": total_estimated,
        "by_layer_role": dict(sorted(by_layer_role.items())),
    }


def _should_compute_stats(
    *,
    stats_mode: str,
    file_size: int,
    tensor_total_bytes: int,
    max_full_load_bytes: int,
    max_total_stat_bytes: int,
) -> tuple[bool, str]:
    if stats_mode == "metadata-only":
        return False, "stats_mode=metadata-only"
    if stats_mode == "force":
        return True, "stats_mode=force"
    if file_size > max_full_load_bytes:
        return False, f"file_size {file_size} exceeds max_full_load_bytes {max_full_load_bytes}"
    if tensor_total_bytes > max_total_stat_bytes:
        return False, (
            f"estimated tensor bytes {tensor_total_bytes} exceeds "
            f"max_total_stat_bytes {max_total_stat_bytes}"
        )
    return True, "artifact within auto stats byte guards"


def audit_artifact(
    artifact_path: str | Path,
    *,
    stats_mode: str = "auto",
    max_full_load_bytes: int = DEFAULT_MAX_FULL_LOAD_BYTES,
    max_tensor_stat_bytes: int = DEFAULT_MAX_TENSOR_STAT_BYTES,
    max_total_stat_bytes: int = DEFAULT_MAX_TOTAL_STAT_BYTES,
    max_time_points: int = DEFAULT_MAX_TIME_POINTS,
    trust_pickle: bool = True,
) -> dict[str, Any]:
    artifact = Path(artifact_path).expanduser()
    stat = artifact.stat()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input": {
            "path": str(artifact.resolve()),
            "size_bytes": stat.st_size,
            "mtime_unix": stat.st_mtime,
            "mtime_utc": _utc_iso(stat.st_mtime),
        },
        "load": {
            "status": "pending",
            "stats_mode": stats_mode,
            "max_full_load_bytes": max_full_load_bytes,
            "max_tensor_stat_bytes": max_tensor_stat_bytes,
            "max_total_stat_bytes": max_total_stat_bytes,
            "trust_pickle": trust_pickle,
        },
        "provenance": {},
        "cache_objects": [],
        "tensors": [],
        "layout_summary": {},
        "stats_aggregates": {},
    }

    try:
        metadata_obj, metadata_load = _load_metadata_object(artifact, trust_pickle=trust_pickle)
    except Exception as exc:
        summary["load"].update({"status": "metadata_load_failed", "error": _truncate(repr(exc), 2000)})
        return summary

    collector = _collect(metadata_obj)
    summary["load"].update(metadata_load)
    summary["provenance"] = _extract_provenance(metadata_obj)
    summary["cache_objects"] = collector.cache_objects
    summary["tensors"] = collector.tensors

    tensor_total_bytes = sum((info.get("estimated_bytes") or 0) for info in collector.tensors)
    compute_stats, reason = _should_compute_stats(
        stats_mode=stats_mode,
        file_size=stat.st_size,
        tensor_total_bytes=tensor_total_bytes,
        max_full_load_bytes=max_full_load_bytes,
        max_total_stat_bytes=max_total_stat_bytes,
    )
    summary["load"]["stats_decision"] = {"compute_stats": compute_stats, "reason": reason}

    if not compute_stats:
        _mark_metadata_only(collector.tensors, reason)
        summary["load"]["status"] = "metadata_only"
        summary["layout_summary"] = _layout_summary(collector.tensors, collector.cache_objects)
        summary["stats_aggregates"] = _stats_aggregates(collector.tensors)
        return summary

    try:
        full_obj, full_load = _load_full_object(artifact, trust_pickle=trust_pickle)
    except Exception as exc:
        _mark_metadata_only(collector.tensors, f"full CPU load failed: {_truncate(repr(exc), 1000)}")
        summary["load"].update({"status": "metadata_only", "stats_load_error": _truncate(repr(exc), 2000)})
        summary["layout_summary"] = _layout_summary(collector.tensors, collector.cache_objects)
        return summary

    summary["load"].update(full_load)
    full_collector = _collect(full_obj)
    full_tensors = {info["path"]: info for info in full_collector.tensors}

    # We need the real tensor objects for stats, not just their metadata records.
    path_to_tensor: dict[str, Any] = {}

    class TensorCollector(_Collector):
        def visit_tensor(self, path: str, tensor: Any, context: dict[str, Any]) -> None:
            path_to_tensor[path] = tensor

    tensor_collector = TensorCollector()
    _traverse(full_obj, path="", collector=tensor_collector)

    for info in collector.tensors:
        tensor = path_to_tensor.get(info["path"])
        nbytes = info.get("estimated_bytes")
        if tensor is None:
            info["stats_status"] = "metadata_only"
            info["stats"] = {"metadata_only": True, "reason": "tensor path missing after full load"}
            continue
        if nbytes is not None and nbytes > max_tensor_stat_bytes:
            info["stats_status"] = "metadata_only"
            info["stats"] = {
                "metadata_only": True,
                "reason": (
                    f"tensor estimated_bytes {nbytes} exceeds "
                    f"max_tensor_stat_bytes {max_tensor_stat_bytes}"
                ),
            }
            continue
        try:
            full_info = full_tensors.get(info["path"], info)
            info.update(
                {
                    "cache_kind": full_info.get("cache_kind", info.get("cache_kind")),
                    "layer_idx": full_info.get("layer_idx", info.get("layer_idx")),
                    "kv_role": full_info.get("kv_role", info.get("kv_role")),
                    "layout": full_info.get("layout", info.get("layout")),
                    "chunk_idx": full_info.get("chunk_idx", info.get("chunk_idx")),
                    "cache_position": full_info.get("cache_position", info.get("cache_position")),
                    "packed_quantized_span": full_info.get(
                        "packed_quantized_span", info.get("packed_quantized_span")
                    ),
                }
            )
            info["stats"] = _cache_tensor_stats(tensor, info, max_time_points=max_time_points)
            info["stats_status"] = "computed"
        except Exception as exc:
            info["stats_status"] = "metadata_only"
            info["stats"] = {"metadata_only": True, "reason": f"stats failed: {_truncate(repr(exc), 1000)}"}

    summary["load"]["status"] = "stats_computed"
    summary["layout_summary"] = _layout_summary(collector.tensors, collector.cache_objects)
    summary["stats_aggregates"] = _stats_aggregates(collector.tensors)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Path to a torch.save KV-cache artifact")
    parser.add_argument("-o", "--output", help="Write JSON summary to this path")
    parser.add_argument(
        "--make-subset",
        help=(
            "Write a bounded tensor-only subset artifact to this path, preserving "
            "source tensor paths for later --stats-mode auto audits"
        ),
    )
    parser.add_argument(
        "--subset-layers",
        default=None,
        help="Comma-separated source layer indices for --make-subset, or auto/auto3 for first/middle/last",
    )
    parser.add_argument(
        "--subset-chunks",
        default=None,
        help="Comma-separated source chunk indices for --make-subset, or auto/auto3 for first/middle/last",
    )
    parser.add_argument(
        "--subset-roles",
        default="key,value",
        help="Comma-separated roles for --make-subset; accepts key,value or k,v",
    )
    parser.add_argument(
        "--subset-cache-kind",
        default="self_attention",
        help="Cache kind for --make-subset selection, default self_attention",
    )
    parser.add_argument("--max-subset-bytes", type=_positive_int, default=DEFAULT_MAX_SUBSET_BYTES)
    parser.add_argument(
        "--stats-mode",
        choices=("auto", "metadata-only", "force"),
        default="auto",
        help="auto computes stats only under byte guards; force ignores the full-file guard",
    )
    parser.add_argument("--max-full-load-bytes", type=_positive_int, default=DEFAULT_MAX_FULL_LOAD_BYTES)
    parser.add_argument("--max-tensor-stat-bytes", type=_positive_int, default=DEFAULT_MAX_TENSOR_STAT_BYTES)
    parser.add_argument("--max-total-stat-bytes", type=_positive_int, default=DEFAULT_MAX_TOTAL_STAT_BYTES)
    parser.add_argument("--max-time-points", type=_positive_int, default=DEFAULT_MAX_TIME_POINTS)
    parser.set_defaults(trust_pickle=True)
    parser.add_argument(
        "--trust-pickle",
        dest="trust_pickle",
        action="store_true",
        help="Allow trusted-pickle metadata fallback for local QVG artifacts (default)",
    )
    parser.add_argument(
        "--no-trust-pickle",
        dest="trust_pickle",
        action="store_false",
        help="Disable trusted-pickle fallback; only weights-only loads are attempted",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.make_subset:
        try:
            manifest = create_subset_artifact(
                args.artifact,
                args.make_subset,
                subset_layers=args.subset_layers,
                subset_chunks=args.subset_chunks,
                subset_roles=args.subset_roles,
                subset_cache_kind=args.subset_cache_kind,
                max_subset_bytes=args.max_subset_bytes,
                trust_pickle=args.trust_pickle,
            )
        except Exception as exc:
            print(json.dumps({"status": "subset_failed", "error": _truncate(repr(exc), 2000)}), file=sys.stderr)
            return 2
        text = json.dumps(manifest, indent=None if args.compact else 2, sort_keys=True, allow_nan=False)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    summary = audit_artifact(
        args.artifact,
        stats_mode=args.stats_mode,
        max_full_load_bytes=args.max_full_load_bytes,
        max_tensor_stat_bytes=args.max_tensor_stat_bytes,
        max_total_stat_bytes=args.max_total_stat_bytes,
        max_time_points=args.max_time_points,
        trust_pickle=args.trust_pickle,
    )
    text = json.dumps(summary, indent=None if args.compact else 2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if summary.get("load", {}).get("status") != "metadata_load_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
