#!/usr/bin/env python3
"""Offline attention functional-distortion probe for Quant-VideoGen caches.

The probe compares full-precision/reference attention against one or more
reconstructed-cache variants. It expects already prepared tensors and does not
import the Quant-VideoGen package entrypoint, so it can run in lightweight CPU
environments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only without torch.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


SCHEMA_VERSION = "qvg-attention-error-probe.v1"
INPUT_SCHEMA_VERSION = "qvg-attention-probe-input.v1"
SMOKE_SCHEMA_VERSION = "qvg-attention-error-probe-smoke.v1"
DEFAULT_TOP_K = 10
DEFAULT_EPS = 1e-8


def _ensure_torch() -> Any:
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")
    return torch


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isfinite(result):
        return result
    return None


def _safe_int(value: Any) -> int | str:
    try:
        return int(value)
    except Exception:
        return str(value)


def _shape(tensor: Any) -> list[int | str]:
    return [_safe_int(dim) for dim in tuple(tensor.shape)]


def _tensor_nbytes(tensor: Any) -> int | None:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return None


def _safe_jsonify(value: Any, *, max_list: int = 64) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if torch is not None:
        if torch.is_tensor(value):
            return {
                "tensor": True,
                "shape": _shape(value),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "estimated_bytes": _tensor_nbytes(value),
            }
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_jsonify(item, max_list=max_list) for key, item in list(value.items())[:max_list]}
    if isinstance(value, (list, tuple)):
        items = [_safe_jsonify(item, max_list=max_list) for item in list(value)[:max_list]]
        if len(value) > max_list:
            items.append({"truncated_items": len(value) - max_list})
        return items
    return str(value)


def _mean(values: Any) -> float | None:
    th = _ensure_torch()
    values = values.float().reshape(-1)
    values = values[th.isfinite(values)]
    if values.numel() == 0:
        return None
    return _safe_float(values.mean().item())


def _min(values: Any) -> float | None:
    th = _ensure_torch()
    values = values.float().reshape(-1)
    values = values[th.isfinite(values)]
    if values.numel() == 0:
        return None
    return _safe_float(values.min().item())


def _max(values: Any) -> float | None:
    th = _ensure_torch()
    values = values.float().reshape(-1)
    values = values[th.isfinite(values)]
    if values.numel() == 0:
        return None
    return _safe_float(values.max().item())


def _canonical_bhsd(tensor: Any, layout: str, *, name: str) -> Any:
    if not torch.is_tensor(tensor):  # type: ignore[union-attr]
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got shape {tuple(tensor.shape)}")
    if layout == "BHSD":
        return tensor.detach().cpu().contiguous()
    if layout == "BSHD":
        return tensor.detach().cpu().permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"{name} layout must be BHSD or BSHD, got {layout!r}")


def _layout_for(record: Mapping[str, Any], tensor_name: str, default_layout: str) -> str:
    layouts = record.get("layouts")
    if isinstance(layouts, Mapping) and tensor_name in layouts:
        return str(layouts[tensor_name])
    return str(record.get("layout", default_layout))


def _tensor_from_record(record: Mapping[str, Any], key: str, *, aliases: Iterable[str] = ()) -> Any:
    for name in (key, *aliases):
        if name in record:
            return record[name]
    raise KeyError(f"probe record is missing required tensor {key!r}")


def _optional_tensor_from_record(record: Mapping[str, Any], key: str, *, aliases: Iterable[str] = ()) -> Any | None:
    for name in (key, *aliases):
        if name in record:
            return record[name]
    return None


def _check_attention_shapes(query: Any, key: Any, value: Any) -> None:
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError(
            "query/key/value must have matching batch and head dimensions: "
            f"q={tuple(query.shape)}, k={tuple(key.shape)}, v={tuple(value.shape)}"
        )
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key head_dim mismatch: {query.shape[-1]} vs {key.shape[-1]}")
    if key.shape[2] != value.shape[2]:
        raise ValueError(f"key/value sequence length mismatch: {key.shape[2]} vs {value.shape[2]}")


def _concat_cache_current(cached: Any, current: Any) -> Any:
    return torch.cat([cached, current], dim=2)  # type: ignore[union-attr]


def _apply_attention_mask(logits: Any, mask: Any | None) -> Any:
    if mask is None:
        return logits
    if not torch.is_tensor(mask):  # type: ignore[union-attr]
        raise TypeError(f"attention_mask must be a tensor, got {type(mask)!r}")
    mask = mask.detach().cpu()
    if mask.dtype == torch.bool:
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)  # type: ignore[union-attr]
    return logits + mask.to(logits.dtype)


def _attention(query: Any, key: Any, value: Any, *, scale: float, mask: Any | None) -> tuple[Any, Any, Any]:
    q = query.float()
    k = key.float()
    v = value.float()
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale  # type: ignore[union-attr]
    logits = _apply_attention_mask(logits, mask)
    probs = torch.softmax(logits, dim=-1)  # type: ignore[union-attr]
    output = torch.einsum("bhqk,bhkd->bhqd", probs, v)  # type: ignore[union-attr]
    return logits, probs, output


def _error_summary(reference: Any, candidate: Any) -> dict[str, float | None]:
    th = _ensure_torch()
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    diff_flat = diff.reshape(-1)
    ref_norm = th.linalg.vector_norm(ref.reshape(-1)).clamp(min=1e-12)
    diff_norm = th.linalg.vector_norm(diff_flat)
    return {
        "mean_abs": _mean(diff.abs()),
        "max_abs": _max(diff.abs()),
        "rmse": _safe_float(th.sqrt(th.mean(diff_flat.square())).item()) if diff_flat.numel() else None,
        "relative_l2": _safe_float((diff_norm / ref_norm).item()),
    }


def _per_head_relative_l2(reference: Any, candidate: Any) -> dict[str, Any]:
    th = _ensure_torch()
    ref = reference.float()
    diff = candidate.float() - ref
    per_head_num = th.linalg.vector_norm(diff, dim=(0, 2, 3))
    per_head_den = th.linalg.vector_norm(ref, dim=(0, 2, 3)).clamp(min=1e-12)
    values = per_head_num / per_head_den
    return {
        "values": [_safe_float(item) for item in values.tolist()],
        "mean": _mean(values),
        "max": _max(values),
    }


def _probability_divergence(reference: Any, candidate: Any, *, eps: float) -> dict[str, float | None]:
    ref = reference.float().clamp_min(eps)
    cand = candidate.float().clamp_min(eps)
    mixture = (0.5 * (ref + cand)).clamp_min(eps)
    kl = (ref * (ref.log() - cand.log())).sum(dim=-1)
    js = 0.5 * (ref * (ref.log() - mixture.log())).sum(dim=-1)
    js = js + 0.5 * (cand * (cand.log() - mixture.log())).sum(dim=-1)
    tv = 0.5 * (reference.float() - candidate.float()).abs().sum(dim=-1)
    abs_diff = (reference.float() - candidate.float()).abs()
    return {
        "kl_ref_to_variant_mean": _mean(kl),
        "kl_ref_to_variant_max": _max(kl),
        "js_mean": _mean(js),
        "js_max": _max(js),
        "total_variation_mean": _mean(tv),
        "total_variation_max": _max(tv),
        "probability_abs_diff_mean": _mean(abs_diff),
        "probability_abs_diff_max": _max(abs_diff),
    }


def _top_token_overlap(reference: Any, candidate: Any, *, top_k: int) -> dict[str, float | int | None]:
    th = _ensure_torch()
    seq_len = int(reference.shape[-1])
    k = max(1, min(int(top_k), seq_len))
    ref_top = th.topk(reference, k=k, dim=-1).indices
    cand_top = th.topk(candidate, k=k, dim=-1).indices
    overlap = (ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
    overlap_fraction = overlap / k
    top1_match = (ref_top[..., 0] == cand_top[..., 0]).float()
    return {
        "k": k,
        "mean_overlap_fraction": _mean(overlap_fraction),
        "min_overlap_fraction": _min(overlap_fraction),
        "top1_match_fraction": _mean(top1_match),
    }


def compute_attention_error_metrics(
    *,
    query: Any,
    current_key: Any,
    current_value: Any,
    cached_key_ref: Any,
    cached_value_ref: Any,
    cached_key_variant: Any | None = None,
    cached_value_variant: Any | None = None,
    current_key_variant: Any | None = None,
    current_value_variant: Any | None = None,
    attention_mask: Any | None = None,
    softmax_scale: float | None = None,
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compare reference attention against one reconstructed-cache variant.

    All tensors must already be in canonical BHSD layout:
    ``[batch, heads, sequence, head_dim]``. Current K/V tokens are appended
    after cached K/V tokens for both reference and variant attention.
    """

    _ensure_torch()
    cached_key_variant = cached_key_ref if cached_key_variant is None else cached_key_variant
    cached_value_variant = cached_value_ref if cached_value_variant is None else cached_value_variant
    current_key_variant = current_key if current_key_variant is None else current_key_variant
    current_value_variant = current_value if current_value_variant is None else current_value_variant

    ref_key = _concat_cache_current(cached_key_ref, current_key)
    ref_value = _concat_cache_current(cached_value_ref, current_value)
    variant_key = _concat_cache_current(cached_key_variant, current_key_variant)
    variant_value = _concat_cache_current(cached_value_variant, current_value_variant)

    _check_attention_shapes(query, ref_key, ref_value)
    _check_attention_shapes(query, variant_key, variant_value)
    if ref_key.shape != variant_key.shape or ref_value.shape != variant_value.shape:
        raise ValueError(
            "reference and variant attention tensors must have matching shapes: "
            f"ref_key={tuple(ref_key.shape)}, variant_key={tuple(variant_key.shape)}, "
            f"ref_value={tuple(ref_value.shape)}, variant_value={tuple(variant_value.shape)}"
        )

    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(int(query.shape[-1]))
    ref_logits, ref_probs, ref_output = _attention(query, ref_key, ref_value, scale=scale, mask=attention_mask)
    variant_logits, variant_probs, variant_output = _attention(
        query, variant_key, variant_value, scale=scale, mask=attention_mask
    )

    output_error = _error_summary(ref_output, variant_output)
    output_error["per_head_relative_l2"] = _per_head_relative_l2(ref_output, variant_output)
    return {
        "attention_logit_error": _error_summary(ref_logits, variant_logits),
        "attention_probability_divergence": _probability_divergence(ref_probs, variant_probs, eps=eps),
        "attention_output_relative_error": output_error,
        "top_attended_token_overlap": _top_token_overlap(ref_probs, variant_probs, top_k=top_k),
        "attention_config": {
            "layout": "BHSD",
            "softmax_scale": _safe_float(scale),
            "top_k": int(top_k),
            "cached_tokens": int(cached_key_ref.shape[2]),
            "current_tokens": int(current_key.shape[2]),
            "total_key_tokens": int(ref_key.shape[2]),
            "query_tokens": int(query.shape[2]),
            "head_dim": int(query.shape[-1]),
        },
    }


def _canonical_record_tensors(record: Mapping[str, Any], *, default_layout: str) -> dict[str, Any]:
    tensor_aliases = {
        "cached_key_ref": ("reference_cached_key", "cached_key_reference", "cached_key"),
        "cached_value_ref": ("reference_cached_value", "cached_value_reference", "cached_value"),
    }
    canonical: dict[str, Any] = {}
    for name in ("query", "current_key", "current_value", "cached_key_ref", "cached_value_ref"):
        tensor = _tensor_from_record(record, name, aliases=tensor_aliases.get(name, ()))
        canonical[name] = _canonical_bhsd(tensor, _layout_for(record, name, default_layout), name=name)

    mask = _optional_tensor_from_record(record, "attention_mask")
    if mask is not None:
        canonical["attention_mask"] = mask.detach().cpu()
    return canonical


def _canonical_variant_tensor(
    variant: Mapping[str, Any],
    record: Mapping[str, Any],
    key: str,
    *,
    default_layout: str,
) -> Any | None:
    aliases = {
        "cached_key": ("cached_key_variant", "reconstructed_cached_key", "quantized_cached_key"),
        "cached_value": ("cached_value_variant", "reconstructed_cached_value", "quantized_cached_value"),
        "current_key": ("current_key_variant",),
        "current_value": ("current_value_variant",),
    }
    tensor = _optional_tensor_from_record(variant, key, aliases=aliases.get(key, ()))
    if tensor is None:
        return None
    layout = str(variant.get("layout", _layout_for(record, key, default_layout)))
    return _canonical_bhsd(tensor, layout, name=key)


def _iter_variants(
    record: Mapping[str, Any],
    *,
    identity_variant_if_missing: bool = False,
) -> list[tuple[str, Mapping[str, Any]]]:
    variants = record.get("variants")
    if isinstance(variants, Mapping):
        return [(str(name), item if isinstance(item, Mapping) else {"cached_key": item}) for name, item in variants.items()]
    if isinstance(variants, list):
        result = []
        for idx, item in enumerate(variants):
            if not isinstance(item, Mapping):
                raise TypeError(f"variant #{idx} must be a mapping, got {type(item)!r}")
            result.append((str(item.get("name", f"variant_{idx}")), item))
        return result
    fallback = {
        "cached_key": _optional_tensor_from_record(record, "cached_key_variant", aliases=("reconstructed_cached_key",)),
        "cached_value": _optional_tensor_from_record(
            record, "cached_value_variant", aliases=("reconstructed_cached_value",)
        ),
    }
    if fallback["cached_key"] is None and fallback["cached_value"] is None:
        if identity_variant_if_missing:
            return [
                (
                    "identity_reference",
                    {
                        "metadata": {
                            "identity_variant": True,
                            "note": "Reference tensors reused as the variant; this checks the offline attention path only.",
                        }
                    },
                )
            ]
        raise ValueError("probe record has no variants to compare")
    return [(str(record.get("variant_name", "variant")), fallback)]


def probe_record(
    record: Mapping[str, Any],
    *,
    default_layout: str,
    top_k: int,
    eps: float,
    identity_variant_if_missing: bool = False,
) -> dict[str, Any]:
    canonical = _canonical_record_tensors(record, default_layout=default_layout)
    variants = []
    for name, variant in _iter_variants(record, identity_variant_if_missing=identity_variant_if_missing):
        metrics = compute_attention_error_metrics(
            query=canonical["query"],
            current_key=canonical["current_key"],
            current_value=canonical["current_value"],
            cached_key_ref=canonical["cached_key_ref"],
            cached_value_ref=canonical["cached_value_ref"],
            cached_key_variant=_canonical_variant_tensor(variant, record, "cached_key", default_layout=default_layout),
            cached_value_variant=_canonical_variant_tensor(variant, record, "cached_value", default_layout=default_layout),
            current_key_variant=_canonical_variant_tensor(variant, record, "current_key", default_layout=default_layout),
            current_value_variant=_canonical_variant_tensor(variant, record, "current_value", default_layout=default_layout),
            attention_mask=canonical.get("attention_mask"),
            softmax_scale=variant.get("softmax_scale", record.get("softmax_scale")),
            top_k=top_k,
            eps=eps,
        )
        variants.append({"name": name, "metrics": metrics, "metadata": _safe_jsonify(variant.get("metadata", {}))})

    return {
        "name": str(record.get("name", "record")),
        "metadata": _safe_jsonify(record.get("metadata", {})),
        "tensor_shapes": {name: _shape(tensor) for name, tensor in canonical.items() if torch.is_tensor(tensor)},  # type: ignore[union-attr]
        "variants": variants,
    }


def _torch_load_kwargs(*, weights_only: bool) -> dict[str, Any]:
    th = _ensure_torch()
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        signature = inspect.signature(th.load)
    except (TypeError, ValueError):
        signature = None
    if signature is None or "weights_only" in signature.parameters:
        kwargs["weights_only"] = weights_only
    return kwargs


def _load_probe_input(path: Path, *, trust_pickle: bool) -> Any:
    th = _ensure_torch()
    return th.load(str(path), **_torch_load_kwargs(weights_only=not trust_pickle))


def _records_from_probe_input(obj: Any) -> list[Mapping[str, Any]]:
    if isinstance(obj, Mapping) and isinstance(obj.get("records"), list):
        records = obj["records"]
    elif isinstance(obj, list):
        records = obj
    elif isinstance(obj, Mapping) and "query" in obj:
        records = [obj]
    else:
        raise ValueError("probe input must be a mapping with records, a record mapping, or a list of records")

    result = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record #{idx} must be a mapping, got {type(record)!r}")
        result.append(record)
    return result


def run_probe_artifact(
    input_path: str | Path,
    *,
    default_layout: str = "BHSD",
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
    record_limit: int | None = None,
    trust_pickle: bool = False,
    identity_variant_if_missing: bool = False,
) -> dict[str, Any]:
    path = Path(input_path).expanduser()
    stat = path.stat()
    obj = _load_probe_input(path, trust_pickle=trust_pickle)
    records = _records_from_probe_input(obj)
    if record_limit is not None:
        records = records[:record_limit]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "input": {
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_unix": stat.st_mtime,
            "mtime_utc": _utc_iso(stat.st_mtime),
            "trust_pickle": trust_pickle,
        },
        "config": {
            "default_layout": default_layout,
            "top_k": int(top_k),
            "eps": _safe_float(eps),
            "record_count": len(records),
            "identity_variant_if_missing": bool(identity_variant_if_missing),
        },
        "records": [
            probe_record(
                record,
                default_layout=default_layout,
                top_k=top_k,
                eps=eps,
                identity_variant_if_missing=identity_variant_if_missing,
            )
            for record in records
        ],
    }


def _list_sorted(values: Iterable[Any]) -> list[Any]:
    return sorted({item for item in values if item is not None})


def smoke_cache_audit_summary(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    input_info = data.get("input", {}) if isinstance(data, Mapping) else {}
    provenance = data.get("provenance", {}) if isinstance(data, Mapping) else {}
    subset = provenance.get("qvg_cache_audit_subset", {}) if isinstance(provenance, Mapping) else {}
    tensors = data.get("tensors", []) if isinstance(data, Mapping) else []
    usable_cache_tensors = [
        item
        for item in tensors
        if isinstance(item, Mapping)
        and item.get("cache_kind") == "self_attention"
        and item.get("kv_role") in {"key", "value"}
        and item.get("layout") in {"BHSD", "BSHD"}
    ]
    roles = _list_sorted(item.get("kv_role") for item in usable_cache_tensors)
    layers = _list_sorted(item.get("layer_idx") for item in usable_cache_tensors)
    chunks = _list_sorted(item.get("chunk_idx") for item in usable_cache_tensors)
    selected_bytes = sum(int(item.get("estimated_bytes") or 0) for item in usable_cache_tensors)

    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_summary": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "schema_version": data.get("schema_version") if isinstance(data, Mapping) else None,
        },
        "bounded_subset": {
            "path": input_info.get("path"),
            "size_bytes": input_info.get("size_bytes"),
            "load_status": data.get("load", {}).get("status") if isinstance(data.get("load"), Mapping) else None,
            "stats_decision": data.get("load", {}).get("stats_decision") if isinstance(data.get("load"), Mapping) else None,
            "byte_guards": {
                "stats": {
                    "max_full_load_bytes": data.get("load", {}).get("max_full_load_bytes")
                    if isinstance(data.get("load"), Mapping)
                    else None,
                    "max_total_stat_bytes": data.get("load", {}).get("max_total_stat_bytes")
                    if isinstance(data.get("load"), Mapping)
                    else None,
                    "max_tensor_stat_bytes": data.get("load", {}).get("max_tensor_stat_bytes")
                    if isinstance(data.get("load"), Mapping)
                    else None,
                },
                "subset": subset.get("byte_guards") if isinstance(subset, Mapping) else None,
            },
            "source_artifact": subset.get("source_artifact") if isinstance(subset, Mapping) else None,
            "source_size_bytes": subset.get("source_size_bytes") if isinstance(subset, Mapping) else None,
        },
        "available_without_tensor_materialization": {
            "self_attention_cache_tensor_count": len(usable_cache_tensors),
            "roles": roles,
            "layers": layers,
            "chunks": chunks,
            "estimated_cache_tensor_bytes": selected_bytes,
            "layouts": _list_sorted(item.get("layout") for item in usable_cache_tensors),
            "statistics_available": [
                "cache K/V tensor metadata",
                "bounded K/V norm and tuple4-radius summaries",
                "bounded per-head summaries",
                "selected chunk drift summaries",
            ],
        },
        "attention_probe_computability": {
            "can_compute_attention_logit_error": False,
            "can_compute_attention_probability_divergence": False,
            "can_compute_attention_output_relative_error": False,
            "can_compute_top_attended_token_overlap": False,
            "reason": (
                "bounded cache stats contain cached K/V tensors or summaries only; they do not contain "
                "query, current_key, current_value, attention masks, or paired BF16/reference versus "
                "quantized/reconstructed cache tensors"
            ),
            "missing_required_tensors": [
                "query",
                "current_key",
                "current_value",
                "cached_key_reference",
                "cached_value_reference",
                "cached_key_variant",
                "cached_value_variant",
            ],
        },
    }


def _write_json(data: Mapping[str, Any], output: str | None, *, compact: bool) -> None:
    text = json.dumps(data, indent=None if compact else 2, sort_keys=True, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="torch.save probe input artifact")
    mode.add_argument("--cache-audit-summary", help="cache_audit JSON to smoke for attention-probe readiness")
    parser.add_argument("-o", "--output", help="Write JSON output to this path")
    parser.add_argument("--default-layout", choices=("BHSD", "BSHD"), default="BHSD")
    parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_TOP_K)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--record-limit", type=_positive_int)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow trusted-pickle torch.load for local probe artifacts; default is weights-only",
    )
    parser.add_argument(
        "--identity-variant-if-missing",
        action="store_true",
        help=(
            "If a record has only reference tensors, compare it against itself. "
            "This validates the offline attention path but does not measure quantized-cache distortion."
        ),
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.cache_audit_summary:
            result = smoke_cache_audit_summary(args.cache_audit_summary)
        else:
            result = run_probe_artifact(
                args.input,
                default_layout=args.default_layout,
                top_k=args.top_k,
                eps=args.eps,
                record_limit=args.record_limit,
                trust_pickle=args.trust_pickle,
                identity_variant_if_missing=args.identity_variant_if_missing,
            )
    except Exception as exc:
        print(json.dumps({"status": "attention_probe_failed", "error": repr(exc)}), file=sys.stderr)
        return 2
    _write_json(result, args.output, compact=args.compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
