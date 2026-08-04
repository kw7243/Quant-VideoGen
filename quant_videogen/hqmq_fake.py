"""Fake HQMQ-style KV-cache adapter for Quant-VideoGen.

This mirrors the inference-only fake-quant contract from the read-only HQMQ
reference tree without importing or modifying that project. The adapter returns
dense reconstructed tensors, so it is for quality-transfer tests only and does
not provide packed-cache memory or latency savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any

import torch


HQMQ_FAKE_TYPES = ("hqmq-fake", "hqmq-fake-med3x")


@dataclass(frozen=True)
class FakeHQMQConfig:
    secondary_size: int = 96
    radius_bits: int = 4
    seed: int = 42
    pad_head_dim: bool = True
    log_radius: bool = False
    median_mult: float | None = None
    max_sim_elements: int = 268_435_456

    @classmethod
    def from_quant_config(cls, quant_type: str, quant_config: Any) -> "FakeHQMQConfig":
        median_mult = getattr(quant_config, "hqmq_median_mult", None)
        if quant_type == "hqmq-fake-med3x" and median_mult is None:
            median_mult = 3.0
        return cls(
            secondary_size=int(getattr(quant_config, "hqmq_secondary_size", 96)),
            radius_bits=int(getattr(quant_config, "hqmq_radius_bits", 4)),
            seed=int(getattr(quant_config, "hqmq_seed", 42)),
            pad_head_dim=bool(getattr(quant_config, "hqmq_pad_head_dim", True)),
            log_radius=bool(getattr(quant_config, "hqmq_log_radius", False)),
            median_mult=None if median_mult is None else float(median_mult),
            max_sim_elements=int(getattr(quant_config, "hqmq_max_sim_elements", 268_435_456)),
        )

    def validate(self) -> None:
        if self.secondary_size <= 0:
            raise ValueError(f"hqmq_secondary_size must be positive, got {self.secondary_size}")
        if self.radius_bits < 0:
            raise ValueError(f"hqmq_radius_bits must be non-negative, got {self.radius_bits}")
        if self.median_mult is not None and self.median_mult <= 0:
            raise ValueError(f"hqmq_median_mult must be positive, got {self.median_mult}")
        if self.max_sim_elements <= 0:
            raise ValueError(f"hqmq_max_sim_elements must be positive, got {self.max_sim_elements}")


def relative_l2_error(reference: torch.Tensor, reconstructed: torch.Tensor) -> float:
    ref = reference.float()
    rec = reconstructed.float()
    denom = ref.norm().clamp(min=1e-12)
    return float((ref - rec).norm().div(denom).detach().cpu().item())


def _make_24cell_codebook() -> torch.Tensor:
    codebook = []
    for axis in range(4):
        for sign in (1.0, -1.0):
            value = [0.0] * 4
            value[axis] = sign
            codebook.append(value)
    for signs in product((1.0, -1.0), repeat=4):
        codebook.append([0.5 * sign for sign in signs])
    result = torch.tensor(codebook, dtype=torch.float32)
    if result.shape != (24, 4):
        raise RuntimeError(f"invalid 24-cell codebook shape {tuple(result.shape)}")
    return result


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamiltonian product for tensors ending in quaternion order (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


class FakeHQMQCacheAdapter:
    """Inference-only HQMQ-style fake quantizer for BHSD K/V tensors."""

    chunk_dim = 4

    def __init__(self, config: FakeHQMQConfig):
        config.validate()
        self.config = config
        self._primary = _make_24cell_codebook()
        self._secondary_raw: torch.Tensor | None = None
        self._secondary_layers = 0
        self._secondary_heads = 0
        self.last_diagnostics: list[dict[str, Any]] = []
        self.last_tile_stats: dict[str, int] | None = None

    @classmethod
    def from_quant_config(cls, quant_type: str, quant_config: Any) -> "FakeHQMQCacheAdapter":
        return cls(FakeHQMQConfig.from_quant_config(quant_type, quant_config))

    def bits_per_value(self) -> float:
        bits_dir = math.log2(24 * self.config.secondary_size)
        return (bits_dir + self.config.radius_bits) / self.chunk_dim

    def effective_bits_per_value(self, outlier_rate: float | None = None) -> float:
        base_bpv = self.bits_per_value()
        if outlier_rate is None:
            return base_bpv
        return (1.0 - outlier_rate) * base_bpv + outlier_rate * 16.0 + (1.0 / self.chunk_dim)

    def quantize_pair(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_diagnostics = []
        self.last_tile_stats = None
        k_quant = self.quantize_k(k, layer_idx=layer_idx)
        v_quant = self.quantize_v(v, layer_idx=layer_idx)
        return k_quant, v_quant

    def quantize_k(self, k: torch.Tensor, *, layer_idx: int = 0) -> torch.Tensor:
        return self._quantize(k, layer_idx=layer_idx, kv=0, name="k")

    def quantize_v(self, v: torch.Tensor, *, layer_idx: int = 0) -> torch.Tensor:
        return self._quantize(v, layer_idx=layer_idx, kv=1, name="v")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantize(x, layer_idx=0, kv=0, name="tensor")

    def _validate_input(self, x: torch.Tensor, layer_idx: int) -> tuple[int, int, int, int]:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"FakeHQMQCacheAdapter expects torch.Tensor, got {type(x)!r}")
        if x.ndim != 4:
            raise ValueError(f"HQMQ fake expects BHSD tensor with 4 dims, got shape {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"HQMQ fake expects floating tensor dtype, got {x.dtype}")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        bsz, heads, seq_len, head_dim = x.shape
        if heads <= 0 or seq_len <= 0 or head_dim <= 0:
            raise ValueError(f"HQMQ fake expects non-empty BHSD shape, got {tuple(x.shape)}")
        if head_dim % self.chunk_dim != 0 and not self.config.pad_head_dim:
            raise ValueError(
                f"HQMQ fake requires head_dim divisible by {self.chunk_dim}; "
                f"got {head_dim}. Enable hqmq_pad_head_dim to pad and truncate."
            )
        return bsz, heads, seq_len, head_dim

    def _ensure_secondary(self, *, heads: int, layers: int) -> None:
        if (
            self._secondary_raw is not None
            and self._secondary_heads == heads
            and self._secondary_layers >= layers
        ):
            return

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        raw = torch.randn(
            layers,
            heads,
            2,
            self.config.secondary_size,
            self.chunk_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self._secondary_raw = raw
        self._secondary_layers = layers
        self._secondary_heads = heads

    def _joint_codebook(self, *, layer_idx: int, kv: int, heads: int, device: torch.device) -> torch.Tensor:
        self._ensure_secondary(heads=heads, layers=layer_idx + 1)
        if self._secondary_raw is None:
            raise RuntimeError("secondary HQMQ codebook was not initialized")

        primary = self._primary.to(device=device)
        secondary = self._secondary_raw[layer_idx, :, kv].to(device=device)
        secondary = secondary / secondary.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        p_expanded = primary.unsqueeze(0).unsqueeze(2).expand(heads, 24, self.config.secondary_size, 4)
        s_expanded = secondary.unsqueeze(1).expand(heads, 24, self.config.secondary_size, 4)
        return _quat_mul(p_expanded, s_expanded).reshape(heads, 24 * self.config.secondary_size, 4)

    def _quantize(self, x: torch.Tensor, *, layer_idx: int, kv: int, name: str) -> torch.Tensor:
        _, heads, _, head_dim = self._validate_input(x, layer_idx)
        original_dtype = x.dtype
        padded_head_dim = ((head_dim + self.chunk_dim - 1) // self.chunk_dim) * self.chunk_dim

        work = x.float()
        if padded_head_dim != head_dim:
            pad = torch.zeros(*work.shape[:-1], padded_head_dim - head_dim, dtype=work.dtype, device=work.device)
            work = torch.cat([work, pad], dim=-1)

        quantized = self._quantize_padded(work, layer_idx=layer_idx, kv=kv, heads=heads)
        tile_stats = self.last_tile_stats or {}
        if self.config.median_mult is not None:
            quantized, outlier_rate = self._restore_outliers(work, quantized)
        else:
            outlier_rate = None

        quantized = quantized[..., :head_dim].to(dtype=original_dtype)
        base_bpv = self.bits_per_value()
        diagnostics = {
            "tensor": name,
            "shape": tuple(x.shape),
            "dtype": str(original_dtype),
            "layer_idx": layer_idx,
            "secondary_size": self.config.secondary_size,
            "radius_bits": self.config.radius_bits,
            "direction_bits_per_chunk": math.log2(24 * self.config.secondary_size),
            "radius_bits_per_chunk": self.config.radius_bits,
            "outlier_metadata_bits_per_chunk": 0 if outlier_rate is None else 1,
            "base_bits_per_value": base_bpv,
            "effective_bits_per_value": self.effective_bits_per_value(outlier_rate),
            "radius_scale_overhead_bits_per_value": 16.0 / padded_head_dim,
            "fake_dense_return": True,
            "head_dim": head_dim,
            "padded_head_dim": padded_head_dim,
            "pad_count": padded_head_dim - head_dim,
            "log_radius": self.config.log_radius,
            "median_mult": self.config.median_mult,
            "outlier_rate": outlier_rate,
            "max_sim_elements": self.config.max_sim_elements,
            "sim_tile_tokens": tile_stats.get("sim_tile_tokens"),
            "sim_codeword_tile": tile_stats.get("sim_codeword_tile"),
            "sim_tile_count": tile_stats.get("sim_tile_count"),
            "max_sim_tile_elements": tile_stats.get("max_sim_tile_elements"),
            "dense_sim_elements": tile_stats.get("dense_sim_elements"),
            "relative_l2": relative_l2_error(x, quantized),
            "finite": bool(torch.isfinite(quantized).all().detach().cpu().item()),
        }
        self.last_diagnostics.append(diagnostics)
        return quantized

    def _quantize_padded(self, x: torch.Tensor, *, layer_idx: int, kv: int, heads: int) -> torch.Tensor:
        bsz, _, seq_len, head_dim = x.shape
        n_chunks = head_dim // self.chunk_dim
        chunked = x.reshape(bsz, heads, seq_len, n_chunks, self.chunk_dim)
        radius = chunked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = chunked / radius

        joint = self._joint_codebook(layer_idx=layer_idx, kv=kv, heads=heads, device=x.device)
        direction_quant = self._nearest_joint_directions(direction, joint)
        radius_quant = self._quantize_radius(radius)
        return (radius_quant * direction_quant).reshape(bsz, heads, seq_len, head_dim)

    def _nearest_joint_directions(self, direction: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        bsz, heads, seq_len, n_chunks, _ = direction.shape
        codewords = joint.shape[1]
        dense_sim_elements = bsz * heads * seq_len * n_chunks * codewords
        tile_denominator = bsz * heads * n_chunks * codewords
        tile_tokens = max(1, min(seq_len, self.config.max_sim_elements // tile_denominator))
        codeword_denominator = bsz * heads * tile_tokens * n_chunks
        codeword_tile = max(1, min(codewords, self.config.max_sim_elements // codeword_denominator))

        output = torch.empty_like(direction)
        max_tile_elements = 0
        tile_count = 0
        joint_for_gather = joint.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand(
            bsz, heads, 1, 1, codewords, self.chunk_dim
        )

        for start in range(0, seq_len, tile_tokens):
            end = min(start + tile_tokens, seq_len)
            tile = direction[:, :, start:end]
            best_scores = None
            idx = None

            for code_start in range(0, codewords, codeword_tile):
                code_end = min(code_start + codeword_tile, codewords)
                sims = torch.einsum("bhtnc,hkc->bhtnk", tile, joint[:, code_start:code_end])
                tile_scores, tile_idx = sims.max(dim=-1)
                tile_idx = tile_idx + code_start
                if best_scores is None or idx is None:
                    best_scores = tile_scores
                    idx = tile_idx
                else:
                    update = tile_scores > best_scores
                    best_scores = torch.where(update, tile_scores, best_scores)
                    idx = torch.where(update, tile_idx, idx)
                max_tile_elements = max(max_tile_elements, sims.numel())
                tile_count += 1

            if idx is None:
                raise RuntimeError("HQMQ fake did not evaluate any codewords")
            gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1, self.chunk_dim)
            output[:, :, start:end] = joint_for_gather.expand(
                bsz, heads, end - start, n_chunks, codewords, self.chunk_dim
            ).gather(dim=-2, index=gather_idx).squeeze(-2)

        self.last_tile_stats = {
            "sim_tile_tokens": tile_tokens,
            "sim_codeword_tile": codeword_tile,
            "sim_tile_count": tile_count,
            "max_sim_tile_elements": max_tile_elements,
            "dense_sim_elements": dense_sim_elements,
        }
        return output

    def _quantize_radius(self, radius: torch.Tensor) -> torch.Tensor:
        if self.config.radius_bits >= 16:
            return radius
        if self.config.radius_bits == 0:
            return radius.mean(dim=-2, keepdim=True).expand_as(radius)

        qmax = (1 << self.config.radius_bits) - 1
        if self.config.log_radius:
            log_radius = torch.log(radius.clamp(min=1e-8))
            log_min = log_radius.amin(dim=-2, keepdim=True)
            log_range = (log_radius.amax(dim=-2, keepdim=True) - log_min).clamp(min=1e-8)
            q = ((log_radius - log_min) / log_range * qmax).round().clamp(0, qmax)
            return torch.exp(log_min + (q / qmax) * log_range)

        r_max = radius.amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = r_max / qmax
        return (radius / scale).round().clamp(0, qmax) * scale

    def _restore_outliers(self, original: torch.Tensor, quantized: torch.Tensor) -> tuple[torch.Tensor, float]:
        bsz, heads, seq_len, head_dim = original.shape
        n_chunks = head_dim // self.chunk_dim
        original_chunks = original.reshape(bsz, heads, seq_len, n_chunks, self.chunk_dim)
        quantized_chunks = quantized.reshape(bsz, heads, seq_len, n_chunks, self.chunk_dim)
        chunk_norms = original_chunks.norm(dim=-1)

        flat = chunk_norms.permute(1, 0, 2, 3).reshape(heads, -1)
        median = flat.median(dim=-1).values
        threshold = median * float(self.config.median_mult)
        outlier_mask = chunk_norms > threshold.view(1, heads, 1, 1)
        restored = torch.where(outlier_mask.unsqueeze(-1), original_chunks, quantized_chunks)
        outlier_rate = float(outlier_mask.float().mean().detach().cpu().item())
        return restored.reshape(bsz, heads, seq_len, head_dim), outlier_rate
