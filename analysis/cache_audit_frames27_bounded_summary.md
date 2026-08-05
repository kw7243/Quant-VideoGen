# Bounded Real-Artifact KV Cache Statistics

Source artifact:

```text
/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt
```

Machine-readable outputs:

- `analysis/cache_audit_frames27_subset_manifest.json`
- `analysis/cache_audit_frames27_bounded_stats.json`

The physical subset tensor artifact is not committed:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset.pt
```

## Byte Guards And Selection

| Item | Value |
| --- | ---: |
| Full artifact size | 7,858,486,627 bytes |
| Full metadata load | `fake_tensor_weights_only` |
| Subset tensor count | 18 |
| Subset file size | 86,271,704 bytes |
| Selected tensor bytes | 86,261,760 bytes |
| Subset byte guard | 134,217,728 bytes |
| Audit total-stat guard | 134,217,728 bytes |
| Audit per-tensor guard | 16,777,216 bytes |
| Stats mode | `auto`, computed |

Selection: self-attention layers 0, 14, and 29; roles K and V; chunks 0, 13, and 26; all 12 heads; full 1,560-token chunk spans; head dim 128. This covers early/mid/late depth and cache age without materializing the 7.86 GB dump. It is enough for the next decision about where to expand the audit, not enough for a final method claim.

## K/V Asymmetry

Direct original-to-quantized error is not measurable from this artifact because no paired BF16 pre-quant cache is present. The error column below is a radius-only 4-bit re-quantization proxy: tuple directions are preserved, so this is not full HQMQ codebook error.

| Role | Tensors | Norm p95 mean | Norm outlier frac | Radius p95 mean | Radius outlier frac | Radius-only rel L2, 4-bit | Temporal rel delta p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K | 9 | 48.255 | 0.0278 | 5.659 | 0.1205 | 0.0189 | 0.8765 |
| V | 9 | 18.260 | 0.0000 | 3.764 | 0.0047 | 0.0123 | 1.3969 |

The subset supports a real K/V role asymmetry: keys have much larger norm and radius outliers, while values have larger relative adjacent-token temporal deltas.

## Layer And Role Outliers

| Layer/role | Norm p95 mean | Norm outlier max | Radius p95 mean | Radius outlier max | Radius-only rel L2, 4-bit | Temporal rel delta p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer 0 K | 20.061 | 0.0000 | 5.629 | 0.2353 | 0.0202 | 0.8313 |
| layer 0 V | 20.770 | 0.0000 | 4.317 | 0.0103 | 0.0130 | 1.3893 |
| layer 14 K | 21.949 | 0.0000 | 5.021 | 0.0111 | 0.0141 | 1.1116 |
| layer 14 V | 13.964 | 0.0000 | 3.052 | 0.0007 | 0.0120 | 1.4865 |
| layer 29 K | 102.754 | 0.0833 | 6.329 | 0.1397 | 0.0224 | 0.6866 |
| layer 29 V | 20.048 | 0.0000 | 3.924 | 0.0060 | 0.0120 | 1.3150 |

Layer 29 K is the clear head/layer outlier: vector norm p95 is about 5x the other sampled K layers, with 8.33% tensor-level vector-norm outliers. Layer 0 K also has strong 4-tuple radius outliers despite ordinary vector-norm p95.

Top per-head norm p95 outlier:

| Layer | Role | Head | Head norm p95 mean | Head norm p95 max |
| ---: | --- | ---: | ---: | ---: |
| 29 | K | 9 | 104.263 | 106.346 |
| 29 | K | 0 | 26.863 | 27.009 |
| 29 | V | 9 | 24.212 | 26.820 |
| 29 | V | 0 | 22.758 | 26.519 |

Head 9 in layer 29 K dominates the per-head outlier picture. The next highest sampled head is roughly 4x smaller.

## Four-Tuple Radius Structure

All sampled tensors have head dim 128, so HQMQ-style 4-coordinate grouping gives 32 tuples per token. Keys have higher radius p95 and much higher radius outlier rate than values. The 2-bit radius-only proxy is also worse for K than V: K mean 0.2144 rel L2 versus V mean 0.1656. This supports focusing mixed precision or outlier protection on K first, especially late-layer K.

The radius-only proxy is most useful as a lower-bound diagnostic. It does not include direction codebook assignment error, attention-output error, or generated-video quality.

## Temporal And Chunk Drift

| Chunk | Tensors | Norm p95 mean | Radius p95 mean | Radius outlier mean | Temporal rel delta p95 | Radius-only rel L2, 4-bit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6 | 31.898 | 4.472 | 0.0585 | 1.1150 | 0.00146 |
| 13 | 6 | 34.131 | 4.838 | 0.0639 | 1.1542 | 0.00145 |
| 26 | 6 | 33.744 | 4.825 | 0.0653 | 1.1409 | 0.04388 |

Chunk 26 has much higher 4-bit radius-only re-quantization error than chunks 0 and 13, while norm/radius magnitudes and within-chunk temporal deltas are comparatively stable. That pattern is consistent with chunk-state or refresh drift, but it needs a targeted follow-up before treating it as a mechanism.

## Supported Next Decision

The next smallest experiment should expand only around the observed structure:

1. Add layers near 27-29 and heads 0/9 for K.
2. Include chunks 24-26 plus one adjacent earlier chunk to test the final-chunk radius-error jump.
3. Add a paired BF16 cache or pre/post fake-HQMQ dump so direct quantization error and attention-output error can be measured.

This data supports video-cache geometry hypotheses. It does not support novelty claims about generic KV quantization, generic head-aware compression, or final video quality.
