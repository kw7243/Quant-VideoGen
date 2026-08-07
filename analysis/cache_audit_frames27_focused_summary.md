# Focused cache anomaly audit summary

Source artifact:

```text
/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt
```

Subset selection:

- Self-attention layers `27,28,29`.
- Roles `key,value`.
- Chunks `23,24,25,26`.
- Source heads `0,9`.

Byte guards:

- `max_subset_bytes`: `134217728`.
- Selected tensor bytes: `19169280`.
- Subset file size: `19182930`.
- Stats run `--max-full-load-bytes`: `134217728`.
- Stats run `--max-total-stat-bytes`: `134217728`.
- Stats run `--max-tensor-stat-bytes`: `16777216`.

Staged runs:

- Subset extraction:
  `staging/20260806_230053-36720c-118e8ab-code`.
- Stats audit:
  `staging/20260806_230149-5bbdd1-118e8ab-code`.

## Findings

Layer 29 K head 9 remains the dominant vector-norm outlier in the focused
window. Across chunks `23-26`, layer 29 K head 9 has vector-norm p95 mean
`105.279` and max `105.674`. The next highest sampled head is layer 28 K head 0
with p95 mean `35.613` and max `35.906`.

Head 0 is not another layer-29 anomaly. Layer 29 K head 0 has p95 mean
`26.876`, about one quarter of layer 29 K head 9. However, head 0 is not a
perfect universal null control because layer 28 K head 0 is the second-highest
sampled head.

The earlier chunk-26 radius-proxy jump is not a one-chunk artifact. The 4-bit
radius-only relative L2 proxy is low at chunk 23 (`0.001434` mean), jumps at
chunk 24 (`0.047553`), and stays high at chunks 25 (`0.047646`) and 26
(`0.047883`). The jump is visible in every focused layer/role group from chunk
23 to 24. Norm and radius magnitudes remain roughly stable across the same
transition.

K/V asymmetry persists in the same direction as the broader bounded subset:

- K vector-norm p95 mean `55.423` versus V `21.484`.
- K tuple4 radius outlier fraction mean `0.221843` versus V `0.001056`.
- K 4-bit radius-only rel L2 proxy mean `0.047273` versus V `0.024985`.
- V temporal relative-delta p95 mean `1.410369` versus K `0.529423`.

The radius proxy is still only a radius-preserving proxy. It is not direct
HQMQ direction-codebook error, attention error, video quality, memory, latency,
or novelty evidence.

## Next Probe

Once attention-capture job `1316140` finishes, the smallest functional probe is
to inspect its capture manifest and run one CPU offline attention-error probe on
captured layer 29 head 9 around the chunk `23 -> 24` transition, with head 0 as
the control and K-only/V-only variants separated. This should measure attention
logit/probability/output sensitivity before any video-quality claim.

Machine-readable files:

- `analysis/cache_audit_frames27_focused_subset_manifest.json`
- `analysis/cache_audit_frames27_focused_bounded_stats.json`
- `analysis/cache_audit_frames27_focused_findings.json`
