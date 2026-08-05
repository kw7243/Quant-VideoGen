# Lab Notebook

## 2026-07-24 - HQMQ Transfer Plan For Quant-VideoGen

Task: plan an experimental transfer of HQMQ into Quant-VideoGen without editing
code, configs, scripts, dependency files, or the HQMQ checkout. All
Quant-VideoGen repo edits for this task were limited to this `notebook.md` file
in the task worktree:

```text
/afs/csail.mit.edu/u/k/kwen1/.treehouse/quant-videogen-d60f5d/1/quant-videogen
```

Firstmate report:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-hqmq-plan/report.md
```

### What I Inspected

Quant-VideoGen:

- `README.md`: project scope, supported integrations, memory table, baseline
  commands, default QVG quant config.
- `pyproject.toml`: package extras for LongCat, Self-Forcing, HY-WorldPlay, and
  LingBot.
- `quant_videogen/compress.py`: `QuantizeFunctions`, `get_quantize_fn`,
  `get_quantize_type`, `compress_kv_cache`.
- `quant_videogen/uncompress.py`: `uncompress_kv_cache` and `extract_num_bits`.
- `quant_videogen/kv_cache.py`: `ChunkedKVCache`, `write`, `read`,
  `store_quantized`, `offload_kv_cache_layer`, `onload_kv_cache_layer`.
- `quant_videogen/functions.py` and `quant_videogen/real/*`: PRQ/k-means
  packed-cache encode/decode path.
- `scripts/*/run_bf16.sh`, `scripts/*/run_qvg.sh`, and
  `scripts/LongCat/base.sh`: existing BF16 and QVG baselines.
- LongCat integration:
  - `experiments/LongCat/run_long_t2v.py`
  - `experiments/LongCat/longcat_video/modules/longcat_video_dit.py`
  - `experiments/LongCat/longcat_video/modules/attention.py`
- Self-Forcing integration:
  - `experiments/Self-Forcing/inference.py`
  - `experiments/Self-Forcing/pipeline/causal_inference.py`
  - `experiments/Self-Forcing/wan/modules/causal_model.py`
  - `experiments/Self-Forcing/wan/configs/__init__.py`
- HY-WorldPlay integration:
  - `experiments/HY-WorldPlay/wan/generate.py`
  - `experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py`
  - `experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py`
- LingBot integration:
  - `experiments/LingBot/generate_fast.py`
  - `experiments/LingBot/wan/image2video_fast.py`
  - `experiments/LingBot/wan/modules/model_fast.py`
  - `experiments/LingBot/wan/configs/__init__.py`
- `logs/qvg_smoke_12627.out` and `.err`: prior smoke-test output.

HQMQ read-only reference checkout:

```text
/data/scratch-fast/kwen1/hqmq
```

Inspected:

- `README.md`
- `REPRODUCTION_REPORT.md`
- `hqmq/__init__.py`
- `src/model_utils.py`
- `src/quantizers/base.py`
- `src/quantizers/hqmq.py`
- `src/quantizers/hqmq_padded.py`
- `src/quantizers/outlier_aware.py`
- `src/quantizers/hqmq_attention.py`
- `src/calibration.py`
- relevant experiment script names under `experiments/`

The HQMQ checkout is dirty and has untracked results/reproduction artifacts. I
treated it as read-only and recommend not editing it for Quant-VideoGen work.

### Executive Recommendation

Do not assume HQMQ can replace Quant-VideoGen's current PRQ/K-means cache
compression as-is. Start with HQMQ as a fake-quant adapter behind
`quant_videogen/compress.py::compress_kv_cache`, using the existing BHSD K/V
cache boundary. This first isolates tensor compatibility and video quality. It
does not prove real memory savings, because fake-quant HQMQ returns dense
BF16/fp16 tensors.

Only move to a real packed HQMQ cache after fake-quant runs show useful video
quality. The packed path needs a new schema for HQMQ indices, radius quantizers,
scales, codebooks, and outlier metadata, plus matching decode support in
`quant_videogen/uncompress.py` or a fused attention adapter.

Best first model target: Self-Forcing if the team wants the smallest one-GPU
cache-dump loop; HY-WorldPlay if the team wants to prioritize the current
HY-WorldPlay quality/memory concern. LingBot and LongCat should wait until the
basic transfer risk is clearer.

### Integration Interface

The exact shared plug-in surface is:

```text
quant_videogen/compress.py::get_quantize_fn
quant_videogen/compress.py::get_quantize_type
quant_videogen/compress.py::compress_kv_cache(k, v, quant_type, quant_config, quantize_fn)
quant_videogen/uncompress.py::uncompress_kv_cache(k_cache, v_cache)
quant_videogen/kv_cache.py::ChunkedKVCache.store_quantized(...)
quant_videogen/kv_cache.py::ChunkedKVCache.read(...)
```

The first HQMQ adapter should accept Quant-VideoGen's existing BHSD contract:

```text
K, V: [batch, heads, sequence_tokens, head_dim]
return: same-shape fake-quantized K, V tensors
```

For BSHD integrations, the existing call sites already permute before
compression:

- Self-Forcing: `CausalInferencePipeline.quantize_kv_cache`
- LingBot: `WanI2VFast._quantize_kv_cache`

HY-WorldPlay already reads BHSD spans in `WanPipeline.quantize_kv_cache`.

### Key Transfer Risks

- HQMQ is an LLM KV-cache method; Quant-VideoGen denoisers have video
  transformer, diffusion, temporal, spatial, camera/action/image conditioning,
  and VAE/text-encoder boundaries that are not covered by HQMQ's README claims.
- Diffusion timestep dependence may make one random calibration-free codebook
  unstable across early noisy steps and late detail steps.
- HY-WorldPlay concatenates rope and prope channels into the cached K/V. HQMQ
  4-value chunks may cross semantic channel groups unless the adapter guards
  the grouping.
- Self-Forcing writes unroped K/V and applies RoPE after reading the full cache;
  HY-WorldPlay stores concatenated rope/prope K/V; LongCat decompresses before
  attention concat. These are different cache semantics.
- HQMQ requires `d_head % 4 == 0`; use a padded path if a target model violates
  this.
- Outlier rates may differ from LLMs. Med3x must be remeasured on video K/V
  rather than assumed.
- Fake-quant HQMQ can test quality but not measured memory savings.
- Packed HQMQ needs a different cache schema from QVG's current PRQ dicts.
- Video quality metrics must cover temporal artifacts, not just framewise
  similarity.

### Phase Plan

Phase 0: inventory and plan. Completed here. No Slurm jobs, dependency installs,
model downloads, or code edits were performed.

Phase 1: tensor-contract probe. Future and blocked on an implementation branch.
Add an isolated HQMQ adapter, feed synthetic and ideally dumped real BHSD K/V
tensors through `quantize_K`/`quantize_V`, and measure shape, dtype, finite
values, rel L2, cosine error, outlier rate, and quantization time.

Phase 2: first fake-quant video run. Future and blocked on Phase 1 plus model
availability. Run one prompt/seed/config through BF16, QVG INT4, QVG INT2, HQMQ
fake `s96_r4`, and HQMQ fake `s96_r4+Med3x`. Use paired settings and compare
videos, final latents if available, cache distortion, timing, and peak memory.

Phase 3: timestep/layer/outlier diagnosis. Future and can run in parallel with
Phase 2 analysis after real caches exist. Collect per-layer, per-head, K/V,
timestep, frame/chunk, norm histogram, max/median, Med3x outlier fraction,
attention-output MSE, and quality deltas.

Phase 4: statistical video benchmark. Future and blocked on a Phase 2 go. Run
at least 3 seeds per prompt/config on a balanced prompt suite, including
`assets/t2v.txt`, `assets/hyworld.png` with HY pose/prompt, and
`experiments/LingBot/examples/05`.

Phase 5: real packed HQMQ cache. Future and blocked on fake-quant evidence.
Implement actual index/radius/codebook/outlier storage and dense fallback decode
before claiming measured cache compression.

Phase 6: fused attention. Future and blocked on packed-cache correctness. Adapt
HQMQ fused attention only if dense decode works and measured memory/latency
justify the extra systems work.

Phase 7: multi-model expansion. Future and blocked on at least one successful
packed path. Expand across LongCat, Self-Forcing, HY-WorldPlay, and LingBot
with paired prompts, seeds, settings, metrics, and saved artifacts.

Parallel future work after Phase 1:

- One crewmate can own Self-Forcing fake-quant and cache dumps.
- One can own HY-WorldPlay rope/prope diagnostics.
- One can own LingBot `--use_chunked_kv` plus `--quant_factor` runs.
- One can own metric scripts and aggregation.
- One can own memory/latency accounting and packed-schema design.

### Decision Tree Summary

Phase 1 continue: HQMQ accepts all target K/V shapes, preserves dtype/shape, and
has finite output with distortion no worse than the current QVG INT4 envelope.
Pivot: layout/head-dim/channel grouping causes localized issues. Kill: HQMQ
cannot operate on target K/V without major model surgery or produces NaN/Inf.

Phase 2 continue: one full fake-quant HQMQ video completes and is within QVG
INT4 quality or better at similar theoretical bits. Pivot: output is usable but
worse than QVG; diagnose outliers, timesteps, channel grouping, or mixed
precision. Kill: video collapses while QVG INT4 remains usable.

Phase 3 continue: failures localize to layers, timesteps, heads, or channel
groups. Pivot: design video-specific calibration or mixed precision. Kill:
failures are broad and insensitive to Med3x, radius bits, secondary size, and
protected layers.

Phase 4 continue: paired prompt/seed statistics show reproducible quality.
Pivot: only one model family works. Kill: gains disappear under a broader prompt
suite.

Phase 5 continue: packed decode matches fake-quant and measured cache memory is
at least 3x smaller than BF16. Pivot: use dense decode and delay fused attention
if memory is good but latency is poor. Kill: metadata/outliers/decode overhead
remove the benefit.

### First Minimal Experiment

Smallest discriminating experiment for the implementation team:

1. Add `hqmq-fake` as a new quantization option behind
   `quant_videogen/compress.py::compress_kv_cache` in an implementation branch.
2. Use an isolated HQMQ source boundary: vendored copy, separate clean worktree,
   or optional import. Do not edit `/data/scratch-fast/kwen1/hqmq`.
3. Run a short Self-Forcing one-prompt, one-seed generation because the path is
   one-GPU and has `DUMP_KV_LEVEL` support.
4. Compare BF16, QVG INT4, QVG INT2, HQMQ fake `s96_r4`, and HQMQ fake
   `s96_r4+Med3x` with identical prompt, seed, frames, scheduler, and
   `local_attn_size`.
5. Measure K/V rel L2, attention-output rel L2 at early/mid/late denoising
   steps, final latent MSE if available, frame PSNR/SSIM/LPIPS versus BF16,
   visual temporal artifacts, quantization time, and peak CUDA memory.

Interpretation:

- Transfers mostly as-is: HQMQ fake completes, quality is within the QVG INT4
  envelope or better, and errors do not cluster by timestep or conditioning
  path.
- Needs video-specific adaptation: tensor tests pass but video quality trails
  QVG or errors concentrate by timestep/layer/channel group.
- Stop: tensor tests fail or the first video collapses while QVG INT4 remains
  usable.

### Reproducibility Notes

- Record Quant-VideoGen commit and HQMQ source commit or vendored-copy commit.
- Record prompt file, prompt index, image/action/pose assets, seed, resolution,
  frame count, chunk size, scheduler, guidance, precision, and quant config.
- Save raw logs, generated videos, metric JSON, cache-stat JSON, and environment
  snapshot.
- Separate theoretical bits from measured memory.
- Use paired runs for BF16/QVG/HQMQ comparisons.
- Use Slurm for future GPU jobs, but this planning task submitted no Slurm jobs.

## 2026-07-27 - Fake HQMQ Cache Adapter

Implemented an optional fake-HQMQ path behind Quant-VideoGen's existing
`compress_kv_cache` boundary. The new public method names are:

```text
hqmq-fake
hqmq-fake-med3x
```

Files changed for the adapter:

- `quant_videogen/hqmq_fake.py`: local pure-PyTorch fake adapter. It mirrors
  the read-only HQMQ fake-quant contract for BHSD K/V tensors, uses 4-value
  quaternion chunks, fixed 24-cell primary directions, deterministic per-layer
  per-head K/V secondary codebooks, scalar radius quantization, optional
  padding for non-multiple-of-4 head dimensions, and optional Med3x outlier
  restoration.
- `quant_videogen/compress.py`: dispatches `hqmq-fake` and
  `hqmq-fake-med3x` through `FakeHQMQCacheAdapter`.
- `quant_videogen/sim/quant/quantize_config.py`: records HQMQ fake defaults
  without changing existing QVG method defaults.
- Integration call sites now pass `layer_idx` into `compress_kv_cache` where
  they already have it: LongCat, Self-Forcing, HY-WorldPlay, and LingBot.
- `tests/test_hqmq_fake.py`: CPU-only smoke test for fake HQMQ tensor shape,
  dtype, determinism, finite diagnostics, and existing dispatch categories.

Validation:

```text
python3 -m py_compile quant_videogen/hqmq_fake.py quant_videogen/compress.py \
  quant_videogen/uncompress.py quant_videogen/sim/quant/quantize_config.py \
  tests/test_hqmq_fake.py
```

passed.

The runnable tensor smoke test could not run in the available shell because
both `/usr/bin/python3` and `/data/scratch-fast/kwen1/venvs/causal_forcing/bin/python`
raise `ModuleNotFoundError: No module named 'torch'`. No dependencies were
installed for this task.

Remaining transfer risks:

- This is fake quant only. It returns dense tensors and does not provide packed
  HQMQ memory or latency savings.
- HQMQ remains an LLM KV-cache method until video quality and attention-output
  distortion are measured on real video-generator caches.
- Video K/V may have timestep, layer, head, rope/prope channel, temporal, and
  conditioning-specific failure modes that language-model runs do not expose.
- Med3x outlier restoration is clearly separated as `hqmq-fake-med3x`; its
  outlier rate must be measured on video caches before treating it as useful.

Do not launch this from the implementation task, but the smallest next
fake-quant cache-dump experiment is:

```bash
export PYTHONPATH=experiments/Self-Forcing:.
KV_DUMP_DIR=results/selfforcing/hqmq-fake_cache_dump/kv_dumps \
DUMP_KV_LEVEL=1 \
torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path assets/t2v.txt \
  --output_folder results/selfforcing/hqmq-fake_cache_dump \
  --num_samples 1 \
  --num_output_frames 180 \
  --local_attn_size 180 \
  --use_ema \
  --save_with_index \
  --quant_type hqmq-fake \
  --cache_num_k_centroids 256 \
  --cache_num_v_centroids 256 \
  --kmeans_max_iters 2 \
  --quant_block_size 64 \
  --num_prq_stages 1
```

## 2026-07-27 - HQMQ Fake Tensor Smoke And Model-Run Gate

Task: run the cheapest useful validation of the fake HQMQ cache adapter in the
scratch checkout:

```text
/data/scratch-fast/kwen1/quant-videogen
```

Repo state at inspection:

```text
d088f42ebb4e828b5380fd5ae4327edfb7c66b99 Add fake HQMQ cache adapter
main...origin/main [ahead 2, behind 7]
```

No fetch, rebase, or merge was run.

Environment inventory:

- `/usr/bin/python3`: Python 3.10.12, no `torch`.
- `/data/scratch-fast/kwen1/venvs/causal_forcing/bin/python`: Python 3.10.12,
  no `torch`.
- `/data/scratch-fast/kwen1/eval_envs/vbench/bin/python`: Python 3.10.20,
  `torch 2.5.1+cu121`, `triton 3.1.0`, but Quant-VideoGen import fails in the
  eager Triton PRQ path with `ValueError: 'N' is not in list`.
- `/data/scratch-fast/kwen1/eval_envs/visionreward/bin/python`: Python 3.10.20,
  `torch 2.1.0+cu121`, `triton 2.1.0`, but Quant-VideoGen import fails with
  `ImportError: cannot import name 'libdevice' from 'triton.language.extra'`.
- `/data/scratch-fast/kwen1/hqmq/.venv/bin/python`: Python 3.11.15,
  `torch 2.10.0+cu128`, `triton 3.6.0`, but the unshimmed repo import lacks
  pure-Python helpers `termcolor` and `loguru`.

Required tensor smoke:

```bash
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_hqmq_fake.py
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/visionreward/bin/python tests/test_hqmq_fake.py
PYTHONPATH=. /data/scratch-fast/kwen1/hqmq/.venv/bin/python tests/test_hqmq_fake.py
```

All unshimmed attempts failed before reaching the adapter because of the
environment/import issues above.

The tensor smoke passed under the existing HQMQ venv after in-memory shims for
only `termcolor` and `loguru`; no files or packages were modified. Diagnostic
results from the same run:

- shape preserved: `True`.
- dtype preserved: `torch.float32` for K and V.
- finite outputs: `True`.
- base `hqmq-fake` relative L2: K `0.2251185178756714`, V
  `0.25770625472068787`.
- padded head dims for the non-multiple-of-4 test: `[12, 12]`.
- deterministic for fixed seed: K `True`, V `True`.
- `hqmq-fake-med3x` outlier rates: K `0.0833333358168602`, V `0.0`.

Static validation:

```bash
/usr/bin/python3 -m py_compile quant_videogen/hqmq_fake.py quant_videogen/compress.py \
  quant_videogen/uncompress.py quant_videogen/sim/quant/quantize_config.py \
  tests/test_hqmq_fake.py experiments/LongCat/run_long_t2v.py \
  experiments/LongCat/longcat_video/modules/longcat_video_dit.py \
  experiments/Self-Forcing/inference.py \
  experiments/Self-Forcing/pipeline/causal_inference.py \
  experiments/HY-WorldPlay/wan/generate.py \
  experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py \
  experiments/LingBot/generate_fast.py experiments/LingBot/wan/image2video_fast.py
git diff --check
```

Both commands passed.

Model-run gate:

- `assets/t2v.txt` exists.
- `ckpts/` is absent in this checkout.
- `ckpts/Self-Forcing/self_forcing_dmd.pt` was not found.
- A 17 GB Wan2.1-T2V-1.3B tree exists at
  `/data/scratch-fast/kwen1/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B`, but the
  Self-Forcing script expects `ckpts/Self-Forcing/Wan2.1-T2V-1.3B`.

Decision: do not stage or submit a Slurm model smoke. The next branch is to
provide or approve locating/linking the Self-Forcing DMD checkpoint and a
complete Self-Forcing runtime environment, then run the smallest staged
one-prompt fake-cache job through Slurm.

## 2026-08-04 - Project layout and plan path standardized

Firstmate moved the registered Quant-VideoGen project checkout under its canonical project home:

```text
/data/scratch-fast/kwen1/firstmate/projects/quant-videogen
```

The old scratch compatibility symlink was intentionally removed:

```text
/data/scratch-fast/kwen1/quant-videogen
```

The project plan was standardized to:

```text
plans/project-plan.md
```

The legacy `videogen_plan.md` path was intentionally removed in favor of the standardized plan path.
Future Quant-VideoGen work should read `plans/project-plan.md` before scoping experiments and update this `notebook.md` with meaningful changes, outcomes, failed ideas, and decisions.

## 2026-08-05 - Upstream HQMQ Cache-Audit Scaffold

Task: continue from official upstream `main`, recover the preserved fake-HQMQ
work, and add a CPU-safe Stage-1 cache-audit scaffold without launching GPU
jobs or duplicating the pending Med3x follow-up.

Isolation and refs:

- Worktree path and git top-level both resolved to
  `/afs/csail.mit.edu/u/k/kwen1/.treehouse/quant-videogen-d60f5d/5/quant-videogen`.
- Branch created: `fm/qvg-upstream-hqmq-cache-audit-scaffold`.
- `git branch -a` showed local preserved refs
  `fm/qvg-hqmq-fake-cache-adapter` and
  `fm/qvg-project-layout-standardization`.
- `git log --oneline --decorate --all --max-count=30` showed:
  - `d1d0991` at `fm/qvg-hqmq-fake-cache-adapter`.
  - `7565bfe` at `fm/qvg-project-layout-standardization`.
  - current upstream/default baseline `0601468`.

Recovered work:

- Cherry-picked the fake-HQMQ adapter and smoke-gate work from the preserved
  branch. The adapter commit conflicted only in `notebook.md` and
  `quant_videogen/sim/quant/quantize_config.py`; resolution kept current
  upstream chunk-quant policy fields and added the HQMQ fake defaults.
- Cherry-picked the layout-standardization commit. The direct cherry-pick
  added `plans/project-plan.md`; the notebook conflict was resolved by keeping
  the recovered fake-smoke gate and adding only the standardized plan-path note,
  not unrelated later Slurm operational notes from the layout branch history.

New Stage-1 scaffold:

- Added `tools/cache_audit.py`, a standalone JSON auditor for `torch.save`
  cache artifacts. It avoids importing `quant_videogen.__init__` because that
  path eagerly imports Triton-heavy compression code.
- Metadata load path uses `torch._subclasses.fake_tensor.FakeTensorMode` and
  tiny QVG pickle stubs for `ChunkedKVCache`, so existing cache dumps can be
  inspected without materializing tensor storage or requiring the full QVG
  runtime stack.
- The JSON schema records input path, file size, mtime, safely visible
  provenance keys, cache-object metadata, tensor path/shape/dtype/device/byte
  estimates, layer/K/V-role/cache-position classification, layout summaries,
  and explicit `metadata_only` or `unsupported_layout` markers.
- Guarded full-stats mode computes numeric percentiles, Med3x-style outlier
  fractions, per-head vector-norm summaries, sampled cache-position summaries,
  HQMQ-style four-tuple radius stats, and temporal deltas when the tensor layout
  is supported and byte limits allow CPU materialization.
- Added `tests/test_cache_audit.py`, which creates a tiny synthetic artifact at
  runtime and checks both the computed-stats schema and the metadata-only guard.

Real artifact evidence:

- The exact brief path
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_cache_frames27.pt`
  was absent.
- The dump exists at:
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt`.
- `stat` reported size `7858486627` bytes and mtime
  `2026-07-28 00:58:36.766679252 -0400`.
- Metadata-only audit command:

```bash
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/cache_audit.py \
  /data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt \
  --stats-mode metadata-only --compact \
  -o results/cache_audit_frames27_metadata.json
```

Summary from the generated JSON:

- Load mode: `fake_tensor_weights_only`; no trusted-pickle fallback was needed.
- Status: `metadata_only`.
- Tensor records: `1740`.
- `ChunkedKVCache` objects: `60`.
- Estimated tensor bytes: `7857930720`.
- First self-attention K chunk:
  `kv_cache[0].k.chunks[0]`, shape `[1, 1560, 12, 128]`, dtype
  `torch.bfloat16`, layout `BSHD`, cache-position token span `[0, 1560)`,
  estimated bytes `4792320`.
- First cache object showed `27` allocated chunks, `frame_seq_length=1560`,
  `num_heads=12`, `head_dim=128`, `layout=BSHD`, and all chunks in `BF16`
  state.
- Cross-attention layer-role summaries were also detected, giving `120`
  layer-role groups across self-attention and cross-attention K/V.

Validation:

```bash
python3 -m py_compile tools/cache_audit.py tests/test_cache_audit.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_cache_audit.py
```

Both passed. `/data/scratch-fast/kwen1/eval_envs/vbench/bin/python -m pytest
tests/test_cache_audit.py` could not run because that environment has no
`pytest` module, so the direct smoke entrypoint was used. No Slurm jobs, GPU
inference runs, or long experiments were launched. The existing Med3x follow-up
job `1295590` was not touched.

Next smallest experiment:

Run the auditor on a deliberately small subset or synthetic extraction that is
below the byte guards with `--stats-mode auto`, then inspect K/V asymmetry,
per-head radius outliers, and temporal-delta percentiles. If full-artifact
statistics are needed, first create a bounded CPU-side extraction of a few
layers, roles, and chunks from the existing dump rather than forcing full
materialization of the 7.86 GB artifact.

## 2026-08-05 - Bounded Real-Artifact Cache Statistics

Task: continue from the pushed cache-audit scaffold and produce bounded
real-artifact evidence for K/V asymmetry, layer/head outliers, HQMQ-style
four-tuple radius structure, and temporal/chunk drift. No Slurm jobs or GPU
inference were launched, and the separate Med3x Slurm job `1295590` was not
touched.

Isolation and base:

```bash
pwd -P
git rev-parse --show-toplevel
git checkout -b fm/qvg-bounded-cache-stats-analysis
git fetch kw7243 fm/qvg-upstream-hqmq-cache-audit-scaffold
git reset --hard FETCH_HEAD
git checkout -B fm/qvg-bounded-cache-stats-analysis
```

Both path checks resolved to:

```text
/afs/csail.mit.edu/u/k/kwen1/.treehouse/quant-videogen-d60f5d/5/quant-videogen
```

Real artifact:

```text
/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt
```

`stat` reported:

```text
7858486627 bytes, mtime 2026-07-28 00:58:36.766679252 -0400
```

Tooling changes:

- Extended `tools/cache_audit.py` with `--make-subset` to write a bounded
  tensor-only subset while preserving original source tensor paths.
- Added aggregate stats for K/V role, layer/role, chunk, per-head summaries,
  and selected chunk drift.
- Added a 4-tuple radius-only quantization error proxy for 2-bit and 4-bit
  radius quantization. This preserves tuple directions and therefore is not
  full HQMQ direction-codebook error.
- Added a smoke test for the subset schema in `tests/test_cache_audit.py`.

Quick metadata command used only for read-only selection planning:

```bash
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/cache_audit.py \
  /data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt \
  --stats-mode metadata-only --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_metadata.json
```

Bounded subset command, run from staged repo:

```bash
/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh -- \
  /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/cache_audit.py \
  /data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt \
  --make-subset /data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset.pt \
  --subset-layers 0,14,29 \
  --subset-chunks 0,13,26 \
  --subset-roles key,value \
  --max-subset-bytes 134217728 \
  --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset_manifest.json
```

Staged repo:

```text
/afs/csail.mit.edu/u/k/kwen1/.treehouse/quant-videogen-d60f5d/5/quant-videogen/staging/20260805_015621-24b6fd-378b932-code
```

Subset:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset.pt
```

Subset manifest summary:

- `selected_tensor_count`: `18`.
- Selection: self-attention layers `0,14,29`, roles `key,value`, chunks
  `0,13,26`.
- `selected_estimated_bytes`: `86261760`.
- Subset file size: `86271704` bytes.
- `max_subset_bytes`: `134217728`.
- Full-artifact stats load mode for extraction: `cpu_weights_only_mmap`.

Bounded stats command, run from staged repo:

```bash
/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh -- \
  /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/cache_audit.py \
  /data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset.pt \
  --stats-mode auto \
  --max-full-load-bytes 134217728 \
  --max-total-stat-bytes 134217728 \
  --max-tensor-stat-bytes 16777216 \
  --max-time-points 32 \
  --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_bounded_stats.json
```

Staged repo:

```text
/afs/csail.mit.edu/u/k/kwen1/.treehouse/quant-videogen-d60f5d/5/quant-videogen/staging/20260805_020448-70b895-378b932-code
```

Committed result copies:

```text
analysis/cache_audit_frames27_subset_manifest.json
analysis/cache_audit_frames27_bounded_stats.json
analysis/cache_audit_frames27_bounded_summary.md
```

Private Firstmate result copies:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset_manifest.json
/data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_bounded_stats.json
```

Main results:

- `--stats-mode auto` computed stats on all 18 selected tensors under the byte
  guards.
- K versus V asymmetry:
  - K vector-norm p95 mean `48.255` versus V `18.260`.
  - K tuple4 radius outlier fraction mean `0.1205` versus V `0.0047`.
  - K radius-only 4-bit relative L2 proxy mean `0.0189` versus V `0.0123`.
  - V temporal relative-delta p95 mean `1.3969` versus K `0.8765`.
- Layer/head outliers:
  - Layer 29 K vector-norm p95 mean `102.754`, with tensor-level
    vector-norm outlier fraction `0.0833`.
  - Layer 29 K head 9 dominates per-head norm p95: mean `104.263`, max
    `106.346`; the next sampled head is about `27`.
  - Layer 0 K has ordinary vector-norm p95 but high tuple4 radius outlier rate:
    mean `0.2290`, max-to-median radius mean `28.64`.
- Four-tuple radius structure:
  - Head dim is `128`, so there are 32 four-tuples per token.
  - K radius p95 mean `5.659`; V radius p95 mean `3.764`.
  - K 2-bit radius-only proxy mean `0.2144`; V `0.1656`.
- Temporal/chunk drift:
  - Chunks 0 and 13 have radius-only 4-bit proxy mean near `0.00145`.
  - Chunk 26 jumps to `0.04388` while norm/radius magnitude and within-chunk
    temporal relative deltas stay roughly stable.
  - This is consistent with a final-chunk or refresh-state difference, but it
    needs a targeted adjacent-chunk follow-up before treating it as mechanism.

Failed or deliberately skipped ideas:

- Did not run `--stats-mode force` on the 7.86 GB artifact.
- Did not materialize all tensors from the full dump.
- Did not submit Slurm jobs or launch GPU inference.
- Did not claim direct quantization error because this artifact lacks a paired
  BF16 pre-quant cache. The reported error is only a radius-preserving proxy.

Next smallest experiment:

Expand the subset only around the observed structure: layers 27-29, heads 0 and
9 for K, and chunks 24-26 plus one adjacent earlier chunk. Add a paired BF16 or
pre/post fake-HQMQ cache dump so direct cache quantization error and
attention-output error can be measured before any video-quality or novelty
claim.

## 2026-08-05 - Attention-Error Probe Scaffold

Task: build a targeted offline scaffold to test whether the bounded real-cache
geometry predicts functional attention distortion. Work was done on branch
`fm/qvg-attention-error-probe-scaffold` from cache-audit base `118e8ab`.

HYPOTHESIS: K/V asymmetry, tuple4-radius structure, layer-29 K head-9 outlier
behavior, and late-chunk drift may predict attention-level functional error.

WHY THIS EXPERIMENT MATTERS: cache MSE and radius-only proxies are not enough
to select a quantizer; attention logits, probabilities, output error, and
top-token overlap are closer to model behavior.

MANIPULATED VARIABLE: synthetic K-only and V-only cache perturbations.

CONTROLLED VARIABLES: query tensors, current K/V tensors, token ordering,
attention scale, and reference cached K/V tensors.

PRIMARY METRIC: attention-output relative L2 error.

SECONDARY METRICS: attention-logit relative L2, probability JS/KL/TV
divergence, and top-attended-token overlap.

EXPECTED RESULT: K perturbations change logits/probabilities/top tokens; V
perturbations leave logits/probabilities unchanged but alter attention output.

FALSIFYING RESULT: metrics fail to separate controlled K-only and V-only
perturbations.

DECISION RULE: proceed to real Self-Forcing capture only if synthetic tests
detect both perturbation modes and the bounded subset smoke makes the real
blockers explicit.

Implementation:

- Added `tools/attention_error_probe.py`, a standalone CPU-friendly offline
  probe. It accepts supplied query/current-K/current-V tensors plus cached K/V
  reference tensors and reconstructed cache variants, normalizes BHSD/BSHD
  layouts, concatenates cached tokens before current tokens, and reports:
  attention-logit error, attention-probability divergence, attention-output
  relative error, and top-attended-token overlap.
- Added `tools/attention_probe_capture.py`, an opt-in bounded capture-schema
  helper for future Self-Forcing hooks. It is disabled by default, reads
  `QVG_ATTENTION_PROBE_*` env vars, supports layer/head/chunk selection,
  enforces `MAX_FILE_BYTES`, normalizes BSHD Self-Forcing tensors to BHSD, and
  splits full attention K/V spans into cached and current tensors.
- Added `tests/test_attention_error_probe.py` synthetic coverage for K-only
  and V-only perturbations, probe-artifact loading, cache-audit smoke, and the
  Self-Forcing capture schema.
- Added `analysis/cache_audit_frames27_attention_probe_smoke.json`, a bounded
  metadata/subset smoke derived from the existing bounded stats JSON.

Bounded smoke command:

```bash
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/attention_error_probe.py \
  --cache-audit-summary analysis/cache_audit_frames27_bounded_stats.json \
  --compact \
  -o analysis/cache_audit_frames27_attention_probe_smoke.json
```

Smoke result:

- The existing bounded subset remains usable for cache-only metadata, K/V
  norm/radius summaries, per-head summaries, and selected chunk drift.
- It cannot compute real attention-logit error, probability divergence,
  attention-output relative error, or top-attended-token overlap because it
  lacks query, current-K, current-V, attention-mask, and paired
  BF16/reference-versus-reconstructed cache tensors.
- The smoke read the committed bounded stats JSON only. I also `stat`-checked
  the bounded subset and full dump paths; I did not broad-load the 7.86 GB full
  dump or materialize the bounded PT subset in this branch.

Recorded byte guards from the smoke:

- Bounded subset path:
  `/data/scratch-fast/kwen1/firstmate/data/qvg-bounded-cache-stats-analysis/cache_audit_frames27_subset.pt`.
- Bounded subset size: `86271704` bytes.
- Source full dump path:
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946/kv_dumps/kv_cache_frames27.pt`.
- Source full dump size: `7858486627` bytes.
- Subset creation guard: `max_subset_bytes=134217728`,
  `selected_estimated_bytes=86261760`.
- Stats guards: `max_full_load_bytes=134217728`,
  `max_total_stat_bytes=134217728`, `max_tensor_stat_bytes=16777216`.

Validation:

```bash
python3 -m py_compile tools/cache_audit.py tools/attention_error_probe.py \
  tools/attention_probe_capture.py tests/test_cache_audit.py \
  tests/test_attention_error_probe.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_attention_error_probe.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_cache_audit.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/attention_error_probe.py \
  --cache-audit-summary analysis/cache_audit_frames27_bounded_stats.json \
  --compact \
  -o analysis/cache_audit_frames27_attention_probe_smoke.json
git diff --check
git diff --cached --check
```

All validation commands passed.

Novelty guard:

- Do not claim generic video-generation KV-cache quantization as novel.
- Do not claim head-aware or static-vs-dynamic video cache compression as
  novel.
- Do not claim Jensen-bias correction for quantized video-diffusion keys as
  novel.
- This branch only scaffolds a narrower test: whether the observed
  four-tuple/radius geometry, K/V asymmetry, layer-29 K head-9 outlier, and
  late-chunk drift predict functional attention distortion in
  Quant-VideoGen/Self-Forcing caches.

No Slurm jobs, GPU inference runs, or modifications to job `1295590` were
performed.

Next experiment:

Add the opt-in capture helper at
`experiments/Self-Forcing/wan/modules/causal_model.py::CausalWanSelfAttention.attn_kv_cache_prerope`
just before `attention(roped_query, k_input, v_all)`, pass `layer_idx` from the
block loop, and run a bounded BF16/reference plus quantized/reconstructed
paired capture around layers 27-29, heads 0 and 9, and chunks 24-26. Then feed
the resulting probe artifacts into `tools/attention_error_probe.py`.
