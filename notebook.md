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

## 2026-08-05 - Self-Forcing Attention Capture Hook

Task: hook the opt-in attention-probe capture scaffold into the real
Self-Forcing cached-attention path without launching GPU inference or touching
the separate Med3x Slurm job `1295590`.

HYPOTHESIS: the locally observed Self-Forcing/QVG cache geometry, especially
K/V asymmetry, layer-29 K head-9 outliers, late-chunk drift, and HQMQ/Med3x
radius structure, may predict functional attention distortion for the actual
cached-versus-current attention call.

WHY THIS EXPERIMENT MATTERS: a bounded real attention-call capture is the
smallest missing input needed before `tools/attention_error_probe.py` can test
logit, probability, top-token, and output distortion instead of only cache
statistics.

MANIPULATED VARIABLE: no model variable was manipulated in this branch. The
code adds an opt-in capture surface for later BF16 and quantized/reconstructed
paired runs.

CONTROLLED VARIABLES: default inference remains unchanged when
`QVG_ATTENTION_PROBE_ENABLED` is unset. Enabled capture defaults to layers
`27,28,29`, heads `0,9`, and chunks `24,25,26`, with a 64 MiB per-file byte
guard.

PRIMARY METRIC: future attention-output relative L2 from
`tools/attention_error_probe.py`.

SECONDARY METRICS: future attention-logit relative L2, probability divergence,
and top-attended-token overlap.

EXPECTED RESULT: if the cache geometry is functionally meaningful, bounded
captures around layers 27-29 and chunks 24-26 should show larger attention
distortion for the sensitive K/head/chunk regions than for nearby controls.

FALSIFYING RESULT: paired captures show no meaningful attention distortion
despite the observed K/V geometry, or distortion does not align with layer/head
or late-chunk structure.

DECISION RULE: only use this evidence to motivate a narrow attention-aware
follow-up if paired captures make the functional metrics computable and the
effect localizes to the predicted regions.

Implementation:

- Added a lazy Self-Forcing hook immediately before
  `attention(roped_query, k_input, v_all)` in
  `experiments/Self-Forcing/wan/modules/causal_model.py::CausalWanSelfAttention.attn_kv_cache_prerope`.
- Passed stable `layer_idx=block_index` from
  `CausalWanModel._forward_inference` through `CausalWanAttentionBlock.forward`
  and `CausalWanSelfAttention.forward` into the hook. The only fallback is
  `layer_idx=-1` for direct module calls outside the model block loop; default
  layer selection skips that fallback.
- Kept capture opt-in through environment variables. With
  `QVG_ATTENTION_PROBE_ENABLED` unset, the capture module is not imported and
  no files are written.
- Updated `tools/attention_probe_capture.py` so enabled capture defaults to the
  bounded first target: layers `27,28,29`, heads `0,9`, chunks `24,25,26`.
  `LAYERS=all`, `HEADS=all`, or `CHUNKS=all` remain available only for
  deliberate debug captures.
- Changed bounded chunk selection so cached and current tensors contain only
  selected chunk spans from the full attention call. Metadata records full
  token counts, selected token counts, selected heads/chunks, cached/current
  chunk indices, dropped prefix tokens, and cache chunk states.
- Captured tensors are the roped query, roped all-key span split into cached
  and current K, V span split into cached and current V, plus optional unroped
  query/key diagnostics. If cache chunks are quantized spans, metadata records
  that `kv_cache.read()` returned reconstructed tensors for those chunks.
- Added CPU coverage for env parsing, default bounded selection, disabled and
  mismatch skip paths, selected chunk slicing, invalid head selection,
  `MAX_CACHE_TOKENS` truncation, and both byte-guard branches.

Validation commands:

```bash
python3 -m py_compile tools/cache_audit.py tools/attention_error_probe.py \
  tools/attention_probe_capture.py tests/test_cache_audit.py \
  tests/test_attention_error_probe.py \
  experiments/Self-Forcing/wan/modules/causal_model.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_attention_error_probe.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_cache_audit.py
mkdir -p /data/scratch-fast/kwen1/firstmate/data/qvg-self-forcing-attention-capture-hook
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tools/attention_error_probe.py \
  --cache-audit-summary analysis/cache_audit_frames27_bounded_stats.json \
  --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-self-forcing-attention-capture-hook/attention_probe_smoke.json
git diff --check
```

All commands passed. The smoke JSON is:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-self-forcing-attention-capture-hook/attention_probe_smoke.json
size: 2006 bytes
```

No runtime attention capture artifact was produced in this branch, so no model
inference and no Slurm job were run. Job `1295590` was not inspected, canceled,
modified, or duplicated. Since no capture artifact exists yet, functional
attention metrics are still not computable.

Byte guards:

- Capture default `QVG_ATTENTION_PROBE_MAX_FILE_BYTES`: `67108864`.
- Cache-audit smoke reused the bounded stats JSON only and did not load the
  7.86 GB cache dump.
- Existing bounded stats guards remain `max_full_load_bytes=134217728`,
  `max_total_stat_bytes=134217728`, and `max_tensor_stat_bytes=16777216`.

Failed or deliberately skipped ideas:

- Did not add global mutable layer counters; the hook uses the local block-loop
  index.
- Did not capture all cache tokens by default. The enabled default is the
  bounded layers/heads/chunks target.
- Did not attempt to synthesize paired BF16-versus-quantized variants inside
  one attention call when the reference cache is not present. Quantized or
  reconstructed cache semantics are recorded in metadata, and paired functional
  comparison still needs a matched BF16/reference capture.
- Did not submit a Slurm capture because the hook and guards were validated
  with CPU tests, and the next GPU run should be a distinct bounded capture
  job only after operator approval.

Novelty guard:

- Do not claim generic autoregressive video-generation KV-cache quantization as
  novel.
- Do not claim video-diffusion key quantization bias correction as novel.
- Do not claim salience-based video KV-cache policies as novel.
- Do not claim broad frame-wise sensitivity or outlier-aware ARVD
  quantization as novel.
- Do not claim VideoLLM sub-2-bit mixed-precision KV-cache quantization as
  novel.
- This branch supports only a narrow infrastructure claim: the project can now
  capture bounded Self-Forcing attention-call tensors needed to test whether
  the already observed local cache geometry predicts functional attention
  distortion.

Next smallest experiment:

Run one staged, distinct, bounded Self-Forcing attention-capture job with:

```bash
QVG_ATTENTION_PROBE_ENABLED=1
QVG_ATTENTION_PROBE_DIR=/data/scratch-fast/kwen1/firstmate/data/qvg-self-forcing-attention-capture-hook/captures
QVG_ATTENTION_PROBE_LAYERS=27,28,29
QVG_ATTENTION_PROBE_HEADS=0,9
QVG_ATTENTION_PROBE_CHUNKS=24,25,26
QVG_ATTENTION_PROBE_MAX_FILE_BYTES=67108864
QVG_ATTENTION_PROBE_VARIANT=bf16_or_quantized_run_label
```

Use the research reproducibility staging helper, do Slurm preflight first, and
do not duplicate the pending Med3x smoke. After the bounded artifacts exist and
remain under the byte guard, run `tools/attention_error_probe.py --input` on the
paired probe artifact or on a small merged BF16/reference-plus-variant artifact.

## 2026-08-05 - Bounded Attention-Capture Slurm Setup

Task: prepare one staged Self-Forcing attention-capture Slurm submission on
branch `fm/qvg-attention-capture-slurm-run`, starting from
`kw7243/fm/qvg-self-forcing-attention-capture-hook`.

HYPOTHESIS: a bounded real Self-Forcing attention-call capture around layers
27-29, heads 0 and 9, and chunks 24-26 is the next input needed to test whether
the observed QVG/Self-Forcing cache geometry predicts functional attention
distortion.

RUN CHOICE: default the submitted run to BF16/reference semantics
(`QUANT_TYPE=none`, `QVG_ATTENTION_PROBE_VARIANT=bf16_reference`) rather than
fake-quantizing the live cache. This preserves reference Q/K/V tensors for
offline variants or a later paired merge. The run remains bounded to one prompt,
27 frames, local attention size 27, and no full KV dump (`DUMP_KV_LEVEL=0`).

Output root:

```text
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/
```

Added submission files:

- `scripts/Self-Forcing/attention_capture_bf16.sbatch`: H200 Slurm job with
  account/QoS `vision-torralba-urops-meng`, `gpu:1`, 8 CPUs, 96G host memory,
  one-hour limit, task-local logs, task-local captures, scratch-backed caches,
  staged-copy checkpoint symlinks, GPU/import preflight, and a post-run capture
  manifest.
- `scripts/Self-Forcing/submit_attention_capture.sh`: staged-only wrapper that
  refuses to run outside `stage_and_run.sh`, checks scratch model/checkpoint
  paths, runs Slurm association/partition and `sbatch --test-only` preflight,
  submits exactly one job, and writes task-local submission metadata.

Required capture bounds enforced by the Slurm job:

```text
QVG_ATTENTION_PROBE_ENABLED=1
QVG_ATTENTION_PROBE_LAYERS=27,28,29
QVG_ATTENTION_PROBE_HEADS=0,9
QVG_ATTENTION_PROBE_CHUNKS=24,25,26
QVG_ATTENTION_PROBE_MAX_FILE_BYTES=67108864
```

Source-side validation before staging:

```bash
bash -n scripts/Self-Forcing/attention_capture_bf16.sbatch \
  scripts/Self-Forcing/submit_attention_capture.sh
python3 -m py_compile tools/attention_probe_capture.py \
  tools/attention_error_probe.py \
  experiments/Self-Forcing/wan/modules/causal_model.py \
  tests/test_attention_error_probe.py
git diff --check
git diff --cached --check
```

All passed. Slurm preflight and submission are intentionally left to the staged
copy, per the reproducibility rule.

Staged submission command:

```bash
/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh \
  --staging-parent /data/scratch-fast/kwen1/qvg-staging \
  -- bash scripts/Self-Forcing/submit_attention_capture.sh
```

Staged copy:

```text
/data/scratch-fast/kwen1/qvg-staging/20260805_222232-7132ad-2359491-code
```

Staged metadata:

```text
git_commit_full: 235949177507a9eca84ecb0dcb85f211cb99418a
git_status_short: clean
command: bash scripts/Self-Forcing/submit_attention_capture.sh
```

Slurm preflight from the staged copy passed:

- Association includes `kwen1|vision-torralba-urops-meng` with QoS
  `vision-torralba-urops-meng`.
- `vision-torralba-h200` is `UP`, node `torralba-h200-1`, GRES
  `gpu:h200:8`, memory `2063711M`.
- Partition policy allows account/QoS `vision-torralba-urops-meng`.
- Requested job resources: account/QoS `vision-torralba-urops-meng`,
  partition `vision-torralba-h200`, `gres=gpu:1`, 8 CPUs, 96G host memory,
  one-hour limit.
- `sbatch --test-only` accepted the request and predicted a possible start at
  `2026-08-08T18:10:09`; Slurm printed that message on stderr, so the wrapper
  was updated after submission to capture stderr into future preflight files.

Submitted Slurm job:

```text
job id: 1316140
state at inspection: PENDING
reason: Priority
StartTime: Unknown
dependency: none
ReqTRES: cpu=8,mem=96G,node=1,billing=5800,gres/gpu=1
WorkDir: /data/scratch-fast/kwen1/qvg-staging/20260805_222232-7132ad-2359491-code
Command: scripts/Self-Forcing/attention_capture_bf16.sbatch
```

Task-local paths:

```text
submission metadata:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/submission_1316140.json

stdout:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/logs/qvg_sf_attncap-1316140.out

stderr:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/logs/qvg_sf_attncap-1316140.err

video/output directory:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/outputs/bf16_attncap_27f_prompt1_1316140

attention capture directory:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/captures/bf16_attncap_27f_prompt1_1316140

capture manifest after terminal success:
/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/capture_manifests/bf16_attncap_27f_prompt1_1316140.json
```

Job `1295590` was not modified, canceled, or used as a dependency. This
submission is a distinct BF16/reference attention-capture job, not the pending
Med3x smoke continuation.

## 2026-08-06 - Attention-Capture Postprocess Readiness Check

Task: build the CPU-only post-capture readiness scaffold for pending
Self-Forcing attention-capture Slurm job `1316140`, starting from
`kw7243/fm/qvg-attention-capture-slurm-run` on branch
`fm/qvg-attention-capture-postprocess-check`.

HYPOTHESIS: a metadata-only readiness command can make the pending or completed
state of job `1316140` artifacts auditable without querying Slurm, loading
large tensor payloads, or conflating reference-only captures with quantized
functional distortion.

WHY THIS EXPERIMENT MATTERS: the next scientific question is narrow: whether
the observed Self-Forcing/QVG cache geometry predicts functional attention
distortion in actual captured attention calls. A readiness gate prevents absent
or partial Slurm artifacts from becoming ambiguous failures.

MANIPULATED VARIABLE: synthetic fixture artifact state for the scaffold tests
and the live presence or absence of task-local job `1316140` files.

CONTROLLED VARIABLES: expected job id, BF16/reference `quant_type=none`,
layers `27,28,29`, heads `0,9`, chunks `24,25,26`, and the 64 MiB per-capture
file guard.

PRIMARY METRIC: machine-readable `ready`/`not_ready` state from
`tools/attention_capture_readiness.py`.

SECONDARY METRICS: validated submission metadata, log paths, output directory,
capture directory, capture manifest schema, capture file sizes, layer/head/chunk
coverage, and generated next `tools/attention_error_probe.py` command.

EXPECTED RESULT: pending job artifacts report `not_ready` with explicit missing
paths; completed reference-only captures emit an identity-reference attention
self-check command; variant-bearing captures emit a distortion-probe command.

FALSIFYING RESULT: missing or pending artifacts crash the checker, or completed
captures can produce a command that broad-loads tensors before byte guards and
manifest checks pass.

DECISION RULE: use this scaffold as the CPU gate after job `1316140` completes;
interpret identity-reference output only as an offline attention-path sanity
check, not as quantization distortion evidence.

Implementation:

- Added `tools/attention_capture_readiness.py`, a stdlib-only metadata checker
  for job `1316140`. It reads task-local submission JSON, expected stdout and
  stderr logs, output and capture directories, the capture manifest, selected
  layers/heads/chunks, and capture file byte guards. It does not query or
  control Slurm and does not load `.pt` tensors.
- Extended `tools/attention_error_probe.py` with opt-in
  `--identity-variant-if-missing`. This lets a completed BF16/reference-only
  capture run the smallest offline attention self-check. It is explicitly not
  a quantized-cache distortion measurement.
- Added `tests/test_attention_capture_readiness.py` with synthetic pending,
  complete reference-only, complete variant-bearing, oversized capture, and
  selection-mismatch fixtures.
- Added synthetic coverage in `tests/test_attention_error_probe.py` for the
  identity-reference mode.

Live job `1316140` readiness command:

```bash
python3 tools/attention_capture_readiness.py --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-postprocess-check/readiness_1316140.json
```

Live result:

- `status=not_ready`.
- Submission metadata exists and matches job `1316140`.
- Attention probe metadata matches expected layers `27,28,29`, heads `0,9`,
  chunks `24,25,26`, and `max_file_bytes=67108864`.
- Preflight files from the staged submission are present.
- Missing pending artifacts: stdout log, stderr log, output directory, capture
  directory, and capture manifest.
- No `tools/attention_error_probe.py` command is emitted until capture artifacts
  are present and pass manifest/byte-guard checks.

Validation:

```bash
python3 -m py_compile tools/attention_capture_readiness.py \
  tools/attention_error_probe.py tests/test_attention_capture_readiness.py \
  tests/test_attention_error_probe.py
python3 tests/test_attention_capture_readiness.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_attention_error_probe.py
python3 tools/attention_capture_readiness.py --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-postprocess-check/readiness_1316140.json
git diff --check
```

All commands passed.

Novelty guard:

- No claim is made about video quality, memory reduction, latency, or generic
  autoregressive video-generation KV-cache quantization.
- The scaffold supports only readiness for testing whether the observed local
  cache geometry predicts functional attention distortion in actual
  Self-Forcing attention calls.

## 2026-08-07 - Focused Attention-Error Decision Gate

Task: build the next CPU-only decision gate for the pending attention-capture
job `1316140`, focused on the cache-anomaly result from the staged audit.

Branch:

```text
fm/qvg-focused-attention-error-gate
```

Scope:

- Focused decision target: layer `29`, target head `9`, control head `0`,
  cached chunk `23`, current chunk `24`.
- K-only and V-only variants must be identifiable separately before the gate
  emits separated attention-error probe commands.
- Missing logs, manifest, capture directory, capture files, byte-guard failure,
  missing focused record, missing chunk transition, or reference-only captures
  remain explicit `not_ready` or `incomplete` states with `decision_state` set
  to a no-claim mode.

Implementation:

- Extended `tools/attention_capture_readiness.py` with a
  `focused_attention_error_gate` block in the existing readiness JSON.
- Added focused CLI options:
  `--focused-layer`, `--focused-target-head`, `--focused-control-head`,
  `--focused-cached-chunk`, and `--focused-current-chunk`.
- The focused gate now reports support for the target/control heads, cached and
  current chunk indices, variant-name availability, K-only variant name, V-only
  variant name, and the smallest safe probe commands when supported.
- Reference-only exact-transition captures emit only an identity offline
  attention-path check and remain `incomplete`; they do not become distortion
  evidence.
- Extended `tools/attention_error_probe.py` with repeatable `--variant-name`,
  so K-only and V-only variants can be probed into separate JSON outputs.
- Extended the Self-Forcing capture manifest summarizer to include
  `variant_names` when a capture file contains variants.
- Added focused CPU fixtures for absent artifacts, exact-transition
  reference-only captures, exact-transition separated K/V variants, and the
  current chunk-23-missing no-claim state.

Live job `1316140` focused readiness command:

```bash
python3 tools/attention_capture_readiness.py --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-focused-attention-error-gate/readiness_1316140_focused.json
```

Live result:

- Top-level readiness: `status=not_ready`, `ready=false`.
- Missing artifacts: stdout log, stderr log, output directory, capture
  directory, and capture manifest.
- Focused gate: `status=not_ready`, `decision_state=no_claim`.
- Focused reason: `capture_artifacts_not_ready_or_invalid`.
- No focused probe command is emitted while these artifacts are absent.

Validation:

```bash
python3 -m py_compile tools/attention_capture_readiness.py \
  tools/attention_error_probe.py tests/test_attention_capture_readiness.py \
  tests/test_attention_error_probe.py
python3 tests/test_attention_capture_readiness.py
/data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_attention_error_probe.py
python3 tools/attention_capture_readiness.py --compact \
  -o /data/scratch-fast/kwen1/firstmate/data/qvg-focused-attention-error-gate/readiness_1316140_focused.json
git diff --check
```

All commands passed.

Guard:

- No Slurm jobs were submitted, canceled, inspected, or modified.
- No GPU inference, model loading, real-video metrics, downloads, checkpoint
  changes, PR, main merge, or claims about video quality, memory reduction,
  latency speedup, systems gains, or novelty were performed.
