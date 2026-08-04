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

## 2026-07-27 - Self-Forcing Assets And HQMQ Fake Model Smoke

Task: prepare the smallest Self-Forcing runtime needed to smoke-test
`hqmq-fake` against a real model path, using scratch for large assets and this
treehouse worktree for code/notebook edits.

Expected Self-Forcing paths from the repo scripts:

- `ckpts/Self-Forcing/Wan2.1-T2V-1.3B`
- `ckpts/Self-Forcing/self_forcing_dmd.pt`

Assets:

- Reused Wan base model from
  `/data/scratch-fast/kwen1/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B`
  (`17,573,839,938` bytes), corresponding to
  `https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B`.
- Downloaded DMD checkpoint to
  `/data/scratch-fast/kwen1/qvg-self-forcing-assets/gdhe17-Self-Forcing/checkpoints/self_forcing_dmd.pt`
  (`5,676,252,553` bytes), from
  `https://huggingface.co/gdhe17/Self-Forcing`, file
  `checkpoints/self_forcing_dmd.pt`.
- Created treehouse-local symlinks under `ckpts/Self-Forcing/` to those scratch
  assets, so staging does not duplicate large files.

Environment:

- Installed missing lightweight dependency `termcolor` into the existing
  scratch environment
  `/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing`.
- CPU-side import check passed through `quant_videogen.compress`; importing
  `utils.wan_wrapper` reaches the expected GPU boundary on the login allocation:
  `No CUDA GPUs are available`.
- `tests/test_hqmq_fake.py` passed with both checked torch environments:
  `/data/scratch-fast/kwen1/eval_envs/vbench/bin/python` and
  `/data/scratch-fast/kwen1/eval_envs/visionreward/bin/python`.
- `py_compile`, `bash -n scripts/Self-Forcing/smoke_hqmq_fake.sbatch`, and
  `git diff --check` passed.

Staging and Slurm:

- Failed first submit from staged path
  `/data/scratch-fast/kwen1/qvg-staging/20260727_175752-d9296b-0601468-code`
  because Slurm rejected the implicit QoS for `vision-torralba-h200`.
- Added explicit `#SBATCH --qos=vision-torralba-urops-meng`.
- Submitted corrected H200-only job `1210884` from staged path
  `/data/scratch-fast/kwen1/qvg-staging/20260727_175919-6cf6b5-0601468-code`.
- Current status at submission check: `PENDING`, reason `(Priority)`.

Smoke command in the Slurm script runs one prompt with `--num_output_frames 27`,
`--local_attn_size 27`, and `--quant_type hqmq-fake`. This is the smallest
configuration I found that should reach Self-Forcing cache quantization after
the 8-chunk warmup while staying much smaller than the 180-frame script.

## 2026-07-27 - Tiled HQMQ Fake Similarity Fix

Task: fix the H200 smoke OOM in `FakeHQMQCacheAdapter._quantize_padded()` while
preserving exact fake-HQMQ outputs versus the previous dense implementation on
small tensors.

Result:

- Added `hqmq_max_sim_elements` to `QuantizeConfig`, defaulting to
  `268,435,456` similarity elements.
- Replaced the all-at-once `sims = einsum(... full codebook ...)` path with
  sequence-token and codeword tiling in `_nearest_joint_directions()`.
- Preserved dense argmax tie behavior by scanning codewords in order and only
  replacing the current best score on strict improvement.
- Recorded `sim_tile_tokens`, `sim_codeword_tile`, `sim_tile_count`,
  `max_sim_tile_elements`, and `dense_sim_elements` in fake-HQMQ diagnostics.

Validation before Slurm:

```bash
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_hqmq_fake.py
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/visionreward/bin/python tests/test_hqmq_fake.py
python3 -m py_compile quant_videogen/hqmq_fake.py \
  quant_videogen/sim/quant/quantize_config.py tests/test_hqmq_fake.py \
  experiments/Self-Forcing/inference.py \
  experiments/Self-Forcing/pipeline/causal_inference.py
bash -n scripts/Self-Forcing/smoke_hqmq_fake.sbatch
git diff --check
```

All passed.

Fresh H200-only smoke submission:

- Fix commit staged: `6de8c1cbc421dee81ef7a1e5c2e9598baf646234`.
- Staged path:
  `/data/scratch-fast/kwen1/qvg-staging/20260727_182815-ee78fa-6de8c1c-code`.
- Slurm job id: `1211191`.
- Partition/QoS: `vision-torralba-h200` /
  `vision-torralba-urops-meng`.
- Status at notebook update: `PENDING`, reason `(Priority)`.
- Output scout will be
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1211191`.

Terminal result for job `1211191`:

- State: `FAILED`, exit code `1:0`, elapsed `00:01:43`, node
  `torralba-h200-1`.
- The run reached `hqmq-fake` after the tiling fix. It printed
  `Layer 00 | K Rel L2: 0.1289 | V Rel L2: 0.1270`.
- New failure: `ChunkedKVCache.write()` asserted
  `end_index - start_index (37440) != data tokens (12)`.
- Root cause: Self-Forcing reads BSHD cache tensors, permutes to BHSD for
  `compress_kv_cache()`, receives dense fake-HQMQ tensors in BHSD, then stores
  them into a BSHD `ChunkedKVCache` without permuting back to BSHD.
- Output scout
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1211191`
  contains only an empty `kv_dumps/` directory and no valid smoke artifacts.
- Next smallest fix: in Self-Forcing `quantize_kv_cache()`, permute dense tensor
  fake-quant outputs back to cache-native BSHD before `store_quantized()`, while
  leaving dict/real-quantized outputs in the packed BHSD schema.

## 2026-07-28 - Self-Forcing HQMQ Dense Layout Handoff Fix

Task: fix the BSHD/BHSD handoff that made H200 smoke job `1211191` fail after
the first layer fake-HQMQ diagnostics.

Result:

- Confirmed the diagnosis from the failed smoke logs: Self-Forcing reads native
  BSHD cache spans, permutes them to BHSD for `compress_kv_cache()`, receives
  dense fake-HQMQ BHSD tensors, then must permute those dense tensors back to
  the cache layout before `store_quantized()`.
- Added the Self-Forcing handoff conversion for dense tensor outputs only.
  Packed dict outputs are still stored unchanged through the existing real-quant
  metadata path.
- Added a CPU regression that imports the layout helper with lightweight stubs
  and verifies dense fake-HQMQ BHSD output is converted before storing into a
  BSHD `ChunkedKVCache`.

Local validation before Slurm:

```bash
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_hqmq_fake.py
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/visionreward/bin/python tests/test_hqmq_fake.py
python3 -m py_compile quant_videogen/logger.py quant_videogen/hqmq_fake.py \
  quant_videogen/compress.py quant_videogen/uncompress.py \
  quant_videogen/sim/quant/lowbit_quantize.py \
  quant_videogen/sim/quant/quantize_config.py tests/test_hqmq_fake.py \
  experiments/LongCat/run_long_t2v.py \
  experiments/LongCat/longcat_video/modules/longcat_video_dit.py \
  experiments/Self-Forcing/inference.py \
  experiments/Self-Forcing/pipeline/causal_inference.py \
  experiments/HY-WorldPlay/wan/generate.py \
  experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py \
  experiments/LingBot/generate_fast.py experiments/LingBot/wan/image2video_fast.py
bash -n scripts/Self-Forcing/smoke_hqmq_fake.sbatch
git diff --check
```

All local validation passed. Asset symlinks for
`ckpts/Self-Forcing/Wan2.1-T2V-1.3B`,
`ckpts/Self-Forcing/self_forcing_dmd.pt`, and `assets/t2v.txt` resolved from the
source checkout.

Staged H200-only smoke submission:

- Code fix commit staged: `d4412343ab1378895ca01dbdd6c6d83f0934d25c`.
- Staged path:
  `/data/scratch-fast/kwen1/qvg-staging/20260727_221003-c2cba7-d441234-code`.
- Slurm job id: `1213946`.
- Partition/QoS: `vision-torralba-h200` /
  `vision-torralba-urops-meng`.
- Status after foreground polling through `2026-07-27 22:15:43` local machine
  time: `PENDING`, reason `(Priority)`, no node assigned.
- Logs will be
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/qvg_sf_hqmq_smoke-1213946.out`
  and
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/qvg_sf_hqmq_smoke-1213946.err`
  after the job starts.
- Output scout will be
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_27f_1213946`.

## 2026-07-28 - Self-Forcing Fake-HQMQ Output Review And Med3x Follow-Up

Task: review completed tiny fake-HQMQ Self-Forcing outputs from H200 job
`1213946` and decide the next bounded smoke.

Existing output review:

- `0-0_ema.mp4`: H.264, 832x480, 16 fps, 105 frames, 6.5625 s, 853,175 bytes.
- `1-0_ema.mp4`: H.264, 832x480, 16 fps, 105 frames, 6.5625 s, 803,257 bytes.
- Contact sheets show nonblank prompt-matched outputs: skateboarder on mountain
  road for prompt 0, papercraft coral reef/fish for prompt 1.
- `kv_cache_frames27.pt` exists at 7,858,486,627 bytes. FakeTensor metadata
  inspection found 30 BSHD `ChunkedKVCache` layers with bf16 chunks and
  30 cross-attention cache layers.
- Job log evidence: H200 startup showed 0 MiB / 143,771 MiB used and Python
  free VRAM 139.29 GB. Quantization times were 5.70 s and 5.55 s; peak memory
  was 20,973.79 MB and 29,327.10 MB; total KV cache memory was 6,581.87 MB and
  7,403.94 MB. `sacct` batch MaxRSS was 59,732,868K against a 96G request.

Decision: the outputs are plausible enough to continue, but the next run should
remain small. The follow-up will keep 27 frames and one sample, limit the prompt
file to one prompt to avoid a second 7.3G KV dump overwrite, and switch
`QUANT_TYPE=hqmq-fake-med3x` to exercise the outlier-preserving fake-HQMQ path.

Prepared instrumentation:

- Parameterized `scripts/Self-Forcing/smoke_hqmq_fake.sbatch` for quant type,
  frame count, local attention, prompt limit, dump level, and quant knobs.
- Added Slurm-side preflight logging for host memory, output/cache filesystem,
  H200 VRAM, CUDA visibility, and a static memory estimate derived from job
  `1213946`.
- Added compact Self-Forcing HQMQ fake diagnostics logging for effective bits,
  outlier rate, tile sizes, tile count, max simulated tile elements, and finite
  status.

Validation before submission passed:

```bash
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/vbench/bin/python tests/test_hqmq_fake.py
PYTHONPATH=. /data/scratch-fast/kwen1/eval_envs/visionreward/bin/python tests/test_hqmq_fake.py
bash -n scripts/Self-Forcing/smoke_hqmq_fake.sbatch
git diff --check
python3 -m py_compile experiments/Self-Forcing/pipeline/causal_inference.py \
  experiments/Self-Forcing/inference.py quant_videogen/hqmq_fake.py \
  quant_videogen/compress.py tests/test_hqmq_fake.py
```

Submission outcome:

- Intended staged command:

```bash
/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh \
  --staging-parent /data/scratch-fast/kwen1/qvg-staging \
  -- env QUANT_TYPE=hqmq-fake-med3x PROMPT_LIMIT=1 \
    RUN_LABEL_PREFIX=hqmq_fake_med3x_27f_prompt1 \
    NUM_OUTPUT_FRAMES=27 LOCAL_ATTN_SIZE=27 NUM_SAMPLES=1 DUMP_KV_LEVEL=1 \
    sbatch scripts/Self-Forcing/smoke_hqmq_fake.sbatch
```

- Direct invocation failed before staging or Slurm submission because the AFS
  helper path was inaccessible: `Permission denied`.
- `tokens` reported no AFS tokens, `klist` reported no Kerberos credential
  cache, `aklog` failed non-interactively, and `kinit -R` could not renew
  because there was no credentials cache.
- No follow-up Slurm job was submitted from the source worktree.
- Next step once credentials are restored: rerun the intended staged command
  above, then inspect `hqmq_fake_med3x_27f_prompt1_<jobid>` logs for
  `HQMQ Fake Diagnostics`, outlier rates, effective bits per value, memory
  estimate, H200 VRAM, host memory, and MP4/KV dump outputs.

## 2026-07-29 - Self-Forcing Med3x Staged Smoke Submitted

Task: resume the blocked one-prompt `hqmq-fake-med3x` follow-up after
Kerberos/AFS tokens were restored.

Pre-submit checks:

- AFS/Kerberos credentials were present and the staging helper was executable.
- Source branch was `fm/qvg-self-forcing-review-next-smoke` at
  `ebe7c9f1ca3e6031f977197dfc6ca809f9eed2c0`, with clean git status.
- Current shell was on CPU allocation `1226987` with no direct GPU visibility,
  as expected for the login/setup context.
- Target partition `vision-torralba-h200` was UP with `torralba-h200-1` exposing
  `gpu:h200:8`, `RealMemory=2063711M`, and `FreeMem=1923610M`.
- Account/QoS check showed `vision-torralba-urops-meng` with QoS
  `vision-torralba-urops-meng`.
- Static memory estimate for `NUM_OUTPUT_FRAMES=27`,
  `LOCAL_ATTN_SIZE=27`, `QUANT_TYPE=hqmq-fake-med3x`, `PROMPT_LIMIT=1`:
  estimated KV cache 7,403.94 MB, estimated peak VRAM 29,323.85 MB, and
  25% margin 36,654.81 MB against prior H200 143,771 MiB total VRAM and
  96G host memory request.

Staged submission:

```bash
/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh \
  --staging-parent /data/scratch-fast/kwen1/qvg-staging \
  -- env QUANT_TYPE=hqmq-fake-med3x PROMPT_LIMIT=1 \
    RUN_LABEL_PREFIX=hqmq_fake_med3x_27f_prompt1 \
    NUM_OUTPUT_FRAMES=27 LOCAL_ATTN_SIZE=27 NUM_SAMPLES=1 DUMP_KV_LEVEL=1 \
    sbatch scripts/Self-Forcing/smoke_hqmq_fake.sbatch
```

- Staged path:
  `/data/scratch-fast/kwen1/qvg-staging/20260729_002945-1402d8-ebe7c9f-code`.
- Staged metadata commit:
  `ebe7c9f1ca3e6031f977197dfc6ca809f9eed2c0`.
- Slurm job id: `1227740`.
- Output path after the job starts:
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results/hqmq_fake_med3x_27f_prompt1_1227740`.
- Logs after the job starts:
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/qvg_sf_hqmq_smoke-1227740.out`
  and
  `/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/qvg_sf_hqmq_smoke-1227740.err`.

Monitoring outcome at `2026-07-29 00:40:47 EDT`:

- Job state: `PENDING`, reason `Resources`.
- Scheduled node list: `torralba-h200-1`.
- `squeue --start` estimated start:
  `2026-07-30T17:20:09`.
- The H200 node was occupied by another `gres/gpu:8` job, so no terminal GPU
  outcome exists yet.

Next check: wait for job `1227740` to start/finish, then inspect the logs for
`SMOKE_CONFIG`, `memory_estimate`, H200 VRAM/CUDA evidence,
`HQMQ Fake Diagnostics`, quantization times, peak/KV memory, output MP4 media
metadata, contact sheet, and KV dump metadata.
