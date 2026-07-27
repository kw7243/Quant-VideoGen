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
