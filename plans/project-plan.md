# Research Program: Optimal KV-Cache Quantization for Autoregressive Video Generation

## 0. Mission

Develop the most effective practical method for quantizing the KV cache of autoregressive video-generation models.

The method should optimize the full tradeoff among:

[
\text{video quality},\quad
\text{long-horizon stability},\quad
\text{cache memory},\quad
\text{peak VRAM},\quad
\text{latency}.
]

Do not assume in advance that the final method should be based on:

* Quant VideoGen (QVG),
* HQMQ,
* scalar quantization,
* vector quantization,
* clustering,
* rotations,
* mixed precision,
* or any particular combination.

QVG and HQMQ are initial candidates because they offer distinct approaches:

* QVG exploits redundancy and clustered residual structure in video caches.
* HQMQ exploits four-dimensional radius–direction geometry using structured quaternion codebooks.

They are starting points, not constraints.

The final method may be:

1. A direct adaptation of an existing method.
2. A video-specific modification of an existing method.
3. A new quantization objective.
4. A new representation or codebook.
5. An adaptive policy choosing among quantizers.
6. A simple method revealed by careful analysis.
7. A systems method enabling compressed attention without full dequantization.

Do not combine methods merely because both are available. Combine methods only when experiments show complementary failure modes.

---

# 1. Central research questions

Answer these in order.

1. What statistical and functional properties distinguish video KV caches from language-model KV caches?
2. Which existing quantization family transfers best to video generation?
3. Which form of quantization error most strongly predicts generated-video degradation?
4. Do K and V require different representations or precisions?
5. Does sensitivity vary by layer, head, cache age, frame, motion regime, or rollout horizon?
6. Which candidate method achieves the best quality–memory frontier?
7. Can that method be improved using video-specific structure?
8. Does the resulting method reduce actual peak VRAM and preserve runtime?
9. Does it generalize across video-model architectures?

---

# 2. Success criteria

A successful final method should satisfy most of the following.

## Algorithmic quality

At equal persistent KV-cache memory, the proposed method should produce better:

* Long-horizon visual quality.
* Temporal consistency.
* Subject consistency.
* Motion quality.
* Object permanence.
* Paired-BF16 fidelity.

Alternatively, at matched quality, it should use meaningfully less cache memory.

## Systems quality

The method should reduce:

* Actual persistent cache bytes.
* Peak allocated GPU memory.
* Peak reserved GPU memory.

Its runtime overhead should be measured and preferably small.

## Generality

The method should work on:

* At least two model architectures.
* Multiple prompt categories.
* Multiple rollout lengths.
* Multiple random seeds.

## Scientific contribution

The paper should explain why the method works.

A raw empirical win without mechanism is less valuable than a result tied to measurable properties such as:

* Directional anisotropy.
* Radius outliers.
* Temporal redundancy.
* Cache-age sensitivity.
* K/V asymmetry.
* Attention sensitivity.
* Error amplification over rollouts.

---

# 3. Codebase selection

The infrastructure choice and algorithm choice must remain separate.

## Candidate codebases

### QVG codebase

Use when it provides the fastest path to:

* Reproducible BF16 generation.
* Existing compressed-cache storage.
* Swappable compression functions.
* Direct QVG comparison.
* Layer-wise and K/V-specific interventions.

### Causal Forcing++ codebase

Use when it provides the fastest path to:

* A currently functioning video model.
* Clean persistent-cache interception.
* Long autoregressive rollouts.
* Frame-wise and chunk-wise model comparisons.
* Stronger or more current model evaluation.

## Time-boxed codebase bakeoff

Spend at most four hours evaluating each repository.

For each repository, determine:

1. Can BF16 inference run?
2. Can the same prompt and seed be reproduced?
3. Where are K and V produced?
4. Where are they written to persistent storage?
5. Where are they read for attention?
6. Are cached keys pre- or post-positional encoding?
7. Can one layer or one K/V role be modified independently?
8. Can a compressed object replace a BF16 tensor?
9. Can exact persistent bytes and peak VRAM be measured?
10. How much code is needed to add a new codec?

## Decision rule

Choose the repository that minimizes time to the first controlled quantization experiment.

If both work:

* Use QVG for rapid codec development and direct baseline comparisons.
* Use Causal Forcing++ later for generalization.

Do not maintain two active implementations during initial method search.

---

# 4. Common codec interface

Implement every quantizer behind the same interface.

```python
class KVCodec:
    def encode(
        self,
        x,
        *,
        layer_id,
        role,
        cache_position,
        metadata=None,
    ):
        ...

    def decode(
        self,
        encoded,
        *,
        output_dtype,
    ):
        ...

    def storage_bytes(self, encoded) -> int:
        ...

    def estimate_ideal_packed_bytes(self, encoded) -> int:
        ...
```

Canonical tensor format:

[
X\in\mathbb{R}^{B\times H\times T\times D}.
]

Repository-specific layout conversion must occur only at the codec boundary.

Required initial codecs:

```text
BF16Codec
UniformScalarCodec
QVGCodec
HQMQCodec
```

Later codecs should be added only when motivated by evidence.

---

# 5. Candidate method families

The search must cover distinct quantization principles rather than many minor variants of one method.

## Family A: Uniform scalar quantization

Required baselines:

* INT8
* INT4
* INT3
* INT2

Test:

* Symmetric versus asymmetric.
* Per-tensor.
* Per-head.
* Per-token.
* Per-channel.
* Groupwise.
* Min/max scaling.
* Percentile clipping.
* MSE-optimal clipping.

Purpose:

* Establish how difficult the cache is to quantize.
* Determine whether complex methods are necessary.
* Provide simple Pareto-frontier baselines.

---

## Family B: Outlier-aware scalar quantization

Examples:

* High-precision outlier channels.
* High-precision outlier tokens.
* High-precision four-tuples.
* Sparse outlier storage.
* Clipping with explicit residual correction.

Questions:

* Are a small number of outliers responsible for most quantization error?
* Are outliers stable by layer, head, K/V role, or frame?
* Does preserving outliers improve video quality or only cache MSE?

---

## Family C: QVG-style clustered residual quantization

Test:

* One-stage centroid subtraction.
* Progressive residual stages.
* Different centroid counts.
* Different chunk lengths.
* K and V independently.
* Different low-bit residual quantizers.
* QVG default configuration.
* QVG high-fidelity configuration.

Questions:

* How much does clustering reduce dynamic range?
* Does clustering help because of spatial, temporal, or semantic redundancy?
* Does lower cache MSE translate into better rollout quality?
* Where does QVG fail?

---

## Family D: HQMQ-style geometric quantization

Test:

* Radius–direction decomposition.
* Multiple secondary-codebook sizes.
* Multiple radius precisions.
* Outlier extraction on and off.
* Multiple random codebook seeds.
* K and V independently.
* Alternative four-dimensional groupings.

Questions:

* Are video KV directions compatible with a data-independent spherical codebook?
* Is angular or radial error dominant?
* Is calibration-free random codebook construction stable?
* Does video geometry differ from language-model geometry?

---

## Family E: Rotation-based quantization

Test cheap transforms before scalar quantization:

* Random orthogonal rotations.
* Hadamard rotations.
* Channel permutations.
* Per-head rotations.
* Fixed data-independent rotations.
* Calibration-derived rotations only as a separate setting.

Questions:

* Does rotation suppress outliers?
* Does it make directions more isotropic?
* Does it improve INT2 or INT4 quantization?
* Does it interact differently with K and V?

---

## Family F: Vector and product quantization

Test:

* K-means vector quantization.
* Product quantization.
* Residual vector quantization.
* Small learned or calibrated codebooks.
* Shared versus per-layer codebooks.
* Shared versus separate K/V codebooks.

Questions:

* Does vector quantization preserve correlated dimensions better than scalar methods?
* Is codebook metadata worthwhile?
* Can codebooks be shared across time or layers?
* Does calibration cost produce enough benefit?

---

## Family G: Temporal predictive quantization

Exploit the temporal structure of video caches.

Candidate representations:

[
\Delta X_t=X_t-X_{t-1},
]

[
\Delta X_t=X_t-\widehat X_{t-1},
]

or prediction from previous frames, chunks, centroids, or cache states.

Test:

* Frame-to-frame delta quantization.
* Chunk-to-chunk delta quantization.
* Periodic keyframes plus quantized deltas.
* Predictive radius coding.
* Predictive centroid coding.

Questions:

* Are temporal deltas significantly easier to quantize?
* Does prediction error accumulate?
* How often are refresh frames required?
* Do scene changes break the representation?

---

## Family H: Mixed precision and adaptive quantization

Possible adaptation axes:

* K versus V.
* Layer.
* Head.
* Cache age.
* Frame or chunk.
* Motion regime.
* Attention sensitivity.
* Quantization residual.
* Scene transitions.

Do not begin here.

First establish which base quantizers are worth allocating.

---

# 6. Experimental principles

## 6.1 Separate algorithmic and systems evaluation

### Fake quantization

Use encode–decode numerically while retaining full-precision storage.

Purpose:

* Fast method screening.
* Reconstruction analysis.
* Video-quality analysis.

Do not claim memory reduction from fake quantization.

### Real quantization

Store packed compressed representations.

Purpose:

* Persistent-byte measurement.
* Peak-VRAM measurement.
* Runtime evaluation.
* Deployment validation.

Implement only for methods that survive algorithmic screening.

---

## 6.2 Use paired runs

For each prompt and seed:

* Generate a BF16 reference.
* Reuse the same prompt.
* Reuse the same generation seed.
* Reuse the same initial latent.
* Reuse the same scheduler configuration.
* Change only the cache codec.

---

## 6.3 Separate immediate and compounding effects

### Teacher-forced evaluation

Hold the future trajectory fixed to the BF16 trajectory.

Measures immediate functional distortion.

### Free-running evaluation

Allow quantized outputs to affect later model inputs.

Measures autoregressive error amplification.

---

## 6.4 Preserve raw outputs

Save:

* Generated videos.
* Per-frame metrics.
* Per-chunk metrics.
* Cache statistics.
* Encoded representations.
* Codec configuration.
* Prompt.
* Seed.
* Git commit.
* Hardware.
* Runtime.
* Peak VRAM.

Do not retain only aggregate scores.

---

# 7. Stage 0: Infrastructure validation

## Goal

Establish a trustworthy BF16 and quantized-cache evaluation harness.

## Initial conditions

Run:

1. Native BF16.
2. BF16 through the codec abstraction.
3. Deliberately weak INT4 quantization.
4. Deliberately aggressive INT2 quantization.
5. Existing QVG configuration, when using its repository.

Use:

* 5 prompts.
* 1 seed each.
* Short rollouts.

## Required checks

1. Native BF16 and codec-wrapper BF16 match.
2. Same-seed runs are reproducible.
3. K and V tensor layouts are documented.
4. Cache positions are correct.
5. Positional encoding placement is documented.
6. K-only quantization works.
7. V-only quantization works.
8. Layer selection works.
9. Quantization errors increase under more aggressive settings.
10. Metrics detect obviously damaged videos.

## Decision tree

### BF16 wrapper differs from native BF16

Stop and debug:

* Layout conversion.
* Cache indexing.
* Concatenation order.
* Chunk boundaries.
* Positional encoding.
* Dtype conversion.
* Cache replacement semantics.

### Quantization has no effect

Verify that the modified cache is actually consumed by attention.

### Small quantization causes catastrophic failure

Check implementation before inferring model sensitivity.

### Proceed criterion

Proceed only when:

* BF16 is trustworthy.
* Controlled perturbations produce controlled effects.
* Cache accounting is auditable.

---

# 8. Stage 1: Video KV-cache audit

## Goal

Identify the properties that should guide method selection.

## Current scaffold

Use `tools/cache_audit.py` for Stage-1 cache artifact inspection. It first
loads `torch.save` artifacts with `FakeTensorMode` and byte guards, then emits a
JSON schema covering file metadata, provenance keys, tensor paths, shapes,
dtypes, devices, byte estimates, layer/K/V/head/cache-position summaries,
HQMQ-style four-tuple radius stats when full stats are safe, temporal deltas,
and explicit `metadata_only` or `unsupported_layout` markers.

Default `auto` mode does not full-load artifacts larger than 1 GiB or with more
than 512 MiB of estimated tensor storage. Use `--stats-mode metadata-only` for
large dumps and `--stats-mode force` only for deliberately bounded CPU audits.

## Data collection

Use:

* 10 prompts.
* Low-, medium-, and high-motion categories.
* Early, middle, and late rollout positions.
* Early, middle, and late layers.
* K and V separately.

## Measure

### Marginal distributions

* Mean.
* Variance.
* Skewness.
* Kurtosis.
* Min/max.
* Percentiles.
* Outlier fraction.
* Dynamic range.

### Structural statistics

* Per-channel variance.
* Inter-channel covariance.
* Effective rank.
* Singular-value decay.
* Head-to-head similarity.
* Token-to-token similarity.
* Frame-to-frame similarity.
* Chunk-to-chunk similarity.

### Four-dimensional geometry

For HQMQ-style groupings:

[
r=|x|_2,
\qquad
u=x/r.
]

Measure:

* Radius distribution.
* Maximum-to-median radius.
* Direction covariance.
* Directional anisotropy.
* Nearest-codeword error.
* Codeword occupancy.
* Random-codebook-seed variance.

### Temporal predictability

Measure:

[
\frac{|X_t-X_{t-1}|_F}
{|X_t|_F},
]

and analogous metrics between:

* Adjacent tokens.
* Adjacent frames.
* Adjacent chunks.
* Same spatial location across frames.

### Functional sensitivity

For controlled noise or quantization:

* Attention-logit error.
* Attention-probability divergence.
* Attention-output error.
* Top-attended-token overlap.

## Decision tree

### Strong temporal redundancy

Prioritize:

* QVG.
* Predictive quantization.
* Delta coding.
* Temporally shared codebooks.

### Strong vector correlation but weak temporal redundancy

Prioritize:

* Vector quantization.
* Product quantization.
* Rotations.
* HQMQ.

### Extreme sparse outliers

Prioritize:

* Outlier-aware scalar methods.
* Rotations.
* HQMQ Med3-style extraction.
* Mixed precision.

### Approximately spherical four-dimensional directions

HQMQ becomes a strong candidate.

### Highly anisotropic directions

Test rotations or data-adapted codebooks before investing deeply in raw HQMQ.

### Strong K/V distribution difference

Plan asymmetric evaluation from the beginning.

---

# 9. Stage 2: Broad quantizer screening

## Goal

Quickly determine which method families deserve full-rollout experiments.

## Screening dataset

Use:

* 10 prompts.
* 1 seed each.
* Representative cache snapshots.
* No full long rollout initially.

## Candidate configurations

Keep the initial search small.

### Scalar

* INT4 per-token.
* INT4 per-channel.
* INT2 per-token.
* INT2 per-channel.
* INT2 with clipping.
* INT2 with outlier preservation.

### QVG

* Default QVG.
* One lower-compute setting.
* One higher-fidelity setting.

### HQMQ

* Small codebook, low-radius precision.
* Medium codebook, medium-radius precision.
* High-fidelity setting.
* Med3 on and off.

### Rotation

* Hadamard plus INT4.
* Hadamard plus INT2.
* Random orthogonal rotation plus INT2.

### Vector methods

* Small product-quantization baseline.
* Small residual-vector-quantization baseline.

### Temporal methods

* Frame delta plus INT4.
* Chunk delta plus INT4.
* Periodic refresh plus INT2 deltas.

## Metrics

For each method and effective storage rate:

* K normalized MSE.
* V normalized MSE.
* Cosine error.
* Attention-logit error.
* Attention-probability divergence.
* Attention-output error.
* Encode time.
* Decode time.
* Actual prototype bytes.
* Ideal packed bytes.
* Metadata fraction.

## Screening score

Do not choose only by cache MSE.

Rank methods separately by:

1. Cache reconstruction.
2. Attention-output preservation.
3. Storage efficiency.
4. Computational cost.
5. Robustness across layers and prompts.

## Decision tree

### One family dominates across most metrics

Advance that family and retain the strongest simple baseline.

### Different families dominate K and V

Advance an asymmetric quantization program.

### QVG has best MSE but another method has better attention preservation

Advance both to rollout evaluation.

### HQMQ performs well only for certain layers or roles

Treat it as a candidate within an allocator, not as a global method.

### Temporal coding is excellent except at scene transitions

Advance it with refresh or event-detection mechanisms.

### Complex methods barely beat scalar INT4

Use scalar INT4 as the main baseline and investigate whether lower bits justify complexity.

### No method works below four effective bits

Study sensitivity, outliers, and mixed precision before inventing another global codec.

---

# 10. Stage 3: Controlled rollout comparison

## Goal

Determine which screening results translate into generated-video quality.

## Methods to advance

Advance at most:

* Three best complex candidates.
* Two strongest scalar baselines.
* BF16.
* Existing QVG baseline.

Do not run every configuration on full videos.

## Dataset

Use:

* 20 prompts.
* 2 seeds per prompt.
* Prespecified motion categories.
* Short and medium rollouts.

## Required modes

For each method:

1. K only.
2. V only.
3. K and V.
4. Teacher-forced.
5. Free-running.

## Metrics

### Paired-BF16 fidelity

* LPIPS.
* PSNR.
* SSIM.
* Latent distance.

### Independent video quality

* Temporal consistency.
* Subject consistency.
* Motion quality.
* Scene consistency.
* Appropriate VBench components.

### Longitudinal behavior

For chunk (t):

[
d_t=d(X_t^{\mathrm{BF16}},X_t^{\mathrm{quant}}).
]

Report:

* Initial divergence.
* Final divergence.
* Maximum divergence.
* Divergence AUC.
* Growth rate.
* Time to failure threshold.

## Decision tree

### Cache MSE ranking matches video-quality ranking

Optimize the best rate–distortion codec directly.

### Attention-output error predicts quality better than cache MSE

Use functional distortion for future method design and allocation.

### A method has worse MSE but better video quality

Investigate which errors are functionally harmless.

This may motivate a new objective.

### K-only and V-only results differ greatly

Make K/V asymmetry a primary research direction.

### Short-rollout results are strong but long-rollout results fail

Study error amplification and cache-age-aware precision.

### A method wins only on specific prompt categories

Study content-adaptive selection rather than claiming universal dominance.

---

# 11. Stage 4: Select the main research branch

Choose exactly one primary branch after Stage 3.

## Branch A: Improve the best global codec

Choose when one codec consistently dominates.

Possible work:

* Better clipping.
* Better group size.
* Better codebooks.
* Better radius coding.
* Better outlier handling.
* Better temporal sharing.
* Lower metadata.

---

## Branch B: K/V-asymmetric quantization

Choose when K and V have clearly different sensitivity or distributions.

Candidate actions:

[
a_K\neq a_V.
]

Examples:

* Higher precision for K.
* Higher precision for V.
* Different quantizer families for K and V.
* Different outlier rules.
* Different grouping.

Do not combine QVG and HQMQ by default. Use different methods only if each is empirically best for its assigned role.

---

## Branch C: Layer- or head-sensitive quantization

Choose when sensitivity is strongly nonuniform.

Test progressively:

1. Per-layer.
2. Per-layer and K/V.
3. Per-head only if layer-wise allocation leaves substantial gains.

Avoid unnecessary granularity.

---

## Branch D: Cache-age-aware quantization

Choose when older and newer cache regions tolerate different distortion.

Candidate policy:

[
b(\tau)
]

or

[
\text{codec}(\tau),
]

where (\tau) is cache age.

---

## Branch E: Event-aware quantization

Choose when failures align with:

* Scene cuts.
* Fast motion.
* Camera motion.
* Occlusion.
* New-object entry.

Use online indicators such as:

* Cache residual norm.
* Centroid displacement.
* Latent-frame difference.
* Assignment entropy.
* Directional shift.
* Motion magnitude.

---

## Branch F: Temporal predictive quantization

Choose when deltas are substantially easier to quantize.

Develop:

* Refresh schedules.
* Error-bounded prediction.
* Scene-change resets.
* Adaptive reference selection.

---

## Branch G: Function-aware quantization objective

Choose when numerical reconstruction metrics are misaligned with video quality.

Possible objectives:

[
D_{\mathrm{logit}},
\quad
D_{\mathrm{prob}},
\quad
D_{\mathrm{attn}},
\quad
D_{\mathrm{rollout}}.
]

Develop a cheap proxy only after demonstrating oracle benefit.

---

# 12. Oracle analysis before adaptive-method engineering

Before implementing a learned or complex adaptive policy, estimate its maximum possible benefit.

For cache block (i), define candidate actions:

[
a_i\in\mathcal A,
]

where (\mathcal A) may include:

* BF16.
* INT4.
* INT2.
* QVG configurations.
* HQMQ configurations.
* Best vector quantizer.
* Best temporal codec.

Estimate distortion:

[
D_i(a_i),
]

and exact storage:

[
R_i(a_i).
]

Solve:

[
\min_{{a_i}}
\sum_i D_i(a_i)
\quad
\text{s.t.}
\quad
\sum_i R_i(a_i)\le M.
]

## Decision tree

### Oracle allocation gives a large gain

Develop a practical predictor or static policy.

### Oracle allocation gives a small gain

Do not build a complicated allocator.

Use the best global codec.

### Oracle gain appears only on calibration prompts

Reduce policy granularity or regularize.

### Oracle selects one codec almost everywhere

That codec should remain the primary method.

### Oracle repeatedly selects different codecs for stable regions

A multi-codec policy may be justified.

---

# 13. Method-combination rule

Do not combine two methods merely because their implementations exist.

A combination is justified only when all are true:

1. The methods have measurably different strengths.
2. Their failure modes are complementary.
3. An oracle combination improves the Pareto frontier.
4. The gain persists on held-out prompts.
5. The added metadata and runtime do not erase the benefit.
6. The combination has a coherent explanation.

Examples of justified combinations:

* One codec is consistently best for K and another for V.
* One codec is best for recent cache and another for old cache.
* One codec handles smooth segments and another handles scene transitions.
* One transform removes outliers before the best low-bit codec.

Examples of unjustified combinations:

* QVG and HQMQ are both available.
* Combining methods slightly reduces MSE on a few snapshots.
* A complicated combination matches a simple INT4 baseline.
* The combination helps only on tuned prompts.

---

# 14. Real-storage implementation

Implement packed storage only for finalists.

## Required accounting

For every method, include:

* Quantized payload.
* Scales.
* Zero-points.
* Codebooks.
* Cluster assignments.
* Outlier flags.
* Outlier values.
* Padding.
* Alignment.
* Per-span metadata.
* Persistent auxiliary state.

Define effective bits:

[
b_{\mathrm{eff}}
================

\frac{
8M_{\mathrm{stored}}
}{
N_{\mathrm{original\ scalars}}
}.
]

Report:

1. Prototype storage.
2. Ideal packed storage.
3. Real packed storage.
4. Compression relative to BF16.
5. Metadata fraction.

Do not label a method “2-bit” when its realized storage is materially larger.

---

# 15. Peak-memory and compressed-attention evaluation

A compressed cache does not automatically imply lower peak VRAM.

Measure:

* Persistent cache bytes.
* Temporary dequantization buffers.
* Attention workspace.
* Peak allocated VRAM.
* Peak reserved VRAM.
* Memory over rollout time.

## Decision tree

### Persistent cache shrinks and peak VRAM shrinks

Proceed to runtime optimization.

### Persistent cache shrinks but peak VRAM does not

Identify whether full BF16 cache materialization is responsible.

Potential systems direction:

* Tiled dequantization.
* Streaming dequantization.
* Fused decode and attention.
* Direct attention over compressed representations.

### Compressed attention is necessary but algorithmic gains are weak

Stop. Do not optimize a noncompetitive quantizer.

### Quantizer is strong and decompression dominates

A fused-kernel contribution is justified.

---

# 16. Cross-model validation

After freezing the method on the development model, evaluate on a second architecture.

Preferred structure:

* Develop on the easier of QVG/Self-Forcing or Causal Forcing++.
* Validate on the other.

Transfer:

* Codec.
* Quantization policy.
* Hyperparameter-selection rule.
* Memory accounting.
* Evaluation harness.

Do not extensively retune the second model before reporting zero-shot or minimally calibrated transfer.

## Decision tree

### Method transfers directly

Claim architectural generality.

### Method transfers after simple model-specific calibration

Describe it as model-calibrated.

### Method works only on one architecture

Determine whether the cause is:

* Cache distribution.
* Positional encoding.
* Frame-wise versus chunk-wise generation.
* Query length.
* Cache update policy.
* Rollout stability.

A narrower method may remain publishable if the mechanism is clear.

---

# 17. Final benchmark

## Models

Minimum:

* Two autoregressive video-generation architectures.

## Prompts

* At least 100 fixed prompts.
* At least 2 seeds per prompt.
* Multiple motion and scene categories.
* No removal of unfavorable examples.

## Baselines

At minimum:

* BF16.
* Uniform INT4.
* Uniform INT2.
* Strong outlier-aware scalar baseline.
* QVG.
* HQMQ.
* Strongest additional screened family.
* Proposed method.

Only include hybrid baselines when experimentally justified.

## Evaluation points

Compare at:

* Matched effective storage.
* Matched paired-BF16 fidelity.
* Matched independent video quality.
* Matched peak VRAM.
* Matched runtime where feasible.

## Human evaluation

Use blinded pairwise evaluation for:

* Overall quality.
* Temporal consistency.
* Subject identity.
* Motion realism.
* Long-term coherence.

---

# 18. Required ablations

The final paper should include ablations relevant to the selected method.

Candidate ablations:

## Representation

* Scalar versus vector.
* Raw versus residual.
* Original versus rotated.
* Absolute versus temporal delta.
* Shared versus separate K/V treatment.

## Quantization granularity

* Global.
* Per-layer.
* Per-head.
* Per-token.
* Per-channel.
* Per-age band.

## Error objective

* Cache MSE.
* Cosine error.
* Logit error.
* Attention-probability error.
* Attention-output error.
* Rollout error.

## Precision

* INT8.
* INT4.
* INT3.
* INT2.
* Mixed precision.

## Robustness

* Prompt seed.
* Quantizer seed.
* Codebook seed.
* Motion category.
* Rollout length.
* Resolution.
* Model.

---

# 19. Statistical requirements

Use prompt as the primary statistical unit.

Report:

* Mean paired difference.
* Median paired difference.
* Prompt-level bootstrap confidence interval.
* Win rate.
* Effect size.
* Per-category results.
* Worst-case failure rate.

Do not treat many cache tensors from the same generated video as independent samples.

---

# 20. Failure analysis

Inspect at least the worst 20 examples for each finalist.

Use a fixed taxonomy:

* Identity drift.
* Object disappearance.
* Object duplication.
* Geometry change.
* Motion freezing.
* Motion instability.
* Camera instability.
* Texture collapse.
* Color drift.
* Flicker.
* Scene-transition failure.
* Prompt deviation.

For each failure, record:

* Method.
* K/V treatment.
* Layer or age sensitivity.
* Quantization error.
* Attention-output error.
* Cache statistics.
* Whether BF16 also fails.

---

# 21. Stage-level decision tree

## After cache audit

### Strong temporal redundancy

Prioritize QVG and temporal coding.

### Favorable four-dimensional spherical geometry

Prioritize HQMQ.

### Sparse extreme outliers

Prioritize outlier-aware methods and rotations.

### Strong cross-dimensional covariance

Prioritize vector or product quantization.

### No dominant structure

Prioritize simple scalar baselines and functional sensitivity analysis.

---

## After offline screening

### One clear winner

Advance it.

### Different winners for K and V

Advance asymmetric policies.

### Different winners by layer or age

Run oracle mixed allocation.

### No method beats INT4 meaningfully

Study whether lower precision is intrinsically unstable before adding complexity.

---

## After rollout experiments

### Best cache metric also gives best videos

Optimize that method directly.

### Functional error predicts videos better

Design around functional error.

### Error compounds only at long horizons

Focus the contribution on rollout-aware quantization.

### Errors are event-specific

Develop event-aware precision or refresh.

---

## After oracle allocation

### Large oracle gain

Develop adaptive or static allocation.

### Small oracle gain

Use the best global method.

### Oracle relies on many codec switches

Check metadata and implementation cost before pursuing multi-codec storage.

---

## After real-storage evaluation

### Quality and memory improve

Proceed to final benchmark.

### Storage improves but peak VRAM does not

Pursue compressed attention only if the quality result is strong.

### Runtime is unacceptable

Profile encode, decode, attention, and memory movement separately.

---

# 22. Stop and pivot criteria

Stop a method family when:

1. It is dominated at matched actual storage.
2. It is unstable across prompts or seeds.
3. Its advantages disappear under full rollouts.
4. Its metadata removes the nominal compression benefit.
5. Its runtime is prohibitive without a plausible fused implementation.
6. Its mechanism does not suggest a repair.

Stop the full project direction when:

1. No method materially beats simple INT4.
2. Lower-bit methods consistently cause unacceptable long-horizon degradation.
3. KV cache is not a meaningful memory bottleneck in the target models.
4. Compressed storage does not reduce peak VRAM and fused attention is infeasible.

Possible pivots:

* KV-cache pruning.
* Cache summarization.
* Sparse attention.
* External memory.
* Joint quantization and eviction.
* Weight and activation quantization.
* Faster attention systems.

---

# 23. First 72-hour plan

## Day 1: Infrastructure and codebase choice

Deliver:

1. BF16 generation.
2. Cache-wrapper BF16 generation.
3. Cache tensor shape and layout report.
4. Cache read/write call graph.
5. K-only and V-only perturbation.
6. Codebase-selection decision.
7. Reproducible environment manifest.

## Day 2: Common baselines

Implement:

* Uniform INT4.
* Uniform INT2.
* Clipped INT2.
* Existing QVG path.
* HQMQ reference path.

Deliver:

* Unit tests.
* Storage accounting.
* Snapshot reconstruction metrics.
* Encode/decode timing.

## Day 3: First comparative result

Run:

* 3 prompts.
* 1 seed.
* BF16.
* INT4.
* INT2.
* QVG.
* HQMQ.
* K only.
* V only.
* K and V.

Deliver:

1. Generated videos.
2. Cache reconstruction metrics.
3. Attention-output metrics.
4. Paired-video metrics.
5. Effective storage.
6. Runtime.
7. Initial method-family ranking.
8. Next-stage decision.

---

# 24. First-week deliverables

Produce:

1. A common codec API.
2. Reproducible BF16 evaluation.
3. Scalar, QVG, and HQMQ implementations.
4. Video KV-distribution audit.
5. Cache-level rate–distortion comparison.
6. Initial rollout comparison.
7. K versus V sensitivity report.
8. Prompt-category failure report.
9. Exact byte accounting.
10. A decision selecting no more than three method families for deeper study.

The first-week conclusion must not assume QVG, HQMQ, or their combination is the answer.

---

# 25. Agent operating protocol

Before every experiment, write:

```text
HYPOTHESIS:
WHY THIS EXPERIMENT MATTERS:
MANIPULATED VARIABLE:
CONTROLLED VARIABLES:
PRIMARY METRIC:
SECONDARY METRICS:
EXPECTED RESULT:
FALSIFYING RESULT:
DECISION RULE:
```

After every experiment, write:

```text
RESULT:
CONFIDENCE:
MAIN EVIDENCE:
FAILURE MODES:
STORAGE BYTES:
PEAK VRAM:
RUNTIME:
DECISION:
NEXT EXPERIMENT:
```

Rules:

1. Run the smallest valid experiment first.
2. Inspect generated videos before scaling.
3. Keep prompts, seeds, and initial latents paired.
4. Do not compare nominal bit labels.
5. Do not claim memory reduction from fake quantization.
6. Do not tune on final evaluation prompts.
7. Do not combine methods without oracle evidence.
8. Do not optimize kernels before establishing algorithmic value.
9. Prefer simple methods when quality is comparable.
10. Stop weak branches quickly.
11. Record negative results.
12. Let experimental evidence determine the method, not the starting codebase or initial paper choices.
