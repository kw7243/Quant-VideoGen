# Research Program: Optimal KV-Cache Quantization For Autoregressive Video Generation

This is the captain's central plan for the video-generation research program.
Add it to the active video-generation repo as a durable project document, preserving the research intent and decision rules.

## 0. Mission

Develop the most effective practical method for quantizing the KV cache of autoregressive video-generation models.

The method should optimize the full tradeoff among video quality, long-horizon stability, cache memory, peak VRAM, and latency.

Do not assume in advance that the final method should be based on Quant-VideoGen, HQMQ, scalar quantization, vector quantization, clustering, rotations, mixed precision, or any particular combination.

QVG and HQMQ are starting candidates because they offer distinct approaches:

- QVG exploits redundancy and clustered residual structure in video caches.
- HQMQ exploits four-dimensional radius-direction geometry using structured quaternion codebooks.

They are starting points, not constraints.
The final method may be a direct adaptation of an existing method, a video-specific modification, a new quantization objective, a new representation or codebook, an adaptive policy, a simple method revealed by careful analysis, or a systems method enabling compressed attention without full dequantization.
Do not combine methods merely because both are available.
Combine methods only when experiments show complementary failure modes.

## 1. Central Research Questions

Answer these in order:

1. What statistical and functional properties distinguish video KV caches from language-model KV caches?
2. Which existing quantization family transfers best to video generation?
3. Which form of quantization error most strongly predicts generated-video degradation?
4. Do K and V require different representations or precisities?
5. Does sensitivity vary by layer, head, cache age, frame, motion regime, or rollout horizon?
6. Which candidate method achieves the best quality-memory frontier?
7. Can that method be improved using video-specific structure?
8. Does the resulting method reduce actual peak VRAM and preserve runtime?
9. Does it generalize across video-model architectures?

## 2. Success Criteria

At equal persistent KV-cache memory, the proposed method should produce better long-horizon visual quality, temporal consistency, subject consistency, motion quality, object permanence, and paired-BF16 fidelity.
Alternatively, at matched quality, it should use meaningfully less cache memory.

The method should reduce actual persistent cache bytes, peak allocated GPU memory, and peak reserved GPU memory.
Runtime overhead must be measured and should preferably be small.

The method should work on at least two model architectures, multiple prompt categories, multiple rollout lengths, and multiple random seeds.

The paper should explain why the method works.
A raw empirical win without mechanism is less valuable than a result tied to measurable properties such as directional anisotropy, radius outliers, temporal redundancy, cache-age sensitivity, K/V asymmetry, attention sensitivity, or error amplification over rollouts.

## 3. Codebase Selection

Keep infrastructure choice and algorithm choice separate.

Use the QVG codebase when it provides the fastest path to reproducible BF16 generation, existing compressed-cache storage, swappable compression functions, direct QVG comparison, and layer-wise or K/V-specific interventions.

Use the Causal Forcing++ codebase when it provides the fastest path to a currently functioning video model, clean persistent-cache interception, long autoregressive rollouts, frame-wise and chunk-wise model comparisons, and stronger or more current model evaluation.

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

Choose the repository that minimizes time to the first controlled quantization experiment.
If both work, use QVG for rapid codec development and direct baseline comparisons, and use Causal Forcing++ later for generalization.
Do not maintain two active implementations during initial method search.

## 4. Common Codec Interface

Implement every quantizer behind the same interface:

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

```text
X in R^{B x H x T x D}
```

Repository-specific layout conversion must occur only at the codec boundary.

Required initial codecs:

- `BF16Codec`
- `UniformScalarCodec`
- `QVGCodec`
- `HQMQCodec`

Later codecs should be added only when motivated by evidence.

## 5. Candidate Method Families

The search must cover distinct quantization principles rather than many minor variants of one method.

### Family A: Uniform Scalar Quantization

Required baselines:

- INT8
- INT4
- INT3
- INT2

Test symmetric versus asymmetric scaling, per-tensor, per-head, per-token, per-channel, groupwise, min/max scaling, percentile clipping, and MSE-optimal clipping.
Use this family to establish how difficult the cache is to quantize, whether complex methods are necessary, and what the simple Pareto frontier looks like.

### Family B: Outlier-Aware Scalar Quantization

Examples include high-precision outlier channels, high-precision outlier tokens, high-precision four-tuples, sparse outlier storage, and clipping with explicit residual correction.

Questions:

- Are a small number of outliers responsible for most quantization error?
- Are outliers stable by layer, head, K/V role, or frame?
- Does preserving outliers improve video quality or only cache MSE?

### Family C: QVG-Style Clustered Residual Quantization

Test one-stage centroid subtraction, progressive residual stages, different centroid counts, different chunk lengths, K and V independently, different low-bit residual quantizers, QVG default configuration, and QVG high-fidelity configuration.

Questions:

- How much does clustering reduce dynamic range?
- Does clustering help because of spatial, temporal, or semantic redundancy?
- Does lower cache MSE translate into better rollout quality?
- Where does QVG fail?

### Family D: HQMQ-Style Geometric Quantization

Test radius-direction decomposition, multiple secondary-codebook sizes, multiple radius precisions, outlier extraction on and off, multiple random codebook seeds, K and V independently, and alternative four-dimensional groupings.

Questions:

- Are video KV directions compatible with a data-independent spherical codebook?
- Is angular or radial error dominant?
- Is calibration-free random codebook construction stable?
- Does video geometry differ from language-model geometry?

### Family E: Rotation-Based Quantization

Test cheap transforms before scalar quantization:

- Random orthogonal rotations.
- Hadamard rotations.
- Channel permutations.
- Per-head rotations.
- Fixed data-independent rotations.
- Calibration-derived rotations only as a separate setting.

Questions:

- Does rotation suppress outliers?
- Does it make directions more isotropic?
- Does it improve INT2 or INT4 quantization?
- Does it interact differently with K and V?

### Family F: Vector And Product Quantization

Test K-means vector quantization, product quantization, residual vector quantization, small learned or calibrated codebooks, shared versus per-layer codebooks, and shared versus separate K/V codebooks.

Questions:

- Does vector quantization preserve correlated dimensions better than scalar methods?
- Is codebook metadata worthwhile?
- Can codebooks be shared across time or layers?
- Does calibration cost produce enough benefit?

### Family G: Temporal Predictive Quantization

Exploit temporal structure of video caches.
Candidate representations include:

```text
Delta X_t = X_t - X_{t-1}
Delta X_t = X_t - hat(X_{t-1})
```

or prediction from previous frames, chunks, centroids, or cache states.

Test frame-to-frame delta quantization, chunk-to-chunk delta quantization, periodic keyframes plus quantized deltas, predictive radius coding, and predictive centroid coding.

Questions:

- Are temporal deltas significantly easier to quantize?
- Does prediction error accumulate?
- How often are refresh frames required?
- Do scene changes break the representation?

### Family H: Mixed Precision And Adaptive Quantization

Possible adaptation axes include K versus V, layer, head, cache age, frame or chunk, motion regime, attention sensitivity, quantization residual, and scene transitions.
Do not begin here.
First establish which base quantizers are worth allocating.

## 6. Experimental Principles

Separate fake quantization from real quantization.

Fake quantization uses encode-decode numerically while retaining full-precision storage.
It is for fast method screening, reconstruction analysis, and video-quality analysis.
Do not claim memory reduction from fake quantization.

Real quantization stores packed compressed representations.
It is for persistent-byte measurement, peak-VRAM measurement, runtime evaluation, and deployment validation.
Implement it only for methods that survive algorithmic screening.

Use paired runs.
For each prompt and seed, generate a BF16 reference and reuse the same prompt, seed, initial latent, and scheduler configuration.
Change only the cache codec.

Separate immediate and compounding effects.
Teacher-forced evaluation holds the future trajectory fixed to the BF16 trajectory and measures immediate functional distortion.
Free-running evaluation lets quantized outputs affect later model inputs and measures autoregressive error amplification.

Preserve generated videos, per-frame metrics, per-chunk metrics, cache statistics, encoded representations, codec configuration, prompt, seed, git commit, hardware, runtime, and peak VRAM.
Do not retain only aggregate scores.

## 7. Stage 0: Infrastructure Validation

Goal: establish a trustworthy BF16 and quantized-cache evaluation harness.

Initial conditions:

1. Native BF16.
2. BF16 through the codec abstraction.
3. Deliberately weak INT4 quantization.
4. Deliberately aggressive INT2 quantization.
5. Existing QVG configuration, when using its repository.

Use five prompts, one seed each, and short rollouts.

Required checks:

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

Decision tree:

- If the BF16 wrapper differs from native BF16, stop and debug layout conversion, cache indexing, concatenation order, chunk boundaries, positional encoding, dtype conversion, and cache replacement semantics.
- If quantization has no effect, verify that the modified cache is actually consumed by attention.
- If small quantization causes catastrophic failure, check implementation before inferring model sensitivity.
- Proceed only when BF16 is trustworthy, controlled perturbations produce controlled effects, and cache accounting is auditable.

## 8. Stage 1: Video KV-Cache Audit

Goal: identify the properties that should guide method selection.

Use 10 prompts, low-, medium-, and high-motion categories, early/middle/late rollout positions, early/middle/late layers, and K and V separately.

Measure marginal distributions: mean, variance, skewness, kurtosis, min/max, percentiles, outlier fraction, and dynamic range.

Measure structural statistics: per-channel variance, inter-channel covariance, effective rank, singular-value decay, head-to-head similarity, token-to-token similarity, frame-to-frame similarity, and chunk-to-chunk similarity.

For HQMQ-style four-dimensional groupings, measure radius distribution, maximum-to-median radius, direction covariance, directional anisotropy, nearest-codeword error, codeword occupancy, and random-codebook-seed variance.

Measure temporal predictability with normalized frame, token, chunk, and same-spatial-location differences.

Measure functional sensitivity through attention-logit error, attention-probability divergence, attention-output error, and top-attended-token overlap.

Decision tree:

- Strong temporal redundancy: prioritize QVG, predictive quantization, delta coding, and temporally shared codebooks.
- Strong vector correlation but weak temporal redundancy: prioritize vector quantization, product quantization, rotations, and HQMQ.
- Extreme sparse outliers: prioritize outlier-aware scalar methods, rotations, HQMQ Med3-style extraction, and mixed precision.
- Approximately spherical four-dimensional directions: HQMQ becomes a strong candidate.
- Highly anisotropic directions: test rotations or data-adapted codebooks before investing deeply in raw HQMQ.
- Strong K/V distribution difference: plan asymmetric evaluation from the beginning.

## 9. Stage 2: Broad Quantizer Screening

Goal: quickly determine which method families deserve full-rollout experiments.

Use 10 prompts, one seed each, representative cache snapshots, and no full long rollout initially.

Keep the search small.
Initial candidates:

- Scalar: INT4 per-token, INT4 per-channel, INT2 per-token, INT2 per-channel, INT2 with clipping, and INT2 with outlier preservation.
- QVG: default QVG, one lower-compute setting, and one higher-fidelity setting.
- HQMQ: small codebook/low-radius precision, medium codebook/medium-radius precision, high-fidelity setting, with Med3 on and off.
- Rotation: Hadamard plus INT4, Hadamard plus INT2, and random orthogonal rotation plus INT2.
- Vector: small product-quantization baseline and small residual-vector-quantization baseline.
- Temporal: frame delta plus INT4, chunk delta plus INT4, and periodic refresh plus INT2 deltas.

For each method and effective storage rate, measure K normalized MSE, V normalized MSE, cosine error, attention-logit error, attention-probability divergence, attention-output error, encode time, decode time, actual prototype bytes, ideal packed bytes, and metadata fraction.

Do not choose only by cache MSE.
Rank methods separately by cache reconstruction, attention-output preservation, storage efficiency, computational cost, and robustness across layers and prompts.

Decision tree:

- If one family dominates across most metrics, advance that family and retain the strongest simple baseline.
- If different families dominate K and V, advance an asymmetric quantization program.
- If QVG has best MSE but another method has better attention preservation, advance both to rollout evaluation.
- If HQMQ performs well only for certain layers or roles, treat it as a candidate within an allocator, not as a global method.
- If temporal coding is excellent except at scene transitions, advance it with refresh or event-detection mechanisms.
- If complex methods barely beat scalar INT4, use scalar INT4 as the main baseline and investigate whether lower bits justify complexity.
- If no method works below four effective bits, study sensitivity, outliers, and mixed precision before inventing another global codec.

## 10. Stage 3: Controlled Rollout Comparison

Goal: determine which screening results translate into generated-video quality.

Advance at most three best complex candidates, two strongest scalar baselines, BF16, and existing QVG.
Do not run every configuration on full videos.

Use 20 prompts, two seeds per prompt, prespecified motion categories, and short and medium rollouts.

For each method, run K only, V only, K and V, teacher-forced, and free-running modes.

Metrics:

- Paired-BF16 fidelity: LPIPS, PSNR, SSIM, and latent distance.
- Independent video quality: temporal consistency, subject consistency, motion quality, scene consistency, and appropriate VBench components.
- Longitudinal behavior: initial divergence, final divergence, maximum divergence, divergence AUC, growth rate, and time to failure threshold.

Decision tree:

- If cache MSE ranking matches video-quality ranking, optimize the best rate-distortion codec directly.
- If attention-output error predicts quality better than cache MSE, use functional distortion for future method design and allocation.
- If a method has worse MSE but better video quality, investigate which errors are functionally harmless.
- If K-only and V-only results differ greatly, make K/V asymmetry a primary direction.
- If short-rollout results are strong but long-rollout results fail, study error amplification and cache-age-aware precision.
- If a method wins only on specific prompt categories, study content-adaptive selection instead of claiming universal dominance.

## 11. Stage 4: Select The Main Research Branch

Choose exactly one primary branch after Stage 3.

Candidate branches:

- Improve the best global codec when one codec consistently dominates.
- K/V-asymmetric quantization when K and V have clearly different sensitivity or distributions.
- Layer- or head-sensitive quantization when sensitivity is strongly nonuniform.
- Cache-age-aware quantization when older and newer cache regions tolerate different distortion.
- Event-aware quantization when failures align with scene cuts, fast motion, camera motion, occlusion, or new-object entry.
- Temporal predictive quantization when deltas are substantially easier to quantize.
- Function-aware quantization objective when numerical reconstruction metrics are misaligned with video quality.

Do not combine QVG and HQMQ by default.
Use different methods only if each is empirically best for its assigned role or region.

## 12. Oracle Analysis Before Adaptive-Method Engineering

Before implementing a learned or complex adaptive policy, estimate its maximum possible benefit.

For each cache block, define candidate actions such as BF16, INT4, INT2, QVG configurations, HQMQ configurations, the best vector quantizer, and the best temporal codec.
Estimate each action's distortion and exact storage, then solve the constrained allocation problem:

```text
min sum_i D_i(a_i)
subject to sum_i R_i(a_i) <= M
```

Decision tree:

- If oracle allocation gives a large gain, develop a practical predictor or static policy.
- If oracle allocation gives a small gain, do not build a complicated allocator; use the best global codec.
- If oracle gain appears only on calibration prompts, reduce policy granularity or regularize.
- If oracle selects one codec almost everywhere, that codec remains primary.
- If oracle repeatedly selects different codecs for stable regions, a multi-codec policy may be justified.

## 13. Method-Combination Rule

A combination is justified only when all are true:

1. The methods have measurably different strengths.
2. Their failure modes are complementary.
3. An oracle combination improves the Pareto frontier.
4. The gain persists on held-out prompts.
5. The added metadata and runtime do not erase the benefit.
6. The combination has a coherent explanation.

Justified combinations include one codec for K and another for V, one codec for recent cache and another for old cache, one codec for smooth segments and another for scene transitions, or one transform that removes outliers before the best low-bit codec.

Unjustified combinations include combining QVG and HQMQ only because both are available, slightly reducing MSE on a few snapshots, matching a simple INT4 baseline with a complicated combination, or helping only on tuned prompts.

## 14. Real-Storage Implementation

Implement packed storage only for finalists.

For every method, include quantized payload, scales, zero points, codebooks, cluster assignments, outlier flags, outlier values, padding, alignment, per-span metadata, and persistent auxiliary state.

Define effective bits as:

```text
b_eff = 8 * M_stored / N_original_scalars
```

Report prototype storage, ideal packed storage, real packed storage, compression relative to BF16, and metadata fraction.
Do not label a method "2-bit" when its realized storage is materially larger.

## 15. Peak-Memory And Compressed-Attention Evaluation

A compressed cache does not automatically imply lower peak VRAM.

Measure persistent cache bytes, temporary dequantization buffers, attention workspace, peak allocated VRAM, peak reserved VRAM, and memory over rollout time.

Decision tree:

- If persistent cache and peak VRAM both shrink, proceed to runtime optimization.
- If persistent cache shrinks but peak VRAM does not, identify whether full BF16 cache materialization is responsible.
- Potential systems directions include tiled dequantization, streaming dequantization, fused decode and attention, and direct attention over compressed representations.
- If compressed attention is necessary but algorithmic gains are weak, stop.
- If the quantizer is strong and decompression dominates, a fused-kernel contribution is justified.

## 16. Cross-Model Validation

After freezing the method on the development model, evaluate on a second architecture.
Develop on the easier of QVG/Self-Forcing or Causal Forcing++, then validate on the other.

Transfer codec, quantization policy, hyperparameter-selection rule, memory accounting, and evaluation harness.
Do not extensively retune the second model before reporting zero-shot or minimally calibrated transfer.

Decision tree:

- If the method transfers directly, claim architectural generality.
- If it transfers after simple model-specific calibration, describe it as model-calibrated.
- If it works only on one architecture, determine whether the cause is cache distribution, positional encoding, frame-wise versus chunk-wise generation, query length, cache update policy, or rollout stability.

## 17. Final Benchmark

Minimum models: two autoregressive video-generation architectures.

Prompts:

- At least 100 fixed prompts.
- At least two seeds per prompt.
- Multiple motion and scene categories.
- No removal of unfavorable examples.

Baselines:

- BF16.
- Uniform INT4.
- Uniform INT2.
- Strong outlier-aware scalar baseline.
- QVG.
- HQMQ.
- Strongest additional screened family.
- Proposed method.

Compare at matched effective storage, matched paired-BF16 fidelity, matched independent video quality, matched peak VRAM, and matched runtime where feasible.

Use blinded pairwise human evaluation for overall quality, temporal consistency, subject identity, motion realism, and long-term coherence.

## 18. Required Ablations

The final paper should include ablations relevant to the selected method.

Candidate ablations:

- Representation: scalar versus vector, raw versus residual, original versus rotated, absolute versus temporal delta, shared versus separate K/V treatment.
- Quantization granularity: global, per-layer, per-head, per-token, per-channel, per-age band.
- Error objective: cache MSE, cosine error, logit error, attention-probability error, attention-output error, rollout error.
- Precision: INT8, INT4, INT3, INT2, mixed precision.
- Robustness: prompt seed, quantizer seed, codebook seed, motion category, rollout length, resolution, and model.

## 19. Statistical Requirements

Use prompt as the primary statistical unit.

Report mean paired difference, median paired difference, prompt-level bootstrap confidence interval, win rate, effect size, per-category results, and worst-case failure rate.
Do not treat many cache tensors from the same generated video as independent samples.

## 20. Failure Analysis

Inspect at least the worst 20 examples for each finalist.

Use a fixed taxonomy:

- Identity drift.
- Object disappearance.
- Object duplication.
- Geometry change.
- Motion freezing.
- Motion instability.
- Camera instability.
- Texture collapse.
- Color drift.
- Flicker.
- Scene-transition failure.
- Prompt deviation.

For each failure, record method, K/V treatment, layer or age sensitivity, quantization error, attention-output error, cache statistics, and whether BF16 also fails.

## 21. Stage-Level Decision Tree

After cache audit:

- Strong temporal redundancy: prioritize QVG and temporal coding.
- Favorable four-dimensional spherical geometry: prioritize HQMQ.
- Sparse extreme outliers: prioritize outlier-aware methods and rotations.
- Strong cross-dimensional covariance: prioritize vector or product quantization.
- No dominant structure: prioritize simple scalar baselines and functional sensitivity analysis.

After offline screening:

- One clear winner: advance it.
- Different winners for K and V: advance asymmetric policies.
- Different winners by layer or age: run oracle mixed allocation.
- No method beats INT4 meaningfully: study whether lower precision is intrinsically unstable before adding complexity.

After rollout experiments:

- Best cache metric also gives best videos: optimize that method directly.
- Functional error predicts videos better: design around functional error.
- Error compounds only at long horizons: focus on rollout-aware quantization.
- Errors are event-specific: develop event-aware precision or refresh.

After oracle allocation:

- Large oracle gain: develop adaptive or static allocation.
- Small oracle gain: use the best global method.
- Oracle relies on many codec switches: check metadata and implementation cost before pursuing multi-codec storage.

After real-storage evaluation:

- Quality and memory improve: proceed to final benchmark.
- Storage improves but peak VRAM does not: pursue compressed attention only if the quality result is strong.
- Runtime is unacceptable: profile encode, decode, attention, and memory movement separately.

## 22. Stop And Pivot Criteria

Stop a method family when it is dominated at matched actual storage, unstable across prompts or seeds, loses advantages under full rollouts, has metadata that removes nominal compression benefit, has prohibitive runtime without a plausible fused implementation, or lacks a mechanism suggesting repair.

Stop the full project direction when no method materially beats simple INT4, lower-bit methods consistently cause unacceptable long-horizon degradation, KV cache is not a meaningful memory bottleneck in target models, or compressed storage does not reduce peak VRAM and fused attention is infeasible.

Possible pivots include KV-cache pruning, cache summarization, sparse attention, external memory, joint quantization and eviction, weight and activation quantization, and faster attention systems.

## 23. First 72-Hour Plan

Day 1: infrastructure and codebase choice.
Deliver BF16 generation, cache-wrapper BF16 generation, cache tensor shape and layout report, cache read/write call graph, K-only and V-only perturbation, codebase-selection decision, and reproducible environment manifest.

Day 2: common baselines.
Implement uniform INT4, uniform INT2, clipped INT2, existing QVG path, and HQMQ reference path.
Deliver unit tests, storage accounting, snapshot reconstruction metrics, and encode/decode timing.

Day 3: first comparative result.
Run three prompts, one seed, BF16, INT4, INT2, QVG, HQMQ, K only, V only, and K and V.
Deliver generated videos, cache reconstruction metrics, attention-output metrics, paired-video metrics, effective storage, runtime, initial method-family ranking, and next-stage decision.

## 24. First-Week Deliverables

Produce:

1. A common codec API.
2. Reproducible BF16 evaluation.
3. Scalar, QVG, and HQMQ implementations.
4. Video KV-distribution audit.
5. Cache-level rate-distortion comparison.
6. Initial rollout comparison.
7. K versus V sensitivity report.
8. Prompt-category failure report.
9. Exact byte accounting.
10. A decision selecting no more than three method families for deeper study.

The first-week conclusion must not assume QVG, HQMQ, or their combination is the answer.

## 25. Agent Operating Protocol

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
