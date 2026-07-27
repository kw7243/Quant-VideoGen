from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    """Configuration for model quantization settings."""

    quant_type: str = "none"
    """Quantization type: 'none', 'naive-fp4', 'kmeans-fp4'."""

    # KV cache quantization parameters
    cache_num_k_centroids: int = 256
    """Number of K-Means centroids for K tensor (used in kmeans and nstages-kmeans)."""

    cache_num_v_centroids: int = 256
    """Number of K-Means centroids for V tensor (used in kmeans and nstages-kmeans)."""

    kmeans_max_iters: int = 100
    """Maximum iterations for K-Means clustering."""

    quant_block_size: int = 16
    """Block size for quantization."""

    # PRQ (nstages-kmeans) specific parameters
    num_prq_stages: int = 4
    """Number of PRQ stages for nstages-kmeans quantization."""

    asymmetric: bool = False
    """Asymmetric residual quantization (per-block min-max + zero point) instead
    of symmetric absmax. Symmetric int2 only uses {-1, 0, 1}, i.e. 3 of 4 codes;
    asymmetric spans the full [0, 2**n - 1] range at the cost of storing a zero
    point next to each scale."""

    # HQMQ fake-quant adapter parameters. These are ignored by existing QVG
    # methods and do not imply packed-cache memory savings.
    hqmq_secondary_size: int = 96
    """Secondary quaternion codebook size for hqmq-fake."""

    hqmq_radius_bits: int = 4
    """Scalar radius quantization bits for hqmq-fake."""

    hqmq_seed: int = 42
    """Seed for deterministic fake HQMQ secondary codebooks."""

    hqmq_pad_head_dim: bool = True
    """Pad non-multiple-of-4 head dimensions before HQMQ fake quantization."""

    hqmq_log_radius: bool = False
    """Use log-spaced radius quantization for hqmq-fake."""

    hqmq_median_mult: float | None = None
    """Median-multiplier outlier restore threshold; hqmq-fake-med3x defaults to 3.0."""

    # ---- Chunk-level quantization policy (see experiments/LingBot-v2) ----
    quant_factor: int = 8
    """Quantize the previous `quant_factor` chunks once every `quant_factor`
    chunks (Self-Forcing QUANT_FACTOR convention). 1 == quantize every chunk."""

    quant_keep_recent_chunks: int = 0
    """Completed chunks held in BF16 before quantization reaches them, so the
    chunk being denoised always attends to clean recent history."""

    quant_sink_keep_chunks: int = 0
    """Leading attention-sink chunks never quantized. The first sink chunk holds
    the initial conditioning frame that every later chunk attends to, so
    quantization error there never decays; 1 is recommended."""
