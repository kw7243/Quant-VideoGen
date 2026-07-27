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
