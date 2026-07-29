#!/usr/bin/env bash
#SBATCH --job-name=qvg_sf_hqmq_smoke
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --qos=vision-torralba-urops-meng
#SBATCH --partition=vision-torralba-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/%x-%j.out
#SBATCH --error=/data/scratch-fast/kwen1/qvg-self-forcing-smoke-logs/%x-%j.err

set -euo pipefail

ENV_PYTHON=/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python
ENV_TORCHRUN=/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/torchrun
OUT_ROOT=/data/scratch-fast/kwen1/qvg-self-forcing-smoke-results
RUN_ID="${SLURM_JOB_ID:-manual}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-27}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-$NUM_OUTPUT_FRAMES}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
DATA_PATH="${DATA_PATH:-assets/t2v.txt}"
PROMPT_LIMIT="${PROMPT_LIMIT:-}"
QUANT_TYPE="${QUANT_TYPE:-hqmq-fake}"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-hqmq_fake_${NUM_OUTPUT_FRAMES}f}"
OUTPUT_FOLDER="${OUT_ROOT}/${RUN_LABEL_PREFIX}_${RUN_ID}"
CACHE_NUM_K_CENTROIDS="${CACHE_NUM_K_CENTROIDS:-256}"
CACHE_NUM_V_CENTROIDS="${CACHE_NUM_V_CENTROIDS:-256}"
KMEANS_MAX_ITERS="${KMEANS_MAX_ITERS:-2}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-64}"
NUM_PRQ_STAGES="${NUM_PRQ_STAGES:-1}"

export NUM_OUTPUT_FRAMES LOCAL_ATTN_SIZE NUM_SAMPLES DATA_PATH PROMPT_LIMIT QUANT_TYPE
export CACHE_NUM_K_CENTROIDS CACHE_NUM_V_CENTROIDS KMEANS_MAX_ITERS
export QUANT_BLOCK_SIZE NUM_PRQ_STAGES
export HF_HOME=/data/scratch-fast/kwen1/.cache/huggingface
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/pip
export PYTHONPATH=experiments/Self-Forcing:.
export TOKENIZERS_PARALLELISM=false
export DUMP_KV_LEVEL="${DUMP_KV_LEVEL:-1}"
export KV_DUMP_DIR="${OUTPUT_FOLDER}/kv_dumps"

mkdir -p "$OUTPUT_FOLDER" "$KV_DUMP_DIR"

RUN_DATA_PATH="$DATA_PATH"
if [ -n "$PROMPT_LIMIT" ]; then
  RUN_DATA_PATH="${OUTPUT_FOLDER}/prompts_first_${PROMPT_LIMIT}.txt"
  head -n "$PROMPT_LIMIT" "$DATA_PATH" > "$RUN_DATA_PATH"
fi

hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SMOKE_CONFIG output_folder=${OUTPUT_FOLDER}"
echo "SMOKE_CONFIG num_output_frames=${NUM_OUTPUT_FRAMES} local_attn_size=${LOCAL_ATTN_SIZE} num_samples=${NUM_SAMPLES}"
echo "SMOKE_CONFIG quant_type=${QUANT_TYPE} data_path=${RUN_DATA_PATH} prompt_limit=${PROMPT_LIMIT:-none} dump_kv_level=${DUMP_KV_LEVEL}"
echo "SMOKE_CONFIG cache_num_k_centroids=${CACHE_NUM_K_CENTROIDS} cache_num_v_centroids=${CACHE_NUM_V_CENTROIDS} kmeans_max_iters=${KMEANS_MAX_ITERS} quant_block_size=${QUANT_BLOCK_SIZE} num_prq_stages=${NUM_PRQ_STAGES}"
free -h
df -h "$OUT_ROOT" "$HF_HOME" || true
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

"$ENV_PYTHON" - <<'PY'
import importlib
import os
import torch

def _meminfo_gb(key):
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(key):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")

print("python_import_check")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("device_total_vram_gb", props.total_memory / 1024**3)

print("host_mem_total_gb", _meminfo_gb("MemTotal:"))
print("host_mem_available_gb", _meminfo_gb("MemAvailable:"))

frames = int(os.environ["NUM_OUTPUT_FRAMES"])
local_attn = int(os.environ["LOCAL_ATTN_SIZE"])
baseline_local_attn = 27
baseline_kv_mb = 7403.94
baseline_other_mb = 21919.91
scale = max(local_attn, 1) / baseline_local_attn
estimated_kv_mb = baseline_kv_mb * scale
estimated_peak_mb = baseline_other_mb + estimated_kv_mb
print(
    "memory_estimate "
    f"based_on_job=1213946 output_frames={frames} local_attn_size={local_attn} "
    f"estimated_kv_cache_mb={estimated_kv_mb:.2f} "
    f"estimated_peak_vram_mb={estimated_peak_mb:.2f} "
    f"estimated_peak_vram_with_25pct_margin_mb={estimated_peak_mb * 1.25:.2f} "
    "previous_27f_batch_maxrss_gb=56.97 requested_host_mem_gb=96"
)

for name in [
    "torchvision",
    "omegaconf",
    "einops",
    "termcolor",
    "diffusers",
    "transformers",
    "safetensors",
    "accelerate",
    "easydict",
    "lmdb",
    "av",
    "quant_videogen.compress",
    "utils.wan_wrapper",
    "pipeline.causal_inference",
]:
    importlib.import_module(name)
    print(name, "OK")
PY

"$ENV_TORCHRUN" --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path "$RUN_DATA_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_samples "$NUM_SAMPLES" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --local_attn_size "$LOCAL_ATTN_SIZE" \
  --use_ema \
  --save_with_index \
  --quant_type "$QUANT_TYPE" \
  --cache_num_k_centroids "$CACHE_NUM_K_CENTROIDS" \
  --cache_num_v_centroids "$CACHE_NUM_V_CENTROIDS" \
  --kmeans_max_iters "$KMEANS_MAX_ITERS" \
  --quant_block_size "$QUANT_BLOCK_SIZE" \
  --num_prq_stages "$NUM_PRQ_STAGES"
