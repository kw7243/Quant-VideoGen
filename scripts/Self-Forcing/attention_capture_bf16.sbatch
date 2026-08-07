#!/usr/bin/env bash
#SBATCH --job-name=qvg_sf_attncap
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --qos=vision-torralba-urops-meng
#SBATCH --partition=vision-torralba-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/logs/%x-%j.out
#SBATCH --error=/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run/logs/%x-%j.err

set -euo pipefail

ENV_PYTHON=/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python
ENV_TORCHRUN=/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/torchrun
DATA_ROOT=/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run
MODEL_ROOT=/data/scratch-fast/kwen1/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B
DMD_CKPT=/data/scratch-fast/kwen1/qvg-self-forcing-assets/gdhe17-Self-Forcing/checkpoints/self_forcing_dmd.pt

RUN_ID="${SLURM_JOB_ID:-manual}"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-bf16_attncap_27f_prompt1}"
RUN_LABEL="${RUN_LABEL_PREFIX}_${RUN_ID}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${DATA_ROOT}/outputs/${RUN_LABEL}}"
CAPTURE_DIR="${QVG_ATTENTION_PROBE_DIR:-${DATA_ROOT}/captures/${RUN_LABEL}}"
CAPTURE_MANIFEST="${CAPTURE_MANIFEST:-${DATA_ROOT}/capture_manifests/${RUN_LABEL}.json}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-27}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-27}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
DATA_PATH="${DATA_PATH:-assets/t2v.txt}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"
QUANT_TYPE="${QUANT_TYPE:-none}"
CACHE_NUM_K_CENTROIDS="${CACHE_NUM_K_CENTROIDS:-256}"
CACHE_NUM_V_CENTROIDS="${CACHE_NUM_V_CENTROIDS:-256}"
KMEANS_MAX_ITERS="${KMEANS_MAX_ITERS:-2}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-64}"
NUM_PRQ_STAGES="${NUM_PRQ_STAGES:-1}"

mkdir -p ckpts/Self-Forcing
ln -sfn "$MODEL_ROOT" ckpts/Self-Forcing/Wan2.1-T2V-1.3B
ln -sfn "$DMD_CKPT" ckpts/Self-Forcing/self_forcing_dmd.pt

export HF_HOME=/data/scratch-fast/kwen1/.cache/huggingface
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/pip
export PYTHONPATH=experiments/Self-Forcing:.
export TOKENIZERS_PARALLELISM=false
export NUM_OUTPUT_FRAMES LOCAL_ATTN_SIZE NUM_SAMPLES DATA_PATH PROMPT_LIMIT QUANT_TYPE
export CACHE_NUM_K_CENTROIDS CACHE_NUM_V_CENTROIDS KMEANS_MAX_ITERS
export QUANT_BLOCK_SIZE NUM_PRQ_STAGES
export DUMP_KV_LEVEL="${DUMP_KV_LEVEL:-0}"
export QVG_ATTENTION_PROBE_ENABLED=1
export QVG_ATTENTION_PROBE_DIR="$CAPTURE_DIR"
export QVG_ATTENTION_PROBE_LAYERS=27,28,29
export QVG_ATTENTION_PROBE_HEADS=0,9
export QVG_ATTENTION_PROBE_CHUNKS=24,25,26
export QVG_ATTENTION_PROBE_MAX_FILE_BYTES=67108864
export QVG_ATTENTION_PROBE_VARIANT="${QVG_ATTENTION_PROBE_VARIANT:-bf16_reference}"

mkdir -p "$OUTPUT_FOLDER" "$CAPTURE_DIR" "$(dirname "$CAPTURE_MANIFEST")"

RUN_DATA_PATH="$DATA_PATH"
if [[ -n "$PROMPT_LIMIT" ]]; then
  RUN_DATA_PATH="${OUTPUT_FOLDER}/prompts_first_${PROMPT_LIMIT}.txt"
  head -n "$PROMPT_LIMIT" "$DATA_PATH" > "$RUN_DATA_PATH"
fi

hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME:-}"
printf 'SLURM_SUBMIT_DIR=%s\n' "${SLURM_SUBMIT_DIR:-}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
printf 'RESEARCH_REPRO_STAGED_DIR=%s\n' "${RESEARCH_REPRO_STAGED_DIR:-}"
printf 'RUN_CONFIG output_folder=%s\n' "$OUTPUT_FOLDER"
printf 'RUN_CONFIG capture_dir=%s\n' "$CAPTURE_DIR"
printf 'RUN_CONFIG capture_manifest=%s\n' "$CAPTURE_MANIFEST"
printf 'RUN_CONFIG num_output_frames=%s local_attn_size=%s num_samples=%s\n' \
  "$NUM_OUTPUT_FRAMES" "$LOCAL_ATTN_SIZE" "$NUM_SAMPLES"
printf 'RUN_CONFIG quant_type=%s data_path=%s prompt_limit=%s dump_kv_level=%s\n' \
  "$QUANT_TYPE" "$RUN_DATA_PATH" "${PROMPT_LIMIT:-none}" "$DUMP_KV_LEVEL"
printf 'ATTENTION_PROBE enabled=%s layers=%s heads=%s chunks=%s max_file_bytes=%s variant=%s\n' \
  "$QVG_ATTENTION_PROBE_ENABLED" "$QVG_ATTENTION_PROBE_LAYERS" \
  "$QVG_ATTENTION_PROBE_HEADS" "$QVG_ATTENTION_PROBE_CHUNKS" \
  "$QVG_ATTENTION_PROBE_MAX_FILE_BYTES" "$QVG_ATTENTION_PROBE_VARIANT"

free -h
df -h "$DATA_ROOT" "$HF_HOME" || true
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

"$ENV_PYTHON" - <<'PY'
import importlib
import os
from pathlib import Path

import torch


def meminfo_gb(key: str) -> float:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(key):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


print("python_import_check")
print("cwd", os.getcwd())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device_name", torch.cuda.get_device_name(0))
    print("device_total_vram_gb", props.total_memory / 1024**3)

print("host_mem_total_gb", meminfo_gb("MemTotal:"))
print("host_mem_available_gb", meminfo_gb("MemAvailable:"))
print("checkpoint_exists", Path("ckpts/Self-Forcing/self_forcing_dmd.pt").exists())
print("wan_model_exists", Path("ckpts/Self-Forcing/Wan2.1-T2V-1.3B").exists())

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
    "tools.attention_probe_capture",
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

"$ENV_PYTHON" - "$CAPTURE_DIR" "$CAPTURE_MANIFEST" <<'PY'
import inspect
import json
import sys
from pathlib import Path

import torch


capture_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])


def load_capture(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(str(path), **kwargs)


def shape(value):
    return list(value.shape) if torch.is_tensor(value) else None


captures = []
for path in sorted(capture_dir.glob("*.pt")):
    obj = load_capture(path)
    records = obj.get("records", []) if isinstance(obj, dict) else []
    record_summaries = []
    for record in records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {})
        variants = record.get("variants")
        variant_names = None
        if isinstance(variants, dict):
            variant_names = [str(name) for name in variants.keys()]
        elif isinstance(variants, list):
            variant_names = [
                str(item.get("name", f"variant_{idx}")) if isinstance(item, dict) else str(item)
                for idx, item in enumerate(variants)
            ]
        record_summaries.append({
            "name": record.get("name"),
            "layer_idx": metadata.get("layer_idx"),
            "chunk_idx": metadata.get("chunk_idx"),
            "selected_heads": metadata.get("selected_heads"),
            "selected_chunks": metadata.get("selected_chunks"),
            "cached_chunk_indices": metadata.get("cached_chunk_indices"),
            "current_chunk_indices": metadata.get("current_chunk_indices"),
            "selected_cached_tokens": metadata.get("selected_cached_tokens"),
            "selected_current_tokens": metadata.get("selected_current_tokens"),
            "query_shape": shape(record.get("query")),
            "cached_key_ref_shape": shape(record.get("cached_key_ref")),
            "current_key_shape": shape(record.get("current_key")),
            "has_variants": bool(record.get("variants")),
            "variant_names": variant_names,
        })
    captures.append({
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "record_count": len(record_summaries),
        "records": record_summaries,
    })

summary = {
    "schema_version": "qvg-attention-capture-manifest.v1",
    "capture_dir": str(capture_dir.resolve()),
    "capture_file_count": len(captures),
    "capture_total_size_bytes": sum(item["size_bytes"] for item in captures),
    "captures": captures,
}
manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"capture_manifest={manifest_path}")
print(f"capture_file_count={summary['capture_file_count']}")
if not captures:
    raise SystemExit("no attention capture artifacts were written")
PY
