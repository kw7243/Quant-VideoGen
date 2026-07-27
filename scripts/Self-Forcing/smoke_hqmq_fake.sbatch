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
OUTPUT_FOLDER="${OUT_ROOT}/hqmq_fake_27f_${RUN_ID}"

export HF_HOME=/data/scratch-fast/kwen1/.cache/huggingface
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/pip
export PYTHONPATH=experiments/Self-Forcing:.
export TOKENIZERS_PARALLELISM=false
export DUMP_KV_LEVEL=1
export KV_DUMP_DIR="${OUTPUT_FOLDER}/kv_dumps"

mkdir -p "$OUTPUT_FOLDER" "$KV_DUMP_DIR"

hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

"$ENV_PYTHON" - <<'PY'
import importlib
import torch

print("python_import_check")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))

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
  --data_path assets/t2v.txt \
  --output_folder "$OUTPUT_FOLDER" \
  --num_samples 1 \
  --num_output_frames 27 \
  --local_attn_size 27 \
  --use_ema \
  --save_with_index \
  --quant_type hqmq-fake \
  --cache_num_k_centroids 256 \
  --cache_num_v_centroids 256 \
  --kmeans_max_iters 2 \
  --quant_block_size 64 \
  --num_prq_stages 1
