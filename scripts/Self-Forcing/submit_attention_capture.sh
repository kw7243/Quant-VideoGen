#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/data/scratch-fast/kwen1/firstmate/data/qvg-attention-capture-slurm-run
STATE_FILE=/data/scratch-fast/kwen1/firstmate/state/qvg-attention-capture-slurm-run.status
JOB_SCRIPT=scripts/Self-Forcing/attention_capture_bf16.sbatch
ACCOUNT=vision-torralba-urops-meng
QOS=vision-torralba-urops-meng
PARTITION=vision-torralba-h200

if [[ -z "${RESEARCH_REPRO_STAGED_DIR:-}" ]]; then
  echo "submit_attention_capture.sh must run through stage_and_run.sh" >&2
  exit 2
fi
if [[ "$(pwd)" != "$RESEARCH_REPRO_STAGED_DIR" ]]; then
  echo "expected cwd to be staged copy: $RESEARCH_REPRO_STAGED_DIR" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"/{logs,outputs,captures,capture_manifests,preflight} "$(dirname "$STATE_FILE")"
printf '%s\n' "working: staged Slurm preflight from ${RESEARCH_REPRO_STAGED_DIR} ($(date -Iseconds))" >> "$STATE_FILE"

echo "PRECHECK staged_copy=$RESEARCH_REPRO_STAGED_DIR"
echo "PRECHECK source_repo=${RESEARCH_REPRO_SOURCE_REPO:-unknown}"
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"

test -x /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python
test -x /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/torchrun
test -e /data/scratch-fast/kwen1/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B
test -e /data/scratch-fast/kwen1/qvg-self-forcing-assets/gdhe17-Self-Forcing/checkpoints/self_forcing_dmd.pt
test -f assets/t2v.txt
bash -n "$JOB_SCRIPT"

ASSOC_OUT="$DATA_ROOT/preflight/slurm_assoc_$(date +%Y%m%d_%H%M%S).txt"
SINFO_OUT="$DATA_ROOT/preflight/sinfo_${PARTITION}_$(date +%Y%m%d_%H%M%S).txt"
PARTITION_OUT="$DATA_ROOT/preflight/partition_${PARTITION}_$(date +%Y%m%d_%H%M%S).txt"
SBATCH_TEST_OUT="$DATA_ROOT/preflight/sbatch_test_$(date +%Y%m%d_%H%M%S).txt"

sacctmgr show assoc user=kwen1 format=User,Account,Partition,QOS%100 -P | tee "$ASSOC_OUT"
grep -F "kwen1|${ACCOUNT}" "$ASSOC_OUT" >/dev/null
grep -F "$QOS" "$ASSOC_OUT" >/dev/null

sinfo -p "$PARTITION" -o "%P %a %D %t %G %m %C %N" | tee "$SINFO_OUT"
grep -F "$PARTITION" "$SINFO_OUT" >/dev/null

scontrol show partition "$PARTITION" | tee "$PARTITION_OUT"
grep -F "AllowAccounts" "$PARTITION_OUT" >/dev/null || true

export RUN_LABEL_PREFIX=bf16_attncap_27f_prompt1
export NUM_OUTPUT_FRAMES=27
export LOCAL_ATTN_SIZE=27
export NUM_SAMPLES=1
export PROMPT_LIMIT=1
export QUANT_TYPE=none
export DUMP_KV_LEVEL=0
export QVG_ATTENTION_PROBE_ENABLED=1
export QVG_ATTENTION_PROBE_LAYERS=27,28,29
export QVG_ATTENTION_PROBE_HEADS=0,9
export QVG_ATTENTION_PROBE_CHUNKS=24,25,26
export QVG_ATTENTION_PROBE_MAX_FILE_BYTES=67108864
export QVG_ATTENTION_PROBE_VARIANT=bf16_reference

echo "SBATCH_REQUEST account=$ACCOUNT qos=$QOS partition=$PARTITION gres=gpu:1 cpus=8 mem=96G time=01:00:00"
sbatch --test-only --chdir="$PWD" "$JOB_SCRIPT" | tee "$SBATCH_TEST_OUT"

SBATCH_OUT="$(sbatch --chdir="$PWD" "$JOB_SCRIPT")"
echo "$SBATCH_OUT"
JOB_ID="$(awk '/Submitted batch job/ {print $4}' <<< "$SBATCH_OUT")"
if [[ -z "$JOB_ID" ]]; then
  echo "Unable to parse Slurm job id from: $SBATCH_OUT" >&2
  exit 3
fi

SUBMISSION_JSON="$DATA_ROOT/submission_${JOB_ID}.json"
cat > "$SUBMISSION_JSON" <<JSON
{
  "job_id": "$JOB_ID",
  "staged_copy": "$RESEARCH_REPRO_STAGED_DIR",
  "source_repo": "${RESEARCH_REPRO_SOURCE_REPO:-}",
  "job_script": "$JOB_SCRIPT",
  "account": "$ACCOUNT",
  "qos": "$QOS",
  "partition": "$PARTITION",
  "gres": "gpu:1",
  "cpus_per_task": 8,
  "mem": "96G",
  "time_limit": "01:00:00",
  "log_stdout": "$DATA_ROOT/logs/qvg_sf_attncap-${JOB_ID}.out",
  "log_stderr": "$DATA_ROOT/logs/qvg_sf_attncap-${JOB_ID}.err",
  "output_dir": "$DATA_ROOT/outputs/bf16_attncap_27f_prompt1_${JOB_ID}",
  "capture_dir": "$DATA_ROOT/captures/bf16_attncap_27f_prompt1_${JOB_ID}",
  "capture_manifest": "$DATA_ROOT/capture_manifests/bf16_attncap_27f_prompt1_${JOB_ID}.json",
  "quant_type": "none",
  "prompt_limit": 1,
  "num_output_frames": 27,
  "local_attn_size": 27,
  "attention_probe": {
    "enabled": 1,
    "layers": "27,28,29",
    "heads": "0,9",
    "chunks": "24,25,26",
    "max_file_bytes": 67108864,
    "variant": "bf16_reference"
  },
  "preflight": {
    "assoc": "$ASSOC_OUT",
    "sinfo": "$SINFO_OUT",
    "partition": "$PARTITION_OUT",
    "sbatch_test": "$SBATCH_TEST_OUT"
  }
}
JSON

printf '%s\n' "working: submitted Slurm job ${JOB_ID} for bounded attention capture ($(date -Iseconds))" >> "$STATE_FILE"
echo "SUBMISSION_JSON=$SUBMISSION_JSON"
