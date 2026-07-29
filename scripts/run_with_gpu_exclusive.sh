#!/usr/bin/env bash
# Stops a GPU-consuming systemd service (default: taco's llama-swap,
# llama-multi-models.service) before running the given command, restarting
# it afterward regardless of success/failure - frees the iGPU's VRAM/compute
# for exclusive use by the wrapped command. Override the service via
# GPU_SERVICE if another GPU consumer needs the same treatment later.
#
# Run this ON taco itself (it's a systemctl --user wrapper, not something
# that makes sense over ssh from elsewhere):
#   ./scripts/run_with_gpu_exclusive.sh python training/train_from_db.py --device cuda ...
#   GPU_SERVICE=some-other.service ./scripts/run_with_gpu_exclusive.sh ...
#
# IMPORTANT CAVEAT: stopping this service alone will NOT fix GPU training on
# taco - see docs/gpu-training-investigation.md for why (short version: it's
# a missing-kernel-support issue in the installed torch/ROCm build, not GPU
# contention). Kept anyway because freeing VRAM/compute for whatever CAN use
# the GPU is useful in its own right, and because stopping the service is
# exactly what was asked for - just don't expect it to unblock YOLO training
# by itself.
set -euo pipefail

SERVICE="${GPU_SERVICE:-llama-multi-models.service}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

echo "[gpu-exclusive] Stopping $SERVICE..."
systemctl --user stop "$SERVICE"

restart_service() {
  echo "[gpu-exclusive] Restarting $SERVICE..."
  systemctl --user start "$SERVICE"
}
trap restart_service EXIT

echo "[gpu-exclusive] Running: $*"
"$@"
