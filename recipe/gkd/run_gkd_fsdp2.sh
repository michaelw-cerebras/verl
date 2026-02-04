#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
# GKD Training Script (FSDP2 Backend)
# ============================================================
# Distill Qwen3-8B teacher to Qwen3-0.6B student
#
# Prerequisites:
#   1. Start teacher server first:
#      cd recipe/gkd/teacher && bash start_server.sh
#
# Usage:
#   bash recipe/gkd/run_gkd_fsdp2.sh
#
# To override parameters:
#   bash recipe/gkd/run_gkd_fsdp2.sh trainer.n_gpus_per_node=4

# ============================================================
# Environment Setup
# ============================================================
ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p "$RAY_TMPDIR"

# Cleanup previous runs
rm -f run*
rm -rf outputs
rm -rf checkpoints
rm -rf wandb

ray stop -f || true

# ============================================================
# Logging
# ============================================================
LOG="run_gkd_fsdp2_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# ============================================================
# GPUs to use (modify as needed)
# ============================================================
export CUDA_VISIBLE_DEVICES=4,5,6

# ============================================================
# Run Training
# ============================================================
# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$VERL_ROOT"

nohup \
env HYDRA_FULL_ERROR=1 \
python3 -u -m verl.trainer.main_ppo \
  --config-path "$SCRIPT_DIR/config" \
  --config-name gkd_qwen3_8b_to_0p6b \
  "$@" \
> "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDFILE"
echo "============================================================"
echo "Started GKD training"
echo "  PID: $(cat $PIDFILE)"
echo "  Log: tail -f $LOG"
echo "============================================================"
echo ""
echo "NOTE: Make sure teacher server is running!"
echo "      cd recipe/gkd/teacher && bash start_server.sh"
echo ""
echo "To stop:"
echo "  kill \$(cat run.pid)"
echo "  ray stop -f"
