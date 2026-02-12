#!/bin/bash
# GSM8K Evaluation Script
#
# This script runs validation-only mode (no training) to evaluate a model on GSM8K.
#
# Usage:
#   bash examples/ppo_trainer/evaluate_gsm8k.sh
#
# Or with overrides:
#   bash examples/ppo_trainer/evaluate_gsm8k.sh \
#       actor_rollout_ref.model.path=/path/to/checkpoint \
#       actor_rollout_ref.rollout.val_kwargs.temperature=0.7
#

set -x

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run evaluation using the eval-only config
python3 -m verl.trainer.main_ppo \
    --config-path=${SCRIPT_DIR} \
    --config-name=gsm8k_eval_only \
    "$@"
