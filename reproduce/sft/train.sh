#!/usr/bin/env bash
set -euo pipefail

ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"   
export RAY_TMPDIR=/workspace/ray_tmp 
mkdir -p $RAY_TMPDIR

rm -f run*
rm -rf outputs 
rm -rf checkpoints

args=(
  # ======================
  # Data
  # ======================
  data.train_files=/workspace/mlf2/verl/reproduce/data/openthoughts3/local_parquet_dir/train.parquet
  data.val_files=/workspace/mlf2/verl/reproduce/data/openthoughts3/local_parquet_dir/test.parquet

  data.prompt_key=extra_info
  data.response_key=extra_info
  "+data.prompt_dict_keys=[question]"
  "+data.response_dict_keys=[answer]"
  data.micro_batch_size_per_gpu=1
  ++data.train_max_samples=5120
  ++data.val_max_samples=100
  ++data.train_batch_size=32
  ++data.max_length=20480
  

  # ======================
  # Model
  # ======================
  model.partial_pretrain=Qwen/Qwen3-0.6B
  ++model.strategy=fsdp2
  ++model.fsdp_config.model_dtype=bf16

  # ======================
  # Trainer
  # ======================
  trainer.project_name=mw_verl_recipe_reasoning
  trainer.experiment_name=openthoughts-sft-qwen3_0p6b
  trainer.total_epochs=1
  trainer.logger='["console","wandb"]'
  trainer.test_freq=50
  trainer.save_freq=50
  ++trainer.n_gpus_per_node=4

  # ============================================
  # Validation / Accuracy
  # Comment out below to disable 
  # validaiton acc during training
  # ⚠️ Acc Eval can be very slow for thinking models
  # ============================================
  # "+trainer.compute_score_module=verl.utils.reward_score.gsm8k"
  # "+trainer.ground_truth_key=[reward_model,ground_truth]"
  # "+trainer.compute_score_kwargs.method=strict"
  # "+trainer.accuracy_val_max_samples=50"
  # "+trainer.gen_batch_size=1"
  # "+trainer.accuracy_max_new_tokens=10240"
)

LOG="run_sft_gsm8k_qwen3_4b_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  "${args[@]}" \
  > "$LOG" 2>&1 < /dev/null &

PYTHON_PID=$!
echo $PYTHON_PID > "$PIDFILE"
echo "Started SFT. PID=$PYTHON_PID"
echo "View logs: tail -f $LOG"



# To stop the background run
# pkill -f "verl.trainer.fsdp_sft_trainer"
# sleep 2
# pkill -9 -f "verl.trainer.fsdp_sft_trainer"
# kill -9 $(cat run.pid) || true
# ray stop -f || true