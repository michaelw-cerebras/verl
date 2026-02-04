#!/usr/bin/env bash
set -xeuo pipefail

ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p "$RAY_TMPDIR"

rm -f run*
rm -rf outputs 
rm -rf checkpoints
rm -rf wandb

ray stop -f || true

LOG="run_grpo_fsdp2_qwen3_0p6b_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# ====== Long-context knobs ======
max_prompt_length=512
max_response_length=12288
max_total_length=$((max_prompt_length + max_response_length))

# vLLM: keep batched tokens tight to avoid KV spikes
vllm_max_batched_tokens=$((max_total_length))         # key change vs your 65536
train_max_token_len_per_gpu=$((max_total_length))     # PPO token cap per GPU
infer_max_token_len_per_gpu=$((2 * max_total_length)) # logprob/infer cap (conservative)

args=(
  # Algorithm
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False

  # Data
  data.train_files=/workspace/mlf2/verl/reproduce/data/openthoughts3/local_parquet_dir/train.parquet
  data.val_files=/workspace/mlf2/verl/reproduce/data/openthoughts3/local_parquet_dir/test.parquet
  data.train_batch_size=12
  data.max_prompt_length=${max_prompt_length}
  data.max_response_length=${max_response_length}
  data.filter_overlong_prompts=True
  data.truncation=left
  data.train_max_samples=4800
  data.val_max_samples=500
  data.val_batch_size=16

  # Model
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B
  actor_rollout_ref.model.use_remove_padding=True
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true

  # ====== Switch backend to FSDP2 ======
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2

  # FSDP2 memory savers (enable first; if perf too slow, turn off optimizer_offload)
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  # Long-context activation control
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=2
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192

  # PPO sizes — let dynamic bsz handle tokens; keep micro small to avoid spikes
  actor_rollout_ref.actor.ppo_mini_batch_size=6
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_max_token_len_per_gpu}

  # Optim
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.lr_warmup_steps=20

  # KL / entropy
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.05
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.005

  # Rollout (vLLM)
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_batched_tokens}
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  actor_rollout_ref.rollout.n=8 # This is important!
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.top_p=1.0
  ++actor_rollout_ref.rollout.stop_token_ids='[151645,151643]'

  # Logprob micro — keep tiny for long seq
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=null
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=null

  # Reward manager & overlong buffer
  # reward_model.reward_manager=dapo
  # +reward_model.reward_kwargs.max_resp_len=${max_response_length}
  # +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
  # +reward_model.reward_kwargs.overlong_buffer_cfg.len=$((max_response_length/4))
  # +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.3
  # +reward_model.reward_kwargs.overlong_buffer_cfg.log=True

  # Trainer
  trainer.critic_warmup=0
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_recipe_reasoning
  trainer.experiment_name=openthoughts-grpo-qwen3_0p6b_fsdp2_n8_no_length_penalty
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.save_freq=100
  trainer.test_freq=50
  trainer.total_epochs=1
  trainer.val_before_train=True
)

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -u -m verl.trainer.main_ppo "${args[@]}" \
> "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDFILE"
echo "Started training. PID=$(cat $PIDFILE)"
echo "View logs: tail -f $LOG"


# To stop the background run
# kill $(cat run.pid) || true
# sleep 5
# kill -9 $(cat run.pid) || true
# ray stop -f || true