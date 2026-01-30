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

LOG="run_gkd_grpo_fsdp2_qwen3_0p6b_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# ====== Sequence length knobs ======
max_prompt_length=512
max_response_length=4096
max_total_length=$((max_prompt_length + max_response_length))

# vLLM settings
vllm_max_batched_tokens=$((max_total_length))
train_max_token_len_per_gpu=$((max_total_length))
infer_max_token_len_per_gpu=$((2 * max_total_length))

# ====== Teacher server settings ======
# Make sure teacher server is running before starting training!
# cd recipe/gkd/teacher && bash start_server.sh
teacher_ip="127.0.0.1"
teacher_port=15555
teacher_workers=1

# ====== GKD + GRPO specific settings ======
rollout_n=4                      # Number of responses per prompt for GRPO
gkd_select_strategy="best"       # all / random / best / worst
gkd_select_k=1                   # Number of responses to apply teacher KL

args=(
  # ====== Algorithm: GRPO ======
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False

  # ====== GKD Configuration ======
  actor_rollout_ref.actor.use_teacher_kl_loss=True
  actor_rollout_ref.actor.gkd_only_mode=False  # Combined mode: GRPO + GKD
  actor_rollout_ref.actor.teacher_kl_coef=0.5  # Balance between GRPO and GKD
  actor_rollout_ref.actor.teacher_kl_temperature=1.0

  # GKD selection strategy (for GRPO + GKD mode)
  actor_rollout_ref.actor.gkd_select_strategy=${gkd_select_strategy}
  actor_rollout_ref.actor.gkd_select_k=${gkd_select_k}

  # Teacher server
  +actor_rollout_ref.teacher.server_ip=${teacher_ip}
  +actor_rollout_ref.teacher.server_port=${teacher_port}
  +actor_rollout_ref.teacher.n_server_workers=${teacher_workers}
  +actor_rollout_ref.teacher.temperature=1.0

  # ====== Data ======
  data.train_files=/workspace/data/train.parquet
  data.val_files=/workspace/data/test.parquet
  data.train_batch_size=12
  data.max_prompt_length=${max_prompt_length}
  data.max_response_length=${max_response_length}
  data.filter_overlong_prompts=True
  data.truncation=left
  data.train_max_samples=4800
  data.val_max_samples=200
  data.val_batch_size=16

  # ====== Model ======
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B
  actor_rollout_ref.model.use_remove_padding=False  # GKD needs logits
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true

  # ====== FSDP2 Backend ======
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2

  # FSDP2 memory optimization
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  # Dynamic batch size
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=2

  # PPO batch sizes
  actor_rollout_ref.actor.ppo_mini_batch_size=6
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_max_token_len_per_gpu}

  # ====== Optimizer ======
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.lr_warmup_steps=20

  # ====== KL / entropy (for GRPO) ======
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.05
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.005

  # ====== Rollout (vLLM) ======
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_batched_tokens}
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  actor_rollout_ref.rollout.n=${rollout_n}  # Multiple responses for GRPO
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  ++actor_rollout_ref.rollout.stop_token_ids='[151645,151643]'

  # Logprob micro batch
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1

  # ====== Reward ======
  reward_model.reward_manager=dapo
  +reward_model.reward_kwargs.max_resp_len=${max_response_length}

  # ====== Trainer ======
  trainer.critic_warmup=0
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_gkd
  trainer.experiment_name=gkd-grpo-qwen3_0p6b_fsdp2
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.save_freq=100
  trainer.test_freq=100
  trainer.total_epochs=2
  trainer.val_before_train=True
)

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -u -m verl.trainer.main_ppo "${args[@]}" \
> "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDFILE"
echo "Started GKD+GRPO training. PID=$(cat $PIDFILE)"
echo "View logs: tail -f $LOG"
echo ""
echo "NOTE: Make sure teacher server is running!"
echo "      cd recipe/gkd/teacher && bash start_server.sh"
echo ""
echo "GKD Settings:"
echo "  - Selection strategy: ${gkd_select_strategy}"
echo "  - Select k: ${gkd_select_k} out of ${rollout_n} responses"


# To stop the background run
# kill $(cat run.pid) || true
# sleep 5
# kill -9 $(cat run.pid) || true
# ray stop -f || true
