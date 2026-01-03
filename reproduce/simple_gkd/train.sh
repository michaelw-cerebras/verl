export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
ray stop -f || true

args=(
  --config-path=/workspace/mlf2/verl/recipe/gkd/config
  --config-name=on_policy_distill_trainer

  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync # default is sync and not in _ROLLOUT_REGISTRY in verl/verl/workers/rollout/base.py
  ++actor_rollout_ref.rollout.n=1
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.actor.ppo_mini_batch_size=32

  data.train_files=local_parquet_dir/train.parquet
  data.val_files=local_parquet_dir/test.parquet
  ++data.train_batch_size=64
  data.prompt_key=prompt

  trainer.total_epochs=1
  trainer.n_gpus_per_node=4
  rollout.n_gpus_per_node=1

  actor_rollout_ref.teacher.server_ip=127.0.0.1
  actor_rollout_ref.teacher.server_port=15555

  trainer.scheduler=one_step_off
  trainer.test_freq=5
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_reproduce
  trainer.experiment_name=gsm8k-gkd-qwen2p5_3b_to_0p5b

)

# Disable MoE related
args+=(
  ++actor_rollout_ref.actor.router_replay.mode=disabled
  ++actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=true
)

LOG="run_gkd_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# RAY_DEBUG=legacy \
# HYDRA_FULL_ERROR=1 \
# CUDA_VISIBLE_DEVICES=1,2,4,5,6 \
# python3 -u -m recipe.gkd.main_gkd "${args[@]}" 2>&1 | tee "$LOG" &

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=1,2,4,5,6 \
  python3 -u -m recipe.gkd.main_gkd "${args[@]}" \
  > "$LOG" 2>&1 < /dev/null &


PYTHON_PID=$!
echo $PYTHON_PID > "$PIDFILE"
echo "Started training. PID=$PYTHON_PID"
echo "View logs: tail -f $LOG"


# Before you run this, start the teacher server
# cd/workspace/mlf2/verl/recipe/gkd/teacher/ && bash start_server.sh



# To stop the background run
# kill $(cat run.pid) || true
# sleep 5
# kill -9 $(cat run.pid) || true
# ray stop -f || true