export TOKENIZERS_PARALLELISM=false
ray stop -f || true

args=(
  --config-path=/home/michaelw/mlf2/verl/recipe/gkd/config
  --config-name=on_policy_distill_trainer

  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async # default is sync and not in _ROLLOUT_REGISTRY in verl/verl/workers/rollout/base.py
  ++actor_rollout_ref.rollout.n=1
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  ++actor_rollout_ref.actor.ppo_mini_batch_size=64

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
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_reproduce
  trainer.experiment_name=gsm8k-gkd-qwen2p5_3b_to_0p5b


)

# Disable MoE related
args+=(
  ++actor_rollout_ref.actor.router_replay.mode=disabled
  ++actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=true
)




HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,5 \
python3 -m recipe.gkd.main_gkd "${args[@]}" \
2>&1 | tee "run_gkd_gsm8k_$(date +%Y%m%d_%H%M%S).log"

# Before you run this, start the teacher server
# cd recipe/gkd/teacher
# bash start_server.sh
