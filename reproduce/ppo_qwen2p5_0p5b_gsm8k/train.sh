export TOKENIZERS_PARALLELISM=false
ray stop -f || true

args=(
  algorithm.adv_estimator=gae
  data.train_files=local_parquet_dir/train.parquet
  data.val_files=local_parquet_dir/test.parquet
  data.train_batch_size=256
  data.max_prompt_length=1024
  data.max_response_length=512
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  critic.optim.lr=1e-5
  critic.model.use_remove_padding=False
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
  critic.model.enable_gradient_checkpointing=False
  critic.ppo_micro_batch_size_per_gpu=2
  critic.model.fsdp_config.param_offload=False
  critic.model.fsdp_config.optimizer_offload=False
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_reproduce
  trainer.experiment_name=gsm8k-ppo-qwen2p5-0p5b
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.save_freq=20
  trainer.test_freq=10
  trainer.total_epochs=15
  actor_rollout_ref.rollout.agent.num_workers=1
  data.dataloader_num_workers=0
  data.filter_overlong_prompts_workers=0
  ray_kwargs.ray_init.num_cpus=16
  # Uncomment the following 3 lines to use eager instead of flash_attention_v2
  # ++actor_rollout_ref.model.override_config.attn_implementation=eager
  # ++critic.model.override_config.attn_implementation=eager
  # ++reward_model.model.override_config.attn_implementation=eager
)

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m verl.trainer.main_ppo "${args[@]}" \
2>&1 | tee "run_ppo_gsm8k_$(date +%Y%m%d_%H%M%S).log"
