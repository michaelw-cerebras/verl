HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone \
--nnodes=1 \
--nproc_per_node=4 \
-m verl.trainer.fsdp_sft_trainer \
  data.train_files=local_parquet_dir/train.parquet \
  data.val_files=local_parquet_dir/test.parquet \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  +data.prompt_dict_keys=[question] \
  +data.response_dict_keys=[answer] \
  data.micro_batch_size=8 \
  model.partial_pretrain=google/gemma-2b-it \
  ++model.override_config.attn_implementation=eager \
  trainer.default_local_dir=sft_gemma2_ckpt \
  trainer.project_name=mw_verl_reproduce \
  trainer.total_epochs=2 \
  "trainer.logger=[console,wandb]" \
  trainer.default_hdfs_dir=null \
  trainer.experiment_name=mw_verl_reproduce_gsm8k-sft-gemma-2-2b-it


# Need to override attention implementation to eager for gemma model
# nnodes = how many machines (physical or VM nodes)
# nproc_per_node = how many processes per machine (almost always = number of GPUs on that machine)