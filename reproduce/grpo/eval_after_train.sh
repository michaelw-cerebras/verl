GPUS="4,5,6,7"
export CUDA_VISIBLE_DEVICES="$GPUS"

rm -rf wandb
rm -f eval*log

echo "[pre] HARD reset GPUs: $GPUS"
for g in ${GPUS//,/ }; do
  nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader,nounits \
  | xargs -r kill -9
done
sleep 2

export VLLM_GPU_MEMORY_UTILIZATION=0.75
export TOKENIZERS_PARALLELISM=false

LOG="eval_$(date +%Y%m%d_%H%M%S).log"
echo "[run] Log: $LOG"

nohup python evaluation.py \
  --checkpoint-dir /workspace/mlf2/verl/reproduce/grpo/checkpoints/mw_verl_recipe_reasoning/openthoughts-grpo-qwen3_0p6b_fsdp2 \
  --step0-model Qwen/Qwen3-0.6B \
  --validation-parquet /workspace/mlf2/verl/reproduce/data/openthoughts3/local_parquet_dir/test.parquet \
  --num-samples 500 \
  --seed 2026 \
  --temperature 0.6 \
  --top-p 0.95 \
  --batch-size 4 \
  --pass-k 3 \
  --vllm-tp-size 1 \
  --vllm-dp-size 4 \
  --max-tokens 20480 \
  --max-concurrent-requests 32 \
  --wandb-project mw_verl_recipe_reasoning \
  --wandb-run-name openthoughts-grpo-qwen3_0p6b_fsdp2 \
  --wandb-run-id 5rfdoe9p \
  --use-vllm-serve \
  > "$LOG" 2>&1 &

disown
echo "[run] Started. Tail: tail -f $LOG"