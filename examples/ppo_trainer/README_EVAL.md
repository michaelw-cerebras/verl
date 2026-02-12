# GSM8K Evaluation-Only Mode

This directory contains a configuration for running validation-only evaluation on GSM8K without training.

## Quick Start

### Basic Evaluation

**Option 1: Use your existing training script (simplest)**
```bash
# Just add trainer.val_only=true to your training script
bash your_train_script.sh trainer.val_only=true trainer.val_before_train=true
```

**Option 2: Use the evaluation-only config**
```bash
# Use the cleaned-up eval config
python3 -m verl.trainer.main_ppo \
    --config-path=examples/ppo_trainer \
    --config-name=gsm8k_eval_only
```

This will evaluate the default model (`Qwen/Qwen2.5-0.5B-Instruct`) on the GSM8K test set.

### Evaluate a Checkpoint

**Using your training script:**
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    actor_rollout_ref.model.path=/path/to/checkpoint/global_step_1000
```

**Using the eval config:**
```bash
python3 -m verl.trainer.main_ppo \
    --config-path=examples/ppo_trainer \
    --config-name=gsm8k_eval_only \
    actor_rollout_ref.model.path=/path/to/checkpoint/global_step_1000
```

### Evaluate with Different Parameters
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    data.val_max_samples=100
```

## How It Works

The evaluation mode is enabled by two key settings:

```yaml
trainer:
  val_only: true        # Only run validation, skip training
  val_before_train: true # Trigger validation immediately
  total_epochs: 0       # No training epochs (optional, for clarity)
```

**Two ways to use this:**

1. **Override your training config** (simplest):
   ```bash
   bash your_train_script.sh trainer.val_only=true trainer.val_before_train=true
   ```

2. **Use the dedicated eval config** (cleaner, removes unnecessary components):
   ```bash
   python3 -m verl.trainer.main_ppo \
       --config-path=examples/ppo_trainer \
       --config-name=gsm8k_eval_only
   ```

Both reuse the exact same validation logic as during training, ensuring consistency.

## Configuration Parameters

### Model Path
Change the model to evaluate:
```bash
actor_rollout_ref.model.path=/path/to/your/model
```

This can be:
- A HuggingFace model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- A local checkpoint path (e.g., `/path/to/checkpoint/global_step_1000`)

### Dataset Path
Change the validation dataset:
```bash
data.val_files=/path/to/test.parquet
```

### Validation Sampling Parameters

All parameters in `rollout.val_kwargs` can be overridden:

#### Greedy Decoding (Default)
```bash
actor_rollout_ref.rollout.val_kwargs.temperature=0
actor_rollout_ref.rollout.val_kwargs.do_sample=false
```

#### Sampling with Temperature
```bash
actor_rollout_ref.rollout.val_kwargs.temperature=0.7
actor_rollout_ref.rollout.val_kwargs.do_sample=true
actor_rollout_ref.rollout.val_kwargs.top_p=0.9
actor_rollout_ref.rollout.val_kwargs.top_k=50
```

#### Majority Voting (Multiple Samples)
```bash
actor_rollout_ref.rollout.val_kwargs.n=5
actor_rollout_ref.rollout.val_kwargs.temperature=0.7
actor_rollout_ref.rollout.val_kwargs.do_sample=true
```

This will generate 5 responses per prompt and report both overall accuracy and majority voting accuracy.

### Limit Number of Samples (Quick Test)
```bash
data.val_max_samples=100
```

### GPU Configuration
```bash
trainer.n_gpus_per_node=2
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### Logging Configuration

#### Change Experiment Name
```bash
trainer.experiment_name=eval_my_checkpoint_step_1000
```

#### Save Validation Outputs
```bash
trainer.validation_data_dir=./validation_outputs
```

This will save all prompts, responses, and scores to a file.

#### Log Sample Generations
```bash
trainer.log_val_generations=10
```

This will log 10 sample generations to console/wandb.

## Output

The evaluation will print metrics to console:

```
================================================================================
Evaluation Results
================================================================================
val-core/openai/gsm8k/acc/mean@1: 0.7234
val-core/openai/gsm8k/acc/best@1/mean: 0.7234
...
================================================================================
```

Key metrics:
- `val-core/openai/gsm8k/acc/mean@N`: Mean accuracy across N samples
- `val-core/openai/gsm8k/acc/maj@N/mean`: Majority voting accuracy (if N > 1)

## Examples

### Example 1: Quick Test on 100 Samples
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    data.val_max_samples=100 \
    trainer.log_val_generations=5
```

### Example 2: Evaluate Training Checkpoint
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    actor_rollout_ref.model.path=checkpoints/my_project/my_experiment/actor/global_step_500 \
    trainer.validation_data_dir=./eval_results/step_500
```

### Example 3: Majority Voting Evaluation
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    trainer.validation_data_dir=./maj_voting_results
```

### Example 4: Multi-GPU Evaluation (Large Model)
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95
```

### Example 5: Evaluate with WandB Logging
```bash
bash your_train_script.sh \
    trainer.val_only=true \
    trainer.project_name=my_evaluation_project \
    trainer.experiment_name=eval_qwen_0.5b_baseline \
    'trainer.logger=["console","wandb"]'
```

## Comparison with Standalone Scripts

This approach differs from the standalone scripts in `scripts/`:

| Feature | YAML Config Approach | Standalone Scripts |
|---------|---------------------|-------------------|
| Code reuse | ✅ 100% training code | ⚠️ Partial reuse |
| Consistency | ✅ Identical to training validation | ⚠️ May differ |
| Setup | ✅ Just YAML changes | ⚠️ Separate scripts |
| Dependencies | ✅ Same as training | ⚠️ Additional deps |
| Flexibility | ⚠️ Hydra overrides only | ✅ Custom CLI args |

**Recommendation**: Use the YAML config approach (this method) for evaluating checkpoints during/after training to ensure validation metrics are exactly comparable to training-time validation.

## Differences from Training Config

The `gsm8k_eval_only.yaml` removes:
- ❌ `teacher` configuration (not needed)
- ❌ `critic` configuration (not needed)
- ❌ `actor.optim` configuration (no training)
- ❌ `train_files`, `off_policy_files` (no training data)
- ❌ `train_batch_size`, `train_max_samples` (no training)
- ❌ GKD-specific flags like `gkd_lambda`, `enable_off_policy`

And adds:
- ✅ `trainer.val_only: true`
- ✅ `trainer.val_before_train: true`
- ✅ `trainer.total_epochs: 0`

## Troubleshooting

### Out of Memory
Reduce GPU memory usage:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
data.val_batch_size=8
```

### Slow Inference
Increase tensor parallelism:
```bash
trainer.n_gpus_per_node=4
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### Different Results from Training
Make sure validation parameters match:
- Check `val_kwargs.temperature`, `do_sample`, etc.
- Ensure same `stop_token_ids` for Qwen3 models
- Verify dataset preprocessing is identical

## Tips

1. **For reproducibility**: Always use `temperature=0` and `do_sample=false` (greedy decoding)
2. **For best accuracy**: Use majority voting with `n=5` or higher
3. **For debugging**: Use `data.val_max_samples=10` and `trainer.log_val_generations=5`
4. **For production**: Save outputs with `trainer.validation_data_dir`
