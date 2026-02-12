# GSM8K Evaluation Scripts

This directory contains standalone scripts for evaluating models on the GSM8K validation dataset without running the full training pipeline.

## Scripts

### 1. `evaluate_gsm8k.py` - HuggingFace Transformers

A basic evaluation script using HuggingFace Transformers.

**Usage:**
```bash
python scripts/evaluate_gsm8k.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0 \
    --do_sample False \
    --max_samples 100 \
    --output_path results.json
```

### 2. `evaluate_gsm8k_vllm.py` - vLLM (Recommended)

An optimized evaluation script using vLLM for faster inference. This is the recommended script for production use.

**Usage:**
```bash
# Using vLLM (default)
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0 \
    --do_sample False \
    --tensor_parallel_size 2 \
    --output_path results.json

# Using HuggingFace as fallback
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --engine hf \
    --batch_size 8
```

## Configuration Parameters

All parameters match the `val_kwargs` configuration in `verl/trainer/config/rollout/rollout.yaml`:

### Model Arguments
- `--model_path`: Path to model checkpoint or HuggingFace model ID (required)
- `--tokenizer_path`: Path to tokenizer (defaults to model_path)
- `--dtype`: Model dtype (float32/float16/bfloat16, default: bfloat16)
- `--engine`: Inference engine - vllm or hf (evaluate_gsm8k_vllm.py only)

### Dataset Arguments
- `--dataset_path`: Path to GSM8K dataset (required)
  - Parquet file: `~/data/gsm8k/test.parquet`
  - Directory with parquet files
  - HuggingFace dataset: `openai/gsm8k`
- `--max_samples`: Maximum number of samples to evaluate (default: all)

### Sampling Parameters (matching rollout.yaml)
- `--temperature`: Sampling temperature (default: 0 for greedy)
- `--do_sample`: Whether to use sampling (True/False, default: False)
- `--top_k`: Top-k sampling (default: -1, disabled)
- `--top_p`: Top-p nucleus sampling (default: 1.0)
- `--n`: Number of responses per sample for majority voting (default: 1)
- `--max_new_tokens`: Maximum tokens to generate (default: 512)

### Engine-Specific Arguments
- `--batch_size`: Batch size for HF inference (default: 8)
- `--tensor_parallel_size`: Tensor parallel size for vLLM (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization for vLLM (default: 0.9)

### Scoring Arguments
- `--extraction_method`: Answer extraction method
  - `strict` (default): Requires `####` format in response
  - `flexible`: Extracts last number from response

### Output Arguments
- `--output_path`: Path to save detailed results in JSON format
- `--verbose`: Print detailed output for each sample

## Dataset Preparation

If you don't have the GSM8K dataset in parquet format, you can preprocess it using:

```bash
python examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k
```

This will download and preprocess the GSM8K dataset into:
- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

## Examples

### Basic Greedy Evaluation
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0 \
    --do_sample False
```

### Sampling with Temperature
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0.7 \
    --do_sample True \
    --top_p 0.9
```

### Majority Voting (Multiple Samples per Prompt)
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0.7 \
    --do_sample True \
    --n 5 \
    --output_path results_maj5.json
```

This will generate 5 responses per prompt and compute both:
- Overall accuracy (all responses)
- Majority voting accuracy (best of 5)

### Multi-GPU with Tensor Parallelism
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-70B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.95
```

### Quick Test on 100 Samples
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_path ~/data/gsm8k/test.parquet \
    --max_samples 100 \
    --verbose
```

### Evaluate Checkpoint from Training
```bash
python scripts/evaluate_gsm8k_vllm.py \
    --model_path /path/to/checkpoint/global_step_1000 \
    --dataset_path ~/data/gsm8k/test.parquet \
    --temperature 0 \
    --do_sample False \
    --output_path checkpoint_1000_results.json
```

## Output Format

The script outputs:
1. **Console output**: Summary statistics including accuracy
2. **JSON file** (if `--output_path` specified): Detailed results including:
   - Configuration parameters
   - Overall accuracy
   - Majority voting accuracy (if n > 1)
   - Individual samples with prompts, responses, scores, and ground truths

Example JSON output:
```json
{
  "config": {
    "model_path": "meta-llama/Llama-3.2-1B-Instruct",
    "temperature": 0.0,
    "do_sample": false,
    ...
  },
  "accuracy": 0.7234,
  "num_samples": 1319,
  "samples": [
    {
      "prompt": "Natalia sold clips to 48 of her friends...",
      "responses": ["Let's think step by step..."],
      "ground_truth": "4",
      "scores": [1.0],
      "correct": true
    },
    ...
  ]
}
```

## Implementation Details

These scripts reuse verl's validation components to ensure consistency:
- `verl.utils.reward_score.gsm8k.compute_score()`: Answer extraction and scoring
- `verl.utils.reward_score.gsm8k.extract_solution()`: Extract numerical answers

The default configuration matches the validation settings in `verl/trainer/config/rollout/rollout.yaml`:
- `temperature: 0` (greedy decoding)
- `do_sample: False` (deterministic)
- `top_p: 1.0`
- `top_k: -1`
- `n: 1` (single response per prompt)

## Tips

1. **For fastest inference**: Use `evaluate_gsm8k_vllm.py` with vLLM engine
2. **For large models**: Increase `tensor_parallel_size` and adjust `gpu_memory_utilization`
3. **For debugging**: Use `--max_samples 10 --verbose` to see detailed outputs
4. **For production**: Use `--output_path` to save detailed results for analysis
5. **For best accuracy**: Use majority voting with `--n 5` or higher

## Requirements

- Python 3.8+
- verl (this package - installed or in PYTHONPATH)
- PyTorch
- Transformers
- datasets
- numpy
- tqdm
- vLLM (optional, for faster inference)

Install dependencies:
```bash
# Install verl and its dependencies
pip install -e .

# Or install additional packages separately
pip install torch transformers datasets numpy tqdm
pip install vllm  # Optional, for faster inference
```

**Note**: The scripts need to be run from the verl repository root or with verl in your PYTHONPATH:
```bash
# Run from repository root
cd /path/to/verl
python scripts/evaluate_gsm8k_vllm.py ...

# Or set PYTHONPATH
export PYTHONPATH=/path/to/verl:$PYTHONPATH
python scripts/evaluate_gsm8k_vllm.py ...
```
