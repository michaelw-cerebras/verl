#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone script to evaluate a model on GSM8K validation dataset using vLLM for faster inference.

This script uses vLLM for accelerated inference and reuses verl's validation logic
to ensure consistency with training-time validation.

Example usage:
    # Using vLLM (faster)
    python scripts/evaluate_gsm8k_vllm.py \
        --model_path meta-llama/Llama-3.2-1B-Instruct \
        --dataset_path ~/data/gsm8k/test.parquet \
        --temperature 0 \
        --do_sample False \
        --tensor_parallel_size 1

    # Using HuggingFace (fallback)
    python scripts/evaluate_gsm8k_vllm.py \
        --model_path meta-llama/Llama-3.2-1B-Instruct \
        --dataset_path ~/data/gsm8k/test.parquet \
        --engine hf
"""

import argparse
import json
import os
from typing import Optional

import datasets
import numpy as np
from tqdm import tqdm

# Import verl's GSM8K scoring function to ensure consistency
from verl.utils.reward_score.gsm8k import compute_score, extract_solution


def load_gsm8k_dataset(dataset_path: str, max_samples: Optional[int] = None):
    """Load GSM8K dataset from parquet file."""
    if dataset_path.endswith('.parquet'):
        dataset = datasets.load_dataset('parquet', data_files=dataset_path, split='train')
    else:
        # Assume it's a directory or HuggingFace dataset
        try:
            dataset = datasets.load_dataset(dataset_path, split='test')
        except:
            # Try loading as openai/gsm8k
            dataset = datasets.load_dataset('openai/gsm8k', 'main', split='test')

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def format_prompt(prompt_data):
    """Format the prompt from the dataset into a string."""
    if isinstance(prompt_data, list):
        # Handle conversational format [{"role": "user", "content": "..."}]
        messages = []
        for msg in prompt_data:
            if msg.get("role") == "user":
                messages.append(msg["content"])
            elif msg.get("role") == "assistant":
                messages.append(msg["content"])
        return "\n".join(messages)
    elif isinstance(prompt_data, str):
        return prompt_data
    else:
        raise ValueError(f"Unsupported prompt format: {type(prompt_data)}")


def generate_responses_vllm(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
):
    """Generate responses using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM is not installed. Please install it with: pip install vllm")

    print(f"Initializing vLLM with model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if (do_sample and top_k > 0) else -1,
        n=n,
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Extract responses
    all_responses = []
    for output in outputs:
        for completion in output.outputs:
            all_responses.append(completion.text)

    return all_responses


def generate_responses_hf(
    model_path: str,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
    dtype: str = "bfloat16",
):
    """Generate responses using HuggingFace Transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype],
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_responses = []

    # Repeat prompts n times
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * n)

    for i in tqdm(range(0, len(repeated_prompts), batch_size), desc="Generating"):
        batch_prompts = repeated_prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
                if top_k > 0:
                    generation_kwargs["top_k"] = top_k

            outputs = model.generate(**inputs, **generation_kwargs)

            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_responses.append(response)

    return all_responses


def compute_accuracy(responses: list[str], ground_truths: list[str], method: str = "strict"):
    """Compute accuracy using verl's GSM8K scoring function."""
    assert len(responses) == len(ground_truths), f"Mismatch: {len(responses)} responses vs {len(ground_truths)} ground truths"

    scores = []
    correct_count = 0

    for response, ground_truth in zip(responses, ground_truths):
        score = compute_score(response, ground_truth, method=method)
        scores.append(score)
        if score > 0:
            correct_count += 1

    accuracy = correct_count / len(scores) if len(scores) > 0 else 0.0
    return accuracy, scores


def majority_voting(responses: list[str], ground_truths: list[str], n: int, method: str = "strict"):
    """Compute accuracy with majority voting when n > 1."""
    num_samples = len(responses) // n
    maj_correct = 0

    for i in range(num_samples):
        sample_responses = responses[i*n:(i+1)*n]
        ground_truth = ground_truths[i*n]

        sample_scores = [compute_score(resp, ground_truth, method=method) for resp in sample_responses]

        if sum(sample_scores) > n / 2:
            maj_correct += 1

    maj_accuracy = maj_correct / num_samples if num_samples > 0 else 0.0
    return maj_accuracy


def save_results(output_path: str, results: dict):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K validation dataset")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace model ID")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "hf"],
                        help="Inference engine to use (vllm or hf)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to GSM8K dataset (parquet file, directory, or 'openai/gsm8k')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all)")

    # Generation arguments (matching rollout.yaml val_kwargs)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--do_sample", type=lambda x: x.lower() == 'true', default=False,
                        help="Whether to use sampling (True/False)")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling (-1 for disabled)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of responses per sample (for majority voting)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")

    # Engine-specific arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for HF inference (ignored for vLLM)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")

    # Scoring arguments
    parser.add_argument("--extraction_method", type=str, default="strict",
                        choices=["strict", "flexible"],
                        help="Answer extraction method (strict requires '####' format)")

    # Output arguments
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save detailed results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each sample")

    args = parser.parse_args()

    print("=" * 80)
    print("GSM8K Evaluation Configuration")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Inference Engine: {args.engine}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Max Samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Dtype: {args.dtype}")
    print(f"Temperature: {args.temperature}")
    print(f"Do Sample: {args.do_sample}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"N (responses per sample): {args.n}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Extraction Method: {args.extraction_method}")
    if args.engine == "vllm":
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
        print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    else:
        print(f"Batch Size: {args.batch_size}")
    print("=" * 80)

    # Load dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_gsm8k_dataset(args.dataset_path, args.max_samples)
    print(f"Loaded {len(dataset)} samples")

    # Prepare prompts and ground truths
    print("\nPreparing prompts and ground truths...")
    prompts = []
    ground_truths = []

    for item in dataset:
        # Format prompt
        if "prompt" in item:
            prompt = format_prompt(item["prompt"])
        elif "question" in item:
            prompt = item["question"] + ' Let\'s think step by step and output the final answer after "####".'
        else:
            raise ValueError(f"Cannot find prompt in dataset item: {item.keys()}")

        prompts.append(prompt)

        # Get ground truth
        if "reward_model" in item and isinstance(item["reward_model"], dict):
            ground_truth = item["reward_model"]["ground_truth"]
        elif "answer" in item:
            ground_truth = extract_solution(item["answer"], method="strict")
        else:
            raise ValueError(f"Cannot find ground truth in dataset item: {item.keys()}")

        ground_truths.append(ground_truth)

    print(f"Prepared {len(prompts)} prompts")

    # Generate responses
    print("\nGenerating responses...")
    if args.engine == "vllm":
        responses = generate_responses_vllm(
            model_path=args.model_path,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            n=args.n,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
        )
    else:
        responses = generate_responses_hf(
            model_path=args.model_path,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            n=args.n,
            dtype=args.dtype,
        )

    # Expand ground truths to match responses if n > 1
    expanded_ground_truths = []
    for gt in ground_truths:
        expanded_ground_truths.extend([gt] * args.n)

    # Compute accuracy
    print("\nComputing accuracy...")
    accuracy, scores = compute_accuracy(responses, expanded_ground_truths, method=args.extraction_method)

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Number of unique samples: {len(prompts)}")
    print(f"Total responses evaluated: {len(responses)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({int(sum(scores))}/{len(scores)})")

    # Compute majority voting accuracy if n > 1
    if args.n > 1:
        maj_accuracy = majority_voting(responses, expanded_ground_truths, args.n, method=args.extraction_method)
        print(f"Majority Voting Accuracy (n={args.n}): {maj_accuracy:.4f}")

    print("=" * 80)

    # Print some examples if verbose
    if args.verbose:
        print("\nSample outputs:")
        num_examples = min(5, len(prompts))
        for i in range(num_examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {prompts[i][:200]}...")
            for j in range(args.n):
                idx = i * args.n + j
                print(f"Response {j+1}: {responses[idx][:300]}...")
                print(f"Score {j+1}: {scores[idx]}")
            print(f"Ground Truth: {ground_truths[i]}")

    # Save results if output path provided
    if args.output_path:
        results = {
            "config": vars(args),
            "accuracy": accuracy,
            "num_samples": len(prompts),
            "num_responses": len(responses),
        }

        if args.n > 1:
            results["majority_voting_accuracy"] = maj_accuracy

        results["samples"] = []
        for i in range(len(prompts)):
            sample_responses = []
            sample_scores = []
            for j in range(args.n):
                idx = i * args.n + j
                sample_responses.append(responses[idx])
                sample_scores.append(scores[idx])

            results["samples"].append({
                "prompt": prompts[i],
                "responses": sample_responses,
                "ground_truth": ground_truths[i],
                "scores": sample_scores,
                "correct": any(s > 0 for s in sample_scores),
            })

        save_results(args.output_path, results)


if __name__ == "__main__":
    main()
