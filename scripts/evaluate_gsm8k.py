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
Standalone script to evaluate a model on GSM8K validation dataset.

This script allows you to evaluate any model checkpoint or HuggingFace model ID
on the GSM8K dataset without running the full training pipeline.

Example usage:
    python scripts/evaluate_gsm8k.py \
        --model_path meta-llama/Llama-3.2-1B-Instruct \
        --dataset_path ~/data/gsm8k/test.parquet \
        --temperature 0 \
        --do_sample False \
        --max_samples 100
"""

import argparse
import json
import os
from typing import Optional

import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import verl's GSM8K scoring function to ensure consistency
from verl.utils.reward_score.gsm8k import compute_score


def load_gsm8k_dataset(dataset_path: str, max_samples: Optional[int] = None):
    """Load GSM8K dataset from parquet file."""
    if dataset_path.endswith('.parquet'):
        dataset = datasets.load_dataset('parquet', data_files=dataset_path, split='train')
    else:
        # Assume it's a directory with parquet files
        dataset = datasets.load_dataset(dataset_path, split='test')

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


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_k: int = -1,
    top_p: float = 1.0,
    n: int = 1,
):
    """Generate responses using the model."""
    all_responses = []

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
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
            else:
                # Greedy decoding
                generation_kwargs["temperature"] = None
                generation_kwargs["top_p"] = None
                generation_kwargs["top_k"] = None

            # Generate n times if needed (for majority voting or multiple samples)
            batch_responses = []
            for _ in range(n):
                outputs = model.generate(**inputs, **generation_kwargs)

                # Decode only the generated part (excluding input)
                for j, output in enumerate(outputs):
                    input_length = inputs.input_ids[j].shape[0]
                    generated_tokens = output[input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    batch_responses.append(response)

            all_responses.extend(batch_responses)

    return all_responses


def compute_accuracy(responses: list[str], ground_truths: list[str], method: str = "strict"):
    """Compute accuracy using verl's GSM8K scoring function."""
    assert len(responses) == len(ground_truths), "Mismatch between responses and ground truths"

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
    """
    Compute accuracy with majority voting when n > 1.

    Args:
        responses: List of all responses (length = num_samples * n)
        ground_truths: List of ground truths (length = num_samples, repeated n times)
        n: Number of responses per sample
        method: Extraction method for GSM8K

    Returns:
        Majority voting accuracy
    """
    num_samples = len(responses) // n
    maj_correct = 0

    for i in range(num_samples):
        sample_responses = responses[i*n:(i+1)*n]
        ground_truth = ground_truths[i*n]  # All n copies should have the same ground truth

        # Get scores for all n responses
        sample_scores = [compute_score(resp, ground_truth, method=method) for resp in sample_responses]

        # Majority vote: if more than half are correct, consider it correct
        if sum(sample_scores) > n / 2:
            maj_correct += 1

    maj_accuracy = maj_correct / num_samples if num_samples > 0 else 0.0
    return maj_accuracy


def save_results(
    output_path: str,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    scores: list[float],
    accuracy: float,
    config: dict,
):
    """Save evaluation results to a JSON file."""
    results = {
        "config": config,
        "accuracy": accuracy,
        "num_samples": len(prompts),
        "samples": [
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": gt,
                "score": score,
            }
            for prompt, response, gt, score in zip(prompts, responses, ground_truths, scores)
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K validation dataset")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace model ID")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to GSM8K dataset (parquet file or dataset directory)")
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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")

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

    # Set tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("GSM8K Evaluation Configuration")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Max Samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Temperature: {args.temperature}")
    print(f"Do Sample: {args.do_sample}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"N (responses per sample): {args.n}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Extraction Method: {args.extraction_method}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
    )

    if args.device == "cpu":
        model = model.to("cpu")

    model.eval()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Load dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_gsm8k_dataset(args.dataset_path, args.max_samples)
    print(f"Loaded {len(dataset)} samples")

    # Prepare prompts and ground truths
    print("\nPreparing prompts...")
    prompts = []
    ground_truths = []

    for item in dataset:
        # Format prompt
        if "prompt" in item:
            prompt = format_prompt(item["prompt"])
        elif "question" in item:
            # Handle raw GSM8K format
            prompt = item["question"] + ' Let\'s think step by step and output the final answer after "####".'
        else:
            raise ValueError(f"Cannot find prompt in dataset item: {item.keys()}")

        # Get ground truth
        if "reward_model" in item and isinstance(item["reward_model"], dict):
            ground_truth = item["reward_model"]["ground_truth"]
        elif "answer" in item:
            # Extract from raw answer
            from verl.utils.reward_score.gsm8k import extract_solution
            ground_truth = extract_solution(item["answer"], method="strict")
        else:
            raise ValueError(f"Cannot find ground truth in dataset item: {item.keys()}")

        # Repeat n times for multiple samples per prompt
        for _ in range(args.n):
            prompts.append(prompt)
            ground_truths.append(ground_truth)

    print(f"Total prompts to evaluate: {len(prompts)}")

    # Generate responses
    print("\nGenerating responses...")
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        n=1,  # We already repeated prompts, so n=1 here
    )

    # Compute accuracy
    print("\nComputing accuracy...")
    accuracy, scores = compute_accuracy(responses, ground_truths, method=args.extraction_method)

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.4f} ({sum(scores)}/{len(scores)})")

    # Compute majority voting accuracy if n > 1
    if args.n > 1:
        maj_accuracy = majority_voting(responses, ground_truths, args.n, method=args.extraction_method)
        print(f"Majority Voting Accuracy (n={args.n}): {maj_accuracy:.4f}")

    print("=" * 80)

    # Print some examples if verbose
    if args.verbose:
        print("\nSample outputs:")
        num_examples = min(5, len(prompts))
        for i in range(num_examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {prompts[i][:200]}...")
            print(f"Response: {responses[i][:300]}...")
            print(f"Ground Truth: {ground_truths[i]}")
            print(f"Score: {scores[i]}")

    # Save results if output path provided
    if args.output_path:
        save_results(
            output_path=args.output_path,
            prompts=prompts[::args.n],  # Save unique prompts only
            responses=responses,
            ground_truths=ground_truths[::args.n],  # Save unique ground truths only
            scores=scores,
            accuracy=accuracy,
            config=vars(args),
        )


if __name__ == "__main__":
    main()
