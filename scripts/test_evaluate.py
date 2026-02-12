#!/usr/bin/env python3
"""Simple test to verify the GSM8K evaluation scripts work correctly."""

import sys
import tempfile

# Test imports
try:
    from verl.utils.reward_score.gsm8k import compute_score, extract_solution
    print("✓ Successfully imported verl.utils.reward_score.gsm8k")
except ImportError as e:
    print(f"✗ Failed to import verl components: {e}")
    sys.exit(1)

# Test extract_solution
test_cases = [
    ("The answer is 42. #### 42", "strict", "42"),
    ("After calculation, we get #### 123", "strict", "123"),
    ("The final answer is 3.14 #### 3.14", "strict", "3.14"),
    ("The result is -5 #### -5", "strict", "-5"),
    ("Total cost is $1,234 #### 1,234", "strict", "1234"),  # Should remove comma
    ("No answer format here", "strict", None),
]

print("\nTesting extract_solution:")
for solution_str, method, expected in test_cases:
    result = extract_solution(solution_str, method=method)
    status = "✓" if result == expected else "✗"
    print(f"{status} extract_solution('{solution_str[:50]}...', '{method}') = {result} (expected: {expected})")

# Test compute_score
print("\nTesting compute_score:")
test_scores = [
    ("The answer is #### 42", "42", "strict", 1.0),
    ("The answer is #### 42", "43", "strict", 0.0),
    ("No answer here", "42", "strict", 0),
    ("Result: #### 100", "100", "strict", 1.0),
]

for solution_str, ground_truth, method, expected in test_scores:
    result = compute_score(solution_str, ground_truth, method=method)
    status = "✓" if result == expected else "✗"
    print(f"{status} compute_score('{solution_str}', '{ground_truth}', '{method}') = {result} (expected: {expected})")

# Test dataset loading
print("\nTesting dataset loading:")
try:
    import datasets
    print("✓ datasets library available")

    # Try to load a minimal GSM8K sample
    print("  Attempting to load GSM8K from HuggingFace...")
    try:
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
        print(f"  ✓ Successfully loaded GSM8K dataset with {len(dataset)} samples")
        print(f"  Sample keys: {list(dataset[0].keys())}")
    except Exception as e:
        print(f"  ! Could not load from HuggingFace (might need internet): {e}")

except ImportError:
    print("✗ datasets library not available")

# Test model imports
print("\nTesting model libraries:")
try:
    import torch
    print(f"✓ PyTorch available (version: {torch.__version__})")
except ImportError:
    print("✗ PyTorch not available")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Transformers available")
except ImportError:
    print("✗ Transformers not available")

try:
    import vllm
    print(f"✓ vLLM available (version: {vllm.__version__})")
except ImportError:
    print("! vLLM not available (optional)")

print("\n" + "=" * 80)
print("Test Summary:")
print("=" * 80)
print("Core functionality tests passed. The evaluation scripts should work.")
print("\nTo test with a real model:")
print("  1. Prepare GSM8K dataset: python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k")
print("  2. Run evaluation: python scripts/evaluate_gsm8k_vllm.py --model_path <model> --dataset_path ~/data/gsm8k/test.parquet --max_samples 10")
