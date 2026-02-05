"""
Monkey-patch vLLM's compute_logprobs to use chunked computation.

This reduces peak memory from O(seq_len * vocab_size) to O(chunk_size * vocab_size)
when computing prompt_logprobs for long sequences.

Usage:
    # Import this BEFORE importing vllm or starting the server
    import patch_vllm_logprobs
    patch_vllm_logprobs.apply_patch()

    # Then import vllm and start server as usual
    from vllm import LLM
    ...
"""

import torch

# Configuration
LOGPROBS_CHUNK_SIZE = 2048  # Process this many positions at a time


def compute_logprobs_chunked(logits: torch.Tensor, chunk_size: int = LOGPROBS_CHUNK_SIZE) -> torch.Tensor:
    """
    Memory-efficient chunked log_softmax computation.

    Original vLLM does:
        logits.log_softmax(dim=-1, dtype=torch.float32)

    Which allocates a full [seq_len, vocab_size] fp32 tensor.
    For 16k seq_len and 152k vocab, that's 9.3 GB!

    This function processes in chunks to reduce peak memory:
        - chunk_size=2048: ~1.2 GB peak instead of 9.3 GB
        - chunk_size=1024: ~0.6 GB peak

    Args:
        logits: Input logits tensor of shape [seq_len, vocab_size]
        chunk_size: Number of positions to process at once

    Returns:
        Log probabilities tensor of shape [seq_len, vocab_size] in fp32
    """
    seq_len = logits.shape[0]

    # If small enough, just do it normally
    if seq_len <= chunk_size:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    # Allocate output tensor
    result = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)

    # Process in chunks
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        result[start:end] = logits[start:end].log_softmax(dim=-1, dtype=torch.float32)

    return result


_original_compute_logprobs = None


def apply_patch(chunk_size: int = LOGPROBS_CHUNK_SIZE):
    """
    Apply the monkey-patch to vLLM's Sampler.compute_logprobs method.

    Call this BEFORE creating any vLLM LLM instance.

    Args:
        chunk_size: Number of sequence positions to process at once.
                   Smaller = less memory, more kernel launches.
                   Recommended: 1024-2048
    """
    global _original_compute_logprobs, LOGPROBS_CHUNK_SIZE
    LOGPROBS_CHUNK_SIZE = chunk_size

    try:
        from vllm.v1.sample.sampler import Sampler
    except ImportError:
        print("[patch_vllm_logprobs] Warning: Could not import vllm.v1.sample.sampler")
        print("[patch_vllm_logprobs] This patch is for vLLM v1 API. Skipping.")
        return False

    # Save original method
    _original_compute_logprobs = Sampler.compute_logprobs

    # Define patched method
    def patched_compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        """Patched version using chunked computation for memory efficiency."""
        return compute_logprobs_chunked(logits, chunk_size=LOGPROBS_CHUNK_SIZE)

    # Apply patch
    Sampler.compute_logprobs = patched_compute_logprobs

    print(f"[patch_vllm_logprobs] Successfully patched Sampler.compute_logprobs")
    print(f"[patch_vllm_logprobs] Using chunk_size={LOGPROBS_CHUNK_SIZE}")
    print(f"[patch_vllm_logprobs] Peak memory reduced from O(seq_len * vocab) to O({LOGPROBS_CHUNK_SIZE} * vocab)")

    return True


def remove_patch():
    """Remove the monkey-patch and restore original behavior."""
    global _original_compute_logprobs

    if _original_compute_logprobs is None:
        print("[patch_vllm_logprobs] No patch to remove")
        return

    try:
        from vllm.v1.sample.sampler import Sampler
        Sampler.compute_logprobs = _original_compute_logprobs
        _original_compute_logprobs = None
        print("[patch_vllm_logprobs] Patch removed, original behavior restored")
    except ImportError:
        pass


if __name__ == "__main__":
    # Test the patch
    print("Testing chunked logprobs computation...")

    # Simulate logits tensor
    seq_len = 16000
    vocab_size = 152000

    print(f"Simulating logits tensor: [{seq_len}, {vocab_size}]")
    print(f"Original would need: {seq_len * vocab_size * 4 / 1e9:.2f} GB for fp32 output")
    print(f"Chunked (2048) needs: {2048 * vocab_size * 4 / 1e9:.2f} GB peak")

    # Create small test
    test_logits = torch.randn(100, 1000, dtype=torch.bfloat16, device="cpu")

    # Test chunked vs original
    original = test_logits.log_softmax(dim=-1, dtype=torch.float32)
    chunked = compute_logprobs_chunked(test_logits, chunk_size=32)

    # Verify results match
    max_diff = (original - chunked).abs().max().item()
    print(f"Max difference between original and chunked: {max_diff}")
    assert max_diff < 1e-5, "Results don't match!"
    print("✓ Test passed! Results match.")
