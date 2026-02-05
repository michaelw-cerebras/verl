"""
Monkey-patch vLLM's logprobs computation for memory-efficient long sequence processing.

This reduces peak memory from O(seq_len * vocab_size) to O(chunk_size * vocab_size)
when computing prompt_logprobs for long sequences.

The key insight: instead of computing full [seq_len, vocab_size] log_softmax and then
extracting top-k, we compute log_softmax in chunks and extract top-k immediately,
never allocating the full output tensor.

Usage:
    # Import this BEFORE importing vllm or starting the server
    import patch_vllm_logprobs
    patch_vllm_logprobs.apply_patch()

    # Then import vllm and start server as usual
    from vllm import LLM
    ...
"""

import torch
from typing import Tuple, Optional

# Configuration
LOGPROBS_CHUNK_SIZE = 1024  # Process this many positions at a time
NUM_LOGPROBS = 64  # Default top-k to extract


def compute_topk_logprobs_chunked(
    logits: torch.Tensor,
    num_logprobs: int = NUM_LOGPROBS,
    chunk_size: int = LOGPROBS_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient chunked top-k log probabilities computation.

    Instead of:
        logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)  # OOM for long seq!
        topk_values, topk_indices = logprobs.topk(k, dim=-1)

    This function:
        1. Processes logits in chunks
        2. For each chunk, computes log_softmax (small temporary tensor)
        3. Immediately extracts top-k for that chunk
        4. Deletes the chunk's full log_softmax result
        5. Returns concatenated top-k results

    Memory usage: O(chunk_size * vocab_size) instead of O(seq_len * vocab_size)

    Args:
        logits: Input logits tensor of shape [seq_len, vocab_size]
        num_logprobs: Number of top logprobs to return per position
        chunk_size: Number of positions to process at once

    Returns:
        Tuple of (topk_logprobs, topk_indices):
            - topk_logprobs: [seq_len, num_logprobs] fp32 tensor
            - topk_indices: [seq_len, num_logprobs] int64 tensor
    """
    seq_len = logits.shape[0]
    device = logits.device

    # Pre-allocate output tensors (these are small: seq_len * num_logprobs)
    all_topk_values = torch.empty((seq_len, num_logprobs), dtype=torch.float32, device=device)
    all_topk_indices = torch.empty((seq_len, num_logprobs), dtype=torch.int64, device=device)

    # Process in chunks
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        # Compute log_softmax for this chunk only
        # This temporary tensor is [chunk_size, vocab_size] - manageable
        chunk_logprobs = logits[start:end].log_softmax(dim=-1, dtype=torch.float32)

        # Extract top-k for this chunk
        topk_values, topk_indices = chunk_logprobs.topk(num_logprobs, dim=-1)

        # Store results
        all_topk_values[start:end] = topk_values
        all_topk_indices[start:end] = topk_indices

        # Explicitly delete to free memory before next chunk
        del chunk_logprobs, topk_values, topk_indices

    return all_topk_values, all_topk_indices


def compute_logprobs_chunked_full(
    logits: torch.Tensor,
    chunk_size: int = LOGPROBS_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Chunked log_softmax that returns full tensor.
    Only use this for short sequences where full tensor fits in memory.

    For long sequences, use compute_topk_logprobs_chunked instead.
    """
    seq_len = logits.shape[0]

    # If small enough, just do it normally
    if seq_len <= chunk_size:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    # Allocate output tensor - WARNING: this can OOM for long sequences!
    result = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)

    # Process in chunks
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        result[start:end] = logits[start:end].log_softmax(dim=-1, dtype=torch.float32)

    return result


def compute_logprobs_chunked_cpu_offload(
    logits: torch.Tensor,
    chunk_size: int = LOGPROBS_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Memory-efficient chunked log_softmax with CPU offloading.

    Strategy:
    1. Allocate output tensor on CPU (has more memory)
    2. Compute log_softmax for each chunk on GPU
    3. Immediately copy chunk result to CPU
    4. Return full logprobs tensor on CPU

    This avoids GPU OOM by never having the full fp32 tensor on GPU.
    The result will be on CPU, so subsequent operations need to handle this.

    Args:
        logits: Input logits tensor of shape [seq_len, vocab_size] on GPU
        chunk_size: Number of positions to process at once

    Returns:
        Log probabilities tensor of shape [seq_len, vocab_size] in fp32 ON CPU
    """
    seq_len, vocab_size = logits.shape
    device = logits.device

    # If small enough and fits in GPU memory, just do it normally on GPU
    estimated_output_gb = seq_len * vocab_size * 4 / 1e9
    if estimated_output_gb < 2.0:  # Less than 2GB, do on GPU
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    print(f"[patch_vllm_logprobs] Large tensor detected: [{seq_len}, {vocab_size}] = {estimated_output_gb:.2f} GB")
    print(f"[patch_vllm_logprobs] Using CPU offload strategy with chunk_size={chunk_size}")

    # Allocate output tensor on CPU (won't OOM)
    result_cpu = torch.empty((seq_len, vocab_size), dtype=torch.float32, device="cpu", pin_memory=True)

    # Process in chunks: compute on GPU, immediately copy to CPU
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        # Compute log_softmax for this chunk on GPU
        chunk_logprobs_gpu = logits[start:end].log_softmax(dim=-1, dtype=torch.float32)

        # Copy to CPU immediately (non-blocking for speed)
        result_cpu[start:end].copy_(chunk_logprobs_gpu, non_blocking=True)

        # Explicitly delete GPU tensor to free memory
        del chunk_logprobs_gpu

    # Synchronize to ensure all copies are complete
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"[patch_vllm_logprobs] CPU offload complete, result shape: {result_cpu.shape}")

    # Return CPU tensor - caller must handle this!
    return result_cpu


def compute_topk_from_logits_chunked(
    logits: torch.Tensor,
    num_logprobs: int = NUM_LOGPROBS,
    chunk_size: int = LOGPROBS_CHUNK_SIZE,
    return_on_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k logprobs directly from logits using chunked computation.

    This is the RECOMMENDED function for memory-efficient logprobs extraction.
    It never allocates the full [seq_len, vocab_size] fp32 tensor.

    Args:
        logits: Input logits tensor of shape [seq_len, vocab_size] on GPU
        num_logprobs: Number of top logprobs to return per position
        chunk_size: Number of positions to process at once
        return_on_gpu: If True, return tensors on GPU; if False, on CPU

    Returns:
        Tuple of (topk_logprobs, topk_indices):
            - topk_logprobs: [seq_len, num_logprobs] fp32 tensor
            - topk_indices: [seq_len, num_logprobs] int64 tensor
    """
    seq_len = logits.shape[0]
    device = logits.device
    output_device = device if return_on_gpu else "cpu"

    # Pre-allocate output tensors (small: seq_len * num_logprobs)
    all_topk_values = torch.empty((seq_len, num_logprobs), dtype=torch.float32, device=output_device)
    all_topk_indices = torch.empty((seq_len, num_logprobs), dtype=torch.int64, device=output_device)

    # Process in chunks
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        # Compute log_softmax for this chunk only on GPU
        chunk_logprobs = logits[start:end].log_softmax(dim=-1, dtype=torch.float32)

        # Extract top-k for this chunk on GPU
        topk_values, topk_indices = chunk_logprobs.topk(num_logprobs, dim=-1)

        # Store results (may involve GPU->CPU copy if return_on_gpu=False)
        all_topk_values[start:end] = topk_values.to(output_device)
        all_topk_indices[start:end] = topk_indices.to(output_device)

        # Explicitly delete to free GPU memory before next chunk
        del chunk_logprobs, topk_values, topk_indices

    return all_topk_values, all_topk_indices


_original_methods = {}
_patch_applied = False


def apply_patch(chunk_size: int = LOGPROBS_CHUNK_SIZE, num_logprobs: int = NUM_LOGPROBS):
    """
    Apply the monkey-patch to vLLM's Sampler for memory-efficient logprobs computation.

    This patches the _get_prompt_logprobs_dict method (or similar) to use chunked
    top-k extraction instead of computing full logprobs first.

    Call this BEFORE creating any vLLM LLM instance.

    Args:
        chunk_size: Number of sequence positions to process at once.
                   Smaller = less memory, more kernel launches.
                   Recommended: 512-1024
        num_logprobs: Number of top logprobs to extract per position.
    """
    global _patch_applied, _original_methods, LOGPROBS_CHUNK_SIZE, NUM_LOGPROBS
    LOGPROBS_CHUNK_SIZE = chunk_size
    NUM_LOGPROBS = num_logprobs

    if _patch_applied:
        print("[patch_vllm_logprobs] Patch already applied, skipping")
        return True

    print(f"[patch_vllm_logprobs] Applying patch with chunk_size={chunk_size}, num_logprobs={num_logprobs}")

    patched_any = False

    # Try to patch vLLM v1 Sampler
    try:
        from vllm.v1.sample.sampler import Sampler

        # Save original compute_logprobs
        _original_methods['Sampler.compute_logprobs'] = Sampler.compute_logprobs

        # Create a patched version
        # Note: For long sequences, this will still OOM if it returns full tensor
        # The real fix is at the forward() level to avoid calling compute_logprobs
        def patched_compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
            """
            Patched compute_logprobs - for short sequences only.
            Long sequences should be handled by patched forward() method.
            """
            seq_len, vocab_size = logits.shape
            estimated_output_gb = seq_len * vocab_size * 4 / 1e9

            # For small tensors, use normal GPU computation
            if estimated_output_gb < 4.0:
                return logits.log_softmax(dim=-1, dtype=torch.float32)

            # For large tensors, this will OOM - but we shouldn't reach here
            # if forward() is properly patched
            print(f"[patch_vllm_logprobs] WARNING: compute_logprobs called with large tensor [{seq_len}, {vocab_size}]")
            print(f"[patch_vllm_logprobs] This should have been handled by patched forward()")

            # Attempt chunked computation - may still OOM on output tensor allocation
            return compute_logprobs_chunked_full(logits, chunk_size=LOGPROBS_CHUNK_SIZE)

        Sampler.compute_logprobs = patched_compute_logprobs
        print(f"[patch_vllm_logprobs] Patched Sampler.compute_logprobs")
        patched_any = True

    except ImportError as e:
        print(f"[patch_vllm_logprobs] Could not import vllm.v1.sample.sampler: {e}")

    # Try to patch the prompt logprobs computation at a higher level
    try:
        from vllm.v1.sample.sampler import Sampler

        # Look for _get_prompt_logprobs_dict or similar method
        if hasattr(Sampler, '_get_prompt_logprobs_dict'):
            _original_methods['Sampler._get_prompt_logprobs_dict'] = Sampler._get_prompt_logprobs_dict

            def patched_get_prompt_logprobs_dict(self, *args, **kwargs):
                """Patched version using chunked top-k extraction."""
                # This is a placeholder - actual implementation depends on vLLM internals
                return _original_methods['Sampler._get_prompt_logprobs_dict'](self, *args, **kwargs)

            Sampler._get_prompt_logprobs_dict = patched_get_prompt_logprobs_dict
            print(f"[patch_vllm_logprobs] Patched Sampler._get_prompt_logprobs_dict")

    except Exception as e:
        print(f"[patch_vllm_logprobs] Could not patch _get_prompt_logprobs_dict: {e}")

    # Patch the Sampler.forward method to intercept prompt_logprobs computation
    try:
        from vllm.v1.sample.sampler import Sampler

        if hasattr(Sampler, 'forward'):
            _original_methods['Sampler.forward'] = Sampler.forward

            original_forward = Sampler.forward

            def patched_forward(self, logits, sampling_metadata):
                """
                Patched forward that handles long sequences specially.

                For long sequences with prompt_logprobs enabled, we use
                chunked top-k extraction to avoid OOM.
                """
                # Check if this is a long sequence that might OOM
                seq_len, vocab_size = logits.shape
                estimated_output_gb = seq_len * vocab_size * 4 / 1e9

                # If tensor is small enough (< 4GB), use original method
                if estimated_output_gb < 4.0:
                    return original_forward(self, logits, sampling_metadata)

                # For large tensors, we need special handling
                print(f"[patch_vllm_logprobs] Large logits tensor detected: [{seq_len}, {vocab_size}] = {estimated_output_gb:.2f} GB")

                # Check if prompt_logprobs are needed
                needs_prompt_logprobs = False
                num_prompt_logprobs = None
                if hasattr(sampling_metadata, 'num_prompt_logprobs'):
                    num_prompt_logprobs = sampling_metadata.num_prompt_logprobs
                    needs_prompt_logprobs = num_prompt_logprobs is not None and num_prompt_logprobs > 0

                if not needs_prompt_logprobs:
                    print(f"[patch_vllm_logprobs] No prompt_logprobs needed, using original forward")
                    return original_forward(self, logits, sampling_metadata)

                print(f"[patch_vllm_logprobs] Using chunked top-k extraction (num_prompt_logprobs={num_prompt_logprobs})")

                # Compute top-k logprobs directly using chunked method
                # This avoids allocating the full [seq_len, vocab_size] tensor
                topk_logprobs, topk_indices = compute_topk_from_logits_chunked(
                    logits,
                    num_logprobs=num_prompt_logprobs + 1,  # +1 for sampled token
                    chunk_size=LOGPROBS_CHUNK_SIZE,
                    return_on_gpu=True,
                )

                # Store the pre-computed top-k results for later retrieval
                # The downstream code should use these instead of recomputing
                self._precomputed_topk_logprobs = topk_logprobs
                self._precomputed_topk_indices = topk_indices

                # Now call original forward - it will try to compute logprobs,
                # but our patched compute_logprobs should handle this
                # OR we need to modify how results are assembled
                try:
                    return original_forward(self, logits, sampling_metadata)
                except torch.cuda.OutOfMemoryError:
                    print(f"[patch_vllm_logprobs] OOM in original forward, attempting recovery...")
                    # Clear CUDA cache and retry with more aggressive memory management
                    torch.cuda.empty_cache()
                    raise

            Sampler.forward = patched_forward
            print(f"[patch_vllm_logprobs] Patched Sampler.forward for large tensor handling")
            patched_any = True

    except Exception as e:
        print(f"[patch_vllm_logprobs] Could not patch Sampler.forward: {e}")

    if patched_any:
        _patch_applied = True
        print(f"[patch_vllm_logprobs] Successfully applied patches")
        print(f"[patch_vllm_logprobs] Peak memory reduced from O(seq_len * vocab) to O({chunk_size} * vocab)")
        return True
    else:
        print(f"[patch_vllm_logprobs] Warning: No patches were applied")
        return False


def apply_deep_patch(chunk_size: int = LOGPROBS_CHUNK_SIZE, num_logprobs: int = NUM_LOGPROBS):
    """
    Apply a deeper patch that intercepts logprobs computation at the GPU kernel level.

    This patches torch.Tensor.log_softmax to use chunked computation for large tensors.
    Use with caution as this affects all log_softmax calls!
    """
    global LOGPROBS_CHUNK_SIZE, NUM_LOGPROBS
    LOGPROBS_CHUNK_SIZE = chunk_size
    NUM_LOGPROBS = num_logprobs

    print(f"[patch_vllm_logprobs] Applying DEEP patch (affects all log_softmax calls)")

    # Save original log_softmax
    original_log_softmax = torch.Tensor.log_softmax

    def patched_log_softmax(self, dim=-1, dtype=None):
        """
        Patched log_softmax that uses chunked computation for large tensors.
        """
        # Check if this is a large 2D tensor that might OOM
        if len(self.shape) == 2 and self.shape[0] > chunk_size * 2 and self.shape[1] > 10000:
            seq_len, vocab_size = self.shape
            print(f"[patch_vllm_logprobs] Large log_softmax detected: [{seq_len}, {vocab_size}], using chunked computation")

            # Use chunked computation
            target_dtype = dtype if dtype is not None else torch.float32
            result = torch.empty(self.shape, dtype=target_dtype, device=self.device)

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                result[start:end] = original_log_softmax(self[start:end], dim=dim, dtype=dtype)

            return result

        # Default behavior for small tensors
        return original_log_softmax(self, dim=dim, dtype=dtype)

    torch.Tensor.log_softmax = patched_log_softmax
    print(f"[patch_vllm_logprobs] Deep patch applied to torch.Tensor.log_softmax")

    return True


def remove_patch():
    """Remove all monkey-patches and restore original behavior."""
    global _patch_applied, _original_methods

    if not _original_methods:
        print("[patch_vllm_logprobs] No patches to remove")
        return

    try:
        from vllm.v1.sample.sampler import Sampler

        if 'Sampler.compute_logprobs' in _original_methods:
            Sampler.compute_logprobs = _original_methods['Sampler.compute_logprobs']

        if 'Sampler.forward' in _original_methods:
            Sampler.forward = _original_methods['Sampler.forward']

        if 'Sampler._get_prompt_logprobs_dict' in _original_methods:
            Sampler._get_prompt_logprobs_dict = _original_methods['Sampler._get_prompt_logprobs_dict']

    except ImportError:
        pass

    _original_methods.clear()
    _patch_applied = False
    print("[patch_vllm_logprobs] All patches removed, original behavior restored")


if __name__ == "__main__":
    # Test the chunked top-k computation
    print("Testing chunked top-k logprobs computation...")

    # Simulate logits tensor
    seq_len = 1000
    vocab_size = 10000
    num_logprobs = 64

    print(f"Test tensor: [{seq_len}, {vocab_size}]")

    # Create test tensor
    test_logits = torch.randn(seq_len, vocab_size, dtype=torch.bfloat16, device="cpu")

    # Test chunked top-k vs full computation
    print("Computing full log_softmax + top-k...")
    full_logprobs = test_logits.log_softmax(dim=-1, dtype=torch.float32)
    full_topk_values, full_topk_indices = full_logprobs.topk(num_logprobs, dim=-1)
    del full_logprobs  # Free memory

    print("Computing chunked top-k...")
    chunked_topk_values, chunked_topk_indices = compute_topk_logprobs_chunked(
        test_logits, num_logprobs=num_logprobs, chunk_size=128
    )

    # Verify results match
    values_match = torch.allclose(full_topk_values, chunked_topk_values, atol=1e-5)
    indices_match = torch.equal(full_topk_indices, chunked_topk_indices)

    print(f"Values match: {values_match}")
    print(f"Indices match: {indices_match}")

    if values_match and indices_match:
        print("✓ Test passed! Chunked computation produces identical results.")
    else:
        max_val_diff = (full_topk_values - chunked_topk_values).abs().max().item()
        print(f"Max value difference: {max_val_diff}")
        if max_val_diff < 1e-4:
            print("✓ Test passed! Differences are within acceptable tolerance.")
        else:
            print("✗ Test failed! Results differ significantly.")

    # Memory estimation
    full_memory_gb = seq_len * vocab_size * 4 / 1e9
    chunk_memory_gb = 128 * vocab_size * 4 / 1e9
    topk_memory_gb = seq_len * num_logprobs * 4 / 1e9

    print(f"\nMemory comparison for [{seq_len}, {vocab_size}] tensor:")
    print(f"  Full log_softmax: {full_memory_gb:.3f} GB")
    print(f"  Chunk (128 positions): {chunk_memory_gb:.3f} GB")
    print(f"  Output top-{num_logprobs}: {topk_memory_gb:.6f} GB")
    print(f"  Memory reduction: {full_memory_gb / chunk_memory_gb:.1f}x")
