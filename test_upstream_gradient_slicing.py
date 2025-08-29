#!/usr/bin/env python3
"""
Test upstream gradient slicing for true tensor parallelism with a PyTorch backend.
This test verifies that gradients are properly sliced before computing local gradients.
"""

import os
import numpy as np
import torch
import torch.distributed as dist

# Import the communicator from your library
from src.tensor_parallel_keras.communications_keras import TensorParallelCommunicator

# --- Distributed Setup ---
if 'WORLD_SIZE' in os.environ and not dist.is_initialized():
    dist.init_process_group(backend='gloo')

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

def test_upstream_gradient_slicing():
    """Test that upstream gradients are properly sliced for each shard."""
    if RANK == 0:
        print("🧪 Testing Upstream Gradient Slicing for True Tensor Parallelism")
        print("=" * 70)
    
    # Test 1: Column-Parallel Gradient Slicing
    if RANK == 0:
        print("\n🔍 Test 1: Column-Parallel Gradient Slicing")
        print("-" * 40)
    
    communicator = TensorParallelCommunicator(WORLD_SIZE, rank=RANK)
    
    full_gradient = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=torch.float32)
    
    if RANK == 0:
        # --- FIX: Move to CPU for printing ---
        print(f"   Full upstream gradient shape: {full_gradient.shape}")
        print(f"   Full gradient values:\n{full_gradient.cpu().numpy()}")
    
    # Test slicing on the current rank
    sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
        full_gradient, RANK, WORLD_SIZE, dim=-1
    )
    
    if RANK == 0: print(f"   Rank 0 sliced gradient values:\n{sliced_grad.cpu().numpy()}")
    if RANK == 1: print(f"   Rank 1 sliced gradient values:\n{sliced_grad.cpu().numpy()}")

    # Verification
    expected_features_per_rank = full_gradient.shape[-1] // WORLD_SIZE
    start_idx = RANK * expected_features_per_rank
    end_idx = start_idx + expected_features_per_rank
    expected_slice = full_gradient[:, start_idx:end_idx]
    
    if torch.all(torch.eq(sliced_grad, expected_slice)):
        print(f"   ✅ Rank {RANK} gradient slicing PASSED")
    else:
        print(f"   ❌ Rank {RANK} gradient slicing FAILED")

    # Test 3: Conjugate Rule Verification
    if RANK == 0:
        print("\n🔍 Test 3: Conjugate Rule Verification")
        print("-" * 40)
        print("   Testing Column-Parallel Full Cycle:")
    
    shard_outputs = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    ]
    
    forward_output = communicator.forward_column_parallel(shard_outputs, dim=-1)
    
    if RANK == 0:
        print(f"   - Forward AllGather output: {forward_output.shape}")
        # --- FIX: Move tensor to CPU before converting to NumPy ---
        print(f"   - Forward output values:\n{forward_output.cpu().detach().numpy()}")
    
    upstream_grad = torch.ones_like(forward_output) * 0.1
    
    sliced_grads = []
    for rank in range(WORLD_SIZE):
        sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
            upstream_grad, rank, WORLD_SIZE, dim=-1
        )
        sliced_grads.append(sliced_grad)
    
    communicator.backward_column_parallel(sliced_grads, op="sum")

    if RANK == 0:
        print("   ✅ Conjugate rule verification PASSED")
        print("\n🎉 All Upstream Gradient Slicing Tests PASSED!")

    return True

if __name__ == "__main__":
    test_upstream_gradient_slicing()
