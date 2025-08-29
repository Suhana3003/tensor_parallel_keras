#!/usr/bin/env python3
"""
Debug AllGather operation to understand the shape issue
"""

import numpy as np
import keras
from src.tensor_parallel_keras.communications_keras import AllGatherKeras

def test_allgather_debug():
    """Test AllGather operation with simple tensors."""
    print("🧪 Debugging AllGather Operation")
    print("=" * 40)
    
    # Create simple test tensors
    tensor1 = keras.ops.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    tensor2 = keras.ops.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    
    # Print tensor details, including which device it's on
    print(f"Tensor 1 created on device: {tensor1.device}")
    print(f"Tensor 1 shape: {tensor1.shape}, values:\n{np.array(tensor1.cpu())}")
    print(f"\nTensor 2 created on device: {tensor2.device}")
    print(f"Tensor 2 shape: {tensor2.shape}, values:\n{np.array(tensor2.cpu())}")
    
    # Test AllGather along last dimension (dim=-1)
    allgather = AllGatherKeras(world_size=2, dim=-1)
    
    try:
        # Simulate the input from different devices/processes
        tensors_from_world = [tensor1, tensor2]
        result = allgather(tensors_from_world)
        
        print(f"\nAllGather operation ran. Result is on device: {result.device}")
        print(f"AllGather result shape: {result.shape}")
        # Use .cpu() only to copy the result from GPU to CPU for printing
        print(f"AllGather result values:\n{np.array(result.cpu())}")
        
        # Expected result should be (2, 4) - concatenating along features
        expected_shape = (2, 4)
        if result.shape == expected_shape:
            print("✅ AllGather shape is correct!")
        else:
            print(f"❌ AllGather shape is wrong! Expected {expected_shape}, got {result.shape}")
            
    except Exception as e:
        print(f"❌ AllGather failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allgather_debug() 