#!/usr/bin/env python3
"""
Test simple Dense layer tensor parallelism execution with a PyTorch backend.
This version is adapted for distributed execution with torchrun.
"""

import os
import numpy as np
import torch
import torch.distributed as dist

# Import Keras and its layers
import keras
from keras.layers import Input, Dense

# Import the TensorParallelKeras wrapper and config utility from your library
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
from src.tensor_parallel_keras.distribution_lib import auto_configure_tensor_parallel


# --- Distributed Setup ---
# Initialize the process group ONLY when launched via torchrun
if 'WORLD_SIZE' in os.environ:
    dist.init_process_group(backend='gloo')

# Get rank for logging purposes
RANK = int(os.environ.get("RANK", 0))

# --- PyTorch Device Detection ---
if RANK == 0:
    print("🔍 PyTorch Device Detection:")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
    # Setting default device is helpful for tensor creation
    torch.set_default_device(DEVICE)
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

if RANK == 0:
    print(f"   - Using device: {DEVICE}")


def create_simple_model():
    """Create a simple Dense model."""
    inputs = Input(shape=(64,), name='input_tensor')
    output = Dense(32, activation='relu', name='dense')(inputs)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_dense_execution():
    """Test simple Dense layer tensor parallelism execution."""
    if RANK == 0:
        print("\nTesting simple Dense layer tensor parallelism execution (PyTorch)...")
    
    # Create model
    model = create_simple_model()
    if RANK == 0:
        print(f"Model created with {len(model.layers)} layers")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # --- MODIFICATION: Call the config utility and print the devices ---
    # This confirms what devices the library should be using internally.
    if RANK == 0:
        tp_config = auto_configure_tensor_parallel()
        print(f"Auto-configured device IDs: {tp_config.get('devices')}")

    # Create tensor parallel model with PyTorch backend
    tp_model = TensorParallelKeras(
        model=model,
        world_size=world_size, # Use detected world size
        distributed_backend='torch'
    )
    if RANK == 0:
        print(f"Tensor parallel model created for world_size={world_size}")
    
    # Create test input
    input_data = np.random.random((8, 64)).astype(np.float32)
    if RANK == 0:
        print(f"Input data shape: {input_data.shape}")
    
    # Run single device model
    single_output = model(input_data)
    
    # Run tensor parallel model
    tp_output = tp_model(input_data)

    # Only perform checks and logging on the main process
    if RANK == 0:
        print(f"Single device output shape: {single_output.shape}")
        print(f"Tensor parallel output shape: {tp_output.shape}")
        
        # Check shapes match
        shape_match = single_output.shape == tp_output.shape
        print(f"Shape match: {shape_match}")
        
        if shape_match:
            # FIX: Detach tensors from the computation graph before converting
            single_np = single_output.cpu().detach().numpy()
            tp_np = tp_output.cpu().detach().numpy()
            
            # Calculate differences
            abs_diff = np.abs(single_np - tp_np)
            
            print(f"Max absolute difference: {np.max(abs_diff):.2e}")
            
            # Check if within tolerance
            tolerance = 1e-5
            within_tolerance = np.max(abs_diff) < tolerance
            
            if within_tolerance:
                print("✅ MATHEMATICAL IDENTITY ACHIEVED! (within tolerance)")
            else:
                print("❌ Mathematical differences detected")
                
            # Show sample values
            print("\nSample values:")
            print(f"  Single device: {single_np[0, :5]}")
            print(f"  Tensor parallel: {tp_np[0, :5]}")
            print(f"  Differences: {abs_diff[0, :5]}")
        else:
            print("❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    if keras.backend.backend() != 'torch':
        raise RuntimeError(
            "This test requires the Keras backend to be set to 'torch'.\n"
            "Please set the environment variable: KERAS_BACKEND=torch"
        )
    test_dense_execution()

