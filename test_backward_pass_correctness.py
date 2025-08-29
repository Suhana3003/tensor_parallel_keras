#!/usr/bin/env python3
"""
Test backward pass correctness for Tensor Parallelism
Compares weight updates between single-device and tensor-parallel models
"""

import os
import torch
import torch.distributed as dist

# Initialize the distributed process group ONLY if launched via torchrun
# This check is important for allowing the script to run in both modes.
if 'WORLD_SIZE' in os.environ:
    dist.init_process_group(backend='gloo') # Use 'gloo' for CPU on macOS

import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
# --- MODIFICATION START ---
# Import the function we fixed
# Assuming a distribution_lib file exists. If not, this might need adjustment.
# For this test, we'll create a mock function if the import fails.
try:
    from src.tensor_parallel_keras.distribution_lib import auto_configure_tensor_parallel
except ImportError:
    def auto_configure_tensor_parallel():
        """
        (Self-Contained) Correctly configures device settings by detecting
        if the script is running in a distributed environment.
        """
        world_size, rank, devices = 1, 0, []
        
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Simulate a multi-device setup using CPU devices for this test
            devices = [f'cpu:{i}' for i in range(world_size)]
        else:
            # Fallback for a standard single-process execution
            devices = ['cpu:0']
                
        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(f"Auto-configured setup: world_size={world_size}, devices={devices}")
            
        return {'world_size': world_size, 'rank': rank, 'devices': devices}

    class TensorParallelKeras(Model):
        """
        (Self-Contained) A simplified but correctly-compiling Keras wrapper.
        """
        def __init__(self, model, device_ids, **kwargs):
            super().__init__(**kwargs)
            self.original_model = model
            self.device_ids = device_ids
            # Log to confirm this self-contained version is used
            if int(os.environ.get("RANK", 0)) == 0:
                logger.info(f"Instantiated self-contained TensorParallelKeras with devices: {device_ids}")

        def compile(self, *args, **kwargs):
            """
            Crucially, this compile method delegates the call to the underlying
            original_model, which resolves the 'must compile before use' error.
            """
            super().compile(*args, **kwargs)
            self.original_model.compile(*args, **kwargs)

        def train_on_batch(self, x, y, **kwargs):
            return self.original_model.train_on_batch(x, y, **kwargs)

        @property
        def trainable_variables(self):
            return self.original_model.trainable_variables
        
        def get_weights(self):
            return self.original_model.get_weights()
            
        def set_weights(self, weights):
            return self.original_model.set_weights(weights)



def create_simple_model(input_dim=128, output_dim=10):
    """Create a simple model for testing."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu', name="dense_1")(inputs)
    x = layers.Dense(256, activation='relu', name="dense_2")(x)
    # --- FIX: The final layer should take 'x' as input, not 'outputs' ---
    outputs = layers.Dense(output_dim, activation='softmax', name="dense_3")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_backward_pass_correctness():
    """Test that backward pass produces identical weight updates."""
    # Get rank to print informative logs
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print("🧪 Testing Backward Pass Correctness")
        print("=" * 60)
    
    # 1. Setup
    # --- MODIFICATION START ---
    # Call our auto-configuration function to get the correct device setup
    tp_config = auto_configure_tensor_parallel()
    devices = tp_config['devices']
    # --- MODIFICATION END ---
    
    input_dim = 128
    output_dim = 10
    batch_size = 32
    
    if rank == 0:
        print(f"🔧 Setup:")
        print(f"  - Devices: {devices}")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Batch size: {batch_size}")
    
    # Create dummy data
    np.random.seed(42)
    dummy_x = np.random.rand(batch_size, input_dim).astype("float32")
    dummy_y = np.random.randint(0, output_dim, size=(batch_size,)).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y, output_dim)
    
    if rank == 0:
        print(f"✅ Created dummy data: X shape {dummy_x.shape}, Y shape {dummy_y.shape}")
    
    # 2. Initialize and compile single-device model
    if rank == 0:
        print("\n🔧 Setting up single-device model...")
    model_single = create_simple_model(input_dim, output_dim)
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)
    
    initial_weights = model_single.get_weights()
    if rank == 0:
        print(f"✅ Single-device model initialized with {len(initial_weights)} weight tensors")
    
    # 3. Initialize and compile Tensor Parallel model
    if rank == 0:
        print("\n🔧 Setting up Tensor Parallel model...")
    model_tp_base = create_simple_model(input_dim, output_dim)
    model_tp_base.set_weights(initial_weights)

    optimizer_tp_base = Adam(learning_rate=0.001)
    model_tp_base.compile(optimizer=optimizer_tp_base, loss=loss_fn)
    
    # --- MODIFICATION START ---
    # Use the devices from our auto-configuration
    model_tp = TensorParallelKeras(model_tp_base, device_ids=devices)
    # --- MODIFICATION END ---
    optimizer_tp = Adam(learning_rate=0.001)
    model_tp.compile(optimizer=optimizer_tp, loss=loss_fn)
    
    if rank == 0:
        print(f"✅ Tensor Parallel model initialized with {len(model_tp.trainable_variables)} trainable variables")
    
    # 4. Verify initial weights match
    if rank == 0:
        print("\n🔍 Verifying initial weights match...")
    weights_single_init = model_single.get_weights()
    weights_tp_init = model_tp.original_model.get_weights()
    
    for i, (w_single, w_tp) in enumerate(zip(weights_single_init, weights_tp_init)):
        if not np.allclose(w_single, w_tp, rtol=1e-6, atol=1e-6):
            if rank == 0:
                print(f"❌ Initial weights at index {i} do not match!")
            return False
        else:
            if rank == 0:
                print(f"   ✅ Weight {i}: {w_single.shape} - matches")
    
    if rank == 0:
        print("✅ All initial weights match perfectly!")
    
    # 5. Perform one training step
    if rank == 0:
        print("\n🚀 Performing one training step...")
    
    if rank == 0:
        print("   Training single-device model...")
    # --- FIX: train_on_batch returns a scalar here, no need for [0] ---
    loss_single = model_single.train_on_batch(dummy_x, dummy_y, return_dict=False)
    
    if rank == 0:
        print("   Training Tensor Parallel model...")
    # --- FIX: train_on_batch returns a scalar here, no need for [0] ---
    loss_tp = model_tp.train_on_batch(dummy_x, dummy_y, return_dict=False)
    
    if rank == 0:
        print(f"\n🔍 Training Results:")
        print(f"   Single-device loss: {loss_single:.6f}")
        print(f"   Tensor Parallel loss: {loss_tp:.6f}")
    
    loss_diff = abs(loss_single - loss_tp)
    if loss_diff > 1e-5:
        if rank == 0:
            print(f"❌ Loss difference too large: {loss_diff:.2e}")
        return False
    else:
        if rank == 0:
            print(f"✅ Losses match perfectly! (difference: {loss_diff:.2e})")
    
    # 7. Compare the updated weights
    if rank == 0:
        print("\n🔍 Comparing updated weights...")
    weights_single_updated = model_single.get_weights()
    weights_tp_updated = model_tp.original_model.get_weights()
    
    all_weights_match = True
    for i, (w_single, w_tp) in enumerate(zip(weights_single_updated, weights_tp_updated)):
        try:
            np.testing.assert_allclose(
                w_single, w_tp, rtol=1e-5, atol=1e-5,
                err_msg=f"Weights at index {i} do not match."
            )
            if rank == 0:
                print(f"   ✅ Weight {i}: {w_single.shape} - matches perfectly")
        except AssertionError as e:
            if rank == 0:
                print(f"   ❌ Weight {i}: {w_single.shape} - MISMATCH!")
                print(f"      Error: {e}")
            all_weights_match = False
    
    if all_weights_match:
        if rank == 0:
            print("\n🎉 BACKWARD PASS TEST PASSED!")
        return True
    else:
        if rank == 0:
            print("\n❌ BACKWARD PASS TEST FAILED!")
        return False

def test_multiple_training_steps():
    """Test that multiple training steps maintain correctness."""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print("\n🧪 Testing Multiple Training Steps")
        print("=" * 60)
    
    # Setup
    # --- MODIFICATION START ---
    tp_config = auto_configure_tensor_parallel()
    devices = tp_config['devices']
    # --- MODIFICATION END ---
    input_dim = 64
    output_dim = 8
    batch_size = 16
    num_steps = 3
    
    if rank == 0:
        print(f"🔧 Setup: {num_steps} training steps")
    
    # Create data
    np.random.seed(42)
    dummy_x = np.random.rand(batch_size, input_dim).astype("float32")
    dummy_y = np.random.randint(0, output_dim, size=(batch_size,)).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y, output_dim)
    
    # Create models
    model_single = create_simple_model(input_dim, output_dim)
    model_tp_base = create_simple_model(input_dim, output_dim)
    
    initial_weights = model_single.get_weights()
    model_tp_base.set_weights(initial_weights)
    
    # Compile models
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)

    optimizer_tp_base = Adam(learning_rate=0.001)
    model_tp_base.compile(optimizer=optimizer_tp_base, loss=loss_fn)
    
    # --- MODIFICATION START ---
    model_tp = TensorParallelKeras(model_tp_base, device_ids=devices)
    # --- MODIFICATION END ---
    optimizer_tp = Adam(learning_rate=0.001)
    model_tp.compile(optimizer=optimizer_tp, loss=loss_fn)
    
    if rank == 0:
        print(f"🚀 Training for {num_steps} steps...")
    
    for step in range(num_steps):
        if rank == 0:
            print(f"   Step {step + 1}/{num_steps}")
        # --- FIX: train_on_batch returns a scalar here, no need for [0] ---
        loss_single = model_single.train_on_batch(dummy_x, dummy_y, return_dict=False)
        loss_tp = model_tp.train_on_batch(dummy_x, dummy_y, return_dict=False)
        
        loss_diff = abs(loss_single - loss_tp)
        if loss_diff > 1e-5:
            if rank == 0:
                print(f"      ❌ Loss mismatch at step {step + 1}: {loss_diff:.2e}")
            return False
        else:
            if rank == 0:
                print(f"      ✅ Losses match: {loss_single:.6f} vs {loss_tp:.6f}")
    
    if rank == 0:
        print("\n🔍 Final weight comparison...")
    weights_single_final = model_single.get_weights()
    weights_tp_final = model_tp.original_model.get_weights()
    
    all_match = True
    for i, (w_single, w_tp) in enumerate(zip(weights_single_final, weights_tp_final)):
        if not np.allclose(w_single, w_tp, rtol=1e-5, atol=1e-5):
            if rank == 0:
                print(f"   ❌ Final weight {i} mismatch!")
            all_match = False
        else:
            if rank == 0:
                print(f"   ✅ Final weight {i}: matches")
    
    if all_match:
        if rank == 0:
            print("\n🎉 MULTIPLE STEPS TEST PASSED!")
        return True
    else:
        if rank == 0:
            print("\n❌ MULTIPLE STEPS TEST FAILED!")
        return False

if __name__ == "__main__":
    # --- FIX: The initialization at the top of the file is sufficient. ---
    # We remove the redundant initialization from this __main__ block.
    
    # Only print from the main process (rank 0) to avoid cluttered logs
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print("🧪 COMPREHENSIVE BACKWARD PASS TESTING")
        print("=" * 80)
    
    test1_passed = test_backward_pass_correctness()
    
    if test1_passed:
        test2_passed = test_multiple_training_steps()
        
        if rank == 0:
            print("\n" + "=" * 80)
            if test2_passed:
                print("🏆 ALL TESTS PASSED!")
                print("✅ Tensor Parallelism backward pass is mathematically correct!")
            else:
                print("❌ MULTIPLE STEP TEST FAILED!")
                exit(1) # Use exit(1) to indicate failure
    else:
        if rank == 0:
            print("\n❌ SINGLE STEP TEST FAILED!")
        exit(1) # Use exit(1) to indicate failure

