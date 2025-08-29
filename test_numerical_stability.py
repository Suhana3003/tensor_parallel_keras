#!/usr/bin/env python3
"""
Tests the numerical correctness and stability of the TensorParallelKeras wrapper
with a PyTorch backend in a distributed environment.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import keras

# Import the TensorParallelKeras wrapper from your library
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# --- Distributed Setup ---
if 'WORLD_SIZE' in os.environ:
    dist.init_process_group(backend='gloo')

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

# --- Configuration ---
BATCH_SIZE = 8
SEQ_LEN = 16
INPUT_DIM = 64
MLP_DIM = 256
TOLERANCE = 1e-6

keras.utils.set_random_seed(42)

def build_test_model():
    """Builds a simple two-layer MLP for testing."""
    inp = keras.Input(shape=(SEQ_LEN, INPUT_DIM))
    x = keras.layers.Dense(MLP_DIM, activation="relu", name="mlp_up")(inp)
    out = keras.layers.Dense(INPUT_DIM, name="mlp_down")(x)
    model = keras.Model(inputs=inp, outputs=out, name="OriginalMLP")
    return model

def compare_model_outputs_and_weights(
    original_model, tp_model, input_data, dummy_target
):
    """
    Compares outputs and weights after a single training step.
    """
    if RANK == 0:
        print("\n" + "-" * 80)
        print("📊 STEP 1: COMPARING FORWARD PASS OUTPUTS 📊")
        print("-" * 80)

    output_original = original_model(input_data)
    output_tp = tp_model(input_data)
    
    if RANK == 0:
        # Convert PyTorch tensors to NumPy arrays for comparison
        output_original_np = output_original.cpu().detach().numpy()
        output_tp_np = output_tp.cpu().detach().numpy()
        forward_pass_diff = np.max(np.abs(output_original_np - output_tp_np))

        print(f"   Original model output calculated. Shape: {output_original.shape}")
        print(f"   Tensor parallel model output calculated. Shape: {output_tp.shape}")
        print(f"\n   Maximum absolute difference in forward pass: {forward_pass_diff:.2e}")

        if forward_pass_diff < TOLERANCE:
            print("   ✅ PASSED: Forward pass outputs are numerically identical.")
        else:
            print("   ❌ FAILED: Forward pass outputs differ.")
            return False

    # --- BACKWARD PASS TEST ---
    if RANK == 0:
        print("\n" + "-" * 80)
        print("📊 STEP 2: COMPARING WEIGHTS AFTER ONE TRAINING STEP 📊")
        print("-" * 80)
        print("   Performing one training step on original model...")
        
    original_model.train_on_batch(input_data, dummy_target)
    
    if RANK == 0:
        print("   Original model weights updated.")
        print("   Performing one training step on tensor parallel model...")

    tp_model.train_on_batch(input_data, dummy_target)

    if RANK == 0:
        print("   Tensor parallel model weights updated.")
        weights_original_after_step = original_model.get_weights()
        weights_tp_after_step = tp_model.original_model.get_weights()

        all_weights_match = True
        for i, (w_orig, w_tp) in enumerate(zip(weights_original_after_step, weights_tp_after_step)):
            weight_diff = np.max(np.abs(w_orig - w_tp))
            param_name = original_model.weights[i].name
            if weight_diff >= TOLERANCE:
                all_weights_match = False
                print(f"   ❌ MISMATCH on parameter '{param_name}': Max difference = {weight_diff:.2e}")

        if all_weights_match:
            print("\n   ✅ PASSED: All weights are numerically identical after one training step.")
        else:
            print("\n   ❌ FAILED: Weights differ between models after one training step.")
        
        return all_weights_match

def run_test():
    """Runs the full numerical stability and correctness test."""
    if RANK == 0:
        print("=" * 80)
        print("🚀 Starting Tensor Parallel Keras PyTorch Numerical Correctness Test 🚀")
        print("=" * 80)

    original_model = build_test_model()
    model_for_tp = build_test_model()
    
    tp_model = TensorParallelKeras(
        model_for_tp,
        world_size=WORLD_SIZE,
        distributed_backend='torch'  # Specify the PyTorch backend
    )
    tp_model.set_weights(original_model.get_weights())

    optimizer_orig = keras.optimizers.Adam(learning_rate=0.001)
    optimizer_tp = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.MeanSquaredError()
    
    original_model.compile(optimizer=optimizer_orig, loss=loss_fn)
    tp_model.compile(optimizer=optimizer_tp, loss=loss_fn)

    # Use NumPy to create data that can be shared across processes
    input_data_np = np.random.normal(size=(BATCH_SIZE, SEQ_LEN, INPUT_DIM)).astype('float32')
    dummy_target_np = np.random.normal(size=(BATCH_SIZE, SEQ_LEN, INPUT_DIM)).astype('float32')
    
    compare_model_outputs_and_weights(
        original_model, tp_model, input_data_np, dummy_target_np
    )

if __name__ == "__main__":
    if keras.backend.backend() != 'torch':
        raise RuntimeError("This test requires the Keras backend to be 'torch'.")
    run_test()
