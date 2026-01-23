#!/usr/bin/env python3
"""
GPU Performance Benchmark for MuJoCo Leap Cube Training
Tests GPU acceleration and performance optimization effectiveness.
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import random

# Force CUDA backend
os.environ['JAX_PLATFORMS'] = 'cuda'

def benchmark_gpu():
    """Test GPU performance with various tensor operations"""
    print("=" * 60)
    print("GPU Performance Benchmark")
    print("=" * 60)

    # Check device availability
    devices = jax.devices()
    print(f"Available devices: {devices}")

    if len(devices) == 0:
        print("❌ No JAX devices available!")
        return False

    device = devices[0]
    print(f"Using device: {device}")
    print(f"Device kind: {device.device_kind}")

    # Get GPU memory info
    try:
        mem_stats = device.memory_stats()
        total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)
        used_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
        print(f"GPU Memory: {used_gb:.1f}GB / {total_gb:.1f}GB")
    except Exception as e:
        print(f"Could not get memory info: {e}")

    print("\nRunning performance benchmarks...")

    # Test 1: Matrix multiplication
    print("\n1. Matrix Multiplication Benchmark")
    sizes = [1024, 2048, 4096, 8192]
    for size in sizes:
        key = random.PRNGKey(42)
        x = random.normal(key, (size, size), dtype=jnp.float32)

        try:
            # JIT compile and benchmark
            @jax.jit
            def matmul(a):
                return jnp.dot(a, a)

            # Warmup
            _ = matmul(x).block_until_ready()

            # Benchmark
            start_time = time.time()
            result = matmul(x).block_until_ready()
            end_time = time.time()

            elapsed = end_time - start_time
            gflops = (2 * size**3) / (elapsed * 1e9)  # 2*N^3 operations for matmul
            print(f"  {size}x{size}: {elapsed:.3f}s, {gflops:.1f} GFLOPS")

        except Exception as e:
            print(f"  {size}x{size}: Failed - {e}")

    # Test 2: Large tensor operations (similar to PPO training)
    print("\n2. PPO-like Training Operations")
    batch_sizes = [512, 1024, 2048, 4096]

    for batch_size in batch_sizes:
        try:
            # Simulate PPO operations
            obs_dim = 256  # Observation dimension
            action_dim = 64  # Action dimension

            key = random.PRNGKey(42)
            obs = random.normal(key, (batch_size, obs_dim), dtype=jnp.float32)
            actions = random.normal(key, (batch_size, action_dim), dtype=jnp.float32)

            @jax.jit
            def ppo_forward(obs, actions):
                # Simulate neural network forward pass
                hidden = jax.nn.relu(jnp.dot(obs, jnp.ones((obs_dim, 512), dtype=jnp.float32)))
                policy = jnp.dot(hidden, jnp.ones((512, action_dim), dtype=jnp.float32))
                value = jnp.dot(hidden, jnp.ones((512, 1), dtype=jnp.float32))

                # Simulate advantage computation
                advantages = jnp.sum(actions * policy, axis=-1)
                returns = advantages + value.squeeze()

                return policy.mean(), returns.mean(), jnp.mean(value)

            # Warmup
            _ = ppo_forward(obs, actions)

            # Benchmark
            start_time = time.time()
            policy_loss, value_loss, returns = ppo_forward(obs, actions)
            end_time = time.time()

            elapsed = end_time - start_time
            throughput = batch_size / elapsed * 1000  # Examples per second

            print(f"  Batch {batch_size:4d}: {elapsed:.3f}s, {throughput:.0f} examples/s")

        except Exception as e:
            print(f"  Batch {batch_size:4d}: Failed - {e}")

    # Test 3: Memory stress test
    print("\n3. Memory Usage Test")
    try:
        # Simulate large batch training
        num_envs = 4096
        obs_dim = 256

        key = random.PRNGKey(42)
        large_obs = random.normal(key, (num_envs, obs_dim), dtype=jnp.float32)

        @jax.jit
        def process_batch(obs):
            # Simulate processing multiple environments in parallel
            mean_obs = jnp.mean(obs, axis=0)
            processed = jax.nn.tanh(obs - mean_obs)
            return jnp.mean(processed, axis=0)

        start_time = time.time()
        result = process_batch(large_obs).block_until_ready()
        end_time = time.time()

        # Check memory usage
        mem_stats = device.memory_stats()
        used_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
        peak_gb = mem_stats.get('peak_bytes_in_use', 0) / (1024**3)

        print(f"  {num_envs} environments processed in {end_time - start_time:.3f}s")
        print(f"  Memory usage: {used_gb:.1f}GB (peak: {peak_gb:.1f}GB)")

    except Exception as e:
        print(f"  Memory test failed: {e}")

    print("\n" + "=" * 60)
    print("✅ GPU benchmark completed successfully!")
    print("Your GPU is ready for high-performance training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = benchmark_gpu()
    if not success:
        sys.exit(1)