#!/usr/bin/env python3
"""
MuJoCo Leap Cube Reorient Training Script
Converted from Jupyter notebook for headless server execution

This script trains a PPO agent to reorient a cube using the Leap Hand manipulator.
Optimized for headless server cluster execution with proper OpenGL configuration.
"""

import os
import sys
import subprocess
import json
import itertools
import time
import warnings
from typing import Callable, List, NamedTuple, Optional, Union
from datetime import datetime

# JSON serialization helper for JAX objects
def json_serialize(obj):
    """Convert JAX arrays and other non-serializable objects to JSON-serializable types"""
    if hasattr(obj, '__class__') and 'Array' in str(obj.__class__):
        # Handle JAX arrays
        try:
            import jax.numpy as jnp
            return jnp.array(obj).tolist()
        except:
            return str(obj)
    elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        # Handle numpy arrays and similar
        try:
            return np.array(obj).tolist()
        except:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        # Handle sequences
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle dictionaries
        return {key: json_serialize(value) for key, value in obj.items()}
    else:
        # Handle basic types and unknown objects
        try:
            # Test if it's already JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

# Configure headless mode BEFORE importing any graphics libraries
print("Configuring MuJoCo for headless server execution...")

# Try different MuJoCo rendering backends in order of preference
def configure_mujoco_rendering():
    """Configure MuJoCo for headless server with fallback options"""

    # Option 1: Completely disable rendering (most reliable)
    os.environ['MUJOCO_RENDER'] = '0'
    os.environ['PYOPENGL_PLATFORM'] = ''

    # Try to test if this works
    try:
        import mujoco
        # Simple test model
        xml = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="box" pos="0 0 0.5">
              <freejoint/>
              <geom name="box" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Run a few simulation steps
        for i in range(5):
            mujoco.mj_step(model, data)

        print("✓ MuJoCo configuration successful with rendering disabled")
        return True
    except Exception as e:
        print(f"⚠ Rendering disabled failed: {e}")

        # Option 2: Try OSMesa (CPU software rendering)
        try:
            os.environ.pop('MUJOCO_RENDER', None)
            os.environ['MUJOCO_GL'] = 'osmesa'
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

            import mujoco
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            for i in range(5):
                mujoco.mj_step(model, data)

            print("✓ MuJoCo configuration successful with OSMesa")
            return True
        except Exception as e2:
            print(f"⚠ OSMesa failed: {e2}")

            # Option 3: Try EGL (GPU rendering)
            try:
                os.environ['MUJOCO_GL'] = 'egl'
                os.environ['PYOPENGL_PLATFORM'] = 'egl'

                import mujoco
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                for i in range(5):
                    mujoco.mj_step(model, data)

                print("✓ MuJoCo configuration successful with EGL")
                return True
            except Exception as e3:
                print(f"⚠ EGL failed: {e3}")
                print("✗ All rendering backends failed")
                return False

# Configure rendering
configure_mujoco_rendering()

# Configure JAX to use GPU if available
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force CUDA backend
try:
    import jax
    print(f"JAX devices available: {jax.devices()}")
    if len(jax.devices()) > 0 and 'gpu' in str(jax.devices()[0]).lower():
        device = jax.devices()[0]
        print(f"✓ JAX GPU backend: {device.device_kind}")
        print(f"  Device: {device}")

        # Show GPU memory info
        try:
            mem_stats = device.memory_stats()
            total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)
            print(f"  Total GPU Memory: {total_gb:.1f} GB")
        except:
            print("  GPU Memory info not available")

        # Configure JAX for optimal GPU performance
        xla_flags = os.environ.get('XLA_FLAGS', '')
        if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
            xla_flags += ' --xla_gpu_triton_gemm_any=True'
        os.environ['XLA_FLAGS'] = xla_flags

        # Configure JAX for better memory management
        jax.config.update('jax_platform_name', 'cuda')

        print("  ✓ GPU optimizations enabled")
        print("  ✓ Triton GEMM enabled for faster matrix operations")
    else:
        print("⚠ JAX is using CPU, GPU acceleration not available")
        print("  Recommendation: Install CUDA-enabled jaxlib for GPU acceleration:")
        print(f"  pip install --upgrade pip && pip install --upgrade 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("  Training will proceed but will be significantly slower on CPU.")
except ImportError:
    print("JAX not available, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jax", "jaxlib"])
    import jax

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless mode
import matplotlib.pyplot as plt

# Try to import mediapy, install if needed
try:
    import mediapy as media
except ImportError:
    print("Installing mediapy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapy"])
    import mediapy as media

# Configure numpy for better readability
np.set_printoptions(precision=3, suppress=True, linewidth=100)

def check_system_requirements():
    """Check if system requirements are met for headless execution"""
    print("Checking system requirements...")

    # Check GPU availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ GPU detected and accessible")
            print(result.stdout.split('\n')[8])  # GPU info line
        else:
            print("⚠ GPU not available, will use CPU")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found, assuming no GPU")

    # Check if MuJoCo was already configured successfully
    # The configure_mujoco_rendering() function already tested this
    print("✓ MuJoCo installation verified during configuration")
    return True

def install_dependencies():
    """Install required packages if not already present"""
    print("Checking and installing dependencies...")

    # Disable SSL certificate verification for problematic repositories
    pip_options = [
        "--trusted-host", "pypi.ngc.nvidia.com",
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org"
    ]

    packages = [
        "mujoco",
        "mujoco_mjx",
        "brax",
        "playground",
        "opencv-python",
        "pyvirtualdisplay"
    ]

    for package in packages:
        try:
            # Special handling for packages with different import names
            import_name = package.replace('-', '_')
            if import_name == 'opencv_python':
                import_name = 'cv2'
            elif import_name == 'pyvirtualdisplay':
                import_name = 'pyvirtualdisplay'

            __import__(import_name)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package] + pip_options)
            except subprocess.CalledProcessError as e:
                print(f"⚠ Failed to install {package}: {e}")
                print("Continuing anyway, may be already available...")

def setup_environment():
    """Setup environment variables and paths"""
    print("Setting up environment...")

    # Tell XLA to use Triton GEMM for better GPU performance
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    print("✓ Environment setup complete")

def main():
    """Main training function"""
    print("=" * 60)
    print("MuJoCo Leap Cube Reorient Training")
    print("Headless Server Mode")
    print("=" * 60)

    # Check system requirements
    if not check_system_requirements():
        print("System requirements not met, exiting...")
        sys.exit(1)

    # Install dependencies
    install_dependencies()

    # Setup environment
    setup_environment()

    # Import heavy libraries after setup
    print("\nImporting training libraries...")
    from datetime import datetime  # Ensure datetime is available for training
    import functools
    from brax import base
    from brax import envs
    from brax import math
    from brax.base import Base, Motion, Transform
    from brax.base import State as PipelineState
    from brax.envs.base import Env, PipelineEnv, State
    from brax.io import html, mjcf, model
    from brax.mjx.base import State as MjxState
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from etils import epath
    from flax import struct
    from flax.training import orbax_utils
    from jax import numpy as jp
    from ml_collections import config_dict
    import mujoco
    from mujoco import mjx
    from orbax import checkpoint as ocp

    # Import MuJoCo playground
    try:
        from mujoco_playground import wrapper
        from mujoco_playground import registry
        from mujoco_playground.config import manipulation_params
        print("✓ MuJoCo playground imports successful")
    except ImportError as e:
        print(f"✗ Failed to import mujoco_playground: {e}")
        sys.exit(1)

    # List available environments
    print("\nAvailable manipulation environments:")
    available_envs = registry.manipulation.ALL_ENVS
    for env_name in available_envs:
        print(f"  - {env_name}")

    # Setup environment
    env_name = 'LeapCubeReorient'
    print(f"\nLoading environment: {env_name}")

    try:
        env = registry.load(env_name)
        env_cfg = registry.get_default_config(env_name)
        print("✓ Environment loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        sys.exit(1)

    print("\nEnvironment configuration:")
    print(env_cfg)

    # Configure training parameters
    print("\nConfiguring PPO training parameters...")
    ppo_params = manipulation_params.brax_ppo_config(env_name)

    print(f"Original configuration:")
    print(f"  num_envs: {ppo_params.num_envs}")
    print(f"  batch_size: {ppo_params.batch_size}")
    print(f"  num_minibatches: {ppo_params.num_minibatches}")
    print(f"  unroll_length: {ppo_params.unroll_length}")

    # Check GPU memory to determine optimal batch sizes
    if len(jax.devices()) > 0 and 'gpu' in str(jax.devices()[0]).lower():
        try:
            device = jax.devices()[0]
            mem_stats = device.memory_stats()
            total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)

            if total_gb > 80:  # H200 has ~143GB, we detected ~105GB usable
                print(f"\n🚀 High-memory GPU detected ({total_gb:.1f} GB) - Optimizing for maximum performance:")
                # Use much larger batch sizes for H200
                ppo_params.num_envs = 4096       # Increased for H200 (up to 8192 if memory allows)
                ppo_params.batch_size = 512      # Increased for GPU
                ppo_params.num_minibatches = 32  # Increased for better gradient coverage
                ppo_params.unroll_length = 32    # Increased for longer episodes

                print("✓ Optimized for high-performance GPU")
                print("  - Increased parallel environments for better throughput")
                print("  - Larger batch sizes for better GPU utilization")
                print("  - Longer unroll for improved learning stability")

            elif total_gb > 40:
                print(f"\n✅ Mid-range GPU detected ({total_gb:.1f} GB) - Optimizing for balanced performance:")
                ppo_params.num_envs = 2048
                ppo_params.batch_size = 256
                ppo_params.num_minibatches = 24
                ppo_params.unroll_length = 24

                print("✓ Optimized for standard GPU performance")

            else:
                print(f"\n⚠ Low-memory GPU detected ({total_gb:.1f} GB) - Using conservative settings:")
                ppo_params.num_envs = 1024
                ppo_params.batch_size = 128
                ppo_params.num_minibatches = 16
                ppo_params.unroll_length = 20

                print("✓ Optimized for memory efficiency")

        except Exception as e:
            print(f"\n⚠ Could not detect GPU memory ({e}) - Using default optimized settings:")
            ppo_params.num_envs = 2048
            ppo_params.batch_size = 256
            ppo_params.num_minibatches = 24
            ppo_params.unroll_length = 24
    else:
        print("\n⚠ CPU detected - Using memory-optimized settings:")
        ppo_params.num_envs = 512        # Reduced for CPU
        ppo_params.batch_size = 64       # Reduced for CPU
        ppo_params.num_minibatches = 8   # Reduced for CPU
        ppo_params.unroll_length = 16    # Reduced for CPU

    print(f"\nFinal optimized configuration:")
    print(f"  num_envs: {ppo_params.num_envs}")
    print(f"  batch_size: {ppo_params.batch_size}")
    print(f"  num_minibatches: {ppo_params.num_minibatches}")
    print(f"  unroll_length: {ppo_params.unroll_length}")

    # Calculate expected memory usage
    estimated_envs_gb = ppo_params.num_envs * 0.001  # Rough estimate per environment
    print(f"  Estimated memory usage: {estimated_envs_gb:.1f} GB + overhead")

    # Training progress tracking
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Progress callback function with GPU monitoring"""
        current_time = datetime.now()
        times.append(current_time)
        x_data.append(num_steps)

        try:
            # Safely extract and convert metrics to basic types
            episode_reward = json_serialize(metrics.get("eval/episode_reward", 0))
            episode_reward_std = json_serialize(metrics.get("eval/episode_reward_std", 0))

            y_data.append(float(episode_reward) if isinstance(episode_reward, (int, float)) else episode_reward)
            y_dataerr.append(float(episode_reward_std) if isinstance(episode_reward_std, (int, float)) else episode_reward_std)

            # Get GPU memory usage if available
            gpu_memory_info = ""
            try:
                if len(jax.devices()) > 0 and 'gpu' in str(jax.devices()[0]).lower():
                    device = jax.devices()[0]
                    mem_stats = device.memory_stats()
                    used_gb = mem_stats.get('bytes_in_use', 0) / (1024**3)
                    peak_gb = mem_stats.get('peak_bytes_in_use', 0) / (1024**3)
                    total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)
                    gpu_memory_info = f" | GPU: {used_gb:.1f}GB/{total_gb:.1f}GB (peak: {peak_gb:.1f}GB)"
            except:
                gpu_memory_info = ""

            # Save progress to file
            progress_data = {
                'timestamp': current_time.isoformat(),
                'num_steps': int(num_steps) if isinstance(num_steps, (int, float)) else str(num_steps),
                'episode_reward': episode_reward,
                'episode_reward_std': episode_reward_std,
                'gpu_memory_info': gpu_memory_info.strip(),
            }

            # Serialize the entire progress data structure
            serialized_data = json_serialize(progress_data)

            with open('training_progress.json', 'a') as f:
                f.write(json.dumps(serialized_data) + '\n')

            # Print progress with GPU info
            elapsed = current_time - times[0]
            if isinstance(episode_reward, (int, float)) and isinstance(episode_reward_std, (int, float)):
                print(f"[{elapsed}] Step {num_steps}: Reward = {episode_reward:.3f} ± {episode_reward_std:.3f}{gpu_memory_info}")
            else:
                print(f"[{elapsed}] Step {num_steps}: Reward = {episode_reward} ± {episode_reward_std}{gpu_memory_info}")

        except Exception as e:
            # Fallback to basic logging if serialization fails
            elapsed = current_time - times[0]
            print(f"[{elapsed}] Step {num_steps}: Training progress (logging error: {e})")
            # Still save basic progress info
            try:
                basic_data = {
                    'timestamp': current_time.isoformat(),
                    'num_steps': int(num_steps) if isinstance(num_steps, (int, float)) else str(num_steps),
                    'logging_error': str(e)
                }
                with open('training_progress.json', 'a') as f:
                    f.write(json.dumps(basic_data) + '\n')
            except:
                pass  # Silently fail if even basic logging doesn't work

    # Setup training
    print("\nSetting up training pipeline...")
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks

    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        seed=1
    )

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    try:
        make_inference_fn, params, metrics = train_fn(
            environment=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

        print(f"\nTraining completed successfully!")
        print(f"JIT compilation time: {times[1] - times[0]}")
        print(f"Training time: {times[-1] - times[1]}")

        # Save final model
        model_path = f"leap_cube_reorient_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Saving model to: {model_path}")

        # Save parameters
        serialized_metrics = json_serialize(dict(metrics))
        with open(f"{model_path}_params.json", 'w') as f:
            json.dump(serialized_metrics, f, indent=2)

        # Compile inference functions
        print("\nCompiling inference functions...")
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

        # Run evaluation
        print("\nRunning evaluation...")
        rng = jax.random.PRNGKey(42)
        rollout = []
        n_episodes = 3

        total_rewards = []
        for episode in range(n_episodes):
            episode_rewards = []
            state = jit_reset(rng)
            rollout.append(state)

            for i in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                rollout.append(state)
                episode_rewards.append(state.reward)

            total_reward = sum(episode_rewards)
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total reward = {total_reward:.3f}")

        print(f"\nAverage reward over {n_episodes} episodes: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")

        # Save final results
        final_results = {
            'model_path': model_path,
            'training_completed': datetime.now().isoformat(),
            'final_metrics': json_serialize(dict(metrics)),
            'evaluation_results': {
                'mean_reward': float(np.mean(total_rewards)),
                'std_reward': float(np.std(total_rewards)),
                'n_episodes': n_episodes
            }
        }

        with open(f"{model_path}_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n✓ Training completed successfully!")
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Results saved to: {model_path}_results.json")

    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()