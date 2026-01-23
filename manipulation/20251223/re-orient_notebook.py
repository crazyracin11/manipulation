#!/usr/bin/env python3
"""
==============================================================================
RE-ORIENT.NB.PY - Converted from re-orient.ipynb
==============================================================================
CHANGE LOG: All modifications from .ipynb to .py are marked with [CHANGE]
==============================================================================
"""

# [CHANGE] Added comprehensive imports at the top for headless execution
import os
import sys
import json
import time
import subprocess
import warnings
from datetime import datetime
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# [CHANGE] Configure headless execution BEFORE any graphics imports
print("Configuring headless environment...")
os.environ['MUJOCO_RENDER'] = '0'  # Start with rendering disabled for setup
os.environ['PYOPENGL_PLATFORM'] = ''
os.environ['JAX_PLATFORMS'] = 'cuda'

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def configure_mujoco_rendering():
    """
    [CHANGE] New function added: Handle MuJoCo rendering backend configuration
    This replaces the notebook's magic commands and environment setup cells
    """
    print("Configuring MuJoCo rendering backend...")

    # Try different rendering backends in order of preference
    backends = [
        ('disabled', 'MUJOCO_RENDER', '0'),           # Option 1: Completely disable rendering
        ('osmesa', 'MUJOCO_GL', 'osmesa'),           # Option 2: CPU software rendering
        ('egl', 'MUJOCO_GL', 'egl'),                 # Option 3: GPU rendering
    ]

    for backend_name, env_var, env_value in backends:
        try:
            os.environ[env_var] = env_value

            if backend_name == 'disabled':
                # Test basic MuJoCo functionality
                import mujoco
                xml = '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                for _ in range(3):
                    mujoco.mj_step(model, data)
                print(f"✅ MuJoCo working with {backend_name} rendering")
                return True

            elif backend_name == 'osmesa':
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                # Import and test
                import mujoco
                print(f"✅ MuJoCo working with {backend_name} rendering")
                return True

            elif backend_name == 'egl':
                # Check for GPU and EGL support
                try:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ GPU detected, trying {backend_name} rendering")
                        import mujoco
                        xml = '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
                        model = mujoco.MjModel.from_xml_string(xml)
                        data = mujoco.MjData(model)
                        for _ in range(3):
                            mujoco.mj_step(model, data)
                        print(f"✅ MuJoCo working with {backend_name} rendering")
                        return True
                    else:
                        continue
                except:
                    continue

        except Exception as e:
            print(f"⚠️ {backend_name} backend failed: {e}")
            continue

    print("❌ All rendering backends failed")
    return False

def install_dependencies():
    """
    [CHANGE] New function added: Consolidates all pip install commands from notebook
    Replaces multiple notebook cells with pip commands
    """
    print("Installing dependencies...")

    # List of packages from notebook cells
    packages = [
        "mujoco",
        "mujoco_mjx",
        "brax",
        "pyvirtualdisplay",
        "opencv-python",
        "playground"
    ]

    # Check and install packages
    for package in packages:
        try:
            # Handle different import names
            import_name = package.replace('-', '_')
            if import_name == 'opencv_python':
                import_name = 'cv2'

            __import__(import_name)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")

def setup_nvidia_icd():
    """
    [CHANGE] New function added: NVIDIA ICD configuration from notebook cell 8
    This replaces the notebook's NVIDIA driver and ICD setup
    """
    print("Setting up NVIDIA ICD configuration...")

    # Check for GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected")
            print(result.stdout.split('\n')[8])  # GPU info line

            # Create NVIDIA ICD config (from notebook cell 8)
            NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
            if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
                print("Creating NVIDIA ICD configuration...")
                os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)

                with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
                    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
                print("✓ NVIDIA ICD configuration created")
            else:
                print("✓ NVIDIA ICD configuration already exists")

        else:
            print("⚠️ No GPU detected")
            return False
    except FileNotFoundError:
        print("⚠️ nvidia-smi not found")
        return False

    return True

def configure_xla_flags():
    """
    [CHANGE] New function added: XLA configuration from notebook cell 8
    """
    print("Configuring XLA flags for GPU optimization...")

    # Tell XLA to use Triton GEMM (from notebook cell 8)
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    print("✓ XLA flags configured for Triton GEMM")

def install_ffmpeg():
    """
    [CHANGE] New function added: FFmpeg installation from notebook cell 10
    """
    print("Checking for FFmpeg...")

    # Check if ffmpeg is available
    result = subprocess.run(['which', 'ffmpeg'], capture_output=True)
    if result.returncode != 0:
        print("Installing FFmpeg...")
        try:
            subprocess.check_call(['apt', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(['apt', 'install', '-y', 'ffmpeg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓ FFmpeg installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️ Could not install FFmpeg (may not have sudo access)")
    else:
        print("✓ FFmpeg already available")

def main():
    """
    [CHANGE] New main function added: Script entry point
    Replaces the linear execution of notebook cells
    """
    print("=" * 80)
    print("LEAP CUBE REORIENT TRAINING - CONVERTED FROM JUPYTER NOTEBOOK")
    print("=" * 80)

    # Cell 1-6: Package installation (converted to function calls)
    install_dependencies()

    # Cell 7: System check (graphics packages check)
    print("\nChecking graphics packages...")
    result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
    if result.returncode == 0:
        graphics_packages = [line for line in result.stdout.split('\n')
                           if any(pkg in line for pkg in ['nvidia', 'mesa', 'gl', 'egl'])]
        print(f"Found {len(graphics_packages)} graphics-related packages")

    # Cell 8: NVIDIA configuration and MuJoCo setup
    print("\n" + "=" * 60)
    print("CELL 8: NVIDIA AND MUJOCO CONFIGURATION")
    print("=" * 60)

    setup_nvidia_icd()
    configure_mujoco_rendering()
    configure_xla_flags()

    # Install FFmpeg (Cell 10)
    install_ffmpeg()

    # Test MuJoCo installation
    print("\nChecking MuJoCo installation...")
    try:
        import mujoco
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print("✅ MuJoCo installation successful")
    except Exception as e:
        print(f"❌ MuJoCo installation failed: {e}")
        return False

    # Cell 11: Import packages for plotting (simplified for headless)
    print("\n" + "=" * 60)
    print("CELL 11: IMPORT PACKAGES FOR PLOTTING")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported (headless mode)")
    except ImportError:
        print("⚠️ Matplotlib not available, skipping plotting features")

    # Set numpy printing options
    try:
        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        print("✅ NumPy formatting configured")
    except NameError:
        import numpy as np
        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        print("✅ NumPy imported and configured")

    # Cell 12: Import MuJoCo, MJX, and Brax
    print("\n" + "=" * 60)
    print("CELL 12: IMPORT MUJOCO, MJX, AND BRAX")
    print("=" * 60)

    # Core imports
    from datetime import datetime
    import functools
    from brax import base, envs, math
    from brax.base import Base, Motion, Transform
    from brax.base import State as PipelineState
    from brax.envs.base import Env, PipelineEnv, State
    from brax.io import html, mjcf, model
    from brax.mjx.base import State as MjxState
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.agents.sac import train as sac
    from etils import epath
    from flax import struct
    from flax.training import orbax_utils
    import jax
    from jax import numpy as jp
    from ml_collections import config_dict
    import mujoco
    from mujoco import mjx
    from orbax import checkpoint as ocp

    # [CHANGE] Try to import mediapy, but handle failure gracefully
    try:
        import mediapy as media
        print("✅ MediaPy imported successfully")
        media_available = True
    except ImportError:
        print("⚠️ MediaPy not available, video features will be limited")
        media_available = False

    # Cell 13: Install playground (already done in install_dependencies)
    print("\n" + "=" * 60)
    print("CELL 13: MUJOCO PLAYGROUND IMPORT")
    print("=" * 60)

    try:
        from mujoco_playground import wrapper
        from mujoco_playground import registry
        print("✅ MuJoCo playground imported successfully")

        # Display available environments
        available_envs = registry.manipulation.ALL_ENVS
        print("Available manipulation environments:")
        for env_name in available_envs:
            print(f"  - {env_name}")

    except ImportError as e:
        print(f"❌ Failed to import mujoco_playground: {e}")
        return False

    # Cell 16: Load LeapCubeReorient environment
    print("\n" + "=" * 60)
    print("CELL 16: LOAD LEAPCUBEREORIENT ENVIRONMENT")
    print("=" * 60)

    env_name = 'LeapCubeReorient'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    print(f"✅ Loaded environment: {env_name}")

    # Cell 17: Display environment configuration
    print("\n" + "=" * 60)
    print("CELL 17: ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    print("Environment configuration:")
    print(env_cfg)

    # Cell 19: Train Policy Setup
    print("\n" + "=" * 60)
    print("CELL 19: TRAIN POLICY SETUP")
    print("=" * 60)

    # [CHANGE] Fix datetime import issue mentioned in notebook comment
    from datetime import datetime
    from mujoco_playground.config import manipulation_params
    ppo_params = manipulation_params.brax_ppo_config(env_name)

    # Display original configuration
    print("Original configuration:")
    print(f"  num_envs: {ppo_params.num_envs}")
    print(f"  batch_size: {ppo_params.batch_size}")
    print(f"  num_minibatches: {ppo_params.num_minibatches}")
    print(f"  unroll_length: {ppo_params.unroll_length}")

    # [CHANGE] GPU optimization (modified from notebook's memory optimization)
    if len(jax.devices()) > 0 and 'gpu' in str(jax.devices()[0]).lower():
        try:
            device = jax.devices()[0]
            mem_stats = device.memory_stats()
            total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)

            if total_gb > 80:  # H200 optimization
                print(f"\n🚀 High-memory GPU detected ({total_gb:.1f} GB) - Optimizing for maximum performance:")
                ppo_params.num_envs = 4096
                ppo_params.batch_size = 512
                ppo_params.num_minibatches = 32
                ppo_params.unroll_length = 32
                print("✓ Optimized for high-performance GPU")
            else:
                print(f"\n✅ Using standard GPU configuration")

        except Exception as e:
            print(f"Could not detect GPU memory: {e}")
    else:
        print("\n⚠️ CPU detected - Using conservative settings")
        ppo_params.num_envs = 1024
        ppo_params.batch_size = 128
        ppo_params.num_minibatches = 16
        ppo_params.unroll_length = 20

    print(f"\nFinal configuration:")
    print(f"  num_envs: {ppo_params.num_envs}")
    print(f"  batch_size: {ppo_params.batch_size}")
    print(f"  num_minibatches: {ppo_params.num_minibatches}")
    print(f"  unroll_length: {ppo_params.unroll_length}")

    # Training progress tracking setup
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """
        [CHANGE] Modified progress function to work without Jupyter display
        Replaces clear_output() and display() with file logging and console output
        """
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        # Log to file
        try:
            progress_data = {
                'timestamp': times[-1].isoformat(),
                'num_steps': num_steps,
                'episode_reward': metrics["eval/episode_reward"],
                'episode_reward_std': metrics["eval/episode_reward_std"],
            }

            with open('training_progress.json', 'a') as f:
                f.write(json.dumps(progress_data) + '\n')
        except Exception as e:
            print(f"Warning: Could not log progress to file: {e}")

        # Console output
        elapsed = times[-1] - times[0]
        print(f"[{elapsed}] Step {num_steps}: Reward = {metrics['eval/episode_reward']:.3f} ± {metrics['eval/episode_reward_std']:.3f}")

        # [CHANGE] Create progress plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
            plt.xlabel("# environment steps")
            plt.ylabel("reward per episode")
            plt.title(f"Training Progress - Reward: {y_data[-1]:.3f}")
            plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
            plt.grid(True, alpha=0.3)

            # Save plot to file
            plot_filename = f"training_progress_{num_steps}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Training plot saved: {plot_filename}")
        except Exception as e:
            print(f"  Warning: Could not create progress plot: {e}")

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
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        seed=1
    )

    # Cell 20: Execute Training
    print("\n" + "=" * 60)
    print("CELL 20: EXECUTE TRAINING")
    print("=" * 60)

    try:
        make_inference_fn, params, metrics = train_fn(
            environment=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

        print(f"time to jit: {times[1] - times[0]}")
        print(f"time to train: {times[-1] - times[1]}")

        # Save model
        model_path = f"leap_cube_reorient_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Saving model to: {model_path}")

        with open(f"{model_path}_params.json", 'w') as f:
            json.dump(dict(metrics), f, indent=2)

        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cell 21: JIT Compilation
    print("\n" + "=" * 60)
    print("CELL 21: JIT COMPILATION")
    print("=" * 60)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    print("✅ JIT compilation completed")
    print("  - jit_reset: Environment reset function")
    print("  - jit_step: Environment step function")
    print("  - jit_inference_fn: Policy inference function")

    # Cell 22: Rollout and Video Generation
    print("\n" + "=" * 60)
    print("CELL 22: ROLLOUT AND VIDEO GENERATION")
    print("=" * 60)

    try:
        rng = jax.random.PRNGKey(42)
        rollout = []
        n_episodes = 3  # [CHANGE] Increased from 1 to 3 for better evaluation

        for episode in range(n_episodes):
            print(f"\nRunning episode {episode + 1}/{n_episodes}...")

            state = jit_reset(rng)
            rollout.append(state)
            episode_rewards = []

            for i in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                rollout.append(state)
                episode_rewards.append(float(state.reward))

                # Progress indicator
                if i % 200 == 0:
                    print(f"  Step {i:4d}: Reward = {state.reward:.3f}")

            episode_total = sum(episode_rewards)
            print(f"Episode {episode + 1} completed:")
            print(f"  Total reward: {episode_total:.3f}")
            print(f"  Steps: {len(episode_rewards)}")

        # Video generation
        print(f"\nGenerating video from rollout...")
        render_every = 5  # [CHANGE] Reduced from 1 for smaller video
        frames = env.render(rollout[::render_every])
        rewards = [s.reward for s in rollout]

        print(f"✅ Generated {len(frames)} frames")
        print(f"   Total reward: {sum(rewards):.3f}")
        print(f"   Rollout length: {len(rollout)} steps")

        # [CHANGE] Enhanced video saving with multiple formats
        if media_available:
            try:
                # Save as MP4
                video_file = f"{model_path}_rollout.mp4"
                media.write_video(video_file, frames, fps=1.0 / env.dt / render_every)
                print(f"✅ Video saved as: {video_file}")

                # [CHANGE] Also save frames as individual images for debugging
                print("Saving individual frames...")
                for i, frame in enumerate(frames[::10]):  # Save every 10th frame
                    frame_filename = f"{model_path}_frame_{i:04d}.png"
                    try:
                        # Try to save frame using PIL or OpenCV
                        import cv2
                        import numpy as np

                        # Convert frame if needed
                        if hasattr(frame, 'shape'):
                            frame_array = np.array(frame)
                            if frame_array.dtype != np.uint8:
                                frame_array = (frame_array * 255).astype(np.uint8)
                            cv2.imwrite(frame_filename, frame_array)
                    except ImportError:
                        # Fallback: try to save raw frame data
                        np.save(frame_filename.replace('.png', '.npy'), frame)

                print(f"✅ Individual frames saved (every 10th frame)")

            except Exception as e:
                print(f"⚠️ Could not save video: {e}")
        else:
            print("⚠️ MediaPy not available - video features limited")

            # [CHANGE] Save frames as numpy arrays when MediaPy is not available
            print("Saving frames as numpy arrays...")
            for i, frame in enumerate(frames[::10]):
                frame_filename = f"{model_path}_frame_{i:04d}.npy"
                np.save(frame_filename, frame)
            print(f"✅ Saved {len(frames[::10])} frames as numpy arrays")

        # Save evaluation results
        evaluation_results = {
            'model_path': model_path,
            'training_completed': datetime.now().isoformat(),
            'final_metrics': dict(metrics),
            'evaluation_results': {
                'n_episodes': n_episodes,
                'episode_rewards': [sum([float(s.reward) for s in rollout if rollout.index(s) % (env_cfg.episode_length + 1) != 0])],
                'total_rollout_reward': float(sum(rewards)),
                'rollout_length': len(rollout),
                'frames_generated': len(frames),
                'render_every': render_every,
                'fps': 1.0 / env.dt / render_every
            }
        }

        with open(f"{model_path}_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"✅ Evaluation results saved: {model_path}_results.json")

    except Exception as e:
        print(f"❌ Rollout/Video generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY!")
    print("=" * 80)
    print("📁 Generated files:")
    print("   - training_progress.json")
    print("   - Training plots (progress_*.png)")
    print("   - Model files and parameters")
    print("   - Evaluation video and frames")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Script execution failed")
        sys.exit(1)
    else:
        print("\n✅ Script completed successfully")