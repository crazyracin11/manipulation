# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive reinforcement learning research project focused on **humanoid robot control using Proximal Policy Optimization (PPO)** and **dexterous manipulation using MuJoCo Playground**. The project combines multiple RL algorithms and environments for experimentation.

## Common Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirement.txt

# For humanoid training (PyTorch-based)
pip install gymnasium[mujoco] torch torchvision tensorboardX

# For manipulation tasks (JAX/Brax-based)
pip install mujoco mujoco_mjx brax mediapy
pip install opencv-python pyvirtualdisplay orbax-checkpoint
```

### PPO-Humanoid Training
```bash
# The main PPO implementation is in the PPO-Humanoid subdirectory
cd PPO-Humanoid
pip install -r req.txt

# Train PPO on Humanoid-v5
python train_ppo.py

# Train with custom hyperparameters
python train_ppo.py --n-envs 64 --n-epochs 2000 --learning-rate 2e-4

# Test pre-trained model (model.pt in PPO-Humanoid directory)
python test_ppo.py

# Monitor training progress
tensorboard --logdir "logs"
```

### Manipulation Tasks (JAX/Brax)
```bash
# Main manipulation training script
python manipulation/re-orient.py

# Alternative scripts with different features
python manipulation/re-orient_fixed_complete.py  # Has checkpoint saving
python manipulation/quick_demo_training.py        # Quick short training
python manipulation/final_demo.py                 # Generate demo video
```

## Architecture Overview

### Two Separate Implementations

This project contains **two separate PPO implementations**:

#### 1. PPO-Humanoid (PyTorch-based)
- Location: `PPO-Humanoid/` subdirectory
- Framework: PyTorch + Gymnasium
- Purpose: Humanoid locomotion in MuJoCo
- Files:
  - `train_ppo.py`: Main training script
  - `test_ppo.py`: Evaluation script
  - `lib/agent_ppo.py`: Actor-critic networks
  - `lib/buffer_ppo.py`: Experience replay with GAE
  - `lib/utils.py`: Utilities and argument parsing

#### 2. Manipulation (JAX/Brax-based)
- Location: `manipulation/` subdirectory
- Framework: JAX + Brax + MuJoCo Playground
- Purpose: Dexterous hand manipulation (LeapCubeReorient)
- Key environment: `LeapCubeReorient` - robotic hand reorienting a cube
- Uses `manipulation_params.brax_ppo_config()` for configuration

### PPO-Humanoid Architecture
```
PPO-Humanoid/
├── train_ppo.py          # Main training loop
├── test_ppo.py           # Evaluation with video rendering
├── lib/
│   ├── agent_ppo.py      # Actor-critic networks (3-layer MLP, 512 units)
│   ├── buffer_ppo.py     # Trajectory storage with GAE advantage computation
│   └── utils.py          # Argument parsing, environment setup, video logging
├── logs/                 # TensorBoard event files
├── checkpoints/          # Model state dicts (best.pt, last.pt)
└── model.pt              # Pre-trained checkpoint (~5.7MB)
```

**Key Patterns:**
- Shared feature extractor with separate policy (actor) and value (critic) heads
- Gaussian policy with diagonal covariance for continuous actions
- GAE (Generalized Advantage Estimation) for variance reduction
- Mixed precision training with `torch.amp.autocast()`
- Gradient clipping at 1.0, KL divergence early stopping

### Manipulation Architecture
```
manipulation/
├── re-orient.py                 # Main JAX/Brax training script
├── re-orient_fixed_complete.py  # With checkpoint saving and video generation
├── quick_demo_training.py       # Short training for quick demos
├── final_demo.py                # Video generation from trained model
├── simple_evaluate.py           # Evaluation utilities
└── *.ipynb                      # Various experiment notebooks
```

**Key Patterns:**
- Uses `from mujoco_playground.config import manipulation_params`
- `ppo_params = manipulation_params.brax_ppo_config(env_name)`
- Returns `make_inference_fn, params, metrics` from training
- For video generation: use `make_inference_fn(state.obs, act_rng)` directly (not wrapped)
- Environment rendering: `env.render(rollout[::render_every])`

## Training Hyperparameters

### PPO-Humanoid (lib/utils.py or command line)
- `--n-envs`: Parallel environments (default: 32)
- `--n-epochs`: Total epochs (default: 1000)
- `--n-steps`: Steps per epoch per environment (default: 2048)
- `--batch-size`: Training batch size (default: 256)
- `--learning-rate`: Policy learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.995)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--clip-ratio`: PPO clipping parameter (default: 0.2)

### Manipulation (Brax config)
- Uses `manipulation_params.brax_ppo_config()` which returns a ConfigDict
- Modify directly: `ppo_params.num_timesteps = 10_000_000`
- Common settings: num_envs=128-8192, batch_size=256
- Environment: `LeapCubeReorient` with episode_length=1000

## Output Structure

### PPO-Humanoid
- `logs/`: TensorBoard event files
- `checkpoints/`: best.pt, last.pt (model state dicts)
- `videos/`: MP4 recordings every --render-epoch

### Manipulation
- `./videos/`: Generated demonstration videos
- `./checkpoints/`: Model checkpoints (.npz format)
- `./training_plots/`: Training progress charts

## Technology Stack

### PPO-Humanoid
- **PyTorch**: Neural networks and autograd
- **Gymnasium**: RL environment interface
- **MuJoCo**: Physics simulation
- **TensorBoardX**: Metrics visualization
- **OpenCV**: Video rendering

### Manipulation
- **JAX**: JIT compilation and autograd
- **Brax**: JAX-based RL framework
- **MuJoCo/MJX**: Physics simulation with JAX acceleration
- **MuJoCo Playground**: Specialized manipulation environments
- **Mediapy**: Video generation

## Important Implementation Notes

### Mixed Precision Training (PPO-Humanoid)
- Uses `torch.amp.autocast()` for forward passes
- GradScaler for gradient scaling
- Significant GPU memory savings

### Environment Vectorization
- PPO-Humanoid: `Gymnasium.AsyncVectorEnv`
- Manipulation: Built-in Brax vectorization

### Video Generation (Manipulation)
When generating videos from trained models in JAX/Brax:
1. The `inference_fn` returned from training already has params baked in
2. Use directly: `ctrl, _ = jit_inference(state.obs, act_rng)`
3. Use original environment (not wrapped) for single-rollout: `jit_reset = jax.jit(env.reset)`
4. Do NOT wrap inference_fn again with params

### Checkpoint Saving (Manipulation)
- Use `orbax-checkpoint` for reliable saving
- Convert JAX params to dict format before saving
- Checkpoint paths must be absolute paths

## Pre-trained Models

- `PPO-Humanoid/model.pt`: Pre-trained Humanoid-v5 model (~1000 epochs)
- `checkpoints/model_step_200000000.npz`: Manipulation model (200M steps, reward=110)
