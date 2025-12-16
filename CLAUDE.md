# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive reinforcement learning research project focused on **humanoid robot control using Proximal Policy Optimization (PPO)**. The project combines multiple RL algorithms and environments for experimentation, with particular emphasis on humanoid locomotion and dexterous manipulation tasks.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirement.txt

# Install Mujoco for humanoid simulation
pip install gymnasium[mujoco]

# For manipulation tasks with MuJoCo playground
pip install mujoco mujoco_mjx brax
pip install pyvirtualdisplay opencv-python
```

### Training and Testing
```bash
# Train PPO on humanoid (main implementation)
python train_ppo.py

# Train with custom hyperparameters
python train_ppo.py --n-envs 64 --n-epochs 2000 --learning-rate 2e-4

# Test pre-trained model
python test_ppo.py

# PPO-Humanoid variant training
cd PPO-Humanoid
python train_ppo.py
python test_ppo.py

# Monitor training progress
tensorboard --logdir "logs"
```

### GPU Acceleration
The project automatically detects CUDA availability. Training requires GPU for reasonable performance:
- Mixed precision training with `torch.amp.autocast`
- Supports multiple parallel environments for efficient data collection

## Architecture Overview

### Core PPO Implementation Structure
```
lib/
├── agent_ppo.py          # Actor-critic networks with PPO logic
├── buffer_ppo.py         # Experience replay with GAE advantage computation
└── utils.py              # Hyperparameter parsing, environment setup, video logging
```

**Key Architectural Patterns:**
- **Agent Architecture**: Shared feature extractor with separate policy (actor) and value (critic) heads
- **Training Loop**: Vectorized environment data collection → batched PPO updates with GAE
- **Advantage Computation**: Generalized Advantage Estimation (GAE) for variance reduction
- **Multi-Environment Training**: Synchronous parallel environment stepping for efficiency

### PPO Training Algorithm Flow
1. **Data Collection**: Collect experiences across `n-envs` parallel environments for `n-steps` each
2. **Advantage Estimation**: Compute returns and advantages using GAE with gamma/gae-lambda parameters
3. **Policy Update**: Multiple PPO update iterations per batch using clipped surrogate objective
4. **Monitoring**: TensorBoard logging, periodic video rendering, checkpoint saving

### Environment Variants
- **Main Project**: General PPO implementation configurable for any Gymnasium environment
- **PPO-Humanoid**: Specialized for Humanoid-v5 with pre-trained model checkpoint (`model.pt`)
- **Manipulation**: Dexterous hand manipulation using MuJoCo Playground (LeapCubeReorient)
- **Experiment Notebooks**: Algorithm comparisons (Policy Gradient, SAC) across various environments

## Key Configuration Parameters

### Training Hyperparameters (lib/utils.py)
- `--n-envs`: Number of parallel environments (default: 32)
- `--n-epochs`: Total training epochs (default: 1000)
- `--n-steps`: Steps per epoch per environment (default: 2048)
- `--batch-size`: Training batch size (default: 256)
- `--learning-rate`: Policy learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.995)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--clip-ratio`: PPO clipping parameter (default: 0.2)

### Output Structure
Training creates timestamped directories:
- `checkpoints/`: Model state dictionaries with optimizer state
- `logs/`: TensorBoard event files with metrics
- `videos/`: MP4 recordings of agent performance every `--render-epoch`

## Technology Stack

### Core Dependencies
- **PyTorch**: Neural networks and automatic differentiation
- **Gymnasium**: RL environment interface with MuJoCo physics
- **TensorBoardX**: Training metrics visualization
- **OpenCV**: Video rendering and processing
- **DM Control**: Additional RL environments for experimentation

### Manipulation Extension
- **MuJoCo/MJX**: High-performance physics simulation with JAX compilation
- **Brax**: JAX-based RL framework for accelerated simulation
- **MuJoCo Playground**: Specialized manipulation environments

## Important Implementation Details

### Mixed Precision Training
The implementation uses NVIDIA's automatic mixed precision (AMP) for memory efficiency:
- `torch.amp.autocast()` for forward passes
- GradScaler for gradient scaling and optimizer stepping
- Significant GPU memory savings for large batch training

### Gradient Clipping and KL Divergence
- Gradient norm clipping at 1.0 to prevent training instability
- KL divergence monitoring with early stopping at `--target-kl` threshold
- Entropy regularization via `--ent-coef` for exploration

### Environment Vectorization
Uses Gymnasium's `AsyncVectorEnv` for parallel environment stepping, providing:
- Efficient data collection on multi-core systems
- Deterministic behavior across training runs
- Easy scaling of environment count based on available compute

## File Organization Patterns

### Modular Training Scripts
Each algorithm implementation follows consistent structure:
- `train_<algorithm>.py`: Main training loop with argument parsing
- `test_<algorithm>.py`: Evaluation script with video rendering
- `lib/agent_<algorithm>.py`: Core algorithm implementation
- `lib/buffer_<algorithm>.py`: Experience replay/Buffer logic
- `lib/utils.py`: Shared utilities and environment wrappers

### Experiment Notebooks
Jupyter notebooks in root directory contain:
- Algorithm tutorials and comparisons
- Environment-specific experiments
- Visualization and analysis tools
- Reference implementations for different RL algorithms

This codebase is designed for both research experimentation and production training, with comprehensive logging, checkpointing, and visualization capabilities built into the training pipeline.