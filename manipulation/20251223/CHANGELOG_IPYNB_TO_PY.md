# Jupyter Notebook to Python Script Conversion - Change Log

## Overview
This document details all modifications made when converting `re-orient.ipynb` to `re-orient_notebook.py`.

## 📋 STRUCTURAL CHANGES

### 1. **Script Entry Point**
**.ipynb**: Linear execution of cells starting from Cell 1
**.py**: Added `main()` function with organized execution flow

### 2. **Dependency Management**
**.ipynb**: Multiple pip install commands in separate cells
**.py**: Consolidated into `install_dependencies()` function

### 3. **Environment Configuration**
**.ipynb**: Magic commands and direct OS calls
**.py**: Structured configuration functions with error handling

## 🔧 DETAILED MODIFICATIONS

### **Cells 1-6: Package Installation**
**Original .ipynb cells:**
```python
!pip install mujoco
!pip install mujoco_mjx
!pip install brax
!pip install pyvirtualdisplay
!pip install opencv-python
```

**Converted .py code:**
```python
def install_dependencies():
    """[CHANGE] New function: Consolidates all pip install commands"""
    packages = ["mujoco", "mujoco_mjx", "brax", "pyvirtualdisplay", "opencv-python", "playground"]
    for package in packages:
        try:
            __import__(package.replace('-', '_').replace('opencv_python', 'cv2'))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
```

**[CHANGE] Added**: Centralized dependency management with error handling

### **Cell 7: Graphics Packages Check**
**Original .ipynb:**
```python
!dpkg -l | grep -E "(nvidia|mesa|gl|egl)"
```

**Converted .py:**
```python
def check_graphics_packages():
    """[CHANGE] New function: Check installed graphics packages"""
    result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
    graphics_packages = [line for line in result.stdout.split('\n')
                        if any(pkg in line for pkg in ['nvidia', 'mesa', 'gl', 'egl'])]
```

**[CHANGE] Added**: Function wrapper and parsing logic

### **Cell 8: NVIDIA Configuration**
**Original .ipynb:**
```python
import distutils.util
import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError('Cannot communicate with GPU...')

NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{...}""")
```

**Converted .py:**
```python
def setup_nvidia_icd():
    """[CHANGE] New function: NVIDIA ICD configuration with error handling"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Create ICD config with proper directory creation
            os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
```

**[CHANGE] Added**:
- Error handling for missing nvidia-smi
- Directory creation before file writing
- Return status for error checking

### **Cell 9-10: MuJoCo and FFmpeg Setup**
**Original .ipynb:**
```python
!apt update && apt install -y libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev
!command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
```

**Converted .py:**
```python
def configure_mujoco_rendering():
    """[CHANGE] New function: Try multiple rendering backends"""
    backends = [
        ('disabled', 'MUJOCO_RENDER', '0'),
        ('osmesa', 'MUJOCO_GL', 'osmesa'),
        ('egl', 'MUJOCO_GL', 'egl'),
    ]
    # Try each backend with error handling

def install_ffmpeg():
    """[CHANGE] New function: FFmpeg installation check"""
    result = subprocess.run(['which', 'ffmpeg'], capture_output=True)
```

**[CHANGE] Added**:
- Multiple rendering backend fallback logic
- FFmpeg availability checking
- Proper error handling for package installations

### **Cell 11-12: Imports**
**Original .ipynb:**
```python
import json
import itertools
import time
import numpy as np
!pip install -q mediapy
import mediapy as media
import matplotlib.pyplot as plt

# Multiple imports in later cell
from datetime import datetime
import functools
# ... many more imports
```

**Converted .py:**
```python
# [CHANGE] Added comprehensive imports at top
import os, sys, json, time, subprocess, warnings
from datetime import datetime
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# [CHANGE] Conditional MediaPy import with fallback
try:
    import mediapy as media
    media_available = True
except ImportError:
    media_available = False
```

**[CHANGE] Added**:
- Conditional MediaPy import
- Headless matplotlib backend configuration
- Warning suppression
- Centralized import organization

### **Cell 19: Training Setup with GPU Optimization**
**Original .ipynb:**
```python
# 2025年12月18日修改：减少内存消耗的参数覆盖
ppo_params.num_envs = 1024        # 从8192减少到1024
ppo_params.batch_size = 128       # 从256减少到128
# ... memory optimization for CPU
```

**Converted .py:**
```python
# [CHANGE] Enhanced GPU optimization
if len(jax.devices()) > 0 and 'gpu' in str(jax.devices()[0]).lower():
    try:
        device = jax.devices()[0]
        mem_stats = device.memory_stats()
        total_gb = mem_stats.get('bytes_limit', 0) / (1024**3)

        if total_gb > 80:  # H200 optimization
            ppo_params.num_envs = 4096
            ppo_params.batch_size = 512
            ppo_params.num_minibatches = 32
            ppo_params.unroll_length = 32
```

**[CHANGE] Added**:
- Dynamic GPU memory detection
- H200-specific optimizations
- Automatic configuration based on available resources

### **Cell 19: Progress Function**
**Original .ipynb:**
```python
def progress(num_steps, metrics):
  clear_output(wait=True)
  times.append(datetime.now())
  # ... progress tracking
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
  display(plt.gcf())
```

**Converted .py:**
```python
def progress(num_steps, metrics):
    """[CHANGE] Modified to work without Jupyter display"""
    # Log to file
    progress_data = {
        'timestamp': times[-1].isoformat(),
        'num_steps': num_steps,
        'episode_reward': metrics["eval/episode_reward"],
    }
    with open('training_progress.json', 'a') as f:
        f.write(json.dumps(progress_data) + '\n')

    # [CHANGE] Console output instead of display
    print(f"[{elapsed}] Step {num_steps}: Reward = {metrics['eval/episode_reward']:.3f}")

    # [CHANGE] Save plots to files instead of displaying
    try:
        plot_filename = f"training_progress_{num_steps}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create progress plot: {e}")
```

**[CHANGE] Modified**:
- Removed `clear_output(wait=True)` and `display(plt.gcf())`
- Added file logging for progress data
- Added console output for real-time monitoring
- Added plot file saving instead of display

### **Cell 22: Rollout and Video Generation**
**Original .ipynb:**
```python
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  # ... rollout logic
frames = env.render(rollout[::render_every])
media.show_video(frames, fps=1.0 / env.dt / render_every)
```

**Converted .py:**
```python
# [CHANGE] Enhanced rollout with better monitoring
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 3  # Increased from 1 for better evaluation

for episode in range(n_episodes):
    print(f"Running episode {episode + 1}/{n_episodes}...")
    # ... enhanced rollout with progress indicators
    if i % 200 == 0:
        print(f"  Step {i:4d}: Reward = {state.reward:.3f}")

# [CHANGE] Enhanced video generation with fallbacks
if media_available:
    try:
        video_file = f"{model_path}_rollout.mp4"
        media.write_video(video_file, frames, fps=1.0 / env.dt / render_every)
        print(f"✅ Video saved as: {video_file}")

        # [CHANGE] Also save individual frames
        for i, frame in enumerate(frames[::10]):
            frame_filename = f"{model_path}_frame_{i:04d}.png"
            cv2.imwrite(frame_filename, frame_array)
    except Exception as e:
        print(f"⚠️ Could not save video: {e}")
else:
    # [CHANGE] Fallback: save frames as numpy arrays
    print("Saving frames as numpy arrays...")
    for i, frame in enumerate(frames[::10]):
        frame_filename = f"{model_path}_frame_{i:04d}.npy"
        np.save(frame_filename, frame)
```

**[CHANGE] Added**:
- Multiple episode evaluation (1→3)
- Progress indicators during rollout
- Multiple video format options
- Individual frame saving
- Fallback when MediaPy is not available
- Comprehensive error handling

## 🆕 NEW FUNCTIONALITY

### 1. **Headless Execution Support**
- Automatic backend detection and fallback
- Non-interactive matplotlib configuration
- Proper environment variable setup

### 2. **Enhanced Error Handling**
- Graceful degradation when optional packages missing
- Comprehensive exception handling throughout
- Status reporting for all operations

### 3. **GPU Optimization**
- Automatic GPU memory detection
- Dynamic parameter adjustment based on hardware
- H200-specific optimizations

### 4. **Comprehensive Logging**
- Progress data saved to JSON
- Training plots saved as images
- Evaluation results with detailed metrics

### 5. **Multiple Output Formats**
- MP4 video files
- Individual frame images
- Numpy array fallback
- Detailed JSON result files

## 📁 GENERATED FILES

**Notebook cells generate:**
- Inline plots (displayed in notebook)
- Media.show_video() (displayed in notebook)

**Python script generates:**
- `training_progress.json` - Progress tracking data
- `training_progress_*.png` - Training progress plots
- `*_rollout.mp4` - Generated video
- `*_frame_*.png` - Individual video frames
- `*_results.json` - Comprehensive evaluation results
- `*_params.json` - Training parameters and metrics

## 🔄 EXECUTION FLOW

**Notebook**: Linear cell execution, manual intervention possible
**Python script**: Automated execution with error recovery, no manual intervention needed

## ✅ BENEFITS OF CONVERSION

1. **Automation**: Fully automated execution without manual cell running
2. **Error Recovery**: Robust error handling and fallback mechanisms
3. **Portability**: Works in headless environments
4. **Scalability**: GPU optimizations and dynamic configuration
5. **Debugging**: Comprehensive logging and intermediate file saving
6. **Flexibility**: Multiple output formats and fallback options

## 🔧 TESTING RECOMMENDATIONS

1. Test with: `python re-orient_notebook.py`
2. Check generated files in current directory
3. Verify video playback (if MediaPy available)
4. Monitor training progress via console output
5. Check training_progress.json for detailed metrics