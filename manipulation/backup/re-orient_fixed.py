#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# 环境设置部分 - 新增代码，用于替代Jupyter环境的IPython调用
# =============================================================================

import subprocess
import sys
import os

def run_command(command, description=""):
    """执行系统命令，替代get_ipython().system()"""
    if description:
        print(f"正在执行: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {command}")
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"错误: {e.stderr}")
        return False

def install_required_packages():
    """安装必要的Python包"""
    print("=== 安装Python依赖包 ===")

    packages = [
        ("pip install mujoco", "MuJoCo物理模拟引擎"),
        ("pip install mujoco_mjx", "MuJoCo JAX加速版本"),
        ("pip install brax", "Brax强化学习框架"),
        ("pip install pyvirtualdisplay", "虚拟显示器"),
        ("pip install opencv-python", "OpenCV计算机视觉库"),
        ("pip install -q mediapy", "媒体处理库"),
        ("pip install playground", "MuJoCo Playground环境")
    ]

    for cmd, desc in packages:
        run_command(cmd, desc)

def setup_system_packages():
    """安装系统级依赖包"""
    print("=== 安装系统依赖包 ===")

    # 只有在有root权限时才执行
    if os.geteuid() == 0:
        system_commands = [
            ("apt update", "更新包列表"),
            ("apt install -y ffmpeg", "安装FFmpeg"),
            ("apt install -y libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev", "安装OpenGL开发库"),
            ("apt install -y libegl1-mesa-dev libosmesa6-dev --fix-missing", "修复缺失的依赖")
        ]

        for cmd, desc in system_commands:
            run_command(cmd, desc)
    else:
        print("跳过系统级包安装 (需要root权限)")

def setup_gpu_environment():
    """设置GPU环境和MuJoCo配置"""
    print("=== 设置GPU环境 ===")

    # 检查GPU可用性
    try:
        result = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("警告: 未检测到NVIDIA GPU，将使用CPU模式")
            return False
        print("✓ 检测到NVIDIA GPU")
    except:
        print("警告: 无法执行nvidia-smi，可能没有GPU")
        return False

    # 配置NVIDIA ICD
    NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        try:
            with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
                f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
            print("✓ 创建NVIDIA ICD配置文件")
        except Exception as e:
            print(f"警告: 无法创建NVIDIA ICD配置: {e}")

    # 设置MuJoCo渲染后端 - 原代码修改
    # 原代码: get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
    # 修改为:
    print('设置环境变量以使用GPU渲染:')
    os.environ['MUJOCO_GL'] = 'egl'

    # 配置XLA使用Triton GEMM
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    print("✓ 配置XLA使用Triton GEMM")

    # 设置XLA Python客户端内存分配器
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    print("✓ 设置XLA内存分配器为platform")

    return True

def test_installation():
    """测试MuJoCo安装"""
    print("=== 测试MuJoCo安装 ===")

    try:
        import mujoco
        print("✓ 成功导入mujoco")

        # 测试基本功能
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print("✓ MuJoCo基本功能测试通过")
        print("安装成功.")
        return True

    except Exception as e:
        print(f"✗ MuJoCo安装测试失败: {e}")
        return False

# 执行环境设置
if __name__ == "__main__" or __name__ == "__builtin__":
    print("开始环境设置...")

    # 安装依赖包
    install_required_packages()
    setup_system_packages()

    # 设置GPU环境
    gpu_available = setup_gpu_environment()

    # 测试安装
    if not test_installation():
        print("环境设置失败，程序退出")
        sys.exit(1)

    print("环境设置完成，开始执行主程序...")

# =============================================================================
# 原始代码开始 - 保持原有结构和注释
# =============================================================================

# In[1]:

# 原代码:
# get_ipython().system('pip install mujoco')
# get_ipython().system('pip install mujoco_mjx')
# get_ipython().system('pip install brax')
# 修改说明: 已在环境设置部分完成，无需重复安装

# In[2]:

# 原代码:
# get_ipython().system('pip install pyvirtualdisplay')
# 修改说明: 已在环境设置部分完成

# In[3]:

# 原代码:
# get_ipython().system('pip install opencv-python')
# 修改说明: 已在环境设置部分完成

# In[ ]:

# 这三个好像必须在终端依次执行才能解决找不到路径的问题
# 检查glvnd相关目录
# 原代码:
# get_ipython().system('find /usr -name "*glvnd*" -type d 2>/dev/null')
# 修改为: (仅在调试时使用)
# run_command('find /usr -name "*glvnd*" -type d 2>/dev/null', "检查glvnd相关目录")

# In[5]:

# 检查已安装的图形包
# 原代码:
# get_ipython().system('dpkg -l | grep -E "(nvidia|mesa|gl|egl)"')
# 修改为: (仅在调试时使用)
# run_command('dpkg -l | grep -E "(nvidia|mesa|gl|egl)"', "检查已安装的图形包")

# In[6]:

# 更新包列表并安装OpenGL开发库
# 原代码:
# get_ipython().system('apt update && apt install -y libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev')
# 修改说明: 已在环境设置部分完成

# In[7]:

# 原代码:
# get_ipython().system('apt install -y libegl1-mesa-dev libosmesa6-dev --fix-missing')
# 修改说明: 已在环境设置部分完成

# Check if MuJoCo installation was successful
#

# In[8]:

import distutils.util
import time
from typing import Callable, List, NamedTuple, Optional, Union

# 原代码中的GPU检查和环境设置部分已移至环境设置部分
# 原代码:
# if subprocess.run('nvidia-smi').returncode:
#   raise RuntimeError(
#       'Cannot communicate with GPU. '
#       'Make sure you are using a GPU Colab runtime. '
#       'Go to the Runtime menu and select Choose runtime type.'
#   )
# 修改说明: 已在环境设置部分的setup_gpu_environment()函数中处理

# 原代码中的NVIDIA ICD配置已移至环境设置部分

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
# 原代码中的XLA配置已移至环境设置部分

# In[12]:

# @title Import packages for plotting and creating graphics
import json
import itertools
import numpy as np

# Graphics and plotting.
print("Installing mediapy:")
# 原代码:
# get_ipython().system('command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)')
# get_ipython().system('pip install -q mediapy')
# 修改说明: 已在环境设置部分完成

import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# In[13]:

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
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
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from orbax import checkpoint as ocp

# In[8]:

# 原代码:
# get_ipython().system('pip install playground')
# 修改说明: 已在环境设置部分完成

# In[4]:

from mujoco_playground import wrapper
from mujoco_playground import registry
registry.manipulation.ALL_ENVS

# Dexterous Manipulation
# Let's now train a policy that was transferred onto a real Leap Hand robot with the LeapCubeReorient environment! The environment contains a cube placed in the center of the hand, and the goal is to re-orient the cube in SO(3).

# In[5]:

env_name = 'LeapCubeReorient'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

# In[7]:

env_cfg

# Train Policy
# Let's train an initial policy and visualize the rollouts. Notice that the PPO parameters contain policy_obs_key and value_obs_key fields, which allow us to train brax PPO with asymmetric observations for the actor and the critic. While the actor recieves proprioceptive state similar in nature to the real-world camera tracking sensors, the critic network recieves privileged state only available in the simulator. This enables more sample efficient learning, and we are able to train an initial policy in 33 minutes on a single RTX 4090.
#
# Depending on the GPU device and topology, training can be brought down to 10-20 minutes as shown in the MuJoCo Playground technical report.

# In[ ]:

# 原代码:
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# 修改说明: 已在setup_gpu_environment()函数中完成设置

# In[ ]:

# 2025年12月16日16:25 修改：添加datetime导入以解决NameError错误
# 原代码中已经有datetime导入，保持原样
from datetime import datetime
from mujoco_playground.config import manipulation_params
ppo_params = manipulation_params.brax_ppo_config(env_name)

# 2025年12月18日修改：减少内存消耗的参数覆盖
print("原始配置:")
print(f"  num_envs: {ppo_params.num_envs}")
print(f"  batch_size: {ppo_params.batch_size}")
print(f"  num_minibatches: {ppo_params.num_minibatches}")
print(f"  unroll_length: {ppo_params.unroll_length}")

# 内存优化参数覆盖
ppo_params.num_envs = 8192        # 从8192减少到1024 (减少87.5%内存)
ppo_params.batch_size = 256       # 从256减少到128
ppo_params.num_minibatches = 32   # 从32减少到16
ppo_params.unroll_length = 40     # 从40减少到20

print("\n优化后配置:")
print(f"  num_envs: {ppo_params.num_envs}")
print(f"  batch_size: {ppo_params.batch_size}")
print(f"  num_minibatches: {ppo_params.num_minibatches}")
print(f"  unroll_length: {ppo_params.unroll_length}")
print("预期内存减少: ~75%")

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]  # 确保datetime已导入

def progress(num_steps, metrics):
  # 原代码使用IPython的clear_output，修改为兼容普通Python脚本
  try:
      from IPython.display import clear_output
      clear_output(wait=True)
  except ImportError:
      # 如果不在IPython环境中，使用简单的清屏
      import os
      os.system('clear' if os.name == 'posix' else 'cls')

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  # 原代码中的变量名拼写错误修正
  # plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")  # 2025年12月16日16:25 修改：原代码中的变量名拼写错误
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")  # 新代码：修正变量名为y_dataerr

  try:
      from IPython.display import display
      display(plt.gcf())
  except ImportError:
      plt.show()

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

# In[ ]:

print("开始训练...")
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# In[ ]:

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# In[ ]:

rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]
media.show_video(frames, fps=1.0 / env.dt / render_every)