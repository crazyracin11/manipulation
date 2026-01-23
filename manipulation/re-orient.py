#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mujoco')
get_ipython().system('pip install mujoco_mjx')
get_ipython().system('pip install brax')


# In[2]:


get_ipython().system('pip install pyvirtualdisplay')


# In[3]:


get_ipython().system('pip install opencv-python')


# In[ ]:


# 这三个好像必须在终端依次执行才能解决找不到路径的问题
# 检查glvnd相关目录
get_ipython().system('find /usr -name "*glvnd*" -type d 2>/dev/null')


# In[5]:


# 检查已安装的图形包
get_ipython().system('dpkg -l | grep -E "(nvidia|mesa|gl|egl)"')


# In[6]:


# 更新包列表并安装OpenGL开发库
get_ipython().system('apt update && apt install -y libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev')


# In[7]:


get_ipython().system('apt install -y libegl1-mesa-dev libosmesa6-dev --fix-missing')


# Check if MuJoCo installation was successful
# 
# 

# In[8]:


import distutils.util
import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')

try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".'
  )

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


# In[12]:


# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
print("Installing mediapy:")
get_ipython().system('command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)')
get_ipython().system('pip install -q mediapy')
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# In[ ]:





# In[13]:


# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
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
import numpy as np
from orbax import checkpoint as ocp


# In[8]:


get_ipython().system('pip install playground')


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


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# In[ ]:


# 2025年12月16日16:25 修改：添加datetime导入以解决NameError错误
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
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  # plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")  # 2025年12月16日16:25 修改：原代码中的变量名拼写错误
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")  # 新代码：修正变量名为y_dataerr

  display(plt.gcf())

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

