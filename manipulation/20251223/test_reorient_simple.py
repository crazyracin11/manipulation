#!/usr/bin/env python
# coding: utf-8

# 简化的测试版本，只测试核心功能

print("=== 测试MuJoCo Playground环境 ===")

import os
import sys
from datetime import datetime

# 设置基本环境变量
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# 设置虚拟显示器和渲染后端
print("设置虚拟显示器...")
try:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1024, 768))
    display.start()
    print("✓ 虚拟显示器启动成功")
except:
    print("警告: 虚拟显示器启动失败，继续尝试...")

print("使用OSMesa渲染后端（避免EGL问题）")
os.environ['MUJOCO_GL'] = 'osmesa'

# 添加无头模式
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    print("导入基础库...")
    import json
    import itertools
    import time
    from typing import Callable, List, NamedTuple, Optional, Union
    import numpy as np

    print("✓ 基础库导入成功")

    print("导入MuJoCo相关库...")
    import mujoco
    from mujoco import mjx
    print("✓ MuJoCo导入成功")

    print("导入Brax相关库...")
    from brax import base, envs, math
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    print("✓ Brax导入成功")

    print("导入JAX相关库...")
    import jax
    from jax import numpy as jp
    print("✓ JAX导入成功")

    print("导入其他必要库...")
    import functools
    from ml_collections import config_dict
    import matplotlib.pyplot as plt
    import mediapy as media
    print("✓ 其他库导入成功")

    print("导入MuJoCo Playground...")
    from mujoco_playground import wrapper
    from mujoco_playground import registry
    print("✓ MuJoCo Playground导入成功")

    # 显示可用环境
    print(f"可用的操作环境: {registry.manipulation.ALL_ENVS}")

    # 设置环境
    env_name = 'LeapCubeReorient'
    print(f"加载环境: {env_name}")
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    print("✓ 环境加载成功")

    print("环境配置:")
    print(env_cfg)

    # 获取训练参数
    from mujoco_playground.config import manipulation_params
    ppo_params = manipulation_params.brax_ppo_config(env_name)
    print("✓ 获取PPO训练参数成功")

    print("训练参数:")
    print(f"  num_envs: {ppo_params.num_envs}")
    print(f"  batch_size: {ppo_params.batch_size}")
    print(f"  num_minibatches: {ppo_params.num_minibatches}")
    print(f"  unroll_length: {ppo_params.unroll_length}")

    print("\n🎉 所有环境测试通过！")
    print("现在可以运行完整的训练了。")

    # 测试基本环境功能（不实际运行训练）
    print("\n=== 测试环境基本功能 ===")

    # 测试环境重置
    print("测试环境重置...")
    jit_reset = jax.jit(env.reset)
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    print("✓ 环境重置成功")
    print(f"观察空间形状: {state.obs.shape}")
    print(f"奖励: {state.reward}")

    # 测试环境步进
    print("测试环境步进...")
    jit_step = jax.jit(env.step)
    act_rng, rng = jax.random.split(rng)
    # 随机动作
    action = jax.random.uniform(act_rng, (env.action_dim,))
    next_state = jit_step(state, action)
    print("✓ 环境步进成功")
    print(f"下一状态奖励: {next_state.reward}")
    print(f"是否结束: {next_state.done}")

    print("\n✅ 所有测试通过！环境完全就绪。")

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 测试失败: {e}")
    sys.exit(1)