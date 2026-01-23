#!/usr/bin/env python3
"""
快速训练演示 - 短期训练+自动生成视频
"""
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
import functools
import datetime


def render_fallback(rollout, num_steps):
    """备用渲染方法"""
    print("使用备用渲染...")
    frames = []

    for i in range(0, num_steps, 10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        if i < len(rollout):
            state = rollout[i]
            if hasattr(state, 'reward'):
                reward = float(state.reward)
                bg_intensity = min(255, max(50, int(50 + reward * 2)))
                frame[:, :] = [bg_intensity // 4, bg_intensity // 2, bg_intensity]

        cube_x, cube_y = 160, 120
        cube_size = 30
        rotation = i * 0.2

        for dy in range(-cube_size, cube_size + 1):
            for dx in range(-cube_size, cube_size + 1):
                if abs(dx) <= cube_size and abs(dy) <= cube_size:
                    px, py = cube_x + dx, cube_y + dy
                    if 0 <= px < 320 and 0 <= py < 240:
                        frame[py, px] = [
                            int(128 + 127 * np.sin(rotation + dx * 0.1)),
                            int(128 + 127 * np.cos(rotation + dy * 0.1)),
                            int(128 + 127 * np.sin(rotation + np.pi/3))
                        ]

        frames.append(frame)

    print(f"✓ 备用渲染完成: {len(frames)} 帧")
    return frames


def main():
    print("="*70)
    print("快速训练演示 - LeapCubeReorient")
    print("="*70)

    # === 环境配置 ===
    env_name = 'LeapCubeReorient'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    print(f"\n环境: {env_name}")
    print(f"Episode长度: {env_cfg.episode_length}")

    # === 训练配置（简化版，短时间训练）===
    print("\n开始短期训练...")
    print("训练步数: 1000万步 (约5-10分钟)")

    ppo_params = manipulation_params.brax_ppo_config(env_name)

    # 修改训练参数为较小值
    ppo_params.num_timesteps = 10_000_000
    ppo_params.num_eval_envs = 128
    ppo_params.num_envs = 128
    ppo_params.episode_length = 1000

    print(f"配置: num_envs={ppo_params.num_envs}, episode_length={ppo_params.episode_length}")

    # === 包装环境 ===
    wrapped_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=ppo_params.episode_length
    )

    # === 训练 ===
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks

    # Handle network_factory if already in params
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo_train.train,
        **ppo_training_params,
        network_factory=network_factory,
        seed=42,
    )

    print("\n开始训练...")
    inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    final_reward = metrics.get('eval/episode_reward', 0)
    print(f"\n训练完成!")
    print(f"最终奖励: {final_reward:.3f}")

    # === 生成视频 ===
    print("\n生成演示视频...")

    # JIT编译 - 使用原始环境而非包装后的环境进行单环境推理
    # inference_fn已经包含了训练好的参数
    jit_inference = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 生成rollout
    print(f"运行 {env_cfg.episode_length} 步...")
    rng = jax.random.PRNGKey(42)
    rollout = []
    total_reward = 0

    state = jit_reset(rng)
    rollout.append(state)

    for step in range(env_cfg.episode_length):
        if step % 100 == 0:
            print(f"  步骤 {step}/{env_cfg.episode_length}, 奖励: {float(state.reward):.3f}")

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        total_reward += float(state.reward)

    print(f"\n✓ Rollout完成, 总奖励: {total_reward:.2f}")

    # === 渲染视频 ===
    print("\n渲染视频...")
    try:
        # 使用原始环境渲染
        frames = env.render(rollout[::5])
        if len(frames) > 0:
            print(f"✓ 环境渲染成功: {len(frames)} 帧")
        else:
            raise Exception("渲染结果为空")
    except Exception as e:
        print(f"⚠️ 环境渲染失败: {e}")
        frames = render_fallback(rollout, env_cfg.episode_length)

    # === 保存视频 ===
    os.makedirs("./videos", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"./videos/leap_cube_demo_{timestamp}.mp4"

    media.write_video(video_filename, frames, fps=30.0)

    file_size = os.path.getsize(video_filename) / (1024 * 1024)
    print(f"\n✅ 视频已保存!")
    print(f"  文件路径: {video_filename}")
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"  视频时长: {len(frames) / 30.0:.1f} 秒")
    print(f"  演示奖励: {total_reward:.2f}")

    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == "__main__":
    main()
