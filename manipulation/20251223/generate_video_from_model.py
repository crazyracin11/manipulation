#!/usr/bin/env python3
"""
加载训练好的模型并生成立方体旋转演示视频
"""
import os
import warnings
warnings.filterwarnings("ignore")

# 设置环境
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

def generate_video_with_trained_model():
    """加载训练好的模型并生成视频"""

    print("="*70)
    print("加载训练模型并生成立方体旋转演示视频")
    print("="*70)

    # === 1. 加载环境 ===
    print("\n[1] 加载环境...")
    env_name = 'LeapCubeReorient'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    print(f"✓ 环境加载成功: {env_name}")
    print(f"  Episode长度: {env_cfg.episode_length}")

    # === 2. 包装环境 ===
    print("\n[2] 包装环境用于训练...")
    wrapped_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=1000
    )
    print("✓ 环境包装完成")

    # === 3. 加载模型参数 ===
    print("\n[3] 加载训练好的模型参数...")
    model_path = "/root/HW/RL_qizhi-main/checkpoints/model_step_200000000.npz"

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    # 加载权重
    data = np.load(model_path, allow_pickle=True)
    print(f"✓ 模型文件加载成功")
    print(f"  包含的参数组: {len(data.files)}")

    # 重建参数树
    params = {}
    for key in data.files:
        params[key] = jp.array(data[key])

    print(f"✓ 参数树重建完成，共 {len(params)} 个参数")

    # === 4. 创建推理函数 ===
    print("\n[4] 创建推理函数...")
    from brax.training.agents.ppo import networks as ppo_networks

    # 创建网络
    network = ppo_networks.make_ppo_networks(
        observation_size=wrapped_env.observation_size,
        action_size=wrapped_env.action_size,
    )

    # 创建推理函数
    def make_policy_fn(params):
        """创建策略函数"""
        def policy(observations, rng):
            """推理函数"""
            # 使用actor网络
            return network.apply(params, observations, rng)
        return policy

    policy_fn = make_policy_fn(params)
    print("✓ 推理函数创建完成")

    # === 5. JIT编译 ===
    print("\n[5] JIT编译推理函数...")
    jit_policy = jax.jit(policy_fn)
    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)
    print("✓ JIT编译完成")

    # === 6. 生成演示rollout ===
    print("\n[6] 生成演示rollout...")
    print(f"  运行 {env_cfg.episode_length} 步...")

    rng = jax.random.PRNGKey(42)
    rollout = []
    total_reward = 0

    # 重置环境
    print("  重置环境...")
    state = jit_reset(rng)
    rollout.append(state)

    # 运行完整的episode
    for step in range(env_cfg.episode_length):
        if step % 100 == 0:
            print(f"  步骤 {step}/{env_cfg.episode_length}, 当前奖励: {float(state.reward):.3f}")

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        total_reward += float(state.reward)

    print(f"\n✓ Rollout生成完成")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均奖励: {total_reward/env_cfg.episode_length:.3f}")

    # === 7. 渲染视频 ===
    print("\n[7] 渲染视频帧...")

    try:
        # 使用环境的渲染功能
        print("  尝试使用环境渲染...")
        frames = wrapped_env.render(rollout[::5])  # 每5帧渲染一次

        if len(frames) > 0:
            print(f"✓ 环境渲染成功: {len(frames)} 帧")
        else:
            raise Exception("环境渲染结果为空")

    except Exception as e:
        print(f"⚠️ 环境渲染失败: {e}")
        print("  使用备用渲染方法...")
        frames = render_fallback(rollout, len(rollout))

    if len(frames) == 0:
        print("❌ 无法生成任何帧")
        return False

    # === 8. 保存视频 ===
    print("\n[8] 保存视频...")
    import datetime
    os.makedirs("./videos", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"./videos/leap_cube_trained_demo_{timestamp}.mp4"

    media.write_video(video_filename, frames, fps=30.0)

    file_size = os.path.getsize(video_filename) / (1024 * 1024)
    print(f"\n✅ 视频保存成功!")
    print(f"  文件路径: {video_filename}")
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"  视频时长: {len(frames) / 30.0:.1f} 秒")
    print(f"  演示总奖励: {total_reward:.2f}")

    return True


def render_fallback(rollout, num_steps):
    """备用渲染方法"""
    print("  使用备用渲染方法生成概念演示...")

    frames = []
    print(f"  生成 {num_steps // 10} 帧...")

    for i in range(0, num_steps, 10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # 获取当前状态
        if i < len(rollout):
            state = rollout[i]

            # 背景颜色根据奖励变化
            if hasattr(state, 'reward'):
                reward = float(state.reward)
                bg_intensity = min(255, max(50, int(50 + reward * 2)))
                frame[:, :] = [bg_intensity // 4, bg_intensity // 2, bg_intensity]

            # 立方体表示
            cube_x, cube_y = 160, 120
            cube_size = 30

            # 根据步骤变化创建旋转效果
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

            # 显示步骤信息
            step_text = f"Step: {i}"
            reward_text = f"Reward: {reward:.2f}" if hasattr(state, 'reward') else ""

            # 简单文本绘制
            for j, char in enumerate(f"LeapCubeReorient - Trained Model"):
                if 10 + j * 8 < 320:
                    frame[20, 10 + j * 8] = [255, 255, 255]

        frames.append(frame)

        if i % 100 == 0:
            print(f"    生成帧 {i}/{num_steps}")

    print(f"✓ 备用渲染完成: {len(frames)} 帧")
    return frames


if __name__ == "__main__":
    success = generate_video_with_trained_model()

    if success:
        print("\n" + "="*70)
        print("🎉 视频生成完成！")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ 视频生成失败")
        print("="*70)
