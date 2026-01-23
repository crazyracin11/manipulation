#!/usr/bin/env python3
"""绘制训练 reward 曲线"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# 训练数据 (从日志中提取)
training_data = [
    (0, -8.163),
    (10813440, 89.896),
    (21626880, 102.543),
    (32440320, 103.621),
    (43253760, 106.479),
    (54067200, 108.446),
    (64880640, 95.210),
    (75694080, 86.749),
    (86507520, 86.844),
    (97320960, 89.496),
    (108134400, 81.967),
    (118947840, 84.861),
    (129761280, 83.423),
    (140574720, 89.893),
    (151388160, 106.783),
    (162201600, 108.714),
    (173015040, 113.027),
    (183828480, 109.535),
    (194641920, 110.829),
    (200000000, 110.065),
]

steps = [x[0] for x in training_data]
rewards = [x[1] for x in training_data]

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(steps, rewards, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Episode Reward', fontsize=12)
plt.title('LeapCubeReorient Training Progress', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 添加起点和终点标注
plt.annotate(f'Start: {rewards[0]:.1f}', xy=(steps[0], rewards[0]), xytext=(steps[0]+5e6, rewards[0]-20),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
plt.annotate(f'Final: {rewards[-1]:.1f}', xy=(steps[-1], rewards[-1]), xytext=(steps[-1]-2e7, rewards[-1]+5),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

# 保存图片
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'/root/HW/RL_qizhi-main/manipulation/training_plots/training_progress_{timestamp}.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"✅ Reward 曲线已保存: {output_path}")

# 保存数据为 JSON
data_path = f'/root/HW/RL_qizhi-main/manipulation/training_plots/training_data_{timestamp}.json'
with open(data_path, 'w') as f:
    json.dump({"steps": steps, "rewards": rewards}, f, indent=2)
print(f"✅ 训练数据已保存: {data_path}")
