#!/usr/bin/env python3
"""
Generate training loss and reward plots from training data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Set matplotlib to non-interactive mode
plt.switch_backend('Agg')

def main():
    print("🚀 Generating Training Loss Plots...")

    if not os.path.exists('training_progress.json'):
        print("❌ training_progress.json not found!")
        return

    # Load training data
    timestamps = []
    steps = []
    rewards = []
    stds = []

    with open('training_progress.json', 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                timestamps.append(entry.get('timestamp', ''))
                steps.append(entry.get('num_steps', 0))
                rewards.append(entry.get('episode_reward', 0))
                stds.append(entry.get('episode_reward_std', 0))
            except:
                continue

    if len(rewards) == 0:
        print("❌ No valid training data found!")
        return

    print(f"✅ Loaded {len(rewards)} training entries")

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Leap Cube Reorient Training Analysis', fontsize=16, fontweight='bold')

    # Convert to arrays
    steps = np.array(steps) / 1e6  # Convert to millions
    rewards = np.array(rewards)
    stds = np.array(stds)

    x_indices = np.arange(len(rewards))

    # Plot 1: Reward over training steps
    axes[0,0].plot(x_indices, rewards, 'b-', linewidth=2, label='Episode Reward')
    axes[0,0].fill_between(x_indices, rewards - stds, rewards + stds, alpha=0.3, color='blue')
    axes[0,0].set_title('Episode Reward Progress')
    axes[0,0].set_xlabel('Training Checkpoint')
    axes[0,0].set_ylabel('Episode Reward')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()

    # Plot 2: Training steps accumulation
    axes[0,1].plot(x_indices, steps, 'g-', linewidth=2)
    axes[0,1].set_title('Training Steps (Millions)')
    axes[0,1].set_xlabel('Training Checkpoint')
    axes[0,1].set_ylabel('Steps (Millions)')
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Reward standard deviation (stability)
    axes[1,0].plot(x_indices, stds, 'r-', linewidth=2)
    axes[1,0].set_title('Reward Standard Deviation')
    axes[1,0].set_xlabel('Training Checkpoint')
    axes[1,0].set_ylabel('Standard Deviation')
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Reward histogram
    axes[1,1].hist(rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Reward Distribution')
    axes[1,1].set_xlabel('Episode Reward')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Training Statistics:
Final Reward: {rewards[-1]:.2f}
Max Reward: {np.max(rewards):.2f}
Improvement: {rewards[-1] - rewards[0]:+.2f}
Final Steps: {steps[-1]:.1f}M
Std Dev: {stds[-1]:.2f}"""

    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()

    # Save plots
    output_file = 'training_loss_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved as: {output_file}")

    # Save PDF version
    pdf_file = 'training_loss_analysis.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✅ High-quality PDF saved as: {pdf_file}")

    # Print summary
    print(f"\n📊 Training Summary:")
    print(f"   Initial Reward: {rewards[0]:.2f}")
    print(f"   Final Reward: {rewards[-1]:.2f}")
    print(f"   Max Reward: {np.max(rewards):.2f}")
    print(f"   Improvement: {rewards[-1] - rewards[0]:+.2f}")
    print(f"   Final Steps: {steps[-1]:.1f}M")
    print(f"   Final Std Dev: {stds[-1]:.2f}")

if __name__ == "__main__":
    main()