import gymnasium as gym
import numpy as np
import torch

from lib.agent_ppo import PPOAgent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #env = gym.make("Humanoid-v5", render_mode="human")
    env = gym.make("Humanoid-v5", render_mode="offscreen")
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load("model.pt"))
    agent.eval()

    obs, _ = env.reset()
    done = False
    frames = []  # 用于保存渲染帧
    while not done:
        # Render the frame
        #env.render()
        # 获取渲染的帧
        frame = env.render()
        #frame = env.render(render_mode="rgb_array")
        frame.append(frame)
        
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    
    # 保存渲染的帧为视频
    if frames:
        frame_height, frame_width, _ = frames[0].shape
        video_out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        for frame in frames:
            video_out.write(frame)

        video_out.release()

    env.close()
