# test_model.py - 根據 config.yaml 載入指定模型並測試其表現

import os
import sys
import time
import yaml
import torch
import numpy as np
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.baby_pong_env import BabyPongBounceEnv as BabyPongEnv
from models.qnet import QNet

# -------------------------
# 載入設定檔
# -------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

test_cfg = cfg['test']
model_path = test_cfg['model_path']
render = test_cfg['render']
episodes = test_cfg['episodes']

# -------------------------
# 測試指定模型的表現
# -------------------------
def evaluate_model(path):
    env = BabyPongEnv(**cfg['env'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNet(state_dim, action_dim)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    total_rewards = []
    font = None
    if render:
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 24)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()
                reward_surface = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 0))
                env.screen.blit(reward_surface, (10, 10))
                pygame.display.update()
                time.sleep(0.03)

            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    env.close()
    avg = np.mean(total_rewards)
    print("------------------------")
    print(f"✅ Average reward over {episodes} episodes: {avg:.2f}")

if __name__ == "__main__":
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案：{model_path}")
    else:
        print(f"📦 測試模型: {model_path}")
        evaluate_model(model_path)