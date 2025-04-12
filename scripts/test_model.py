# test_model.py - 顯示旋轉球的可視化版本

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

# 可視化樣式設定
THEME = {
    "bg": (20, 20, 30),
    "ball": (255, 100, 100),
    "paddle": (100, 255, 100),
    "text": (255, 255, 100),
    "font": "Courier New",
    "font_size": 20
}

TIME_SCALE = 1  # 可調整為 0.25 ～ 5.0

# -------------------------
# 測試指定模型表現
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
        font = pygame.font.SysFont(THEME['font'], THEME['font_size'])

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frame = 0
        spin_history = []

        while not done:
            if render:
                env.render()

                # 文字資訊顯示
                reward_surface = font.render(f"Episode {ep+1} | Reward: {total_reward:.2f}", True, THEME['text'])
                speed_now = np.linalg.norm([env.ball_vx, env.ball_vy])
                base_speed = cfg['env']['base_speed']
                speed_ratio = speed_now / base_speed
                speed_surface = font.render(f"Speed: {speed_now:.3f} ({speed_ratio:.1f}x)", True, THEME['text'])

                env.screen.blit(reward_surface, (10, 10))
                env.screen.blit(speed_surface, (10, 35))

                # 旋轉視覺化：畫圓弧箭頭表示旋轉方向與強度
                spin = getattr(env, 'spin', 0.0)
                spin_text = font.render(f"Spin: {spin:+.2f}", True, THEME['text'])
                env.screen.blit(spin_text, (10, 60))

                pygame.display.update()
                time.sleep(0.03 / TIME_SCALE)

            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            frame += 1

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