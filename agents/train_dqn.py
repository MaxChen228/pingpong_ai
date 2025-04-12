# train_dqn.py - 使用 config.yaml 管理訓練與環境參數

import pygame
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
from collections import deque
from envs.baby_pong_env import BabyPongBounceEnv as BabyPongEnv
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# 🧾 使用說明
# -------------------------
# 1. 所有超參設定集中於 config.yaml 檔案中
# 2. 你可以調整參數而不需要更動此程式碼
# 3. 執行方式：python agents/train_dqn.py
# 4. TensorBoard 查看訓練過程：tensorboard --logdir=runs

# -------------------------
# 模型架構：Q-learning 使用的神經網路
# -------------------------
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------
# 讀取參數設定
# -------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 初始化環境（從 config['env']）
env = BabyPongEnv(**cfg['env'])

# 初始化模型與參數
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = QNet(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
memory = deque(maxlen=cfg['train']['memory_size'])
epsilon = 1.0
rewards = []
start_episode = 0
writer = SummaryWriter(log_dir="runs/baby_pong")

# 嘗試載入模型（如有設定）
if cfg['train']['load_model'] and os.path.exists(cfg['train']['checkpoint_path']):
    checkpoint = torch.load(cfg['train']['checkpoint_path'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint['epsilon']
    start_episode = checkpoint['episode'] + 1
    print(f"✔ Loaded checkpoint from episode {checkpoint['episode']}")
else:
    print("🆕 Starting from scratch")

# 訓練函數
def train():
    if len(memory) < cfg['train']['batch_size']:
        return

    batch = random.sample(memory, cfg['train']['batch_size'])
    states, actions, rewards_, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards_ = torch.tensor(rewards_, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)

    q_values = model(states)
    next_q_values = model(next_states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q = rewards_ + cfg['train']['gamma'] * next_q_value * (~dones)

    loss = nn.MSELoss()(q_value, expected_q)
    writer.add_scalar("Loss", loss.item(), len(rewards))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主訓練迴圈
for episode in range(start_episode, start_episode + cfg['train']['episodes']):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train()
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    epsilon = max(cfg['train']['min_epsilon'], epsilon * cfg['train']['epsilon_decay'])

    print(f"Episode {episode}: Total reward = {total_reward}")

    # 儲存模型
    if (episode + 1) % cfg['train']['save_every'] == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f"checkpoints/model_ep{episode}_{timestamp}.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode
        }, model_path)
        print(f"💾 Saved model to {model_path}")

    # 渲染動畫畫面
    if (episode + 1) % cfg['train']['render_every'] == 0:
        state, _ = env.reset()
        done = False
        print("--- RENDER ---")
        while not done:
            env.render()
            time.sleep(0.05)
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, _, done, _, _ = env.step(action)

# 收尾
env.close()
plt.plot(rewards, label='Reward')
plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), label='Moving Avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.legend()
plt.grid()
plt.show()
plt.savefig(f"plots/reward_curve_{time.strftime('%Y%m%d_%H%M%S')}.png")  # 儲存圖像
