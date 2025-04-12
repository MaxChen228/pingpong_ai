# train_dqn.py 完整代碼 - 使用 Target Network 改進 DQN (完整註解版)

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

# 定義Q學習神經網路模型
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

# 從設定檔中讀取配置
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 初始化環境
env = BabyPongEnv(**cfg['env'])

# 初始化模型及目標模型
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = QNet(state_dim, action_dim)
target_model = QNet(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())  # 初始化目標模型權重與主模型一致
target_model.eval()  # 目標模型設為評估模式，不進行梯度更新

optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
memory = deque(maxlen=cfg['train']['memory_size'])  # 經驗回放記憶庫
epsilon = 1.0  # ε-greedy探索初始值
rewards = []
start_episode = 0
writer = SummaryWriter(log_dir="runs/baby_pong")
target_update_freq = cfg['train'].get('target_update_freq', 10)  # 目標網路更新頻率

# 載入先前訓練過的模型（若有）
if cfg['train']['load_model'] and os.path.exists(cfg['train']['checkpoint_path']):
    checkpoint = torch.load(cfg['train']['checkpoint_path'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint['epsilon']
    start_episode = checkpoint['episode'] + 1
    target_model.load_state_dict(model.state_dict())
    print(f"✔ Loaded checkpoint from episode {checkpoint['episode']}")
else:
    print("🆕 Starting from scratch")

# 訓練函數，使用batch更新模型

def train():
    # 確認記憶庫中有足夠樣本
    if len(memory) < cfg['train']['batch_size']:
        return

    # 從記憶庫中隨機抽取batch
    batch = random.sample(memory, cfg['train']['batch_size'])
    states, actions, rewards_, next_states, dones = zip(*batch)

    # 轉換成tensor
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards_ = torch.tensor(rewards_, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)

    # 計算當前狀態的Q值
    q_values = model(states)
    # 使用目標模型計算下一狀態的Q值（固定目標）
    with torch.no_grad():
        next_q_values = target_model(next_states)
        next_q_value = next_q_values.max(1)[0]

    # 依照實際採取的行動取得預測Q值
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    # 計算目標Q值（基於Bellman方程）
    expected_q = rewards_ + cfg['train']['gamma'] * next_q_value * (~dones)

    # 計算損失函數
    loss = nn.MSELoss()(q_value, expected_q)
    writer.add_scalar("Loss", loss.item(), len(rewards))

    # 更新模型參數
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主訓練迴圈
for episode in range(start_episode, start_episode + cfg['train']['episodes']):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # ε-greedy策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

        # 執行動作，獲取下一狀態和獎勵
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train()
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    epsilon = max(cfg['train']['min_epsilon'], epsilon * cfg['train']['epsilon_decay'])

    # 定期更新目標網路
    if (episode + 1) % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"🔄 Updated target network at episode {episode}")

    # 定期儲存模型
    if (episode + 1) % cfg['train']['save_every'] == 0:
        model_path = f"checkpoints/model_traget_ep{episode}.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode
        }, model_path)
        print(f"💾 Saved model to {model_path}")

    print(f"Episode {episode}: Total reward = {total_reward}")

    # 定期渲染環境觀察訓練狀態
    if (episode + 1) % cfg['train']['render_every'] == 0:
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            time.sleep(0.05)
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, _, done, _, _ = env.step(action)

# 結束環境，關閉資源
env.close()
