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
losses = []
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
    with torch.no_grad():
        next_q_values = target_model(next_states)
        next_q_value = next_q_values.max(1)[0]

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    expected_q = rewards_ + cfg['train']['gamma'] * next_q_value * (~dones)

    loss = nn.MSELoss()(q_value, expected_q)
    writer.add_scalar("Loss", loss.item(), len(rewards))
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 建立存圖資料夾
os.makedirs("plot", exist_ok=True)

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

    if (episode + 1) % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"🔄 Updated target network at episode {episode}")

    if (episode + 1) % cfg['train']['save_every'] == 0:
        model_path = f"checkpoints/model_traget2_ep{episode}.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode
        }, model_path)
        print(f"💾 Saved model to {model_path}")

    print(f"Episode {episode}: Total reward = {total_reward}")

    if (episode + 1) % cfg['train']['render_every'] == 0:
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[警告] 嘗試關閉視窗，但我假裝沒看到。")
                    # 不退出訓練，只跳過這個事件
                    continue
            time.sleep(0.05)
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, _, done, _, _ = env.step(action)

# 繪製 reward 與 loss 圖表
window = 50
smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards, alpha=0.3, label='Reward')
plt.plot(range(window - 1, len(rewards)), smoothed_rewards, label='Smoothed Reward')
plt.title('Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses, alpha=0.3, label='Loss')
plt.plot(range(window - 1, len(losses)), smoothed_losses, label='Smoothed Loss')
plt.title('Loss over Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("plot/training_metrics.png")

# 結束環境，關閉資源
env.close()
