# compare_models.py - 比較多個模型在同一環境下的表現

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.baby_pong_env import BabyPongBounceEnv as BabyPongEnv
from models.qnet import QNet

# -------------------------
# 載入設定檔
# -------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# -------------------------
# 設定要比較的模型檔案清單
# -------------------------
model_files = [
    "checkpoints/model_traget2_ep399.pth",
    "checkpoints/model_traget2_ep799.pth",
    "checkpoints/model_traget2_ep1199.pth",
    "checkpoints/model_traget2_ep1599.pth",
    "checkpoints/model_traget2_ep1999.pth"
]

# -------------------------
# 測試每個模型表現
# -------------------------
def evaluate_model(path, episodes=10):
    env = BabyPongEnv(**cfg['env'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = QNet(state_dim, action_dim)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    env.close()
    return np.mean(total_rewards), np.std(total_rewards)

# -------------------------
# 主程式：比較結果 + 繪圖
# -------------------------
labels = []
avgs = []
stds = []

print("📊 Model comparison results:")
for path in model_files:
    if not os.path.exists(path):
        print(f"❌ 找不到模型：{path}")
        continue
    name = os.path.basename(path)
    avg, std = evaluate_model(path, episodes=10)
    print(f"{name}: Avg = {avg:.2f}, Std = {std:.2f}")
    labels.append(name)
    avgs.append(avg)
    stds.append(std)

# 畫圖比較
plt.figure(figsize=(10, 6))
plt.barh(labels, avgs, xerr=stds, color="skyblue")
plt.xlabel("Average Reward")
plt.title("Model performance comparison (± std)")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()
