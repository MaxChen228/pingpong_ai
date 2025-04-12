# 🏓 PINGPONG-AI - Project Overview

## 🧠 Goal
Train a reinforcement learning agent (DQN-based) to control a paddle in a 2D pong-like environment. The project features a fully modular, configurable, and visualizable AI training framework.

---

## 📁 Directory Structure
```
PINGPONG-AI/
├── agents/                # Main training logic
│   └── train_dqn.py
├── checkpoints/           # Saved model checkpoints (.pth)
├── envs/                  # Custom environment
│   ├── __init__.py
│   └── baby_pong_env.py   # BabyPongBounceEnv definition
├── models/                # Neural network models
│   └── qnet.py            # QNet class (used in DQN)
├── plots/                 # Saved reward curves
├── runs/                  # TensorBoard logs
├── scripts/               # Utility tools
│   ├── list_checkpoints.py
│   └── test_model.py      # Model evaluation with live render
├── config.yaml            # Central config file (env/train/test)
├── requirements.txt       # All dependencies
└── 說明.txt / README.md   # Documentation
```

---

## 🛠️ How to Use

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start training
```bash
python agents/train_dqn.py
```
All parameters are configurable in `config.yaml`.

### 3️⃣ Launch TensorBoard to monitor training
```bash
tensorboard --logdir=runs
```
Then open [http://localhost:6006](http://localhost:6006)

### 4️⃣ Evaluate trained model
```bash
python scripts/test_model.py
```
- Renders gameplay and displays reward value on screen.
- Reads model path and render toggle from `config.yaml > test:` section.

---

## ⚙️ Config File Breakdown
All parameters are centralized in `config.yaml`, including:
- `env`: Pong environment settings
- `train`: DQN hyperparameters
- `test`: Evaluation settings

Feel free to create variants (e.g., `config_easy.yaml`, `config_hard.yaml`) to manage experiments.

---

## ✨ Highlights
- ✅ Full modular training structure (train/test separation)
- ✅ Configurable environment (paddle size, bounce speed, penalties)
- ✅ Fully config-driven, no hardcoded numbers
- ✅ TensorBoard visualization for reward, loss, epsilon
- ✅ Reward curves are auto-saved
- ✅ Test render includes live reward HUD

---

## 🧩 Suggested Extensions
- Support PPO / A2C / other algorithms
- Export training sessions as video
- Implement human vs. AI mode
- Add compare_models.py for side-by-side analysis
- YAML-based experiment manager for difficulty presets

---

# 🏓 PINGPONG-AI 專案說明文件

## 🧠 專案目標
訓練一個強化學習模型（DQN）在 2D Pong 環境中控制擋板反彈球，
並逐步強化 AI 學習能力，提升表現，同時打造一個模組化、可視化、可管理的完整訓練系統。

---

## 📁 專案結構總覽
```
PINGPONG-AI/
├── agents/                # 訓練主程式
│   └── train_dqn.py       
├── checkpoints/           # 模型儲存檔案 (.pth)
├── envs/                  # 自訂環境
│   ├── __init__.py
│   └── baby_pong_env.py   # BabyPongBounceEnv 定義
├── models/                # 神經網路模型
│   └── qnet.py            # QNet 類別（DQN用）
├── plots/                 # 儲存 reward 曲線圖
├── runs/                  # TensorBoard log
├── scripts/               # 工具腳本
│   ├── list_checkpoints.py
│   └── test_model.py      # 模型測試與可視化
├── config.yaml            # 所有參數設定（env/train/test）
└── requirements.txt       # 安裝所需套件
 說明.txt               # 補充中文說明（另附）
```

---

## 🔧 如何使用

### 1️⃣ 安裝環境
```bash
pip install -r requirements.txt
```

### 2️⃣ 開始訓練模型
```bash
python agents/train_dqn.py
```
你可以在 `config.yaml` 中修改訓練參數與環境設定。

### 3️⃣ 開啟 TensorBoard 觀察訓練過程
```bash
tensorboard --logdir=runs
```
打開瀏覽器前往：http://localhost:6006

### 4️⃣ 測試模型表現
```bash
python scripts/test_model.py
```
- 渲染畫面會顯示 AI 行為與當前 reward 數值。
- 使用的模型、是否開啟動畫、測試次數皆來自 `config.yaml > test:` 區段。

---

## ⚙️ config.yaml 說明
將所有參數集中管理，包含：
- `env`: BabyPong 環境設定
- `train`: DQN 訓練流程設定
- `test`: 測試模型設定

你可以自由建立多份 `config_*.yaml` 切換不同實驗場景。

---

## 🎯 特色亮點
- ✅ 完全模組化訓練流程（train/test 分離）
- ✅ 自訂 Pong 環境支援難度遞增、中心判罰、完美擋球 bonus
- ✅ config.yaml 控制所有參數（零硬編碼）
- ✅ TensorBoard 可視化 reward/loss/epsilon
- ✅ 每次訓練可自動儲存 reward 圖表與模型
- ✅ 可直接測試模型並顯示即時 reward 數值

---

## 🧩 未來擴充建議
- 支援 PPO / A2C 等演算法切換
- 訓練過程自動存成影片
- 加入人機對戰模式
- 模型對比工具（compare_models.py）
- YAML 分組管理多版本訓練實驗（如 easy / hard 難度）

# 🧰 Git 使用說明（給初學者的乒乓 AI 專案指南）

本說明文件是為了協助你學會如何使用 Git 和 GitHub 來管理你在本專案的所有程式碼、模型、訓練記錄和實驗設定。

---

## ✅ 一、為什麼要用 Git？

| 問題 | Git 解法 |
|------|-----------|
| 想要保留歷史版本 | Git 幫你做快照（commit）
| 程式壞掉想回到昨天 | Git 可以切回昨天那一版
| 想把程式傳給別人 | GitHub 幫你雲端備份
| 同時做很多實驗 | Git tag / branch 幫你分開版本

---

## ✅ 二、初始化 Git 專案

1. 打開 Terminal / VSCode 終端機
2. 移動到你的 `pingpong-ai` 專案資料夾中

```bash
cd pingpong-ai
```

3. 初始化 Git：
```bash
git init
```
這樣會產生一個 `.git/` 隱藏資料夾，代表這是個 Git 專案了。

---

## ✅ 三、建立 .gitignore

`.gitignore` 是告訴 Git：「這些東西不要加進版本裡」的清單。

請建立 `.gitignore` 檔案，內容如下：

```gitignore
# Python cache
__pycache__/
*.pyc

# 虛擬環境
venv/

# 模型檔案與中繼檔
*.pth
checkpoints/
runs/
plots/

# Jupyter
.ipynb_checkpoints/

# VS Code 設定檔
.vscode/
```

然後加入這份檔案：
```bash
git add .gitignore
git commit -m "🧹 建立 .gitignore 清單"
```

---

## ✅ 四、版本控制流程（最簡）

每當你改了程式，可以執行這三步：

```bash
git add .                  # 把所有變動加入追蹤
git commit -m "💡 寫清楚你改了什麼"
git push                   # 推上 GitHub（前提是已設定好 remote）
```

---

## ✅ 五、連結 GitHub 倉庫

1. 到 [GitHub.com](https://github.com) 建立一個新 repo
2. 不要勾選 README 或 gitignore（因為你本地已經有了）

3. 在本地 terminal 裡設定 remote：
```bash
git remote add origin https://github.com/你的帳號/pingpong-ai.git
git branch -M main
git push -u origin main
```

---

## ✅ 六、建立里程碑（Milestone）與版本標籤（Tag）

```bash
git tag v1.0 -m "第一版：完成基本訓練 + 可視化"
git push origin v1.0
```

你可以為每一個重要階段做標記，例如：
- `v1.0` 基本 DQN 訓練結構
- `v1.1` 加入可視化與 tensorboard
- `v2.0` 加入對戰或 PPO

---

## ✅ 七、檢查狀態與回溯指令

```bash
git status       # 查看目前有哪些變動

git log          # 查看歷史 commit 記錄

git diff         # 查看尚未 commit 的變動

git checkout .   # 丟掉所有未儲存的修改

git checkout abc1234 train_dqn.py  # 從歷史版本還原特定檔案
```

---

## ✅ 八、常見錯誤解法

| 錯誤訊息 | 解法 |
|-----------|------|
| `fatal: not a git repository` | 沒有先 `git init` |
| `refusing to merge unrelated histories` | `git pull origin main --allow-unrelated-histories` |
| `You must specify a repository to clone` | `git clone` 時忘記貼 GitHub 連結 |

---

## ✅ 九、進階建議

| 功能 | 工具 |
|------|------|
| 多版本訓練記錄 | 多份 config.yaml + commit 分開記錄 |
| 比較 reward 曲線 | 儲存成 plot + commit 對應 tag |
| 發表專案頁面 | 編輯 README.md，推上 GitHub |

---

## 🏁 總結
你現在可以用 Git：
- 儲存每一次訓練進度
- 回顧歷史實驗表現
- 與未來的你或其他人分享完整過程

> Git 就是你訓練 AI 的黑盒紀錄器 ✨
你會用它，就能重現、比較、展示，讓這個專案變成專業級武器。

如果還不懂任何一行，歡迎你對我說：「幫我做」，我可以幫你從 `git init` 一路帶到 `git tag v3.0` 😎