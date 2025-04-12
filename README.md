# 🏓 pingpong_ai

這是一個使用深度強化學習 (DQN) 訓練 AI 控制 2D Pong 遊戲中球拳的專案。它包含了課程式難度調整、獎勵設計以及完整的視覺化功能。

---

## 🎯 專案目標

訓練一個能夠在 Pong 環境中控制球拳的智能代理。專案特色包括：

- 基於 DQN 的強化學習架構
- 可調整的課程式難度設計
- 自定義獎勵機制
- 完整的訓練過程視覺化

---

## 🗂️ 專案結構

```
pingpong_ai/
├── agents/                # 訓練邏輯
│   └── train_dqn.py
├── checkpoints/           # 模型檢查點 (.pth)
├── envs/                  # 自定義環境
│   ├── __init__.py
│   └── baby_pong_env.py   # BabyPongBounceEnv 定義
├── models/                # 神經網絡模型
│   └── qnet.py            # QNet 類別 (DQN)
├── plots/                 # 儲存獎勵曲線
├── scripts/               # 輔助腳本
├── config.yaml            # 訓練配置
├── requirements.txt       # 依賴項
└── README.md              # 本文件
```

---

## 🚀 快速開始

1. 安裝依賴項：

   ```bash
   pip install -r requirements.txt
   ```

2. 開始訓練：

   ```bash
   python agents/train_dqn.py
   ```

3. 閱覽訓練視覺化：

   訓練過程中會自動產生視覺化圖表，儲存於 `plots/` 目錄下。

---

## 🧐 技術細節

- 使用 DQN (Deep Q-Network) 作為強化學習算法
- 自定義 Pong 環境 `BabyPongBounceEnv`
- 可調整的訓練配置，儲存在 `config.yaml`
- 模型檢查點儲存於 `checkpoints/`

---

## 📈 視覺化

訓練過程中會產生獎勵曲線，助於分析模型的學習情況。圖表儲存在 `plots/` 目錄中。

---

## 🤝 貢獻

歡迎提出問題或貢獻代碼。請透過 Issues 或 Pull Requests 聯絡我們。

---

## 📄 授權

本專案採用 MIT 授權條款。