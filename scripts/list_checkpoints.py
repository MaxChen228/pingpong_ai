# scripts/list_checkpoints.py

import os

def list_checkpoints(path="checkpoints"):
    files = sorted(f for f in os.listdir(path) if f.endswith(".pth"))
    for i, file in enumerate(files):
        print(f"[{i}] {file}")

if __name__ == "__main__":
    list_checkpoints()
