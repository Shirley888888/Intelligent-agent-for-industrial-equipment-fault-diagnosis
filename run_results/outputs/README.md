# Intelligent Agent for Industrial Equipment Fault Diagnosis — Course Assignment

This repository contains code and instructions for the ETTh1 oil temperature prediction assignment from the lecture "从 Agent 原型到真实模型训练".

Contents:
- requirements.txt — Python dependencies
- train.py — data pipeline, baseline, Linear, MLP, CNN, LSTM implementations and a simple training loop
- models.py — PyTorch model classes
- utils.py — data downloading and dataset utilities
- results/ — output folder for saved metrics and figures

Quick start (recommended):
1. Create a Python environment and install dependencies:
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

2. Run a quick demo (short training, CPU-friendly):
   python train.py --model mlp --epochs 2 --device cpu

3. To reproduce full experiments, increase --epochs and tune hyperparameters.

What to submit to GitHub:
- All code (train.py, models.py, utils.py)
- results/metrics.csv and figures (pred_vs_true.png)
- A short report (can add REPORT.md) summarizing three structural changes and the results table described in the course.

Notes:
- This package will download ETTh1 data from the public ETT dataset repository. If your network blocks GitHub raw files, download the CSV manually and place it under data/ETTh1.csv.
- Do NOT commit any secrets (API keys, tokens) to the repository.

"如何推送到 GitHub":
- Initialize repo and push (replace <YOUR_REMOTE_URL> with your repo URL):
  git init
  git add .
  git commit -m "Add ETTh1 training code and demo"
  git remote add origin <YOUR_REMOTE_URL>
  git branch -M main
  git push origin main

If you want, I can also run the demo here and produce results; you asked to prepare a downloadable repository package — I'll finish creating files and then provide a compressed zip and exact commands to push locally.