"""统一汇总：从 metrics.csv 生成跨轮次 Test 结果表和对比曲线。

输出：
  outputs/unified_test_table.csv / .md
  outputs/figures/unified_mae_heatmap.png
  outputs/figures/unified_best_per_model.png
  outputs/figures/unified_structure_ablation.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

metrics_path = os.path.join(OUT_DIR, 'metrics.csv')
df = pd.read_csv(metrics_path)

# 1) 统一 Test 结果表：每个模型在全部轮次中的最优一次 + 各轮次明细
df['mae'] = pd.to_numeric(df['mae'], errors='coerce')
model_order = ['baseline', 'linear', 'mlp', 'cnn', 'lstm']

best_rows = []
for m in model_order:
    sub = df[df['model'] == m]
    if sub.empty:
        continue
    idx = sub['mae'].idxmin()
    best = sub.loc[idx]
    best_rows.append({
        'model': m,
        'best_round': best['round'],
        'best_structure': best['structure'],
        'best_mae': round(best['mae'], 4),
        'best_mse': round(float(best['mse']), 4),
        'params': int(best['params']),
        'best_epoch': best['best_epoch'],
        'train_time_s': round(float(best['train_time_s']), 2),
    })
best_df = pd.DataFrame(best_rows)

# 2) 明细表（模型 × 轮次 MAE/MSE）
pivot_mae = df.pivot_table(index='model', columns='round', values='mae', aggfunc='first')
pivot_mae = pivot_mae.reindex(model_order)
pivot_mse = df.pivot_table(index='model', columns='round', values='mse', aggfunc='first')
pivot_mse = pivot_mse.reindex(model_order)

# 保存统一表
table_csv = os.path.join(OUT_DIR, 'unified_test_table.csv')
with open(table_csv, 'w', encoding='utf-8-sig') as f:
    f.write('# 每个模型在 r1-r7 中的最佳表现\n')
    best_df.to_csv(f, index=False)
    f.write('\n# 各模型 × 轮次 MAE 明细\n')
    pivot_mae.to_csv(f)
    f.write('\n# 各模型 × 轮次 MSE 明细\n')
    pivot_mse.to_csv(f)

# Markdown 版本
table_md = os.path.join(OUT_DIR, 'unified_test_table.md')
with open(table_md, 'w', encoding='utf-8') as f:
    f.write('## 统一 Test 结果：各模型最佳表现\n\n')
    f.write(best_df.to_markdown(index=False))
    f.write('\n\n## 各模型 × 轮次 MAE 明细\n\n')
    f.write(pivot_mae.round(4).to_markdown())
    f.write('\n\n## 各模型 × 轮次 MSE 明细\n\n')
    f.write(pivot_mse.round(4).to_markdown())

# 3) 热力图：模型 × 轮次 MAE
plt.figure(figsize=(10, 4.5))
rounds = [c for c in pivot_mae.columns if isinstance(c, str)]
if 'round' in rounds:
    rounds.remove('round')
# 保证 r1..r7 顺序
rounds = sorted(rounds, key=lambda x: int(x.replace('r', '')))
mat = pivot_mae[rounds].values
im = plt.imshow(mat, cmap='RdYlGn_r', aspect='auto')
plt.xticks(np.arange(len(rounds)), rounds)
plt.yticks(np.arange(len(model_order)), model_order)
plt.colorbar(im, label='MAE')
for i in range(len(model_order)):
    for j in range(len(rounds)):
        v = mat[i, j]
        if not np.isnan(v):
            plt.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8)
plt.title('统一 Test 结果：MAE 热图（模型 × 轮次）')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_mae_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4) 每个模型最佳 MAE 柱状图
plt.figure(figsize=(8, 4.5))
colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
plt.bar(best_df['model'], best_df['best_mae'], color=colors)
for i, (m, v) in enumerate(zip(best_df['model'], best_df['best_mae'])):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
plt.axhline(1.4606, color='gray', ls='--', lw=1.0, label='Baseline MAE=1.4606')
plt.ylabel('Best Test MAE')
plt.title('各模型在 r1-r7 中的最优 MAE')
plt.ylim(1.2, 2.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_best_per_model.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5) 结构消融对比图（关键发现）
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# MLP：隐藏层宽度 ablation（r1 hidden=64 vs r2/r3/r4 hidden=128；均 layers=2）
r1_mlp = df[(df['model'] == 'mlp') & (df['round'] == 'r1')]
r2_mlp = df[(df['model'] == 'mlp') & (df['round'] == 'r2')]
xs = ['hidden=64\n(r1)', 'hidden=128\n(r2-r4)']
ys = [r1_mlp['mae'].values[0] if len(r1_mlp) else np.nan,
      r2_mlp['mae'].values[0] if len(r2_mlp) else np.nan]
axes[0].bar(xs, ys, color=['tab:orange', 'tab:orange'])
axes[0].set_title('MLP 宽度消融（layers=2）')
axes[0].set_ylabel('MAE')
axes[0].axhline(1.4606, color='gray', ls='--', label='baseline')
for i, v in enumerate(ys):
    if not np.isnan(v):
        axes[0].text(i, v + 0.03, f'{v:.4f}', ha='center', fontsize=9)

# CNN：kernel 消融（r2 kernel=3 vs r3/r4 kernel=5；均 channels=32）
r2_cnn = df[(df['model'] == 'cnn') & (df['round'] == 'r2')]
r3_cnn = df[(df['model'] == 'cnn') & (df['round'] == 'r3')]
xs2 = ['kernel=3\n(r2)', 'kernel=5\n(r3-r4)']
ys2 = [r2_cnn['mae'].values[0] if len(r2_cnn) else np.nan,
       r3_cnn['mae'].values[0] if len(r3_cnn) else np.nan]
axes[1].bar(xs2, ys2, color=['tab:green', 'tab:green'])
axes[1].set_title('CNN 卷积核消融（channels=32）')
axes[1].axhline(1.4606, color='gray', ls='--', label='baseline')
for i, v in enumerate(ys2):
    if not np.isnan(v):
        axes[1].text(i, v + 0.03, f'{v:.4f}', ha='center', fontsize=9)

# LSTM：层数+隐藏维度消融（r3:1层/32 vs r4:2层/64）
r3_lstm = df[(df['model'] == 'lstm') & (df['round'] == 'r3')]
r4_lstm = df[(df['model'] == 'lstm') & (df['round'] == 'r4')]
xs3 = ['1层/h=32\n(r3)', '2层/h=64\n(r4)']
ys3 = [r3_lstm['mae'].values[0] if len(r3_lstm) else np.nan,
       r4_lstm['mae'].values[0] if len(r4_lstm) else np.nan]
axes[2].bar(xs3, ys3, color=['tab:red', 'tab:red'])
axes[2].set_title('LSTM 结构消融')
axes[2].axhline(1.4606, color='gray', ls='--', label='baseline')
for i, v in enumerate(ys3):
    if not np.isnan(v):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

for ax in axes:
    ax.set_ylim(1.2, 2.5)
    ax.legend(fontsize=8)
plt.suptitle('结构变化消融：MLP width / CNN kernel / LSTM layers×hidden', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_structure_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()

print('统一汇总已生成：')
print(f'  {table_csv}')
print(f'  {table_md}')
print(f'  {os.path.join(FIG_DIR, "unified_mae_heatmap.png")}')
print(f'  {os.path.join(FIG_DIR, "unified_best_per_model.png")}')
print(f'  {os.path.join(FIG_DIR, "unified_structure_ablation.png")}')
