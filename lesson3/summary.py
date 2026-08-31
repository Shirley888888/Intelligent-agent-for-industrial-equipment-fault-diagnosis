"""统一汇总：从 metrics.csv 生成跨轮次 Validation 结果表和对比曲线。

流程说明（修正版）：
  - 模型结构/超参的选择依据是 **Validation** 指标（val_mae / val_mse）；
  - Test 只在最终方案确定后由 evaluate.py 做一次正式评价，
    summary.py 会把 outputs/final_test_results.csv（若存在）追加为
    “最终 Test 正式评价”小节，该结果不再用于调参。
  - 旧切分方式（先滑窗后切分，边界样本重叠）产生的 r1-r7 结果
    已归档为 outputs/metrics_legacy_oldsplit.csv，本脚本不再读取。

输出：
  - outputs/unified_test_table.csv / .md        （Validation 模型选择表）
  - outputs/figures/unified_mae_heatmap.png
  - outputs/figures/unified_best_per_model.png
  - outputs/figures/unified_structure_ablation.png
  - outputs/final_test_table.md                 （最终 Test 正式评价，若存在）
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

# ---- 指标列兼容：新流程用 val_mae / val_mse；若读取到旧格式文件则用 mae / mse ----
mae_col = 'val_mae' if 'val_mae' in df.columns else 'mae'
mse_col = 'val_mse' if 'val_mse' in df.columns else 'mse'
metric_label = 'Validation MAE' if mae_col == 'val_mae' else 'Test MAE'
split_note = ('（新切分方式：先按时间切分原始序列，再各段内滑窗）'
              if mae_col == 'val_mae' else '（旧切分方式：先滑窗后切分，边界样本重叠）')
df[mae_col] = pd.to_numeric(df[mae_col], errors='coerce')
df[mse_col] = pd.to_numeric(df[mse_col], errors='coerce')

model_order = ['baseline', 'linear', 'mlp', 'cnn', 'lstm']

# 1) 统一结果表：每个模型在全部轮次中的最优一次 + 各轮次明细
best_rows = []
for m in model_order:
    sub = df[df['model'] == m]
    if sub.empty:
        continue
    idx = sub[mae_col].idxmin()
    best = sub.loc[idx]
    best_rows.append({
        'model': m,
        'best_round': best['round'],
        'best_structure': best['structure'],
        'best_val_mae': round(best[mae_col], 4),
        'best_val_mse': round(float(best[mse_col]), 4),
        'params': int(best['params']),
        'best_epoch': best['best_epoch'],
        'train_time_s': round(float(best['train_time_s']), 2),
    })
best_df = pd.DataFrame(best_rows)

# 2) 明细表（模型 × 轮次 MAE/MSE）
pivot_mae = df.pivot_table(index='model', columns='round', values=mae_col, aggfunc='first')
pivot_mae = pivot_mae.reindex(model_order)
pivot_mse = df.pivot_table(index='model', columns='round', values=mse_col, aggfunc='first')
pivot_mse = pivot_mse.reindex(model_order)

# 保存统一表（CSV）
table_csv = os.path.join(OUT_DIR, 'unified_test_table.csv')
with open(table_csv, 'w', encoding='utf-8-sig') as f:
    f.write(f'# {metric_label}（模型选择依据）{split_note}\n')
    f.write('# 每个模型在所有轮次中的最佳表现\n')
    best_df.to_csv(f, index=False)
    f.write(f'\n# 各模型 × 轮次 {metric_label} 明细\n')
    pivot_mae.to_csv(f)
    f.write(f'\n# 各模型 × 轮次 {metric_label.split()[0]} MSE 明细\n')
    pivot_mse.to_csv(f)

# Markdown 版本
table_md = os.path.join(OUT_DIR, 'unified_test_table.md')
with open(table_md, 'w', encoding='utf-8') as f:
    f.write(f'## 统一 {metric_label} 结果：各模型最佳表现\n\n')
    f.write(f'> 数据来源：{metric_label}{split_note}\n\n')
    f.write(best_df.to_markdown(index=False))
    f.write(f'\n\n## 各模型 × 轮次 {metric_label} 明细\n\n')
    f.write(pivot_mae.round(4).to_markdown())
    f.write(f'\n\n## 各模型 × 轮次 MSE 明细\n\n')
    f.write(pivot_mse.round(4).to_markdown())

# 3) 热力图：模型 × 轮次 MAE
plt.figure(figsize=(10, 4.5))
rounds = [c for c in pivot_mae.columns if isinstance(c, str)]
if 'round' in rounds:
    rounds.remove('round')
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
plt.title(f'统一 {metric_label} 热图（模型 × 轮次）')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_mae_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4) 每个模型最佳 MAE 柱状图（含 baseline 虚线）
plt.figure(figsize=(8, 4.5))
colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
plt.bar(best_df['model'], best_df['best_val_mae'], color=colors)
for i, (m, v) in enumerate(zip(best_df['model'], best_df['best_val_mae'])):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
bl = best_df[best_df['model'] == 'baseline']
if len(bl):
    base_mae = bl['best_val_mae'].iloc[0]
    plt.axhline(base_mae, color='gray', ls='--', lw=1.0, label=f'Baseline MAE={base_mae:.4f}')
plt.ylabel(f'Best {metric_label}')
plt.title(f'各模型在所有轮次中的最优 {metric_label}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_best_per_model.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5) 结构消融对比图：动态取每个模型最早出现的两种不同结构做对比
def _structure_pairs(df, model):
    """返回该模型按轮次最早出现的两种不同结构及其指标。"""
    sub = df[df['model'] == model].sort_values('round')
    seen, pairs = {}, []
    for _, r in sub.iterrows():
        s = r['structure']
        if s not in seen:
            seen[s] = (r['round'], float(r[mae_col]))
    items = list(seen.items())[:2]
    return [(label, meta) for label, meta in items]

abl_configs = [
    ('mlp', 'MLP 结构对比', ['tab:orange', 'tab:orange']),
    ('cnn', 'CNN 结构对比', ['tab:green', 'tab:green']),
    ('lstm', 'LSTM 结构对比', ['tab:red', 'tab:red']),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (mdl, title, cs) in zip(axes, abl_configs):
    pairs = _structure_pairs(df, mdl)
    if len(pairs) >= 1:
        labels = [f'{s[:18]}\n({meta[0]})' for s, meta in pairs]
        vals = [meta[1] for _, meta in pairs]
        ax.bar(labels, vals, color=cs[:len(labels)])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
    if len(bl):
        ax.axhline(base_mae, color='gray', ls='--', label='baseline')
    ax.set_title(f'{title}（{metric_label}）')
    ax.set_ylabel('MAE')
    ax.legend(fontsize=8)
plt.suptitle('结构变化消融（动态：每种结构取最早出现的轮次）', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'unified_structure_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()

print('统一汇总已生成：')
print(f'  {table_csv}')
print(f'  {table_md}')
print(f'  {os.path.join(FIG_DIR, "unified_mae_heatmap.png")}')
print(f'  {os.path.join(FIG_DIR, "unified_best_per_model.png")}')
print(f'  {os.path.join(FIG_DIR, "unified_structure_ablation.png")}')

# 6) 最终 Test 正式评价表（若 evaluate.py 已运行）
final_csv = os.path.join(OUT_DIR, 'final_test_results.csv')
final_md = os.path.join(OUT_DIR, 'final_test_table.md')
if os.path.exists(final_csv):
    fd = pd.read_csv(final_csv)
    with open(final_md, 'w', encoding='utf-8') as f:
        f.write('## 最终方案 Test 正式评价（每次最终方案确定后仅评价一次）\n\n')
        f.write('> 以下 Test MAE/MSE 由 evaluate.py 产生，不用于任何调参，'
                '作为最终方案的正式证据。\n\n')
        f.write(fd.round({'test_mae': 4, 'test_mse': 4}).to_markdown(index=False))
    # 同时把正式评价追加到统一表 Markdown 末尾
    with open(table_md, 'a', encoding='utf-8') as f:
        f.write('\n\n---\n\n')
        f.write(open(final_md, encoding='utf-8').read())
    print(f'  {final_md}（最终 Test 正式评价）')
else:
    print('\n[提示] 尚未生成 outputs/final_test_results.csv。'
          '确定最终方案后运行 evaluate.py --model <模型> 生成 Test 正式评价。')
