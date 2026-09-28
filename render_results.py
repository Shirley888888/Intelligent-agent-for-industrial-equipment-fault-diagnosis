"""modify:single-source. All active paper numbers are derived from experiment.json."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FINAL=ROOT/'results/final'

def save_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def table(rows,cols):
    def val(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(val(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n'
def assemble():
    # modify:audit-freeze-integrity - this private helper must not rewrite an existing formal run.
    if (FINAL/'experiment.json').exists():
        raise FileExistsError('Formal experiment already frozen; use reproduction outputs for comparisons.')
    f=json.loads((FINAL/'forecast_evaluation.json').read_text());r=json.loads((FINAL/'rag_evaluation.json').read_text());c=json.loads((ROOT/'output/all_case_results.json').read_text())
    frozen=json.loads((ROOT/'configs/risk_thresholds.json').read_text())
    hashes={}
    for rel in ['configs/experiment.yaml','agent/models/config.yaml','configs/risk_thresholds.json','tests/rag_queries.json','agent/rag_tool.py','agent/models/best_model.pt','agent/models/scaler.pkl','data/ETTh1.csv']:
        hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    for p in (ROOT/'agent/models/embedding').rglob('*'):
        if p.is_file():hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for n in c:hashes[f'tests/cases/{n}/input.csv']=hashlib.sha256((ROOT/f'tests/cases/{n}/input.csv').read_bytes()).hexdigest()
    f.update({'rag':r,'cases':c,'risk_calibration':frozen,'input_sha256':hashes,
      'knowledge_source':'25 manually authored English project demonstration records; not certified SOPs or a curated industry benchmark.',
      'embedding_model':{'id':'sentence-transformers/all-MiniLM-L6-v2','revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','language_scope':'English queries and knowledge'},
      'metric_definitions':{'retrieval_accuracy':'accepted rank-1 relevant / 9 answerable queries',
        'Hit@3':'any relevant document in raw top3 / 9 answerable queries',
        'MRR@3':'mean reciprocal first relevant rank, 0 if absent in top3, denominator 9',
        'multi_all_evidence':'all annotated relevant documents accepted in top3 / 3 multi queries',
        'no_answer_rejection':'no top3 document above threshold / 3 unanswerable queries'}})
    save_json(FINAL/'experiment.json',f);return f

def main(assemble_inputs=False):
    d=assemble() if assemble_inputs else json.loads((FINAL/'experiment.json').read_text())
    fig=ROOT/'paper/figures';tab=ROOT/'paper/tables';fig.mkdir(parents=True,exist_ok=True);tab.mkdir(parents=True,exist_ok=True)
    modelrows=d['metrics'];cases=d['cases'];rag=d['rag']
    caserows=[{'Case':n,'Model':v['model'],'Peak_C':v['temperature_metric'],'Rise_C':v['temperature_rise_metric'],'Slope_C_per_h':v['slope_metric'],'Risk':v['risk_level'],'Diagnosis':v['diagnosis']['level']} for n,v in cases.items()]
    ragrows=[{'Backend':k,**v['metrics']} for k,v in rag.items()]
    for name,rows in [('Table_Model_Comparison',modelrows),('Table_Case_Studies',caserows),('Table_RAG',ragrows)]:
        pd.DataFrame(rows).to_csv(tab/(name+'.csv'),index=False)
        (tab/(name+'.md')).write_text(table(rows,list(rows[0])),encoding='utf-8')
    records=[{'backend':b,'query_id':q['id'],'category':q['category'],'query':q['query'],'gold_ids':';'.join(q['relevant_ids']),'rank':i+1,**h,'refused':q['refused']} for b,v in rag.items() for q in v['records'] for i,h in enumerate(q['top3'])]
    pd.DataFrame(records).to_csv(FINAL/'rag_top3_records.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for metric,unit in [('MAE','C'),('MSE','C squared')]:
        f,ax=plt.subplots(figsize=(7,4),layout='constrained');vals=[r[metric] for r in modelrows]
        bars=ax.bar([r['Model'] for r in modelrows],vals,color=['#8796A5','#53799C','#53799C','#247C76','#53799C'])
        ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylim(0,max(vals)*1.18);ax.set_ylabel(f'{metric} ({unit})');ax.set_title(d['experiment_id'])
        f.savefig(fig/f'Figure_Model_Comparison_{metric}.png',dpi=180);plt.close(f)
    for name,v in cases.items():
        f,ax=plt.subplots(figsize=(7,4),layout='constrained');h=np.asarray(v['raw_input_signal'])[:,-1]
        ax.plot(range(-95,1),h,label='History OT');ax.plot(range(1,25),v['forecast_output'],label='LSTM forecast')
        for level,color in [('medium','#AA7722'),('high','#AB3333')]:ax.axhline(d['risk_calibration']['thresholds']['temperature'][level],color=color,linestyle='--',label=f'Temperature {level}')
        ax.set(xlabel='Hour relative to forecast origin',ylabel='Oil temperature (C)',title=f"{name}: forecast risk {v['risk_level']} (training-period demo)");ax.legend(fontsize=8)
        f.savefig(fig/f'Figure_Case_{name}.png',dpi=180);plt.close(f)
    f,ax=plt.subplots(figsize=(7,4),layout='constrained');ks=['retrieval_accuracy_top1_thresholded','Hit@3_raw','MRR@3_raw','no_answer_rejection_rate'];x=np.arange(4)
    for i,(b,v) in enumerate(rag.items()):ax.bar(x+(i-.5)*.36,[v['metrics'][k] for k in ks],.36,label=b)
    ax.set_xticks(x,['Top1 accepted','Hit@3 raw','MRR@3 raw','No-answer refusal']);ax.set_ylim(0,1.14);ax.set_ylabel('Score');ax.legend();f.savefig(fig/'Figure_RAG.png',dpi=180);plt.close(f)
    # Active compatibility exports are generated from the same source, never hand edited.
    for dst,rows in [('outputs/metrics.csv',modelrows),('results/model_comparison.csv',modelrows),('results/case_study_results.csv',caserows)]:
        p=ROOT/dst;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False)
    save_json(ROOT/'outputs/metrics.json',{r['Model']:r for r in modelrows})
    # modify:audit-output-isolation - rendering must not overwrite live case outputs.
    summary='\n'.join([table(modelrows,['Model','MAE','MSE','PeakError']),table(caserows,list(caserows[0])),table(ragrows,['Backend','retrieval_accuracy_top1_thresholded','Hit@3_raw','MRR@3_raw','no_answer_rejection_rate'])])
    (ROOT/'paper/V8_PAPER_RESULTS.md').write_text('# 正式实验结果\n\n数据源：`results/final/experiment.json`，实验编号 '+d['experiment_id']+'。\n\n'+summary,encoding='utf-8')
    lstm=next(r for r in modelrows if r['Model']=='LSTM');base=modelrows[0]
    draft=f'''# 题目
基于油温预测与语义检索的变压器热风险多步骤智能体研究（初稿）

# 摘要
本文在已有 ETTh1 项目上集成时序预测、训练分布分位数风险评价与检索增强证据输出。使用七变量96小时历史预测未来24小时油温，比较 Last Value、Linear、CNN1D、LSTM 和 TCN。知识库包含25条人工编写的英文演示知识，比较TF-IDF与句向量检索。所有结果由同一正式实验文件生成。当前工作定位为热风险辅助分析，不验证具体故障分类。

# 引言
工业设备油温预测可以提供趋势信息，但预测数值需要结合风险规则与知识证据解释。本项目将预测、风险计算、知识检索和规则诊断串联为可追溯工作流。本文拟讨论预测误差、检索覆盖率和拒答行为。相关工作与真实工业文献综述待后续补充，本稿不编造引用。

# 方法
## 数据与预测
ETTh1按时间顺序以70%/15%/15%划分，行数分别为{d['split_rows']}，滑窗数分别为{d['split_windows']}。每个划分内部独立构造96输入、24输出、步长1的窗口，不跨边界。StandardScaler只在训练集拟合。输入变量是HUFL、HULL、MUFL、MULL、LUFL、LULL、OT，输出为OT。
保留现有Linear、CNN1D、LSTM、TCN类与训练流程，Last Value重复最后一个OT值24步。正式实验直接评价已打包的四个checkpoint，本次没有重新训练，也没有通过测试集选择权重。Agent按既有设计使用LSTM，部署权重与对比评价权重完全一致。各checkpoint的训练配置、best epoch及校验值随包提供。

## 风险分位数
预先固定所有指标的中/高分位数为0.90/0.95，不使用stable、slow_rise、fast_rise的结果调整。先遍历训练集连续24小时窗口，计算最大温度、最大值减首值的温升、最大正向逐小时斜率，再计算分位数。验证集只统计固定规则触发频次，不搜索阈值。任一指标达到高阈值为HIGH，否则任一达到中阈值为MEDIUM，其余为LOW。常数分布使用严格大于。配置、数据哈希和校准结果存入冻结文件，Agent运行前校验一致性。
注意预测曲线通常比真实序列平滑，因此训练原始序列斜率分布与预测斜率分布可能不同，本稿不把分位数等级当作经验证的故障概率。历史anomaly模块仍使用原项目启发式规则，与预测risk等级分开报告。

## 知识库与检索
在原5条知识基础上扩充到25条，涵盖热趋势、风机/油泵/散热器、负载、环境、传感器、油质、维护和模型使用边界。知识为项目演示条目，不冒充厂商SOP或标准。保留原分块函数：40词、重叠8词；当前短条目各形成一个块，代码仍支持长文分块。
TF-IDF采用词1至2-gram，余弦相似度阈值0.08。Embedding采用预训练all-MiniLM-L6-v2，归一化句向量点积即余弦相似度，阈值预设0.45。模型权重随包提供，模型标识及revision写入正式结果。两种方法均返回Top-3及逐条accepted标记，不做隐式后端降级，不用这12条测试题优化阈值。当前知识和Query均为英文，不声明中文检索效果。
模型资料：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Agent工作流
保留Forecast、Risk、RAG、Diagnosis及Observation日志。原有检索失败后的条件补检保持原样。本稿仅称为固定多步骤Agent工作流，不声称实现开放式LLM ReAct推理或验证了多轮检索收益。

# 实验
## 正式实验协议
实验编号为{d['experiment_id']}。先冻结风险阈值，再评价固定checkpoint、两个检索后端与三个展示案例。采用CPU float32推理及float64反标准化/指标计算，因此与历史GPU导出值可能有微小差异；历史数值全部退出正式表图。数据、配置、知识库代码、测试Query、模型文件均保存SHA-256用于复核。
未来训练输出写入独立的outputs/training_runs目录，不自动覆盖正式结果。正式源文件为results/final/experiment.json，README、论文表格和图由render_results.py生成。不得将archive中的任何旧表混入本文。

## RAG测试设计与指标
人工标注12条Query，直接问法、同义改写、多知识点、无答案各3条。先固定标注再运行测试。每条保存两个后端各自的Top-3、分数、阈值通过标记和相关性标记。多知识点Query标注全部必需知识ID。
检索正确率定义为9条有答案问题中“Top-1相关且过阈值”的比例。Hit@3为原始前三名至少命中一条标注知识的比例；MRR@3为前三名首次相关结果名次倒数的平均，未命中计0，分母都是9。额外报告过阈值Hit@3、多知识点全证据覆盖率、无答案拒答率以及有答案误拒数。MRR只检查Top-3，不能表述为全排名MRR。无答案拒答指无任何候选通过阈值，不等同于生成答案事实核验。
这12条属于小规模人工测试，不足以证明跨领域泛化；两个后端阈值不同，拒答率不是等误报率条件下的对比。无答案题包含主题相关但缺少具体数值的难负例，如特定设备保护动作值。

## 本轮实现审计
本轮保持冻结实验数值和阈值不变，新增复跑输出隔离、严格时间戳检查、后端阈值选择修复，以及语义检索批处理和分位数计算向量化。实现修补前源码保留用于原结果追溯；等价性测试采用独立复跑记录，不改写原始实验数值。详细分析见reports/AUDIT_AND_ANALYSIS.md。性能优化不等于检索正确率或预测精度提高。

## 三案例与有效性边界
保留原包stable、slow_rise、fast_rise输入文件，名称描述历史趋势，不预设预测risk应为LOW/MEDIUM/HIGH。它们位于训练时段，只是阈值冻结后独立执行的固定展示，并非未见样本测试集。模型评价只使用时间末段测试集。ETTh1没有本项目所需的故障类型标签，因此不能报告真实故障诊断准确率。规则单元测试仅验证代码逻辑。

# 结果
{summary}
LSTM相对Last Value的MAE变化为{(lstm['MAE']/base['MAE']-1)*100:.2f}%，MSE变化为{(lstm['MSE']/base['MSE']-1)*100:.2f}%。Embedding在本测试中的无答案误接收数为{rag['embedding']['metrics']['no_answer_false_accepts']}，该失败保留原始记录，不修改Query或阈值掩盖。三个案例risk分别为{', '.join(n+'='+v['risk_level'] for n,v in cases.items())}，不强求三个不同等级。
图见figures目录。所有数字只来自同一experiment.json。

# 结论
现有项目已完成统一的预测评价、冻结风险规则和双后端RAG检索记录。初步结果说明该固定流程可以提供可追溯的预测与知识证据，但仍存在拒答失败、测试样本少、无真实故障标签、展示案例来自训练时段等限制。后续工作优先补充独立RAG测试集和有标签现场验证，多轮检索与ReAct扩展另行讨论。
'''
    (ROOT/'paper/PAPER_DRAFT.md').write_text(draft,encoding='utf-8')
    readme=f'''# ETTh1 工业油温 Agent：实验收口版

本包在原项目基础上迭代，保留原模型、工具类和多步骤工作流。
唯一正式实验：`{d['experiment_id']}`。来源：`results/final/experiment.json`。旧输出仅在`archive/pre_closeout/`追溯，禁止混入论文。

## 运行
在项目根目录打开终端。已有GPU环境先保留原CUDA版PyTorch，再安装其余依赖；`requirements-reproduced.txt`记录本次CPU评价环境，并非要求替换你现有的GPU环境。

```bash
python -m pip install -r requirements.txt
python run_agent.py --device cuda:0
python tests/evaluate_rag.py
python tests/test_risk_boundaries.py
python tests/test_diagnosis.py
python tests/test_closeout.py
```

数据、四个预测模型、Agent的LSTM权重、句向量模型以及冻结风险文件均已打包。无须重新训练或下载Embedding模型。`prepare_embedding.py`仅在模型文件缺失时用于联网恢复。
`run_agent.py`默认按stable、slow_rise、fast_rise顺序运行并保存到output/。这些文件属于复运行输出；论文仍以冻结的experiment.json为准。

## 完整复核
```bash
python calibrate_risk.py
python evaluate_final.py
python tests/evaluate_rag.py
python run_agent.py --device cpu
python tests/test_closeout.py
python render_results.py
```
`calibrate_risk.py`已有冻结结果时校验后复用，不根据案例调参。`render_results.py`只从正式源重新生成Markdown、CSV和PNG，不把临时复跑结果自动提升为新正式实验。Excel是同一源的冻结导出副本，非另一次实验。原GPU训练入口保留，新训练输出到独立training_runs目录。

## 本轮审计与优化

详见 `reports/AUDIT_AND_ANALYSIS.md` 和 `reports/REVISION_CHANGELOG.md`。
普通复跑的预测评价、RAG记录写入 `results/reruns/` 独立目录，不覆盖正式结果。
原正式实验对应的旧源码存于 `archive/closeout_v1_source/`，映射见 `results/final/source_snapshot.json`；当前修补源码及实测记录见 `results/audit/`。
运行 `python tests/test_revision.py` 检查修补行为，运行 `python analyze_results.py` 重建详细分析。
新上传包中的 `outputs/training_runs/20260929_035240/` 仅为训练运行记录，不作为第二套正式实验。

## 当前正式结果
{summary}
## 阅读入口
- `MODIFICATION_SUMMARY.md`：分轮修改清单、文件位置和验证结果。
- `paper/PAPER_DRAFT.md`：中文论文框架及方法、实验粗稿。
- `paper/V8_PAPER_RESULTS.md`：正式数字。
- `paper/tables/Final_Experiment.xlsx`：统一Excel表。
- `results/final/rag_top3_records.csv`：两后端共72条候选记录；JSON保留完整文本。
- `configs/risk_thresholds.json`：训练校准、验证统计、数据校验值。

## 解释边界
知识库为25条英文人工演示知识，检索模型为预训练[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。阈值预设，未用测试题调参。Embedding仍有无答案误接收，不能把相似候选当作有效答案保证。案例来自训练时段，不作为独立泛化证据。risk与历史anomaly/diagnosis分别输出。ReAct仅按多步骤工作流描述。当前不声称验证真实故障分类准确率。
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    # Hash active generated artifacts after all numeric text/plots are written.
    artifacts=[ROOT/'README.md',ROOT/'paper/PAPER_DRAFT.md',ROOT/'paper/V8_PAPER_RESULTS.md',*tab.glob('*.csv'),*tab.glob('*.md'),*tab.glob('*.xlsx'),*fig.glob('*.png')]
    save_json(FINAL/'generated_manifest.json',{'experiment_sha256':hashlib.sha256((FINAL/'experiment.json').read_bytes()).hexdigest(),'artifacts':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}})
if __name__=='__main__':main()
