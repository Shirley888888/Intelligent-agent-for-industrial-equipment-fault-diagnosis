"""Detailed descriptive analysis of the frozen experiment; never trains or tunes."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def md(rows,cols):
    def v(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(v(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n'

def indicators(x):
    return {'temperature':x.max(1),'rise':x.max(1)-x[:,0],'slope':np.maximum(np.diff(x,axis=1).max(1),0)}
def levels(x,t):
    vs=indicators(x);r=np.zeros(len(x),dtype=int)
    for k,v in vs.items():
        med=v>t[k]['medium'] if t[k].get('constant') else v>=t[k]['medium']
        hi=v>t[k]['high'] if t[k].get('constant') else v>=t[k]['high']
        r=np.maximum(r,med.astype(int));r=np.maximum(r,hi.astype(int)*2)
    return r

def main():
    d=json.loads((ROOT/'results/final/experiment.json').read_text());out=ROOT/'results/audit';out.mkdir(parents=True,exist_ok=True)
    fig=ROOT/'reports/figures';fig.mkdir(parents=True,exist_ok=True)
    z=np.load(ROOT/'results/final/predictions.npz');y=z['truth'];base=z['Persistence'];names=[r['Model'] for r in d['metrics']]
    horizons=[];summary=[];segments=[]
    for name in names:
        e=z[name]-y;b=base-y
        for h in range(24):horizons.append({'Model':name,'Hour':h+1,'MAE':float(abs(e[:,h]).mean()),'MSE':float((e[:,h]**2).mean()),'Bias':float(e[:,h].mean())})
        summary.append({'Model':name,'MAE':float(abs(e).mean()),'MAE_improvement_pct':float(100*(1-abs(e).mean()/abs(b).mean())),
          'MSE_improvement_pct':float(100*(1-(e**2).mean()/(b**2).mean())),
          'window_MAE_win_rate':float((abs(e).mean(1)<abs(b).mean(1)).mean()),
          'signed_peak_bias':float((z[name].max(1)-y.max(1)).mean()),
          'slope_MAE':float(abs(indicators(z[name])['slope']-indicators(y)['slope']).mean())})
        for label,sl in [('h01_06',slice(0,6)),('h07_12',slice(6,12)),('h13_18',slice(12,18)),('h19_24',slice(18,24))]:
            segments.append({'Model':name,'Horizon':label,'MAE':float(abs(e[:,sl]).mean()),'MSE':float((e[:,sl]**2).mean())})
    pd.DataFrame(horizons).to_csv(out/'horizon_metrics.csv',index=False);pd.DataFrame(summary).to_csv(out/'model_analysis.csv',index=False);pd.DataFrame(segments).to_csv(out/'horizon_groups.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    f,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for n in names:ax.plot(range(1,25),[r['MAE'] for r in horizons if r['Model']==n],label=n)
    ax.set(xlabel='Forecast hour',ylabel='MAE (C)',title='Frozen experiment: error by forecast horizon');ax.legend(ncol=3);f.savefig(fig/'horizon_mae.png',dpi=180);plt.close(f)
    categories=[];failures=[]
    for backend,ev in d['rag'].items():
        for cat in ['direct','paraphrase','multi','unanswerable']:
            qs=[q for q in ev['records'] if q['category']==cat]
            categories.append({'Backend':backend,'Category':cat,'N':len(qs),
              'Top1_correct_n':sum(q['top1_correct'] for q in qs) if cat!='unanswerable' else 'N/A',
              'Hit3_n':sum(q['hit_at_3'] for q in qs) if cat!='unanswerable' else 'N/A',
              'All_evidence_n':sum(q['all_relevant_accepted'] for q in qs) if cat=='multi' else 'N/A',
              'Refused_n':sum(q['refused'] for q in qs)})
        for q in ev['records']:
            if (q['relevant_ids'] and not q['top1_correct']) or (not q['relevant_ids'] and not q['refused']):
                failures.append({'Backend':backend,'Query':q['id'],'Category':q['category'],
                  'Gold':','.join(q['relevant_ids']) or '(no answer)','Top1':q['top3'][0]['document_id'],
                  'Score':q['top3'][0]['score'],'Accepted':q['top3'][0]['accepted']})
    pd.DataFrame(categories).to_csv(out/'rag_category_metrics.csv',index=False);pd.DataFrame(failures).to_csv(out/'rag_failure_analysis.csv',index=False)
    t=d['risk_calibration']['thresholds'];raw=pd.read_csv(ROOT/'data/ETTh1.csv')['OT'].to_numpy();a,b,c=d['split_rows'];riskdist=[]
    for name,part in [('train',raw[:a]),('validation',raw[a:a+b]),('test',raw[a+b:])]:
        x=np.lib.stride_tricks.sliding_window_view(part,24);lv=levels(x,t)
        riskdist.append({'Split':name,'Windows':len(x),**{k:int((lv==i).sum()) for i,k in enumerate(['LOW','MEDIUM','HIGH'])}})
    # modify:audit-boundaries - risk labels use original CSV, not float32-normalized reconstruction.
    # Tiny inverse-scaling roundoff can flip >= comparisons exactly at quantile boundaries.
    raw_truth=np.lib.stride_tricks.sliding_window_view(raw[a+b:],24)[96:]
    assert raw_truth.shape==y.shape
    truthlev=levels(raw_truth,t);predlev=levels(z['LSTM'],t);conf=np.zeros((3,3),dtype=int)
    for i,j in zip(truthlev,predlev):conf[i,j]+=1
    pd.DataFrame(conf,index=['true_LOW','true_MEDIUM','true_HIGH'],columns=['pred_LOW','pred_MEDIUM','pred_HIGH']).to_csv(out/'forecast_risk_agreement.csv')
    smooth=[]
    for name,x in [('test_truth_raw_csv',raw_truth),('LSTM_forecast',z['LSTM'])]:
        vs=indicators(x)
        for k,v in vs.items():smooth.append({'Source':name,'Indicator':k,'q50':float(np.quantile(v,.5)),'q90':float(np.quantile(v,.9)),'q95':float(np.quantile(v,.95)),'medium_trigger_count':int((v>=t[k]['medium']).sum())})
    pd.DataFrame(riskdist).to_csv(out/'risk_distribution.csv',index=False);pd.DataFrame(smooth).to_csv(out/'forecast_indicator_distribution.csv',index=False)
    # Examine uploaded new run without selecting/promoting it based on test scores.
    import torch
    training=[];newrun=ROOT/'outputs/training_runs/20260929_035240'
    for name in names[1:]:
        old=torch.load(ROOT/f'outputs/checkpoints/{name}_best.pt',map_location='cpu',weights_only=False)
        new=torch.load(newrun/f'checkpoints/{name}_best.pt',map_location='cpu',weights_only=False)
        equal=old['model_state'].keys()==new['model_state'].keys() and all(torch.equal(old['model_state'][k],new['model_state'][k]) for k in old['model_state'])
        training.append({'Model':name,'weights_exactly_equal':equal,'formal_best_epoch':old['best_epoch'],'new_best_epoch':new['best_epoch'],'formal_val_loss':old['best_val_loss'],'new_val_loss':new['best_val_loss']})
    perf=json.loads((out/'revision_checks.json').read_text()) if (out/'revision_checks.json').exists() else {}
    audit={'experiment_id':d['experiment_id'],'model_analysis':summary,'rag_categories':categories,'rag_failures':failures,'risk_distribution':riskdist,'risk_agreement_confusion_matrix':conf.tolist(),'indicator_distribution':smooth,'uploaded_training_comparison':training,'performance':perf}
    (out/'analysis.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    l=next(r for r in summary if r['Model']=='LSTM');lt=[r for r in horizons if r['Model']=='LSTM'];bt=[r for r in horizons if r['Model']=='Persistence']
    better=[r['Hour'] for r,bv in zip(lt,bt) if r['MAE']<bv['MAE']]
    timing=perf.get('timing',{});timingtext='见revision_checks.json。'
    if timing:timingtext=f"训练分位数计算：原循环中位耗时{timing['loop_median_s']:.6f}s，向量化{timing['vectorized_median_s']:.6f}s，约{timing['loop_median_s']/timing['vectorized_median_s']:.2f}倍。12条句向量查询：逐条{timing['sequential_median_s']:.6f}s，批量{timing['batch_median_s']:.6f}s，约{timing['sequential_median_s']/timing['batch_median_s']:.2f}倍。"
    report=f'''# 项目逐项审计、结果分析与优化报告

分析对象：本次上传的ETTh1_Industrial_Agent_V8_Experiment_Closeout(1).zip。正式实验保持为 `{d['experiment_id']}`。本轮实现修订标识为 `AUDIT_R2_20260929`，不是第二套正式实验。

## 一、结论
六项主体要求已有对应交付，但原版本仍有输出覆盖、阈值切换、时间轴校验和复现链路缺口。本轮已针对现有文件修补。预测与检索准确率并未因本次工程优化宣称提高；既有误检、误接收、同级风险结果均保留。LSTM的整体MAE优于Last Value，但该优势不能直接解释为故障诊断能力。

## 二、原六项要求逐项检查

|要求|原包状态|核查依据与本轮补充|
|---|---|---|
|知识库20–30条、Embedding及TF-IDF|主体完成|25条短知识，真实预训练句向量，原基线仍在；修复切换后端时误用阈值，新增批量接口，未修改知识以迎合测试题。|
|12条Query、四类、Top-3和指标|主体完成|四类各3条，两后端共72条候选；MRR明确为MRR@3；补充逐类别及失败分析，隔离复跑文件。|
|训练/验证固定阈值，不用案例调参|完成但完整性检查可加强|仍使用固定q90/q95；验证只观察；新增冻结数值重算核验，向量化与原循环逐值一致；无案例搜索。|
|唯一正式实验并统一表图|文件数值一致，流程有缺口|原评价脚本可能覆盖results/final；现改为只写results/reruns，渲染不再覆盖实时案例输出；新GPU运行仅作记录，未自动替换正式数据。|
|ReAct不大改|完成|原工具顺序与条件補检保留；本轮未开展多轮检索或LLM规划扩展。|
|论文初稿框架与方法/实验粗稿|完成|paper/PAPER_DRAFT.md含七个一级标题，补充本轮审计范围及来源追溯说明。|

三条硬性规则：原项目迭代而非重建；保留原模型类、核心函数与接口；新增脚本仅用于要求内的复核与分析。逐轮清单见reports/REVISION_CHANGELOG.md，完整项目统一打包。

## 三、预测结果详细分析

{md(summary,['Model','MAE','MAE_improvement_pct','MSE_improvement_pct','window_MAE_win_rate','signed_peak_bias','slope_MAE'])}

改进百分比以Last Value为参照，正数表示误差降低，负数表示更差。window_MAE_win_rate是每个滑窗MAE严格小于Last Value的比例，不能解释为统计显著性。signed_peak_bias是预测峰值减真实峰值的均值；slope_MAE为每个窗口最大正斜率误差绝对值的均值，是补充分析量，不替换原MaxRiseRateError的定义。

LSTM的MAE相对Last Value降低{l['MAE_improvement_pct']:.2f}%，MSE降低{l['MSE_improvement_pct']:.2f}%。其窗口MAE胜率为{l['window_MAE_win_rate']*100:.2f}%，说明整体平均占优不代表每个窗口占优。原始模型、输入和训练产物均未改动。

### 预测步长

{md([r for r in segments if r['Model'] in ('Persistence','LSTM')],['Model','Horizon','MAE','MSE'])}

LSTM在24个单独预测步长中，有{len(better)}个步长的MAE低于Last Value；对应步长为{better}。完整结果见results/audit/horizon_metrics.csv及reports/figures/horizon_mae.png。不能仅凭整体24步均值判断短期和长期表现一致。

时间滑窗重叠：2494个窗口并不是2494个相互独立样本，同一真实小时会作为不同预测起点的目标重复出现。因此本报告只做描述性比较，不把窗口当独立样本计算显著性，也不把单次训练称为多种子稳定提升。

### 新上传GPU训练记录

{md(training,['Model','weights_exactly_equal','formal_best_epoch','new_best_epoch','formal_val_loss','new_val_loss'])}

该表对比checkpoint张量而非仅比较文件名或序列化字节。无论是否一致，新训练运行都未被直接提升为正式实验。CPU与GPU、推理内核及浮点精度会造成小数差异；不得把两套导出数值拼到同一张论文表中。本轮独立复跑用于数值等价核查，正式数值仍固定。

## 四、RAG结果详细分析

{md(categories,['Backend','Category','N','Top1_correct_n','Hit3_n','All_evidence_n','Refused_n'])}

Top1/Hit@3/MRR@3只以9条有答案Query为分母；无答案拒答以3条为分母。多知识点任务“命中一条”与“覆盖所有必需证据”分别统计。类别只有3题，每题就影响33.33个百分点，不能做广泛泛化结论。

### 失败记录

{md(failures,['Backend','Query','Category','Gold','Top1','Score','Accepted'])}

- TF-IDF的同义表达问题说明词面重合不足；Q12无关足球问题仍命中，说明常见词也可能制造较高相似度。保留原TF-IDF设置作为基线，未在已看过的测试集上选择停用词或重调阈值。
- Embedding的Q06没有检索到预设的环境温度知识，说明句向量不能保证处理间接表达。
- Q08涉及缺失样本和重复时间戳。Embedding找到了部分知识，但未覆盖完整标注。Hit@3成功不能代替多知识点完成率。
- Q11要求具体设备的数值保护阈值，而知识库只有一般性热风险建议。主题相关不等于包含答案。当前Embedding仍误接收此题，不能通过提高测试题阈值或对T17等字符串写特判来伪装修复。

本轮优化只批量编码、修正后端阈值选择和验证输入；不改变知识文本、题目、标注、分数阈值或排名政策。拒答性能不足仍是论文应披露的实测局限。下一项有效研究应先建立与这12题独立的开发集及新盲测集，明确设备事实/数值答案覆盖规则，然后固定方案后一次性盲测；本轮没有生成新题后再声称独立验证。

## 五、风险阈值与案例解释

|指标|MEDIUM q90|HIGH q95|
|---|---:|---:|
|24h最大温度|{t['temperature']['medium']:.6f}|{t['temperature']['high']:.6f}|
|24h峰值减首值|{t['rise']['medium']:.6f}|{t['rise']['high']:.6f}|
|最大正小时斜率|{t['slope']['medium']:.6f}|{t['slope']['high']:.6f}|

{md(riskdist,['Split','Windows','LOW','MEDIUM','HIGH'])}

这里使用每个划分内所有连续24小时真实窗口，因此验证/测试各2590个窗口；模型评价需要额外96小时历史，因此测试预测窗口为2494个，二者分母不能混用。阈值取自训练总体，验证和测试风险分布明显不同的可能原因包括时段/季节分布变化，单凭本数据不能断言唯一成因。q90/q95是单指标分位数，采用任一指标触发规则后，组合HIGH比例并不必然等于5%。

### 预测平滑与风险敏感性

{md(smooth,['Source','Indicator','q50','q90','q95','medium_trigger_count'])}

预测误差最小不等于突变风险捕获最好。比较同一组2494个目标窗口的真实指标和LSTM预测指标，可直接观察预测峰值/斜率的分布差异。若预测斜率更平滑，原始序列分位数阈值可能很少被预测触发；应如实报告，不能据此用展示案例调低阈值。

预测风险与真实未来窗口按同一规则得到的等级列联表（行=真实规则等级，列=预测等级）：

{md([{'True':k,'Pred_LOW':int(conf[i,0]),'Pred_MEDIUM':int(conf[i,1]),'Pred_HIGH':int(conf[i,2])} for i,k in enumerate(['LOW','MEDIUM','HIGH'])],['True','Pred_LOW','Pred_MEDIUM','Pred_HIGH'])}

真实规则等级使用原始CSV对应目标窗口重新计算，避免浮点标准化/反标准化的微小舍入在分位数等号边界造成错误等级；原回归表保持正式数据不变。此表衡量规则等级一致性，不是故障诊断混淆矩阵。当前{int((truthlev==1).sum())}个真实MEDIUM窗口中，{int(conf[1,0])}个被预测为LOW，说明仅看总体预测MAE改善会掩盖中风险漏检；这些窗口相互重叠，不应视为独立故障事件。如果某一真实等级没有样本，该等级召回率无定义，不能用0或100%代替。当前ETTh1没有真实故障类型标签，不能把规则单元测试的PASS称为故障诊断准确率。

固定案例的预测risk依次为LOW、LOW、MEDIUM，而组合历史异常与预测风险后的diagnosis为NORMAL、WARNING、ANOMALY。这是两层不同规则的输出，不互相矛盾。原案例来自训练时段且部分重叠，只用于展示。保留这些输入能比较版本行为，但不能证明未见场景泛化。

## 六、本轮完成的优化与收益

1. 结果隔离：普通评价只能写新的复跑目录，禁止写正式、论文和归档目录；渲染不再把旧正式案例写回实时output。
2. 正式来源追溯：原数值与原源码哈希不改写，旧源码保留为小型快照；当前代码由新的实现清单及等价性验证管理，避免通过更新旧哈希掩盖来源变更。
3. RAG后端：切换TF-IDF时自动选用0.08；Embedding仍为0.45。批量接口保留单条retrieve兼容接口，空白Query和非法参数明确报错。
4. 风险计算：使用滑动窗口视图向量化，和原循环逐值一致；加载冻结结果时重算训练分位数验证，防止仅修改阈值数值而未修改配置的情况。
5. 输入与模型：CSV要求96行、唯一连续小时；训练数据同样验证时间轴及有限值；scaler在比较前验证维度、有限值、正比例与目标索引。
6. 运行解耦：导入RAG和风险模块不再强制导入整个Agent；CLI添加main保护，测试和案例共用输入校验。

{timingtext}

性能测试为CPU双线程、预热后5次中位数；RAG不包括模型首次载入，分位数测试包括读CSV。它只反映当前验证环境，不能直接推断用户RTX 5060上的速度。语义检索批量与逐条的Top-3顺序和accepted标记一致；浮点分数差见revision_checks.json。计算提速不代表模型精度提高。

## 七、论文收口建议及尚未证明的事项

当前可以陈述：固定ETTh1协议下的模型比较、两种检索方式的小规模案例对照、训练分位数风险和可追溯多步骤工作流。应避免声称工业级故障识别已获验证、ReAct多轮推理得到验证、Embedding能够可靠回答所有设备问题。

尚未完成的研究证据（不是本轮代码缺陷）：独立专家审核知识库、大规模盲测RAG集、真实故障标签、跨设备数据和重复种子统计。本轮不通过额外功能或随意调参掩盖这些限制。若未来优化模型精度，应先固定验证集目标与搜索预算，再固定一个候选后评价测试集；当前测试已多次查看，后续“改进”应新增未见数据才能作强结论。

## 八、复现入口

```bash
python calibrate_risk.py
python tests/test_risk_boundaries.py
python tests/test_diagnosis.py
python tests/test_revision.py
python tests/evaluate_rag.py
python evaluate_final.py
python run_agent.py --device cuda:0
python tests/test_closeout.py
python analyze_results.py
```

正式结果继续读取results/final/experiment.json；新评价在results/reruns；本报告与细表在results/audit和reports。完整逐文件修改清单见reports/REVISION_CHANGELOG.md。
'''
    (ROOT/'reports/AUDIT_AND_ANALYSIS.md').write_text(report,encoding='utf-8')
    print(json.dumps({'LSTM':l,'better_horizons':better,'training':training,'risk_distribution':riskdist},indent=2))
if __name__=='__main__':main()
