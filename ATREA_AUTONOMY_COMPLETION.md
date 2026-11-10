# ATREA 自主研究闭环补全说明

## 已补充的代码能力
1. `atrea/scheduler.py`：常驻调度器按 `Asia/Shanghai` 每天 08:00 触发，持久化当天成功标记，进程重启后不会重复执行当天任务。
2. `ExperienceStore.retrieve_experience()`：下一轮规划可读取历史实验事实与经验模式；`run_research_cycle` 在提出假设前读取并传给双参数 `hypothesis_builder(papers, memory)`（兼容旧单参数构建器）。
3. 每轮实验报告保存到数据库并写入 `agent_workspace/reports/<batch_id>.md`，新增论文摘要、历史经验使用情况、失败分析字段。
4. `atrea/ablation.py`：增加用户指定的三组消融实验定义与运行器：
   - A1 固定时间重训，不搜索论文、不改架构、不使用经验；
   - A2 联网论文搜索并修改架构，不读取经验库；
   - A3 与 A2 相同，但读取经验库，并要求运行器显式记录 `experience_considered=true`。
   结果保存逐组 JSON 与 `three_arm_summary.csv`。
5. 新增 `tests_agent/test_atrea_autonomy.py`。

## 运行方式
在项目根目录启动每日调度器（将 `my_project.daily_job:run` 替换为你提供真实数据和模型的回调）：
```bash
python -m atrea.scheduler --callback my_project.daily_job:run
```
回调签名为 `run(now=带 Asia/Shanghai 时区的 datetime)`。回调内部应调用 `atrea.orchestrator.run_if_due(...)`，或调用 `run_research_cycle(...)` 执行当天研究流程。

三组实验由宿主项目提供真实训练运行器：
```python
from atrea.ablation import run_three_arm_ablation
from atrea.store import ExperienceStore

store = ExperienceStore("agent_workspace/atrea_experience.sqlite3")
result = run_three_arm_ablation(
    store=store,
    experiment_runner=your_real_experiment_runner,
    paper_searcher=your_paper_searcher,
    query="time series forecasting foundation model",
    dataset="ETTh1",
)
print(result["summary_csv"])
```

## 重要边界
- 调度器实现了每日北京时间 08:00 的调度，但必须由操作系统/服务器保持进程运行；也可由 Windows Task Scheduler、cron 或 systemd 在 08:00 调用一次性入口。
- 论文发现当前使用 arXiv API，关键词提取只是线索，不是“全网最优模型”排名。要实现可靠的最优论文筛选，还需接入 Semantic Scholar/Crossref/OpenAlex/Hugging Face/GitHub 元数据，并对同行评审状态、引用、代码、许可证、复现成本和任务适配度进行排序。
- ATREA 的训练与评估仍由宿主 `experiment_runner` 注入。没有你实际 ETTh1/工业油温训练函数、数据提供器、GPU 环境和门禁回调，不能声称已经自动训练或自动替换真实生产模型。
- 模型激活必须连接既有版本管理和候选门禁。不得仅凭论文摘要自动下载并执行第三方代码，也不得跳过冻结验证集、统计检验、资源/安全检查。
- A1/A2/A3 是本次要求的三臂实验框架；它与项目原有四臂在线漂移回放实验目的不同，不应混淆。需要实际运行器后，生成的指标才是有效实验结果。
