# ATREA 主动经验论文检索机制

## 工作方式

- 研究循环开始时不再调用 `retrieve_experience()`，也不把历史实验、经验模式或经验论文自动塞入 Agent 上下文。
- 规划器可以通过第三个参数 `experience_tool` 使用工具；也可以将参数命名为 `experience_tool`。
- 工具名称：`SearchExperience`；调用方式：`experience_tool.search_experience(query, limit=5, dataset="ETTh1")`。
- Agent 应先根据当前任务、误差诊断和外部论文判断是否需要历史经验。只有判断有帮助时才调用工具；不需要时不调用。
- 搜索返回经验论文题目、主题、标签、相关性分数、正文和 Markdown 文件路径。搜索结果是研究证据，不是直接可部署的模型结论。
- 兼容旧版一参数/二参数规划器，但旧版规划器不会获得搜索工具。要启用自主检索，应将规划器升级为 `hypothesis_builder(papers, memory, experience_tool)` 或 `hypothesis_builder(papers, experience_tool=...)`。

## 经验论文归档

每次研究循环完成后，系统将本轮假设、方法、实验结果、误差分析、讨论和结论归档为一篇 Markdown 经验论文：

`<workspace目录>/experience_library/YYYY-MM-DD_<编号>.md`

目录文件：

`<workspace目录>/experience_library/目录.md`

目录由 SQLite 中的经验论文记录自动重建，记录论文编号、题目、主题、数据集、入库时间和文件路径。新增论文入库成功后立即刷新目录。

## 规划器伪代码

```python
def hypothesis_builder(papers, memory, experience_tool):
    # memory is intentionally empty; do not assume prior experience was injected.
    if prior_experience_would_help(papers):
        result = experience_tool.search_experience(
            "ETTh1 oil temperature forecasting late horizon error", limit=5,
            dataset="ETTh1")
        # Incorporate only relevant evidence and record its limitations.
    return build_falsifiable_plan(papers)
```

`prior_experience_would_help` 和 `build_falsifiable_plan` 由实际 LLM/规划器实现；此示例仅说明工具调用接口。
