# Industrial RAG System

完全离线的工业知识库 RAG 教学/工程化项目。

## 完整链路
Industrial Documents → Chunk + Overlap → Embedding → Vector Index → Query Embedding → Cosine Similarity → Top-K → Similarity Threshold → Evidence → Evidence-based Answer

## 知识库
KB-017、KB-018、KB-022、KB-026、KB-031、KB-044。

## 安装
```bash
pip install -r requirements.txt
```

## 构建知识库
```bash
python build_kb.py
```

## 运行 12 个测试
```bash
python run_tests.py
```

结果保存到：
- outputs/retrieval_results.json
- outputs/retrieval_results.csv

## 交互查询
```bash
python industrial_rag.py
```

## 核心规则
Top-K 仅代表排序靠前，不代表证据一定有效。系统继续使用 Similarity Threshold 过滤；全部低于阈值时返回：
“知识库证据不足，无法判断，不编造故障结论。”

本项目不调用 OpenAI 或任何外部 LLM API。
