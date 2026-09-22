from config import KB_FILE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIMILARITY_THRESHOLD
from rag.knowledge_base import IndustrialKnowledgeBase
from rag.retriever import IndustrialRetriever
from rag.generator import build_prompt, simulate_llm

class IndustrialRAG:
    def __init__(self, top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.knowledge_base = IndustrialKnowledgeBase(KB_FILE, CHUNK_SIZE, CHUNK_OVERLAP).build()
        self.retriever = IndustrialRetriever(
            self.knowledge_base.chunks,
            self.knowledge_base.vectors,
            self.knowledge_base.embedding_model
        )

    def ask(self, query):
        top, evidence = self.retriever.retrieve(query, self.top_k, self.similarity_threshold)
        return {
            "query": query,
            "top_k_results": top,
            "evidence": evidence,
            "prompt": build_prompt(query, evidence),
            "answer": simulate_llm(query, evidence)
        }

if __name__ == "__main__":
    rag = IndustrialRAG()
    query = input("请输入工业设备问题：")
    result = rag.ask(query)
    print("\nTop-K:")
    for x in result["top_k_results"]:
        print(x.kb_id, round(x.similarity,4), x.text)
    print("\n有效 Evidence:")
    for x in result["evidence"]:
        print(x.kb_id, round(x.similarity,4))
    print("\nRAG Answer:")
    print(result["answer"])
