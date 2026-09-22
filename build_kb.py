from config import KB_FILE, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_FILE, VECTOR_INDEX_FILE
from rag.knowledge_base import IndustrialKnowledgeBase

def main():
    print("="*70)
    print("Industrial RAG Knowledge Base Builder")
    print("="*70)
    kb = IndustrialKnowledgeBase(KB_FILE, CHUNK_SIZE, CHUNK_OVERLAP).build()
    print(f"原始知识条目: {len(kb.documents)}")
    print(f"Chunk 数量: {len(kb.chunks)}")
    print(f"Embedding 维度: {kb.vectors.shape[1]}")
    for c in kb.chunks:
        print(f"{c.chunk_id} | {c.text}")
    kb.save(CHUNK_FILE, VECTOR_INDEX_FILE)
    print(f"\nChunk Index: {CHUNK_FILE}")
    print(f"Vector Index: {VECTOR_INDEX_FILE}")

if __name__ == "__main__":
    main()
