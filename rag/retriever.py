from dataclasses import dataclass
from typing import List, Tuple
from .chunker import Chunk
from .embedding import LocalSemanticEmbedding, cosine_similarity

@dataclass
class RetrievalResult:
    kb_id: str
    chunk_id: str
    text: str
    similarity: float

class IndustrialRetriever:
    def __init__(self, chunks: List[Chunk], vectors, embedding_model: LocalSemanticEmbedding):
        self.chunks = chunks
        self.vectors = vectors
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int, similarity_threshold: float) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold 必须位于 [0,1]")
        qv = self.embedding_model.encode(query)
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            sim = cosine_similarity(qv, vector.tolist())
            results.append(RetrievalResult(chunk.kb_id, chunk.chunk_id, chunk.text, float(sim)))
        results.sort(key=lambda x: x.similarity, reverse=True)
        top = results[:top_k]
        valid = [x for x in top if x.similarity >= similarity_threshold]
        return top, valid
