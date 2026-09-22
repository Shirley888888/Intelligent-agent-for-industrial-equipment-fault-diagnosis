import json
from pathlib import Path
from typing import List
import numpy as np
from .chunker import Chunk, chunk_documents
from .embedding import LocalSemanticEmbedding

class IndustrialKnowledgeBase:
    def __init__(self, kb_file: Path, chunk_size: int, overlap: int):
        self.kb_file = Path(kb_file)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = []
        self.chunks: List[Chunk] = []
        self.embedding_model = LocalSemanticEmbedding()
        self.vectors = None

    def load_documents(self):
        if not self.kb_file.exists():
            raise FileNotFoundError(f"知识库不存在: {self.kb_file}")
        with open(self.kb_file, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        return self.documents

    def build_chunks(self):
        if not self.documents:
            self.load_documents()
        self.chunks = chunk_documents(self.documents, self.chunk_size, self.overlap)
        return self.chunks

    def build_vectors(self):
        if not self.chunks:
            self.build_chunks()
        self.vectors = np.asarray([self.embedding_model.encode(c.text) for c in self.chunks], dtype=np.float32)
        return self.vectors

    def build(self):
        self.load_documents()
        self.build_chunks()
        self.build_vectors()
        return self

    def save(self, chunk_file: Path, vector_file: Path):
        chunk_file, vector_file = Path(chunk_file), Path(vector_file)
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump([{"kb_id":c.kb_id,"chunk_id":c.chunk_id,"text":c.text} for c in self.chunks], f, ensure_ascii=False, indent=2)
        np.savez_compressed(vector_file, vectors=self.vectors)
