from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Chunk:
    kb_id: str
    chunk_id: str
    text: str

def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 50, overlap: int = 10) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    for document in documents:
        kb_id = document["id"]
        text = document["text"].strip()
        if not text:
            continue
        start, chunk_index = 0, 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(kb_id, f"{kb_id}-C{chunk_index:03d}", chunk_text))
            if end >= len(text):
                break
            start = end - overlap
            chunk_index += 1
    return chunks
