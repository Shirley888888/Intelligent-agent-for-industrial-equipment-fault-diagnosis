from __future__ import annotations

# modify:rag-observation - preserve the project as an offline, reproducible RAG demo.
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_KNOWLEDGE = [
    {"id": "KB-THERMAL-001", "text": "Stable transformer oil temperature with no unusual rise should continue routine trend monitoring and normal inspection."},
    {"id": "KB-THERMAL-002", "text": "A gradual oil-temperature rise should trigger closer monitoring of transformer load, cooling effectiveness, ambient conditions and sensor consistency."},
    {"id": "KB-THERMAL-003", "text": "A rapid oil-temperature rise or high predicted temperature requires prompt review of load and cooling conditions and escalation under the site's approved operating procedure."},
    {"id": "KB-SENSOR-001", "text": "Unexpected temperature jumps should be checked against sensor quality, timestamp continuity and correlated operating signals before maintenance conclusions are made."},
    {"id": "KB-COOLING-001", "text": "Thermal risk investigation should include cooling path availability, fan or pump operation where applicable, and recent loading changes."},
]


@dataclass
class RAGTool:
    cfg: dict[str, Any]

    def __post_init__(self):
        # modify:rag-config - all retrieval hyperparameters come from config.yaml.
        self.top_k = int(self.cfg["top_k"])
        self.threshold = float(self.cfg["similarity_threshold"])
        self.chunk_size = int(self.cfg["chunk_size"])
        self.chunk_overlap = int(self.cfg["chunk_overlap"])
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG chunk_size must be > 0 and chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunks = self._chunk_documents(DEFAULT_KNOWLEDGE)
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([x["text"] for x in self.chunks])

    def _chunk_documents(self, docs):
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for doc in docs:
            words = doc["text"].split()
            for i, start in enumerate(range(0, len(words), step)):
                text = " ".join(words[start:start + self.chunk_size]).strip()
                if text:
                    chunks.append({"id": f'{doc["id"]}-C{i+1}', "text": text})
                if start + self.chunk_size >= len(words):
                    break
        return chunks

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        # modify:rag-observation - return accepted and rejected Top-K evidence for auditability.
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:self.top_k]
        return [
            {
                "id": self.chunks[int(idx)]["id"],
                "score": float(scores[int(idx)]),
                "accepted": bool(scores[int(idx)] >= self.threshold),
                "text": self.chunks[int(idx)]["text"],
            }
            for idx in order
        ]
