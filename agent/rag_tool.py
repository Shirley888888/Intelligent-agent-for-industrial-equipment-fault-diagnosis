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
    {'id': 'KB-COOLING-002', 'text': 'Cooling fan failure can reduce heat removal. Inspect fan status, power supply and control alarms through authorized procedures.'},
    {'id': 'KB-COOLING-003', 'text': 'Oil pump failure or reduced circulation can impair cooling. Compare pump indications with oil flow and maintenance records.'},
    {'id': 'KB-COOLING-004', 'text': 'Blocked radiator surfaces or obstructed airflow can reduce cooling effectiveness. Arrange inspection and approved cleaning.'},
    {'id': 'KB-LOAD-001', 'text': 'Increased electrical load can raise transformer oil temperature. Compare load trends with thermal trends before attributing a rise to a fault.'},
    {'id': 'KB-LOAD-002', 'text': 'Unbalanced phase currents may contribute to uneven heating. Review phase current measurements and loading records.'},
    {'id': 'KB-AMBIENT-001', 'text': 'Higher ambient air temperature can elevate oil temperature without an internal fault. Compare ambient and oil trends over the same period.'},
    {'id': 'KB-SENSOR-002', 'text': 'A missing measurement is not a zero temperature. Flag missing values and suspend forecasts until input quality meets requirements.'},
    {'id': 'KB-SENSOR-003', 'text': 'Duplicate timestamps or out-of-order samples distort temporal slopes. Check chronological ordering and hourly sampling before forecasting.'},
    {'id': 'KB-SENSOR-004', 'text': 'Persistent disagreement between temperature sensors may indicate calibration drift. Compare independent instruments and calibration history.'},
    {'id': 'KB-OIL-001', 'text': 'Low oil level can compromise thermal performance. Check level indications and potential leaks using approved inspection procedures.'},
    {'id': 'KB-OIL-002', 'text': 'Visible oil leakage requires recording its location and escalation to authorized maintenance staff. Do not infer leak severity from temperature alone.'},
    {'id': 'KB-OIL-003', 'text': 'Moisture in insulating oil can degrade insulation performance. Use appropriate oil analysis records rather than temperature alone to assess moisture.'},
    {'id': 'KB-OIL-004', 'text': 'Dissolved gas analysis supports investigation of internal transformer abnormalities. Oil temperature alone cannot identify gas concentrations or fault types.'},
    {'id': 'KB-ELECTRICAL-001', 'text': 'Abnormal bushing heating needs independent inspection and electrical evidence. Oil temperature forecasts cannot locate a bushing defect.'},
    {'id': 'KB-ELECTRICAL-002', 'text': 'Unusual transformer noise or vibration should be recorded with operating conditions and reviewed by qualified maintenance personnel.'},
    {'id': 'KB-PROTECTION-001', 'text': 'A protection alarm requires following the site approved response procedure. A forecasting agent must not override protection interlocks.'},
    {'id': 'KB-MAINTENANCE-001', 'text': 'Record maintenance time, affected components and load conditions. Compare pre-maintenance and post-maintenance temperature trends.'},
    {'id': 'KB-MODEL-001', 'text': 'Forecast uncertainty increases for operating conditions outside training coverage. Flag distribution shifts and seek additional evidence.'},
    {'id': 'KB-MODEL-002', 'text': 'Training-only normalization avoids data leakage. Validation and test observations must use the scaler fitted on training data.'},
    {'id': 'KB-RISK-001', 'text': 'Quantile thermal risk indicates unusual temperature, rise or slope relative to training data. It is not a validated fault label or equipment trip setting.'},
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
        # modify:rag-embedding - keep TF-IDF baseline and add real sentence embeddings.
        self.backend = self.cfg.get("backend", "tfidf")
        # modify:audit-config - each backend owns its threshold; reject invalid config.
        if self.backend == "tfidf":
            self.threshold = float(self.cfg.get("tfidf_threshold", self.threshold))
        if self.top_k <= 0 or not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("top_k must be positive; cosine threshold must be finite and in [-1, 1].")
        texts = [x["text"] for x in self.chunks]
        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(texts)
        elif self.backend == "embedding":
            from sentence_transformers import SentenceTransformer
            from pathlib import Path
            model_path = Path(__file__).resolve().parents[1] / self.cfg["embedding_model"]
            if not model_path.is_dir():
                raise FileNotFoundError(f"Embedding model missing: {model_path}; run prepare_embedding.py. No TF-IDF fallback.")
            self.encoder = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
            self.matrix = self.encoder.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"Unknown retrieval backend: {self.backend}")

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
        # modify:audit-batch - preserve retrieve API; batch encoder implemented once.
        return self.retrieve_many([query])[0]

    def retrieve_many(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        """Encode queries in one batch; identical ranking/filter policy to retrieve."""
        if not isinstance(queries, (list, tuple)) or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("queries must be a list of nonempty strings")
        if not queries:
            return []
        if self.backend == "tfidf":
            q = self.vectorizer.transform(queries)
            scores = (q @ self.matrix.T).toarray()
        else:
            q = self.encoder.encode(queries, normalize_embeddings=True, show_progress_bar=False)
            scores = q @ self.matrix.T
        return [self._rank_scores(row) for row in scores]

    def _rank_scores(self, scores):
        order = np.argsort(-scores, kind="stable")[:self.top_k]
        return [{"id": self.chunks[int(idx)]["id"], "score": float(scores[int(idx)]),
                 "accepted": bool(scores[int(idx)] >= self.threshold),
                 "text": self.chunks[int(idx)]["text"]} for idx in order]
