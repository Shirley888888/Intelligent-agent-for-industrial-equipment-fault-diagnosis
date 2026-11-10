from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .common import canonical_json, require, sha256_json


def _tokens(text: str) -> list[str]:
    """Small dependency-free tokenizer: English terms + Chinese character bigrams."""
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9][a-z0-9_./-]*", text)
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for segment in cjk:
        if len(segment) == 1:
            words.append(segment)
        else:
            words.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return words


class KnowledgeCatalog:
    """Versioned knowledge metadata; only approved and effective entries retrieve.

    Retrieval is TF-IDF lexical baseline. Embedding retrieval and reranking are
    intentionally deferred until the real manual/SOP corpus is supplied.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _events(self) -> list[dict[str, Any]]:
        events = []
        prev_hash = "0" * 64
        seen = set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_hash = event.pop("event_hash", None)
                if event.get("prev_hash") != prev_hash or event_hash != sha256_json(event):
                    raise ValueError(f"Knowledge catalog hash-chain error at line {line_no}")
                if event["event_id"] in seen:
                    raise ValueError(f"Duplicate knowledge event at line {line_no}")
                seen.add(event["event_id"])
                event["event_hash"] = event_hash
                events.append(event)
                prev_hash = event_hash
        return events

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        events = self._events()
        import uuid
        body = {"sequence": len(events) + 1, "event_id": str(uuid.uuid4()),
                "event_type": event_type, "created_at": self._now(),
                "prev_hash": events[-1]["event_hash"] if events else "0" * 64,
                "payload": payload}
        event = {**body, "event_hash": sha256_json(body)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _documents(self) -> dict[tuple[str, str], dict[str, Any]]:
        docs: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self._events():
            p = event["payload"]
            key = (p["document_id"], p["version"])
            if event["event_type"] == "document_registered":
                docs[key] = dict(p["document"])
            elif event["event_type"] == "document_status_changed":
                if key not in docs:
                    raise ValueError(f"Status event references missing document: {key}")
                docs[key].update(p["changes"])
        return docs

    def register_document(self, document: dict[str, Any]) -> dict[str, str]:
        required = ("document_id", "version", "title", "source_uri", "equipment_scope",
                    "effective_from", "content")
        for field in required:
            require(field in document, f"Knowledge document missing {field}")
        require(bool(str(document["document_id"]).strip()) and bool(str(document["version"]).strip()),
                "document_id/version must be nonempty")
        require(bool(str(document["source_uri"]).strip()), "source_uri is required")
        require(isinstance(document["equipment_scope"], list) and document["equipment_scope"],
                "equipment_scope must be a nonempty list")
        date.fromisoformat(str(document["effective_from"]))
        if document.get("effective_to"):
            date.fromisoformat(str(document["effective_to"]))
        key = (str(document["document_id"]), str(document["version"]))
        docs = self._documents()
        require(key not in docs, f"Knowledge document version already exists: {key}")
        clean = dict(document)
        clean["status"] = "draft"
        clean["registered_at"] = self._now()
        clean["registered_by"] = str(document.get("registered_by", "unknown"))
        clean["content_sha256"] = __import__("hashlib").sha256(
            str(document["content"]).encode("utf-8")).hexdigest()
        clean.setdefault("keywords", [])
        clean.setdefault("summary", "")
        self._append("document_registered", {"document_id": key[0], "version": key[1],
                                               "document": clean})
        return {"document_id": key[0], "version": key[1], "status": "draft",
                "content_sha256": clean["content_sha256"]}

    def approve_document(self, document_id: str, version: str, *, reviewer: str,
                         approval_ref: str) -> dict[str, str]:
        docs = self._documents()
        key = (document_id, version)
        require(key in docs, f"Unknown knowledge document: {key}")
        doc = docs[key]
        require(doc["status"] == "draft", "Only draft knowledge documents can be approved")
        require(bool(reviewer.strip()) and bool(approval_ref.strip()),
                "reviewer and approval_ref are required")
        self._append("document_status_changed", {
            "document_id": document_id, "version": version,
            "changes": {"status": "approved", "approved_by": reviewer,
                        "approval_ref": approval_ref, "approved_at": self._now()},
        })
        return {"document_id": document_id, "version": version, "status": "approved"}

    def supersede_document(self, document_id: str, version: str, *, reviewer: str,
                           replacement_version: str) -> dict[str, str]:
        docs = self._documents()
        key = (document_id, version)
        require(key in docs and docs[key]["status"] == "approved",
                "Only approved documents can be superseded")
        require((document_id, replacement_version) in docs,
                "Replacement document version must be registered first")
        self._append("document_status_changed", {
            "document_id": document_id, "version": version,
            "changes": {"status": "superseded", "superseded_by": replacement_version,
                        "superseded_by_reviewer": reviewer, "superseded_at": self._now()},
        })
        return {"document_id": document_id, "version": version, "status": "superseded"}

    def search(self, query: str, *, equipment_id: str | None = None,
               as_of: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        require(bool(query.strip()), "query must be nonempty")
        require(type(top_k) is int and top_k > 0, "top_k must be positive")
        day = date.fromisoformat(as_of) if as_of else date.today()
        docs = []
        for doc in self._documents().values():
            if doc["status"] != "approved":
                continue
            if date.fromisoformat(doc["effective_from"]) > day:
                continue
            if doc.get("effective_to") and date.fromisoformat(doc["effective_to"]) < day:
                continue
            scope = set(str(x) for x in doc["equipment_scope"])
            if equipment_id and "*" not in scope and equipment_id not in scope:
                continue
            docs.append(doc)
        query_tokens = _tokens(query)
        if not query_tokens or not docs:
            return []
        doc_tokens = [_tokens(" ".join([str(d.get("title", "")), str(d.get("summary", "")),
                                        str(d.get("content", "")),
                                        " ".join(map(str, d.get("keywords", [])))])) for d in docs]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        n_docs = len(docs)

        def vector(tokens: list[str]) -> dict[str, float]:
            tf = Counter(tokens)
            return {term: (count / max(1, len(tokens))) *
                    (math.log((n_docs + 1) / (df[term] + 1)) + 1.0)
                    for term, count in tf.items()}

        qv = vector(query_tokens)
        qnorm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        hits = []
        for doc, tokens in zip(docs, doc_tokens):
            dv = vector(tokens)
            dnorm = math.sqrt(sum(x * x for x in dv.values())) or 1.0
            score = sum(qv.get(term, 0.0) * value for term, value in dv.items()) / (qnorm * dnorm)
            if score > 0:
                hits.append({"document_id": doc["document_id"], "version": doc["version"],
                             "title": doc["title"], "source_uri": doc["source_uri"],
                             "content_sha256": doc["content_sha256"], "score": score,
                             "equipment_scope": doc["equipment_scope"],
                             "approval_ref": doc.get("approval_ref"),
                             "snippet": str(doc.get("summary") or doc.get("content", ""))[:240]})
        return sorted(hits, key=lambda x: (-x["score"], x["document_id"]))[:top_k]

    def verify_integrity(self) -> dict[str, Any]:
        events = self._events()
        return {"valid": True, "event_count": len(events),
                "document_versions": len(self._documents()),
                "last_hash": events[-1]["event_hash"] if events else "0" * 64}
