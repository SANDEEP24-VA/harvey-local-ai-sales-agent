"""Local embedding and semantic vector search using SentenceTransformers.

100% offline, zero cloud API calls, running locally on CPU or NVIDIA CUDA GPU.
Provides persistent SQLite vector storage with cosine similarity search for
product documentation, prospect research evidence, and sales knowledge retrieval.
"""

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "embeddings")


class LocalEmbeddingEngine:
    """Local vector embeddings engine using SentenceTransformers and local SQLite storage."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str | None = None,
        db_path: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.db_path = db_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sellbird.db")
        self._model = None
        self._device = device
        self._use_fallback = False
        self._init_vector_table()

    def _fallback_vector(self, text: str, dim: int = 384) -> list[float]:
        """Deterministic, offline-safe fallback vector generator for testing and airgapped environments."""
        vec = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _init_vector_table(self):
        """Create vector storage table in SQLite if it does not already exist."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                        id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ke_entity ON knowledge_embeddings(entity_type, entity_id)")
        finally:
            conn.close()

    def _get_model(self):
        if self._model is not None or self._use_fallback:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Loading local embedding model '{self.model_name}' on {self._device} (cache: {self.cache_dir})")
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self._device)
            return self._model
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}); falling back to deterministic local hash vectorizer.")
            self._use_fallback = True
            return None

    def embed_text(self, text: str) -> list[float]:
        """Generate local embedding vector for input text."""
        model = self._get_model()
        if model is not None:
            try:
                vec = model.encode(text, convert_to_numpy=True)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding error ({e}); using fallback.")
        return self._fallback_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate local embedding vectors for multiple texts in batch."""
        if not texts:
            return []
        model = self._get_model()
        if model is not None:
            try:
                vecs = model.encode(texts, convert_to_numpy=True, batch_size=32)
                return [v.tolist() for v in vecs]
            except Exception as e:
                logger.warning(f"SentenceTransformer batch encoding error ({e}); using fallback.")
        return [self._fallback_vector(t) for t in texts]

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vector lists."""
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm_a = sum(a * a for a in v1) ** 0.5
        norm_b = sum(b * b for b in v2) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    def index_document(
        self,
        entity_type: str,
        entity_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 500,
    ) -> int:
        """Split text into semantic chunks, embed each, and store in SQLite."""
        if not text or not text.strip():
            return 0

        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for p in raw_paragraphs:
            if len(p) <= chunk_size:
                chunks.append(p)
            else:
                parts = p.split(". ")
                cur = ""
                for part in parts:
                    if len(cur) + len(part) < chunk_size:
                        cur += (". " if cur else "") + part
                    else:
                        if cur:
                            chunks.append(cur.strip())
                        cur = part
                if cur:
                    chunks.append(cur.strip())

        if not chunks:
            chunks = [text[:chunk_size]]

        embeddings = self.embed_batch(chunks)
        meta_json = json.dumps(metadata or {})
        inserted = 0

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    chunk_id = hashlib.sha256(f"{entity_type}:{entity_id}:{idx}:{chunk}".encode()).hexdigest()[:24]
                    conn.execute("""
                        INSERT OR REPLACE INTO knowledge_embeddings
                        (id, entity_type, entity_id, chunk_text, embedding_json, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (chunk_id, entity_type, entity_id, chunk, json.dumps(emb), meta_json))
                    inserted += 1
        finally:
            conn.close()

        return inserted

    def search(
        self,
        query: str,
        entity_type: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Perform local semantic vector search against stored document chunks."""
        query_vec = self.embed_text(query)
        conn = sqlite3.connect(self.db_path)
        candidates = []
        try:
            with conn:
                if entity_type:
                    rows = conn.execute("""
                        SELECT id, entity_type, entity_id, chunk_text, embedding_json, metadata_json
                        FROM knowledge_embeddings
                        WHERE entity_type = ?
                    """, (entity_type,)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT id, entity_type, entity_id, chunk_text, embedding_json, metadata_json
                        FROM knowledge_embeddings
                    """).fetchall()

                for row_id, etype, eid, chunk_text, emb_str, meta_str in rows:
                    emb = json.loads(emb_str)
                    sim = self.cosine_similarity(query_vec, emb)
                    if sim >= min_similarity:
                        candidates.append({
                            "id": row_id,
                            "entity_type": etype,
                            "entity_id": eid,
                            "chunk_text": chunk_text,
                            "similarity": round(float(sim), 4),
                            "metadata": json.loads(meta_str or "{}"),
                        })
        finally:
            conn.close()

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]
