"""Local embedding, semantic vector search, and hybrid RAG engine using SentenceTransformers and SQLite FTS5.

100% offline, zero cloud API calls, running locally on CPU or NVIDIA CUDA GPU.
Provides persistent SQLite vector storage, BM25 full-text indexing, Reciprocal Rank Fusion (RRF),
strict multi-product isolation, and untrusted prompt-injection fencing for sales intelligence.
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


def chunk_document(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[str]:
    """Structure-aware semantic chunker respecting headings, paragraphs, bullet points, and sentences.

    Preserves paragraph and heading boundaries, avoiding mid-sentence truncation and introducing
    bounded overlap when a paragraph or section exceeds chunk_size.
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()
    # Split by major blocks (paragraphs, markdown headings)
    block_pattern = re.compile(r"(?:\n\s*\n|(?<=\n)(?=#{1,4}\s))")
    raw_blocks = [b.strip() for b in block_pattern.split(clean_text) if b.strip()]

    chunks: list[str] = []

    for block in raw_blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
        else:
            # Split large block by sentence boundaries
            sentence_pattern = re.compile(r"(?<=[.!?])\s+")
            sentences = [s.strip() for s in sentence_pattern.split(block) if s.strip()]
            cur = ""
            for sent in sentences:
                if not cur:
                    cur = sent
                elif len(cur) + len(sent) + 1 <= chunk_size:
                    cur = f"{cur} {sent}"
                else:
                    chunks.append(cur.strip())
                    if chunk_overlap > 0 and len(cur) > chunk_overlap:
                        overlap_text = cur[-chunk_overlap:].strip()
                        space_idx = overlap_text.find(" ")
                        if space_idx != -1 and space_idx < len(overlap_text) - 10:
                            overlap_text = overlap_text[space_idx + 1:]
                        cur = f"{overlap_text} {sent}"
                    else:
                        cur = sent
            if cur and cur.strip():
                chunks.append(cur.strip())

    return [c for c in chunks if c.strip()]


class LocalEmbeddingEngine:
    """Production-grade local vector embeddings, BM25 FTS5 indexing, and hybrid RAG retrieval."""

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
        self.model_status = "unloaded"  # "loaded", "unavailable", "fallback"
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
        """Create vector storage and FTS5 tables in SQLite if they do not already exist."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                        id TEXT PRIMARY KEY,
                        product_id TEXT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Ensure product_id column exists if table pre-existed
                cols = [r[1] for r in conn.execute("PRAGMA table_info(knowledge_embeddings)").fetchall()]
                if "product_id" not in cols:
                    conn.execute("ALTER TABLE knowledge_embeddings ADD COLUMN product_id TEXT NULL")

                conn.execute("CREATE INDEX IF NOT EXISTS idx_ke_entity ON knowledge_embeddings(entity_type, entity_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ke_product ON knowledge_embeddings(product_id)")

                # Create FTS5 table
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                        chunk_id UNINDEXED,
                        product_id,
                        entity_type,
                        entity_id,
                        chunk_text,
                        tokenize = 'porter unicode61'
                    )
                """)

                # Triggers to keep FTS in sync with knowledge_embeddings
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_ke_fts_insert AFTER INSERT ON knowledge_embeddings
                    BEGIN
                        INSERT INTO knowledge_fts (chunk_id, product_id, entity_type, entity_id, chunk_text)
                        VALUES (NEW.id, COALESCE(NEW.product_id, ''), NEW.entity_type, NEW.entity_id, NEW.chunk_text);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_ke_fts_delete AFTER DELETE ON knowledge_embeddings
                    BEGIN
                        DELETE FROM knowledge_fts WHERE chunk_id = OLD.id;
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_ke_fts_update AFTER UPDATE ON knowledge_embeddings
                    BEGIN
                        DELETE FROM knowledge_fts WHERE chunk_id = OLD.id;
                        INSERT INTO knowledge_fts (chunk_id, product_id, entity_type, entity_id, chunk_text)
                        VALUES (NEW.id, COALESCE(NEW.product_id, ''), NEW.entity_type, NEW.entity_id, NEW.chunk_text);
                    END
                """)
        finally:
            conn.close()

    def _get_model(self):
        if self._model is not None or self._use_fallback:
            return self._model
        try:
            # Explicitly force offline mode to prevent outbound network calls during inference
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Loading local embedding model '{self.model_name}' on {self._device} (cache: {self.cache_dir})")
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self._device,
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self._device,
                )
            self.model_status = "loaded"
            return self._model
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}); falling back to deterministic local hash vectorizer.")
            self._use_fallback = True
            self.model_status = "fallback"
            return None

    @property
    def is_semantic_available(self) -> bool:
        """True when genuine transformer model is loaded, False when in fallback/degraded mode."""
        self._get_model()
        return self._model is not None and not self._use_fallback

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

    # ── Ingestion Pipeline ─────────────────────────────────────────────

    def index_document(
        self,
        entity_type: str,
        entity_id: str,
        text: str,
        metadata_or_product_id: dict[str, Any] | str | None = None,
        metadata: dict[str, Any] | None = None,
        source_url: str = "",
        verification_status: str = "verified_observation",
        is_global: bool = False,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        product_id: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Split text into semantic chunks, embed each, and store in SQLite with FTS auto-sync."""
        if not text or not text.strip():
            return 0

        # Support both signatures: (..., product_id, metadata, ...) and (..., metadata, ...)
        if isinstance(metadata_or_product_id, dict):
            metadata = metadata or metadata_or_product_id
        elif isinstance(metadata_or_product_id, str):
            product_id = product_id or metadata_or_product_id

        if isinstance(metadata, dict):
            product_id = product_id or metadata.get("product_id")
            if "source_url" in metadata and not source_url:
                source_url = metadata.get("source_url", "")
            if "verification_status" in metadata and verification_status == "verified_observation":
                verification_status = metadata.get("verification_status") or verification_status
            if "is_global" in metadata and not is_global:
                is_global = bool(metadata.get("is_global", False))

        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return 0

        embeddings = self.embed_batch(chunks)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        base_meta = dict(metadata or {})
        base_meta.update({
            "product_id": product_id or "",
            "source_url": source_url,
            "verification_status": verification_status,
            "is_global": is_global,
            "content_hash": content_hash,
            "embedding_model": self.model_name if self.is_semantic_available else "deterministic_fallback",
        })

        inserted = 0
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    chunk_meta = dict(base_meta)
                    chunk_meta["chunk_index"] = idx
                    chunk_id = hashlib.sha256(f"{entity_type}:{entity_id}:{idx}:{chunk}".encode()).hexdigest()[:24]

                    conn.execute("""
                        INSERT OR REPLACE INTO knowledge_embeddings
                        (id, product_id, entity_type, entity_id, chunk_text, embedding_json, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        product_id,
                        entity_type,
                        entity_id,
                        chunk,
                        json.dumps(emb),
                        json.dumps(chunk_meta),
                    ))
                    inserted += 1
        finally:
            conn.close()

        return inserted

    def index_product(self, product: Any) -> int:
        """Index a Product model into individual semantic chunks (overview, features, benefits, objection playbooks)."""
        total = 0
        p_id = getattr(product, "id", "")
        p_name = getattr(product, "name", "Product")

        # 1. Product core overview
        overview_parts = [
            f"# {p_name}",
            getattr(product, "description", ""),
            f"Value Proposition: {getattr(product, 'value_proposition', '')}",
            f"Pricing: {getattr(product, 'pricing_model', '')} ({getattr(product, 'pricing_details', '')})",
            f"Entry Offer: {getattr(product, 'entry_offer', '')}",
        ]
        total += self.index_document(
            entity_type="product_doc",
            entity_id=f"{p_id}_overview",
            text="\n\n".join(p for p in overview_parts if p.strip()),
            product_id=p_id,
            metadata={"product_name": p_name, "section": "overview"},
        )

        # 2. Features
        features = getattr(product, "features", [])
        if features:
            feat_text = f"# {p_name} - Key Features\n" + "\n".join(f"- {f}" for f in features)
            total += self.index_document(
                entity_type="product_feature",
                entity_id=f"{p_id}_features",
                text=feat_text,
                product_id=p_id,
                metadata={"product_name": p_name, "section": "features"},
            )

        # 3. Benefits & Pain Points
        benefits = getattr(product, "benefits", [])
        pain_points = getattr(product, "pain_points", [])
        if benefits or pain_points:
            bp_parts = [f"# {p_name} - Benefits and Pain Points"]
            if benefits:
                bp_parts.append("## Business Benefits\n" + "\n".join(f"- {b}" for b in benefits))
            if pain_points:
                bp_parts.append("## Solved Pain Points\n" + "\n".join(f"- {p}" for p in pain_points))
            total += self.index_document(
                entity_type="product_benefit",
                entity_id=f"{p_id}_benefits",
                text="\n\n".join(bp_parts),
                product_id=p_id,
                metadata={"product_name": p_name, "section": "benefits"},
            )

        # 4. Objection Response Playbooks
        objection_responses = getattr(product, "objection_responses", {})
        if objection_responses and isinstance(objection_responses, dict):
            for trigger, response in objection_responses.items():
                obj_text = f"Objection: {trigger}\nCounter-Strategy & Evidence: {response}"
                total += self.index_document(
                    entity_type="objection_playbook",
                    entity_id=f"{p_id}_obj_{hashlib.md5(trigger.encode()).hexdigest()[:8]}",
                    text=obj_text,
                    product_id=p_id,
                    metadata={"product_name": p_name, "objection_trigger": trigger},
                )

        return total

    def index_product_icp(self, icp: Any) -> int:
        """Index a ProductICP model into searchable ICP criteria."""
        p_id = getattr(icp, "product_id", "")
        parts = [
            f"# Ideal Customer Profile for Product {p_id}",
            f"Target Industries: {', '.join(getattr(icp, 'target_industries', []))}",
            f"Target Job Titles: {', '.join(getattr(icp, 'job_titles', []))}",
            f"Company Size: {getattr(icp, 'company_size', '')}",
            f"Revenue Range: {getattr(icp, 'revenue_range', '')}",
            f"Known Technologies: {', '.join(getattr(icp, 'technologies', []))}",
            f"Pain Points: {', '.join(getattr(icp, 'pain_points', []))}",
            f"Buying Signals: {', '.join(getattr(icp, 'buying_signals', []))}",
            f"Exclusion Criteria: {', '.join(getattr(icp, 'exclusion_criteria', []))}",
        ]
        return self.index_document(
            entity_type="product_icp",
            entity_id=f"{p_id}_icp",
            text="\n".join(parts),
            product_id=p_id,
            metadata={"product_id": p_id, "section": "icp"},
        )

    def index_research_cache_item(
        self,
        item: Any,
        product_id: str | None = None,
        prospect_id: str | None = None,
    ) -> int:
        """Index crawled website content from ResearchCacheItem."""
        text = getattr(item, "extracted_text", "")
        url = getattr(item, "url", "")
        domain = getattr(item, "domain", "")
        item_id = getattr(item, "id", "") or hashlib.md5(url.encode()).hexdigest()[:12]

        return self.index_document(
            entity_type="web_research",
            entity_id=item_id,
            text=text,
            product_id=product_id,
            source_url=url,
            verification_status="verified_observation",
            metadata={"domain": domain, "prospect_id": prospect_id or ""},
        )

    def index_buying_signal(
        self,
        signal: Any,
        prospect_id: str | None = None,
    ) -> int:
        """Index a classified buying signal observation."""
        code = getattr(signal, "signal_code", "")
        evidence = getattr(signal, "supporting_evidence", "")
        result = getattr(signal, "result", "")
        url = getattr(signal, "source_url", "")
        pid = getattr(signal, "product_id", None)
        status = getattr(signal, "verification_status", "verified_observation")

        text = f"Observed Buying Signal [{code}]: {result}\nEvidence: {evidence}"
        return self.index_document(
            entity_type="buying_signal",
            entity_id=getattr(signal, "id", "") or hashlib.md5(f"{code}:{evidence}".encode()).hexdigest()[:12],
            text=text,
            product_id=pid,
            source_url=url,
            verification_status=status,
            metadata={"signal_code": code, "prospect_id": prospect_id or getattr(signal, "prospect_id", "")},
        )

    def index_sales_playbook(self, name: str, markdown_content: str) -> int:
        """Index generic sales playbooks or agent skills as globally accessible knowledge."""
        return self.index_document(
            entity_type="sales_playbook",
            entity_id=f"playbook_{name}",
            text=markdown_content,
            product_id=None,
            is_global=True,
            metadata={"playbook_name": name},
        )

    def delete_document(self, entity_type: str, entity_id: str) -> int:
        """Delete all chunks for a given entity (triggers clean FTS)."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM knowledge_embeddings WHERE entity_type = ? AND entity_id = ?",
                    (entity_type, entity_id)
                )
                return cur.rowcount
        finally:
            conn.close()

    # ── Retrieval Engine ───────────────────────────────────────────────

    def search_dense(
        self,
        query: str,
        product_id: str | None = None,
        prospect_id: str | None = None,
        entity_type: str | None = None,
        top_k: int = 10,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Perform dense semantic vector search with strict product and prospect isolation."""
        if not self.is_semantic_available:
            return []

        query_vec = self.embed_text(query)
        conn = sqlite3.connect(self.db_path)
        candidates = []
        try:
            with conn:
                # Build SQL query with product isolation
                clauses = []
                params: list[Any] = []

                if product_id:
                    # Match specific product OR explicitly global documents
                    clauses.append("(product_id = ? OR json_extract(metadata_json, '$.is_global') = 1)")
                    params.append(product_id)
                else:
                    # If no product specified, match globally accessible or neutral documents
                    clauses.append("(product_id IS NULL OR product_id = '' OR json_extract(metadata_json, '$.is_global') = 1)")

                if entity_type:
                    clauses.append("entity_type = ?")
                    params.append(entity_type)

                where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                sql = f"""
                    SELECT id, product_id, entity_type, entity_id, chunk_text, embedding_json, metadata_json
                    FROM knowledge_embeddings
                    {where_sql}
                """

                rows = conn.execute(sql, params).fetchall()
                for row_id, prod_id, etype, eid, chunk_text, emb_str, meta_str in rows:
                    meta = json.loads(meta_str or "{}")

                    # Prospect-level isolation
                    if prospect_id and meta.get("prospect_id") and meta.get("prospect_id") != prospect_id:
                        continue

                    emb = json.loads(emb_str)
                    sim = self.cosine_similarity(query_vec, emb)
                    if sim >= min_similarity:
                        candidates.append({
                            "chunk_id": row_id,
                            "product_id": prod_id,
                            "entity_type": etype,
                            "entity_id": eid,
                            "chunk_text": chunk_text,
                            "similarity": round(float(sim), 4),
                            "metadata": meta,
                            "source_url": meta.get("source_url", ""),
                            "verification_status": meta.get("verification_status", "verified_observation"),
                        })
        finally:
            conn.close()

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    def search_lexical(
        self,
        query: str,
        product_id: str | None = None,
        prospect_id: str | None = None,
        entity_type: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword search using SQLite FTS5 with product isolation."""
        # Sanitize query for FTS5 syntax
        clean_terms = re.findall(r"\w+", query)
        if not clean_terms:
            return []
        fts_query = " OR ".join(clean_terms[:8])

        conn = sqlite3.connect(self.db_path)
        candidates = []
        try:
            with conn:
                clauses = ["knowledge_fts MATCH ?"]
                params: list[Any] = [fts_query]

                if product_id:
                    clauses.append("(k.product_id = ? OR k.product_id = '' OR k.product_id IS NULL)")
                    params.append(product_id)

                if entity_type:
                    clauses.append("k.entity_type = ?")
                    params.append(entity_type)

                where_sql = " AND ".join(clauses)
                sql = f"""
                    SELECT k.chunk_id, k.product_id, k.entity_type, k.entity_id, k.chunk_text, k.rank,
                           e.metadata_json
                    FROM knowledge_fts k
                    LEFT JOIN knowledge_embeddings e ON k.chunk_id = e.id
                    WHERE {where_sql}
                    ORDER BY k.rank
                    LIMIT ?
                """
                params.append(top_k * 2)

                rows = conn.execute(sql, params).fetchall()
                for cid, prod_id, etype, eid, chunk_text, rank, meta_str in rows:
                    meta = json.loads(meta_str or "{}")

                    # Product isolation check for FTS rows (verify metadata if global)
                    is_glob = bool(meta.get("is_global", False))
                    if product_id:
                        if prod_id and prod_id != product_id and not is_glob:
                            continue
                        if not prod_id and not is_glob:
                            continue
                    else:
                        if prod_id and not is_glob:
                            continue

                    if prospect_id and meta.get("prospect_id") and meta.get("prospect_id") != prospect_id:
                        continue

                    candidates.append({
                        "chunk_id": cid,
                        "product_id": prod_id,
                        "entity_type": etype,
                        "entity_id": eid,
                        "chunk_text": chunk_text,
                        "bm25_rank": float(rank),
                        "metadata": meta,
                        "source_url": meta.get("source_url", ""),
                        "verification_status": meta.get("verification_status", "verified_observation"),
                    })
                    if len(candidates) >= top_k:
                        break
        finally:
            conn.close()

        return candidates

    def search_hybrid(
        self,
        query: str,
        product_id: str | None = None,
        prospect_id: str | None = None,
        entity_type: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining Dense Semantic Vectors and Lexical FTS5 via Reciprocal Rank Fusion (RRF).

        If the neural embedding model is unavailable, gracefully falls back to lexical retrieval in degraded mode.
        """
        # 1. Run Dense Search
        dense_results = []
        semantic_ok = self.is_semantic_available
        if semantic_ok:
            try:
                dense_results = self.search_dense(
                    query=query,
                    product_id=product_id,
                    prospect_id=prospect_id,
                    entity_type=entity_type,
                    top_k=top_k * 2,
                    min_similarity=0.20,
                )
            except Exception as e:
                logger.warning(f"Dense search encountered error ({e}); falling back to lexical.")
                semantic_ok = False

        # 2. Run Lexical Search
        lexical_results = self.search_lexical(
            query=query,
            product_id=product_id,
            prospect_id=prospect_id,
            entity_type=entity_type,
            top_k=top_k * 2,
        )

        # If semantic is unavailable, return lexical results labeled degraded
        if not semantic_ok or not dense_results:
            results = []
            for r in lexical_results[:top_k]:
                item = dict(r)
                item["retrieval_mode"] = "lexical_only_degraded"
                item["score"] = round(1.0 / (60.0 + len(results) + 1), 4)
                results.append(item)
            return results

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = 1 / (60 + dense_rank) + 1 / (60 + lexical_rank)
        rrf_scores: dict[str, float] = {}
        items_by_id: dict[str, dict] = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            items_by_id[cid] = dict(item)
            items_by_id[cid]["dense_rank"] = rank

        for rank, item in enumerate(lexical_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            if cid not in items_by_id:
                items_by_id[cid] = dict(item)
            items_by_id[cid]["lexical_rank"] = rank

        fused = []
        for cid, score in rrf_scores.items():
            if score >= min_score:
                item = items_by_id[cid]
                item["score"] = round(score, 5)
                item["rrf_score"] = round(score, 5)
                item["retrieval_mode"] = "hybrid_rrf"
                fused.append(item)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    # Legacy alias compatibility
    def search(
        self,
        query: str,
        entity_type: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Legacy search interface preserved for backwards compatibility."""
        return self.search_dense(
            query=query,
            entity_type=entity_type,
            top_k=top_k,
            min_similarity=min_similarity,
        )


class RAGContextBuilder:
    """Constructs token-budgeted, evidence-grounded prompt context with prompt-injection defense."""

    def __init__(
        self,
        engine: LocalEmbeddingEngine,
        max_tokens: int = 1500,
    ):
        self.engine = engine
        self.max_tokens = max_tokens
        # Rough ratio: 1 token ≈ 4 characters
        self.max_chars = max_tokens * 4

    def build_context(
        self,
        query: str,
        product_id: str | None = None,
        prospect_id: str | None = None,
        entity_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve relevant knowledge chunks and construct a bounded context block with provenance and citations."""
        from sellbird.untrusted import UNTRUSTED_RULE, fence

        chunks = self.engine.search_hybrid(
            query=query,
            product_id=product_id,
            prospect_id=prospect_id,
            entity_type=entity_type,
            top_k=top_k,
        )

        if not chunks:
            return {
                "context_block": "",
                "citations": [],
                "chunks": [],
                "has_untrusted": False,
            }

        context_lines = ["### GROUNDED SALES & PRODUCT KNOWLEDGE"]
        citations = []
        has_untrusted = False
        current_chars = len(context_lines[0])

        seen_texts = set()

        for idx, chunk in enumerate(chunks, start=1):
            text = chunk["chunk_text"].strip()
            # Deduplicate nearly identical chunks
            norm_key = re.sub(r"\s+", " ", text.lower())[:80]
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)

            etype = chunk.get("entity_type", "knowledge")
            url = chunk.get("source_url", "")
            status = chunk.get("verification_status", "verified_observation")
            source_label = f"Source {idx}: [{etype}] {url or chunk.get('product_id', '')}".strip()

            # Untrusted external web content must be fenced
            is_untrusted = etype in ("web_research", "ats_posting", "email_inbound")
            if is_untrusted:
                has_untrusted = True
                formatted_chunk = fence(text, source=source_label)
            else:
                formatted_chunk = f"[{source_label} - {status}]\n{text}"

            if current_chars + len(formatted_chunk) > self.max_chars:
                break

            context_lines.append(formatted_chunk)
            current_chars += len(formatted_chunk) + 2

            citations.append({
                "source": url or etype,
                "entity_type": etype,
                "quote": text[:150],
                "verification_status": status,
                "rrf_score": chunk.get("score", 0.0),
            })

        if has_untrusted:
            context_lines.append(f"\n{UNTRUSTED_RULE}")

        return {
            "context_block": "\n\n".join(context_lines),
            "citations": citations,
            "chunks": chunks,
            "has_untrusted": has_untrusted,
        }
