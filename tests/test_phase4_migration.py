"""Tests for Safe Database Migration v19 and FTS5 Auto-Sync.

Verifies:
- Safe migration from fresh DB to v19
- Upgrade of existing version 18 database to v19
- Safe migration when knowledge_embeddings already pre-existed without product_id column
- Preserves all existing records and backfills product_id from metadata_json
- FTS5 table creation and auto-sync triggers on INSERT, UPDATE, and DELETE
"""

import os
import sqlite3
import tempfile
import pytest

from sellbird.state import StateManager, MIGRATIONS


@pytest.mark.asyncio
async def test_fresh_database_migration_to_v19():
    """Verify that a fresh database adopts all migrations cleanly up to v19."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        state = StateManager(db_path=temp_db)
        await state.init_db()

        async with state._connect() as db:
            cur = await db.execute("PRAGMA user_version")
            v = (await cur.fetchone())[0]
            assert v == len(MIGRATIONS)
            assert v == 19

            # Verify knowledge_embeddings schema
            cur = await db.execute("PRAGMA table_info(knowledge_embeddings)")
            cols = {r[1]: r[2] for r in await cur.fetchall()}
            assert "id" in cols
            assert "product_id" in cols
            assert "entity_type" in cols
            assert "chunk_text" in cols
            assert "embedding_json" in cols

            # Verify knowledge_fts virtual table exists
            cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_fts'")
            fts = await cur.fetchone()
            assert fts is not None
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


@pytest.mark.asyncio
async def test_migration_from_v18_to_v19():
    """Verify that an existing version 18 database upgrades cleanly to v19 without data loss."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        # Pre-seed database at version 18 using synchronous sqlite3
        conn = sqlite3.connect(temp_db)
        with conn:
            # Execute migrations 1 to 18
            for script in MIGRATIONS[:18]:
                conn.executescript(script)
            conn.execute("PRAGMA user_version = 18")

            # Insert sample product
            conn.execute("INSERT INTO products (id, name) VALUES ('prod_01', 'Test Product')")
        conn.close()

        # Run StateManager.init_db to perform v19 migration
        state = StateManager(db_path=temp_db)
        await state.init_db()

        async with state._connect() as db:
            cur = await db.execute("PRAGMA user_version")
            assert (await cur.fetchone())[0] == 19

            # Verify product still exists
            cur = await db.execute("SELECT name FROM products WHERE id = 'prod_01'")
            row = await cur.fetchone()
            assert row and row[0] == "Test Product"

            # Verify knowledge_embeddings is present with product_id
            cur = await db.execute("PRAGMA table_info(knowledge_embeddings)")
            cols = [r[1] for r in await cur.fetchall()]
            assert "product_id" in cols
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


@pytest.mark.asyncio
async def test_migration_with_preexisting_knowledge_embeddings_without_product_id():
    """Verify that if knowledge_embeddings existed before migration without product_id, it is preserved and backfilled."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        # Seed DB at v18 with old knowledge_embeddings schema
        conn = sqlite3.connect(temp_db)
        with conn:
            for script in MIGRATIONS[:18]:
                conn.executescript(script)
            conn.execute("PRAGMA user_version = 18")

            # Create old-style knowledge_embeddings table (no product_id column)
            conn.execute("""
                CREATE TABLE knowledge_embeddings (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert pre-existing row with product_id inside metadata_json
            conn.execute(
                "INSERT INTO knowledge_embeddings (id, entity_type, entity_id, chunk_text, embedding_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                ("old_chk_1", "product_doc", "p1", "Automated database migration and failover", "[]", '{"product_id": "prod_legacy_01"}')
            )
        conn.close()

        # Execute migration
        state = StateManager(db_path=temp_db)
        await state.init_db()

        async with state._connect() as db:
            # 1. Verify user_version is 19
            cur = await db.execute("PRAGMA user_version")
            assert (await cur.fetchone())[0] == 19

            # 2. Verify product_id column was added and backfilled
            cur = await db.execute("SELECT id, product_id, chunk_text FROM knowledge_embeddings WHERE id = 'old_chk_1'")
            row = await cur.fetchone()
            assert row is not None
            assert row[1] == "prod_legacy_01", f"Expected product_id to be backfilled, got {row[1]}"

            # 3. Verify existing row was backfilled into knowledge_fts
            cur = await db.execute("SELECT chunk_id, product_id FROM knowledge_fts WHERE knowledge_fts MATCH 'migration'")
            fts_row = await cur.fetchone()
            assert fts_row is not None
            assert fts_row[0] == "old_chk_1"
            assert fts_row[1] == "prod_legacy_01"

            # 4. Verify triggers synchronize subsequent mutations
            # A. Insert new row
            await db.execute(
                "INSERT INTO knowledge_embeddings (id, product_id, entity_type, entity_id, chunk_text, embedding_json) VALUES (?, ?, ?, ?, ?, ?)",
                ("new_chk_2", "prod_new", "product_doc", "p2", "Real-time telemetry and metrics streaming", "[]")
            )
            await db.commit()
            cur = await db.execute("SELECT chunk_id FROM knowledge_fts WHERE knowledge_fts MATCH 'telemetry'")
            assert (await cur.fetchone())[0] == "new_chk_2"

            # B. Update row
            await db.execute("UPDATE knowledge_embeddings SET chunk_text = 'High-frequency telemetry log streaming' WHERE id = 'new_chk_2'")
            await db.commit()
            cur = await db.execute('SELECT chunk_id FROM knowledge_fts WHERE knowledge_fts MATCH \'"high-frequency"\'')
            assert (await cur.fetchone())[0] == "new_chk_2"

            # C. Delete row
            await db.execute("DELETE FROM knowledge_embeddings WHERE id = 'new_chk_2'")
            await db.commit()
            cur = await db.execute("SELECT COUNT(*) FROM knowledge_fts WHERE chunk_id = 'new_chk_2'")
            assert (await cur.fetchone())[0] == 0
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
