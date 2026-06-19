#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests"]
# ///
"""ochat: persistent-memory terminal chat for Ollama."""

import argparse
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "gemma4:12b"
EMBED_MODEL = "nomic-embed-text"

DATA_DIR = Path.home() / ".local" / "share" / "ochat"
THREADS_DIR = DATA_DIR / "threads"
MEMORY_DB_PATH = DATA_DIR / "memory.db"
EXTRACTION_LOG_PATH = DATA_DIR / "extraction.log"

CONTEXT_TOKEN_BUDGET = 8192
CHARS_PER_TOKEN = 4
RETRIEVAL_TOP_K = 8
RETRIEVAL_MIN_SIMILARITY = 0.45
DEDUP_SIMILARITY_THRESHOLD = 0.92
DEFAULT_THINK = "off"
EXTRACTION_JOIN_TIMEOUT_SECONDS = 5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_facts(query_embedding, facts, k=RETRIEVAL_TOP_K, min_similarity=RETRIEVAL_MIN_SIMILARITY):
    scored = []
    for fact in facts:
        similarity = cosine_similarity(query_embedding, fact["embedding"])
        if similarity >= min_similarity:
            scored.append((similarity, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]


def is_duplicate_fact(candidate_embedding, existing_embeddings, threshold=DEDUP_SIMILARITY_THRESHOLD):
    for embedding in existing_embeddings:
        if cosine_similarity(candidate_embedding, embedding) > threshold:
            return True
    return False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_messages_to_budget(messages, budget_tokens=CONTEXT_TOKEN_BUDGET):
    selected = []
    used = 0
    for message in reversed(messages):
        cost = estimate_tokens(message["content"])
        if selected and used + cost > budget_tokens:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return selected


def thread_path(name: str) -> Path:
    return THREADS_DIR / f"{name}.json"


def load_thread(path: Path, name: str) -> dict:
    if not path.exists():
        return {"name": name, "messages": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            raise ValueError("missing 'messages' key")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
        path.rename(corrupt_path)
        print(
            f"warning: {path.name} was corrupt ({exc}); moved to "
            f"{corrupt_path.name} and starting a fresh thread",
            file=sys.stderr,
        )
        return {"name": name, "messages": []}


def save_thread(path: Path, thread: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(thread, f, indent=2)
    os.replace(tmp_path, path)


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source_thread TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def insert_fact(conn, text, embedding, source_thread):
    conn.execute(
        "INSERT INTO facts (text, embedding, source_thread, created_at) VALUES (?, ?, ?, ?)",
        (
            text,
            np.asarray(embedding, dtype=np.float32).tobytes(),
            source_thread,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_all_facts(conn):
    rows = conn.execute(
        "SELECT id, text, embedding, source_thread, created_at FROM facts"
    ).fetchall()
    return [
        {
            "id": row[0],
            "text": row[1],
            "embedding": np.frombuffer(row[2], dtype=np.float32),
            "source_thread": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def delete_fact(conn, fact_id):
    cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()
    return cursor.rowcount > 0
