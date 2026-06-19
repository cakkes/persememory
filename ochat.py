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
